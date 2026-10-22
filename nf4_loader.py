import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from . import native_backend as NB

FORMAT = "vathography-nf4-v1"
MARKER = NB.SEE_THROUGH_MARKER


def nf4_decode(packed, absmax, quant_map, shape, blocksize):
    """bitsandbytes nf4 decoding in plain torch (no bitsandbytes needed): two 4-bit indices per byte, high nibble
    first, looked up in the 16-value table, times the block's fp32 absmax, then bf16. Bit-identical to
    bitsandbytes' dequantize_4bit on every layer of the shipped file."""
    p = packed.view(-1)
    idx = torch.stack(((p >> 4) & 0xF, p & 0xF), dim=1).view(-1).long()
    return (quant_map[idx].view(-1, blocksize) * absmax.view(-1, 1)).view(shape).to(torch.bfloat16)


class NF4Linear(torch.nn.Module):
    """A Linear layer of the see-through backbone: nf4 weight (or plain bf16 for the two layers kept unquantised),
    the see-through LoRA added unmerged. Same computation as the Marigold V2 reference (bitsandbytes Linear4bit +
    PEFT LoRA): input to bf16, weight decoded to bf16, linear, + LoRA in fp32, output back to the input dtype.
    The tensors are parameters so ComfyUI counts, loads and unloads them; a layer left in RAM by a partial load
    moves its tensors to the input's device on each call."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        for name in ("packed", "absmax", "quant_map", "weight", "bias", "lora_A", "lora_B"):
            self.register_parameter(name, None)
        self.shape, self.blocksize = (out_features, in_features), None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        p = lambda t: torch.nn.Parameter(t, requires_grad=False)
        if prefix + "weight.absmax" in state_dict:
            qs = json.loads(bytes(state_dict.pop(prefix + "weight.quant_state.bitsandbytes__nf4").tolist()).decode("utf-8"))
            assert qs["quant_type"] == "nf4" and qs["dtype"] == "bfloat16", (prefix, qs)
            self.shape, self.blocksize = tuple(qs["shape"]), int(qs["blocksize"])
            self.packed = p(state_dict.pop(prefix + "weight"))
            self.absmax = p(state_dict.pop(prefix + "weight.absmax"))
            self.quant_map = p(state_dict.pop(prefix + "weight.quant_map"))
        elif prefix + "weight" in state_dict:
            self.weight = p(state_dict.pop(prefix + "weight"))
        else:
            missing_keys.append(prefix + "weight")
        if prefix + "bias" in state_dict:
            self.bias = p(state_dict.pop(prefix + "bias"))

    def forward(self, x):
        dev, xb = x.device, x.to(torch.bfloat16)
        if self.packed is not None:
            w = nf4_decode(self.packed.to(dev), self.absmax.to(dev), self.quant_map.to(dev), self.shape, self.blocksize)
        else:
            w = self.weight.to(dev, torch.bfloat16)
        out = F.linear(xb, w, None if self.bias is None else self.bias.to(dev, torch.bfloat16))
        if self.lora_A is not None:
            out = out + ((xb.float() @ self.lora_A.to(dev).float().t()) @ self.lora_B.to(dev).float().t()).to(out.dtype)
        return out.to(x.dtype)


def _lora_is_see_through(lora):
    t = lora.get("diffusion_model.img_in.lora_B.weight")
    if t is None:
        return False
    return abs(float(t.float().abs().sum()) - NB.LORA_FINGERPRINTS["see-through"]) < 0.05


class VathographySeeThroughLoader:
    CATEGORY = "image/geometry estimation"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The see-through model, ready for 'ModelSamplingAuraFlow' and the Marigold V2 Tiled Depth node.",)
    DESCRIPTION = ("Loads the Marigold V2 see-through model (depth behind glass) the way it was trained: the 4-bit (nf4) "
                   "Qwen-Image-Edit-2509 backbone plus the see-through LoRA. Replaces 'Load Diffusion Model' and "
                   "'LoraLoaderModelOnly' in the see-through workflow. No extra Python package needed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "backbone": (folder_paths.get_filename_list("diffusion_models"),
                         {"tooltip": "marigold_v2_see_through_qwen_image_edit_2509_nf4.safetensors (models/diffusion_models)."}),
            "lora": (folder_paths.get_filename_list("loras"),
                     {"tooltip": "marigold_v2_depth_see_through.safetensors, the see-through LoRA (models/loras)."}),
        }}

    def load(self, backbone, lora):
        path = folder_paths.get_full_path_or_raise("diffusion_models", backbone)
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        if meta.get("format") != FORMAT or meta.get("double_quant") != "false" or "unet_config" not in meta:
            raise ValueError(f"'{backbone}' is not the see-through nf4 backbone. Choose "
                             "marigold_v2_see_through_qwen_image_edit_2509_nf4.safetensors.")
        lsd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora), safe_load=True)
        if not _lora_is_see_through(lsd):
            raise ValueError(f"'{lora}' is not the see-through LoRA. Choose marigold_v2_depth_see_through.safetensors "
                             "(the regular depth model uses the int8 backbone and the template's loaders instead).")

        unet_config = json.loads(meta["unet_config"])
        model_config = comfy.model_detection.model_config_from_unet_config(unet_config)
        if model_config is None:
            raise RuntimeError("this ComfyUI version does not recognise the stored Qwen-Image model settings")
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        manual_cast = comfy.model_management.unet_manual_cast(torch.bfloat16, load_device, model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(torch.bfloat16, manual_cast, device=load_device)
        base_ops = comfy.ops.pick_operations(torch.bfloat16, manual_cast, model_config=model_config)

        class NF4Ops(base_ops):
            Linear = NF4Linear

        model_config.custom_operations = NF4Ops
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = model_config.get_model(sd, "")
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        model.load_model_weights(sd, "")
        mods = dict(model.diffusion_model.named_modules())
        n_l = 0
        for k in [k for k in lsd if k.endswith(".lora_A.weight")]:
            name = k[len("diffusion_model."):-len(".lora_A.weight")]
            m = mods.get(name)
            if not isinstance(m, NF4Linear):
                raise ValueError(f"the LoRA targets '{name}', which is not a Linear layer of this backbone")
            m.lora_A = torch.nn.Parameter(lsd[k], requires_grad=False)
            m.lora_B = torch.nn.Parameter(lsd[k.replace(".lora_A.", ".lora_B.")], requires_grad=False)
            n_l += 1
        bad = [n for n, m in mods.items() if isinstance(m, NF4Linear) and m.packed is None and m.weight is None]
        if bad:
            raise ValueError(f"the backbone file lacks weights for {len(bad)} layers (e.g. {bad[0]})")
        model.vathography_model = MARKER
        n_q = sum(1 for m in mods.values() if isinstance(m, NF4Linear) and m.packed is not None)
        print(f"[tiled] see-through loader: {n_q} nf4 layers, LoRA on {n_l} layers, backbone {backbone}", flush=True)
        return (patcher,)


NODE_CLASS_MAPPINGS = {"VathographySeeThroughLoader": VathographySeeThroughLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"VathographySeeThroughLoader": "Marigold V2 See-through Loader - Vathography"}
