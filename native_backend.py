import logging

import numpy as np
import torch

import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
from comfy_extras.nodes_custom_sampler import Guider_Basic

SIGMAS = (0.5, 0.0)            # = the template's ManualSigmas "0.5, 0"
SAMPLER = "euler"              # = the template's KSamplerSelect
VAE_FALLBACK_TEXT = "Ran out of memory when regular VAE"   # core's own warning, comfy/sd.py (encode and decode)


class _FallbackWatch(logging.Handler):

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.hits = []

    def emit(self, record):
        msg = record.getMessage()
        if VAE_FALLBACK_TEXT in msg:
            self.hits.append(msg)


def check_model(model):
    """The template's ModelSamplingAuraFlow(sampling = img_to_img_velocity) step must have been applied: without it
    the one-step formula is wrong and the map is garbage. Returns the sampling object's class name."""
    ms = model.get_model_object("model_sampling")
    if not isinstance(ms, comfy.model_sampling.IMG_TO_IMG_VELOCITY):
        raise ValueError(
            f"the MODEL's sampling is {type(ms).__name__}, not IMG_TO_IMG_VELOCITY. Wire it through "
            "'ModelSamplingAuraFlow' with sampling = img_to_img_velocity (shift 1.73), as in the Marigold V2 template.")
    return type(ms).__name__


# Fingerprints of the two Marigold V2 depth models, so the node applies the matching recipe by itself.
# LoRA: sum of |values| of `img_in`'s lora_B (float32); VAE: the decoder output bias (`decoder.head.2.bias`).
LORA_KEY = "diffusion_model.img_in.weight"
LORA_FINGERPRINTS = {"depth": 3005.3411, "see-through": 3126.4729}
VAE_KEY = "decoder.head.2.bias"
VAE_FINGERPRINTS = {"depth": (0.00994873046875, 0.0007476806640625),
                    "see-through": (0.010009765625, 0.000492095947265625)}
MODEL_FILES = {"depth": "marigold_v2_depth_log_stage2", "see-through": "marigold_v2_depth_see_through"}
SEE_THROUGH_MARKER = "see-through-nf4"     # set on the model by the See-through Loader


def _lora_family(model):
    for patch in model.patches.get(LORA_KEY, []):
        weights = getattr(patch[1], "weights", None)
        if not weights or not torch.is_tensor(weights[0]):
            continue
        s = float(weights[0].float().abs().sum())
        for name, ref in LORA_FINGERPRINTS.items():
            if abs(s - ref) < 0.05:
                return name
    return None


def _vae_family(vae):
    try:
        b = vae.first_stage_model.state_dict()[VAE_KEY].float().cpu()
    except (AttributeError, KeyError):
        return None
    for name, ref in VAE_FINGERPRINTS.items():
        if abs(float(b[0]) - ref[0]) < 1e-6 and abs(float(b[1]) - ref[1]) < 1e-6:
            return name
    return None


def identify_model(model, vae):
    """Which Marigold V2 depth model is wired: returns (recipe, warnings). A LoRA and a VAE from different models
    is refused (the map would be wrong without any visible error); an unknown LoRA falls back to the depth recipe."""
    lora, vf = _lora_family(model), _vae_family(vae)
    warnings = []
    if getattr(model.model, "vathography_model", None) == SEE_THROUGH_MARKER:
        if model.patches.get(LORA_KEY):
            raise ValueError("a LoRA loader was added after the See-through Loader, which already applies the "
                             "see-through LoRA. Remove the extra LoRA loader.")
        if vf and vf != "see-through":
            raise ValueError(f"the See-through Loader is wired with the {vf} model's VAE ({MODEL_FILES[vf]}_vae). "
                             "Load marigold_v2_depth_see_through_vae.safetensors.")
        if vf is None:
            warnings.append("the VAE is not one of the two Marigold V2 depth VAEs this node knows; the see-through recipe is used")
        return "see-through", warnings
    if lora and vf and lora != vf:
        raise ValueError(
            f"the LoRA is the {lora} model ({MODEL_FILES[lora]}) but the VAE is the {vf} model ({MODEL_FILES[vf]}_vae). "
            "Load the LoRA and the VAE of the same model.")
    if lora is None:
        warnings.append("the LoRA is not one of the two Marigold V2 depth models this node knows: the depth recipe is used"
                        + (f" (the VAE is the {vf} model)" if vf else ""))
        return "depth", warnings
    if vf is None:
        warnings.append(f"the VAE is not one of the two Marigold V2 depth VAEs this node knows; the {lora} recipe is used")
    if lora == "see-through":
        warnings.append("the see-through LoRA runs on a non-nf4 backbone: it loses part of the depth behind glass. "
                        "Load it with the 'Marigold V2 See-through Loader' instead")
    return lora, warnings


def to_image(bgr_u8):
    """uint8 BGR (H, W, 3) -> IMAGE tensor [1, H, W, 3] float32 0..1, RGB."""
    rgb = bgr_u8[:, :, ::-1]                                   # channel flip (= cv2.cvtColor BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]   # astype copies -> positive strides


def run(model, vae, conditioning, inputs, log=None, progress=None):
    """inputs = list of uint8 BGR arrays whose H and W are multiples of 16. Returns (list of float32 (H, W) raw
    predictions, info dict). progress(step, total) is called after every encode / step / decode."""
    def say(m):
        if log is not None:
            log(m)

    for k, x in enumerate(inputs):
        assert x.dtype == np.uint8 and x.ndim == 3 and x.shape[2] == 3, (k, x.dtype, x.shape)
        assert x.shape[0] % 16 == 0 and x.shape[1] % 16 == 0, f"input {k} is {x.shape[1]}x{x.shape[0]}, not /16"
    n = len(inputs); total = 3 * n; done = 0
    watch = _FallbackWatch()
    root = logging.getLogger()
    info = dict(n=n, encode_fallback=0, decode_fallback=0)

    # ---- 1. encode all
    latents = []
    root.addHandler(watch)
    try:
        for k, x in enumerate(inputs):
            comfy.model_management.throw_exception_if_processing_interrupted()
            before = len(watch.hits)
            lat = vae.encode(to_image(x))
            info["encode_fallback"] += len(watch.hits) - before
            latents.append(lat.to(comfy.model_management.intermediate_device()))
            done += 1
            if progress: progress(done, total)
            if k == 0:
                say(f"[native] encode: input {x.shape[1]}x{x.shape[0]} -> latent {tuple(lat.shape)} {lat.dtype}")
    finally:
        root.removeHandler(watch)
    say(f"[native] encoded {n} input(s); VAE tiled-encode fallbacks: {info['encode_fallback']}")

    # ---- 2. one sampler step each (= SamplerCustomAdvanced with DisableNoise / BasicGuider / euler / 0.5, 0)
    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    sampler = comfy.samplers.sampler_object(SAMPLER)
    sigmas = torch.FloatTensor(list(SIGMAS))
    samples = []
    for k, lat in enumerate(latents):
        comfy.model_management.throw_exception_if_processing_interrupted()
        lat = comfy.sample.fix_empty_latent_channels(guider.model_patcher, lat)
        noise = comfy.sample.prepare_empty_noise(lat)
        out = guider.sample(noise, lat, sampler, sigmas, denoise_mask=None, callback=None,
                            disable_pbar=True, seed=0)
        samples.append(out.to(comfy.model_management.intermediate_device()))
        done += 1
        if progress: progress(done, total)
    del latents
    say(f"[native] sampled {n} latent(s): {SAMPLER}, sigmas {SIGMAS}, no noise")

    # ---- 3. decode all
    preds = []
    root.addHandler(watch)
    try:
        for k, s in enumerate(samples):
            comfy.model_management.throw_exception_if_processing_interrupted()
            before = len(watch.hits)
            img = vae.decode(s)
            info["decode_fallback"] += len(watch.hits) - before
            if img.ndim == 5:
                img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
            raw = (2.0 * img[0].float() - 1.0).mean(dim=-1).cpu().numpy().astype(np.float32)
            preds.append(raw)
            done += 1
            if progress: progress(done, total)
            if k == 0:
                say(f"[native] decode: latent {tuple(s.shape)} -> raw {raw.shape[1]}x{raw.shape[0]} "
                    f"range {raw.min():.4f}..{raw.max():.4f} | IMAGE dtype {img.dtype}")
    finally:
        root.removeHandler(watch)
    say(f"[native] decoded {n} prediction(s); VAE tiled-decode fallbacks: {info['decode_fallback']}")
    return preds, info
