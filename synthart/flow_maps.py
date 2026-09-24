"""A real pretrained flow map with a small inference-only setup.

DMF predicts interval-average velocity u(x,t,r), unlike instantaneous velocity
v(x,t) in flow matching. x_r = x_t + (r-t)u(x_t,t,r) can span a large interval.
This class is ImageNet-conditioned; it does not implement FMRG reward guidance.
"""
from PIL import Image
from .images import device_name


class FlowMapGenerator:
    def __init__(self, *, device=None):
        import torch
        from huggingface_hub import hf_hub_download
        from diffusers import AutoencoderKL
        from ._dmf_loader import load_architecture
        DMFT_models = load_architecture()
        self.device = device_name(device)
        self.model_id = "kyungmnlee/DMF"
        self.dtype = torch.float32
        self.model = DMFT_models["DMFT-XL/2"](input_size=32, num_classes=1000,
            use_cfg_embedding=True, use_logvar=True, dmf_depth=20, attn_func="torch_sdpa", qk_norm=False)
        checkpoint = hf_hub_download(self.model_id, "dmf_xl_2_256.pt")
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device).eval()

    def generate(self, prompt="", *, class_id=207, seed=0, steps=4):
        """Generate a 256px ImageNet class (207 = golden retriever); text is unsupported."""
        import torch
        if prompt:
            raise ValueError("DMF uses ImageNet class_id, not a text prompt")
        if not 0 <= class_id < 1000 or steps < 1:
            raise ValueError("class_id must be 0..999 and steps must be positive")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(1, 4, 32, 32, generator=rng).to(self.device)
        y = torch.tensor([class_id], device=self.device)
        times = torch.linspace(1, 0, steps + 1, device=self.device)
        with torch.inference_mode():
            for t, r in zip(times[:-1], times[1:]):
                velocity = self.model(x, t[None], r[None], y)
                x = x + (r - t) * velocity
            pixels = (self.vae.decode(x / 0.18215).sample[0] + 1) / 2
        array = pixels.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        return Image.fromarray(array)

    def describe(self):
        return {"backend": "dmf", "model": self.model_id, "checkpoint": "dmf_xl_2_256.pt"}
