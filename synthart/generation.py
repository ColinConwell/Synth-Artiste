"""Text, image, and adapter conditioning behind one small generate interface."""
import base64
import io
import os
from pathlib import Path
from PIL import Image
from .images import as_image, device_name

OPENAI_MODEL = "gpt-image-2.5-flare"
GEMINI_MODEL = "gemini-3.1-flash-image"
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class ImageGenerator:
    """Return a PIL image. API seeds are unsupported and are rejected explicitly.

    Pass reference for native API editing or SD image-to-image. With ip_adapter=True,
    reference instead becomes an image prompt through learned cross-attention.
    """
    def __init__(self, backend="openai", *, model=None, device=None,
                 ip_adapter=False, lora=None, lora_weight=None, lora_scale=1.0):
        if backend not in {"openai", "gemini", "diffusers"}:
            raise ValueError("Choose openai, gemini, or diffusers")
        if backend != "diffusers" and (ip_adapter or lora):
            raise ValueError("Local adapters require the diffusers backend")
        self.backend = backend
        self.model_id = model or {"openai": OPENAI_MODEL, "gemini": GEMINI_MODEL,
                                  "diffusers": SD_MODEL}[backend]
        self.ip_adapter = ip_adapter
        self.lora = lora
        self.lora_weight, self.lora_scale = lora_weight, lora_scale
        if backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(timeout=240, max_retries=2)
        elif backend == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        else:
            import torch
            from diffusers import AutoPipelineForText2Image
            self.device = device_name(device)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
            if ip_adapter:
                xl = "StableDiffusionXL" in type(self.pipe).__name__
                self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models" if xl else "models",
                    weight_name="ip-adapter_sdxl.bin" if xl else "ip-adapter_sd15.bin")
            if lora:
                kwargs = {"weight_name": lora_weight} if lora_weight else {}
                self.pipe.load_lora_weights(lora, adapter_name="style", **kwargs)
                self.pipe.set_adapters("style", adapter_weights=lora_scale)
                if lora.startswith("latent-consistency/lcm-lora"):
                    from diffusers import LCMScheduler
                    self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

    def generate(self, prompt, *, reference=None, seed=None, steps=25,
                 strength=0.6, guidance=7.5, size=512, quality="low"):
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        if self.backend != "diffusers" and seed is not None:
            raise ValueError("Hosted image APIs do not offer a reproducible seed here")
        if self.backend == "openai":
            kwargs = dict(model=self.model_id, prompt=prompt, size="1024x1024", quality=quality)
            if reference is None:
                response = self.client.images.generate(**kwargs)
            else:
                buffer = io.BytesIO()
                as_image(reference).save(buffer, format="PNG")
                response = self.client.images.edit(image=("reference.png", buffer.getvalue(), "image/png"), **kwargs)
            if not response.data or not response.data[0].b64_json:
                raise RuntimeError("OpenAI returned no image")
            return as_image(Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))))
        if self.backend == "gemini":
            from google.genai import types
            contents = [prompt] if reference is None else [prompt, as_image(reference)]
            response = self.client.models.generate_content(model=self.model_id, contents=contents,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]))
            for part in response.parts or []:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    return as_image(Image.open(io.BytesIO(part.inline_data.data)))
            raise RuntimeError("Gemini returned no image (inspect provider safety or quota status)")
        import torch
        if steps < 1 or size < 64 or size % 8:
            raise ValueError("steps must be positive; size must be a multiple of 8, at least 64")
        if not 0 < strength <= 1:
            raise ValueError("strength must be in (0, 1]")
        kwargs = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                      height=size, width=size)
        if seed is not None:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        pipe = self.pipe
        if self.ip_adapter:
            if reference is None:
                raise ValueError("IP-Adapter requires a reference image")
            pipe.set_ip_adapter_scale(strength)
            kwargs["ip_adapter_image"] = as_image(reference)
        elif reference is not None:
            if int(steps * strength) < 1:
                raise ValueError("steps * strength must permit at least one denoising step")
            from diffusers import AutoPipelineForImage2Image
            pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
            kwargs.update(image=as_image(reference).resize((size, size)), strength=strength)
        return pipe(**kwargs).images[0]

    def describe(self):
        return {"backend": self.backend, "model": self.model_id,
                "ip_adapter": self.ip_adapter, "lora": self.lora,
                "lora_weight": self.lora_weight, "lora_scale": self.lora_scale}
