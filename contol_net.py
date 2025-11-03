import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
from controlnet_aux import CannyDetector
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

controlnet_model_path = "models/sd-controlnet-canny"
sd_model_path = "models/stable-diffusion-v1-5"

os.makedirs(controlnet_model_path, exist_ok=True)
os.makedirs(sd_model_path, exist_ok=True)

if device == "cuda":
    dtype = torch.float16
else:
    dtype = torch.float32

if os.listdir(controlnet_model_path):
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=dtype)
else:
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
    controlnet.save_pretrained(controlnet_model_path)

if os.listdir(sd_model_path):
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_path, controlnet=controlnet, torch_dtype=dtype
    )
else:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype
    )
    pipe.save_pretrained(sd_model_path)

pipe.to(device)
pipe.safety_checker = lambda images, **kwargs: (images, [False]*len(images))

img_path = "data/noibat.png"
assert os.path.exists(img_path), f"File not found: {img_path}"

img = Image.open(img_path).convert("RGB")
img = img.resize((512, 512))  # todo resize later

canny = CannyDetector()
control_image = canny(np.array(img))

prompt = (
    "keep the original colors, cinematic, photo of a bat, film look, realistic lighting, "
    "sharp focus, 35mm lens, soft shadows, detailed texture "
)

num_inference_steps = 50   # balance between speed and quality
guidance_scale = 2.5       # strong guidance
strength = 0.1          # keep original structure

if device == "cuda":
    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            image=img,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
else:
    result = pipe(
        prompt=prompt,
        image=img,
        control_image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    ).images[0]

output_path = "output.png"
result.save(output_path)
print(f"Image saved as {output_path}")
