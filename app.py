import gradio as gr
import modin.pandas as pd
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16) if torch.cuda.is_available() else AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe = pipe.to(device)

def resize(value,img):
    img = Image.open(img)
    img = img.resize((value,value))
    return img

def infer(source_img, prompt, steps, seed, Strength):
    generator = torch.Generator(device).manual_seed(seed)  
    if int(steps * Strength) < 1:
        steps = math.ceil(1 / max(0.10, Strength))
    source_image = resize(512, source_img)
    source_image.save('source.png')
    image = pipe(prompt, image=source_image, strength=Strength, guidance_scale=0.0, num_inference_steps=steps).images[0]
    return image

gr.Interface(fn=infer, inputs=[
    gr.Image(sources=["upload", "webcam", "clipboard"], type="filepath", label="Raw Image."), 
    gr.Textbox(label = 'Prompt Input Text. 77 Token (Keyword or Symbol) Maximum'),
    gr.Slider(1, 5, value = 2, step = 1, label = 'Number of Iterations'),
    gr.Slider(label = "Seed", minimum = 0, maximum = 987654321987654321, step = 1, randomize = True), 
    gr.Slider(label='Strength', minimum = 0.1, maximum = 1, step = .05, value = .5)], 
    outputs='image', title = "Stable Diffusion XL Turbo Image to Image Pipeline CPU", description = "For more information on Stable Diffusion XL Turbo see https://huggingface.co/stabilityai/sdxl-turbo <br><br>Upload an Image, Use your Cam, or Paste an Image. Then enter a Prompt, or let it just do its Thing, then click submit. For more informationon about Stable Diffusion or Suggestions for prompts, keywords, artists or styles see https://github.com/Maks-s/sd-akashic", 
    article = "Code Monkey: <a href=\"https://huggingface.co/Manjushri\">Manjushri</a>").queue(max_size=10).launch()