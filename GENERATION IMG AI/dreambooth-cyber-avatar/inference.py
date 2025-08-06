# Génération d'une image avec le modèle entraîné
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

prompt = "portrait of a cyberpunk character with neon lights, anime male caracter , digital background"
image = pipe(prompt).images[0]
image.save("output.png")
