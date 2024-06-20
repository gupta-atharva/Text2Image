# from flask import Flask, request, render_template
# from PIL import Image
# import io
# import base64
# import torch
# from transformers import CLIPTokenizer
# import model_loader
# import pipeline

# app = Flask(__name__)

# DEVICE = "cpu"
# tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
# model_file = "../data/v1-5-pruned-emaonly.ckpt"
# models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         prompt = request.form["prompt"]
#         uncond_prompt = ""  # Also known as negative prompt
#         do_cfg = True
#         cfg_scale = 8  # min: 1, max: 14
#         strength = 0.9

#         # SAMPLER
#         sampler = "ddpm"
#         num_inference_steps = 50
#         seed = 42

#         output_image = pipeline.generate(
#             prompt=prompt,
#             uncond_prompt=uncond_prompt,
#             input_image=None,
#             strength=strength,
#             do_cfg=do_cfg,
#             cfg_scale=cfg_scale,
#             sampler_name=sampler,
#             n_inference_steps=num_inference_steps,
#             seed=seed,
#             models=models,
#             device=DEVICE,
#             idle_device="cpu",
#             tokenizer=tokenizer,
#         )

#         image = Image.fromarray(output_image)
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()

#         return render_template("index.html", img_data=img_str)

#     return render_template("index.html", img_data=None)

# if __name__ == "__main__":
#     app.run(debug=False)


import streamlit as st
from PIL import Image
import model_loader
import pipeline
from transformers import CLIPTokenizer
import torch
import numpy as np

DEVICE = "cpu"
MODEL_FILE = "../data/v1-5-pruned-emaonly.ckpt"
TOKENIZER_VOCAB = "../data/tokenizer_vocab.json"
TOKENIZER_MERGES = "../data/tokenizer_merges.txt"

# Load models and tokenizer
st.write(f"Using device: {DEVICE}")
tokenizer = CLIPTokenizer(TOKENIZER_VOCAB, merges_file=TOKENIZER_MERGES)
models = model_loader.preload_models_from_standard_weights(MODEL_FILE, DEVICE)

# Streamlit interface
st.title("Text to Image Generator")

prompt = st.text_input("Enter a prompt for the image generation:", "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.")
uncond_prompt = ""  # Also known as negative prompt
do_cfg = st.checkbox("Use Classifier-Free Guidance (CFG)", value=True)
cfg_scale = st.slider("CFG Scale", min_value=1, max_value=14, value=8)
# strength = st.slider("Strength", min_value=0.0, max_value=1.0, value=0.9)
sampler = st.selectbox("Sampler", ["ddpm", "other_sampler_option"])  # Add other sampler options if available
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=100, value=50)
seed = st.number_input("Random Seed", value=42, step=1)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=None,
            strength=0.9,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        
        # Convert the output to an image
        image = Image.fromarray(output_image)

        # Display the image
        st.image(image, caption="Generated Image")

