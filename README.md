**Overview:**

A world where you could describe an image in natural language, and an AI could
generate that image for you with stunning accuracy and detail.
This model creates realistic images based on textual
descriptions.
It is capable of understanding and translating
textual descriptions into
high-fidelity images, capturing intricate details and nuances while
maintaining coherence
with the input description. It is also adaptable to various
domains, from everyday
objects to complex scenes and landscapes.

**Installation:**

clone the repository into a directory.

inside the directory make a folder named "Pretrainedmodeldata".

Download vocab.json and merges.txt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer and save them in this folder.
Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in this folder.

write the following commands in ur terminal.

pip install -r requirements.txt

pip install streamlit

streamlit run app.py

**Dependencies:**

torch==2.0.1

numpy==1.24.4

tqdm==4.65.0

transformers==4.33.2

lightning==2.0.9

pillow==9.5.0

Flask
