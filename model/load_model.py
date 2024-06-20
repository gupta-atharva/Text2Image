from VAE import CLIP
from VAE import VariationalAutoEncoder
from VAE import VariationalAutoDecoder
from VAE import Diffusion

import model.converter as converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = converter.load_from_standard_weights(ckpt_path, device)

    encoder = VariationalAutoEncoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VariationalAutoDecoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }