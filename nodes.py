import os
import torch

from omegaconf import OmegaConf
from .models.depth_normal_pipeline_clip import DepthNormalEstimationPipeline
from .models.unet_2d_condition import UNet2DConditionModel
from diffusers import  DDIMScheduler, AutoencoderKL

import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms
from contextlib import  nullcontext

import model_management as mm
import comfy.utils
import folder_paths
script_directory = os.path.dirname(os.path.abspath(__file__))

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError

class geowizard_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE",),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                    ], {
                        "default": 'fp16'
                    }),
            },
        }

    RETURN_TYPES = ("GEOWIZMODEL",)
    RETURN_NAMES = ("geowizard_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "champWrapper"

    def loadmodel(self, vae, dtype):
        mm.soft_empty_cache()
 
        custom_config = {
            'dtype': dtype,
            'vae': vae
        }

        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))

            dtype = convert_dtype(dtype)
            
            from diffusers.loaders.single_file_utils import (convert_ldm_vae_checkpoint, create_vae_diffusers_config)
            
            sd = vae.get_sd()
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            self.vae = AutoencoderKL(**converted_vae_config)
            self.vae.load_state_dict(converted_vae, strict=False)


            model_path = os.path.join(folder_paths.models_dir,'diffusers', 'geowizard')
            if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="lemonaddie/geowizard", ignore_patterns=["*vae*", "*.ckpt", "*.pt", "*.png", "*non_ema*", "*safety_checker*", "*.bin"], 
                                    local_dir=model_path, local_dir_use_symlinks=False)
            
            unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet')
            scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
            feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder='feature_extractor')
            image_enc = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder='image_encoder')

            self.model = DepthNormalEstimationPipeline(
                            vae=self.vae,
                            image_encoder=image_enc,
                            feature_extractor=feature_extractor,
                            unet=unet,
                            scheduler=scheduler)
            self.model = self.model.to(dtype)
            if mm.XFORMERS_IS_AVAILABLE:
                self.model.enable_xformers_memory_efficient_attention()

        return (self.model,)
    
class geowizard_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "geowizard_model": ("GEOWIZMODEL",),
            "image": ("IMAGE",),
            "steps": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1}),
            "ensemble_size": ("INT", {"default": 3, "min": 1, "max": 200, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "domain": (
            [   
                'outdoor',
                'indoor',
                'object'
            ], {
               "default": 'indoor'
            }),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
    
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("depth", "normal",)
    FUNCTION = "process"
    CATEGORY = "champWrapper"

    def process(self, geowizard_model, image, domain, ensemble_size, steps, seed, keep_model_loaded):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipe = geowizard_model
        dtype = pipe.dtype

        torch.manual_seed(seed)
        
        
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            image = image.permute(0, 3, 1, 2)

            B, C, H, W = image.shape
            orig_H, orig_W = H, W
            if W % 64 != 0:
                W = W - (W % 64)
            if H % 64 != 0:
                H = H - (H % 64)
            if orig_H % 64 != 0 or orig_W % 64 != 0:
                image = F.interpolate(image, size=(H, W), mode="bicubic")
           
            B, C, H, W = image.shape

            pipe = pipe.to(device)
            depth_maps = []
            normal_maps = []
            if B > 1:
                batch_pbar = comfy.utils.ProgressBar(B)
            for img in image:
                pipe_out = pipe(
                    image[0],
                    device=device,
                    denoising_steps=steps,
                    ensemble_size=ensemble_size,
                    processing_res=H,
                    batch_size=0,
                    domain=domain,
                    seed = seed,
                    show_progress_bar=True,
                )

                depth = pipe_out.depth_pred_tensor
                depth = depth.repeat(3,1,1).unsqueeze(0).permute(0, 2, 3, 1)
                depth_maps.append(depth)
                normal_colored = pipe_out.normal_colored
                normal_tensor = transforms.ToTensor()(normal_colored).unsqueeze(0).permute(0, 2, 3, 1)
                normal_maps.append(normal_tensor)
                if B > 1:
                    batch_pbar.update(1)

            depth_out = torch.cat(depth_maps, dim=0)
            depth_out = torch.clamp(depth, 0.0, 1.0)
            depth_out = 1.0 - depth

            normal_out = torch.cat(normal_maps, dim=0)

            return (depth_out, normal_out)


NODE_CLASS_MAPPINGS = {
    "geowizard_model_loader": geowizard_model_loader,
    "geowizard_sampler": geowizard_sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "geowizard_model_loader": "Geowizard Model Loader",
    "geowizard_sampler": "Geowizard Sampler",
}
