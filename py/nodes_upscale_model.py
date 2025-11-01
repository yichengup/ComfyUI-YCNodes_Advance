import logging
import math
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import torch
import comfy.utils
import folder_paths

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass


class UpscaleModelLoaderYC:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_name": (folder_paths.get_filename_list("upscale_models"), ),
            }
        }
    
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "YCNode/Image"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return (out, )


class ImageUpscaleWithModelYC:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "display": "number",
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "YCNode/Image"
    OUTPUT_NODE = False

    def upscale(self, upscale_model, image, tile_size=0):
        device = model_management.get_torch_device()
        
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)
        
        if tile_size == 0:
            tile_size = self._calculate_optimal_tile_size(
                upscale_model, 
                image, 
                device
            )
            logging.info(f"[Upscale Optimized] Auto tile size: {tile_size}x{tile_size}")
        else:
            logging.info(f"[Upscale Optimized] Manual tile size: {tile_size}x{tile_size}")
        
        tile = tile_size
        overlap = max(16, min(64, tile // 16))
        logging.info(f"[Upscale Optimized] Dynamic overlap: {overlap} ({overlap/tile*100:.1f}% of tile)")
        
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        
        oom = True
        retry_count = 0
        max_retries = 3
        s = None
        last_error = None
        
        while oom and retry_count < max_retries:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], 
                    tile_x=tile, tile_y=tile, 
                    overlap=overlap
                )
                
                pbar = comfy.utils.ProgressBar(steps)
                
                s = comfy.utils.tiled_scale(
                    in_img, 
                    lambda a: upscale_model(a),
                    tile_x=tile, 
                    tile_y=tile, 
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    output_device=device,
                    pbar=pbar
                )
                
                oom = False
                logging.info(f"[Upscale Optimized] Success! tile={tile}x{tile}, overlap={overlap}")
                
            except model_management.InterruptProcessingException:
                upscale_model.to("cpu")
                raise
            
            except model_management.OOM_EXCEPTION as e:
                retry_count += 1
                old_tile = tile
                last_error = e
                
                tile = int(tile * 0.7)
                overlap = max(8, overlap // 2)
                
                logging.warning(
                    f"[Upscale Optimized] OOM retry {retry_count}/{max_retries}, "
                    f"tile: {old_tile}->{tile}, overlap: {overlap}"
                )
                
                if tile < 128:
                    upscale_model.to("cpu")
                    raise RuntimeError(
                        f"Out of memory! Cannot process this image.\n"
                        f"Image size: {image.shape[1]}x{image.shape[2]} (batch:{image.shape[0]})\n"
                        f"Upscale: {upscale_model.scale}x\n"
                        f"Output size: {int(image.shape[1]*upscale_model.scale)}x{int(image.shape[2]*upscale_model.scale)}"
                    ) from e
            
            except RuntimeError as e:
                if isinstance(e.__cause__, model_management.InterruptProcessingException):
                    upscale_model.to("cpu")
                    raise model_management.InterruptProcessingException() from None
                
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in ['cuda', 'out of memory', 'tile', 'size']):
                    retry_count += 1
                    old_tile = tile
                    last_error = e
                    
                    tile = int(tile * 0.6)
                    overlap = max(8, overlap // 2)
                    
                    logging.warning(
                        f"[Upscale Optimized] CUDA error retry {retry_count}/{max_retries}, "
                        f"tile: {old_tile}->{tile}"
                    )
                    
                    if tile < 128:
                        upscale_model.to("cpu")
                        raise RuntimeError(
                            f"Processing failed! Tile reduced to minimum.\n"
                            f"Error: {str(e)}\n"
                            f"Image size: {image.shape[1]}x{image.shape[2]}\n"
                            f"Last tile: {old_tile}x{old_tile}"
                        ) from e
                else:
                    upscale_model.to("cpu")
                    raise RuntimeError(
                        f"Model processing error.\n"
                        f"Error: {str(e)}\n"
                        f"Image size: {image.shape[1]}x{image.shape[2]}\n"
                        f"Tile: {tile}x{tile}\n"
                        f"Upscale: {upscale_model.scale}x"
                    ) from e
            
            except Exception as e:
                if isinstance(e, model_management.InterruptProcessingException) or \
                   isinstance(e.__cause__, model_management.InterruptProcessingException):
                    upscale_model.to("cpu")
                    raise model_management.InterruptProcessingException() from None
                
                upscale_model.to("cpu")
                logging.error(f"[Upscale Optimized] Unexpected error: {type(e).__name__}: {str(e)}")
                raise RuntimeError(
                    f"Unexpected error!\n"
                    f"Type: {type(e).__name__}\n"
                    f"Message: {str(e)}\n"
                    f"Image size: {image.shape[1]}x{image.shape[2]} (batch:{image.shape[0]})\n"
                    f"Upscale: {upscale_model.scale}x\n"
                    f"Tile: {tile}x{tile}\n"
                    f"Device: {device}"
                ) from e
        
        if s is None:
            upscale_model.to("cpu")
            raise RuntimeError(
                f"All retries failed!\n"
                f"Last error: {str(last_error) if last_error else 'Unknown'}"
            )
        
        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        
        return (s,)
    
    def _calculate_optimal_tile_size(self, upscale_model, image, device):
        try:
            free_memory = model_management.get_free_memory(device)
            model_memory = model_management.module_size(upscale_model.model)
            
            available_memory = free_memory - model_memory * 1.5
            
            scale = max(upscale_model.scale, 1.0)
            bytes_per_pixel = image.element_size() * 3
            bytes_per_pixel_output = bytes_per_pixel * (scale ** 2)
            
            total_bytes_per_pixel = bytes_per_pixel + bytes_per_pixel_output
            
            max_tile_pixels = int((available_memory * 0.7) / total_bytes_per_pixel)
            max_tile_size = int(math.sqrt(max_tile_pixels))
            
            tile_size = max(256, min(768, max_tile_size))
            tile_size = (tile_size // 64) * 64
            
            if max_tile_size > 768:
                logging.info(
                    f"[Upscale Optimized] VRAM can support larger tile ({max_tile_size}), "
                    f"but limited to 768 for compatibility. Set manually if needed."
                )
            
            logging.info(
                f"[Upscale Optimized] VRAM analysis:\n"
                f"  - Free VRAM: {free_memory / 1024**3:.2f} GB\n"
                f"  - Model size: {model_memory / 1024**3:.2f} GB\n"
                f"  - Available for tiles: {available_memory / 1024**3:.2f} GB\n"
                f"  - Upscale factor: {scale}x\n"
                f"  - Optimal tile: {tile_size}x{tile_size}"
            )
            
            return tile_size
            
        except Exception as e:
            logging.warning(f"[Upscale Optimized] Cannot calculate optimal tile, using default 512: {str(e)}")
            return 512


NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader_Optimized": UpscaleModelYC,
    "ImageUpscaleWithModel_Optimized": ImageUpscaleWithModelYC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleModelLoader_Optimized": "Load Upscale Model YC",
    "ImageUpscaleWithModel_Optimized": "Upscale Image YC",
}


