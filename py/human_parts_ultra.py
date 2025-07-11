import os
import torch
import numpy as np
from PIL import Image, ImageEnhance
import folder_paths

# 整合所需的函数
def log(message:str, message_type:str='info'):
    name = 'LayerStyle'

    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# 😺dzNodes: {name} -> {message}")

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image2mask(image:Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def mask2image(mask:torch.Tensor) -> Image:
    """将掩码Tensor转换为PIL图像"""
    # 确保mask是二维的 (如果是多维的，取第一个通道)
    if len(mask.shape) > 2:
        mask_np = mask[0].cpu().numpy()
    else:
        mask_np = mask.cpu().numpy()
    
    # 归一化并转换为uint8
    mask_np = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
    
    # 创建PIL图像
    mask_image = Image.fromarray(mask_np).convert("L")
    
    # 转换为带透明度的RGBA图像
    rgba_image = Image.new("RGBA", mask_image.size, color=(255, 255, 255, 0))
    rgba_image.putalpha(mask_image)
    
    return rgba_image

def RGB2RGBA(image:Image, mask:Image) -> Image:
    return Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size, (0, 0, 0, 0)), mask)

models_dir_path = os.path.join(folder_paths.models_dir, "onnx", "human-parts")
model_url = "https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir_path, "deeplabv3p-resnet50-human.onnx")


class LS_HumanPartsUltra:
    """
    This node is used to get a mask of the human parts in the image.

    The model used is DeepLabV3+ with a ResNet50 backbone trained
    by Keras-io, converted to ONNX format.
    """

    def __init__(self):
        self.NODE_NAME = 'HumanPartsUltra'

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "human_parts_ultra"
    CATEGORY = "YCNode/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face": ("BOOLEAN", {"default": False, "label_on": "enabled(脸)", "label_off": "disabled(脸)"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled(头发)", "label_off": "disabled(头发)"}),
                "top_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(身体)", "label_off": "disabled(身体)"}),
                "bottom_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(衣服)", "label_off": "disabled(衣服)"}),
                "torso_skin": ("BOOLEAN", {"default": False, "label_on": "enabled(配饰)", "label_off": "disabled(配饰)"}),
                "background": ("BOOLEAN", {"default": False, "label_on": "enabled(背景)", "label_off": "disabled(背景)"}),
                "brightness": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.01, "display": "slider"}),
                "refinement_edges": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1, "display": "slider"}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
            }
        }

    def human_parts_ultra(self, image, face, hair, top_clothes, bottom_clothes,
                          torso_skin, background, brightness, refinement_edges, 
                          black_point, white_point, process_detail):
        """
        Return a Tensor with the mask of the human parts in the image.
        """
        import onnxruntime as ort

        # 确保模型目录存在
        os.makedirs(models_dir_path, exist_ok=True)

        # 检查模型文件是否存在，如果不存在则提示下载
        if not os.path.exists(model_path):
            log(f"Model file not found at {model_path}. Please download from {model_url}", message_type='error')
            log(f"You can download it manually and place it in {models_dir_path}", message_type='info')
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        ret_images = []
        ret_masks = []
        
        for img in image:
            orig_image = tensor2pil(img).convert('RGB')

            human_parts_mask, _ = self.get_mask(orig_image, model=model, rotation=0, background=background,
                                          face=face, hair=hair, glasses=False,
                                          top_clothes=top_clothes, bottom_clothes=bottom_clothes,
                                          torso_skin=torso_skin, left_arm=False, right_arm=False,
                                          left_leg=False, right_leg=False,
                                          left_foot=False, right_foot=False)
            
            # 将结果转换为PIL图像以便进行亮度调整
            _mask_pil = tensor2pil(human_parts_mask).convert('L')
            brightness_image = ImageEnhance.Brightness(_mask_pil)
            _mask_pil = brightness_image.enhance(factor=brightness)
            
            # 转回Tensor以便进一步处理
            _mask = image2mask(_mask_pil)
            
            if process_detail:
                # 简化后的处理逻辑，去除了VITMatte相关处理
                _mask = self.process_mask_edges(_mask, refinement_edges)
                # 应用黑白点调整
                _mask = self.apply_black_white_points(_mask, black_point, white_point)
            
            # 转换为PIL图像用于创建RGBA图像
            _mask_pil = mask2image(_mask)
            
            # 创建RGBA图像
            ret_image = RGB2RGBA(orig_image, _mask_pil)
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(_mask)
        
        # 确保返回的是正确的Tensor格式
        if len(ret_images) > 0:
            ret_image_tensor = torch.cat(ret_images, dim=0)
            ret_mask_tensor = torch.cat(ret_masks, dim=0)
        else:
            # 如果没有图像，返回空的tensor
            ret_image_tensor = torch.zeros((0, 3, 64, 64))
            ret_mask_tensor = torch.zeros((0, 1, 64, 64))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (ret_image_tensor, ret_mask_tensor,)
    
    def process_mask_edges(self, mask, refinement_edges):
        """简化的边缘处理函数"""
        # 获取原始形状以便后续恢复
        original_shape = mask.shape
        
        # 转换为numpy数组，保留原始维度
        mask_np = mask.cpu().numpy()
        
        # 应用简单的高斯模糊来平滑边缘 (只对最后两个维度应用滤波)
        from scipy.ndimage import gaussian_filter
        # 对每个通道单独处理
        for i in range(mask_np.shape[0]):
            mask_np[i] = gaussian_filter(mask_np[i], sigma=refinement_edges/10)
        
        # 转回Tensor，保持原始形状
        return torch.from_numpy(mask_np).to(mask.device)
    
    def apply_black_white_points(self, mask, black_point, white_point):
        """应用黑白点调整"""
        # 获取原始形状以便后续恢复
        original_shape = mask.shape
        
        # 转换为numpy数组
        mask_np = mask.cpu().numpy()
        
        # 应用黑白点调整
        mask_np = np.clip((mask_np - black_point) / (white_point - black_point), 0, 1)
        
        # 转回Tensor，保持原始形状
        return torch.from_numpy(mask_np).to(mask.device)

    def get_mask(self, pil_image:Image, model, rotation:float, **kwargs) -> tuple:
        """
        Return a Tensor with the mask of the human parts in the image.

        The rotation parameter is not used for now. The idea is to propose rotation to help
        the model to detect the human parts in the image if the character is not in a casual position.
        Several tests have been done, but the model seems to fail to detect the human parts in these cases,
        and the rotation does not help.
        """

        # classes used in the model
        classes = {
            "background": 0,
            "hair": 2,
            "glasses": 4,
            "top_clothes": 5,
            "bottom_clothes": 9,
            "torso_skin": 10,
            "face": 13,
            "left_arm": 14,
            "right_arm": 15,
            "left_leg": 16,
            "right_leg": 17,
            "left_foot": 18,
            "right_foot": 19,
        }

        original_size = pil_image.size  # to resize the mask later
        # resize to 512x512 as the model expects
        pil_image = pil_image.resize((512, 512))
        center = (256, 256)

        if rotation != 0:
            pil_image = pil_image.rotate(rotation, center=center)

        # normalize the image
        image_np = np.array(pil_image).astype(np.float32) / 127.5 - 1
        image_np = np.expand_dims(image_np, axis=0)

        # use the onnx model to get the mask
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: image_np})
        result = np.array(result[0]).argmax(axis=3).squeeze(0)

        score: int = 0

        mask = np.zeros_like(result)
        for class_name, enabled in kwargs.items():
            if enabled and class_name in classes:
                class_index = classes[class_name]
                detected = result == class_index
                mask[detected] = 255
                score += mask.sum()

        # back to the original size
        mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
        if rotation != 0:
            mask_image = mask_image.rotate(-rotation, center=center)

        mask_image = mask_image.resize(original_size)

        # and back to numpy...
        mask = np.array(mask_image).astype(np.float32) / 255

        # add 2 dimensions to match the expected output
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        # ensure to return a "binary mask_image"

        del image_np, result  # free up memory, maybe not necessary
        return (torch.from_numpy(mask.astype(np.uint8)), score)


NODE_CLASS_MAPPINGS = {
    "HumanPartsUltra": LS_HumanPartsUltra
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanPartsUltra": "YC Human Parts Ultra(Advance)"
}
