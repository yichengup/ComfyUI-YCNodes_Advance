import torch
import torchvision.transforms.v2 as T
import numpy as np
from PIL import Image
import os
import sys

# 检查是否已安装insightface
IS_INSIGHTFACE_INSTALLED = False
try:
    from insightface.app import FaceAnalysis
    IS_INSIGHTFACE_INSTALLED = True
except ImportError:
    raise ImportError("请安装insightface库以使用人脸检测功能: pip install insightface onnxruntime")

# 定义InsightFace类
class InsightFace:
    def __init__(self, provider="CPU", name="buffalo_l"):
        import folder_paths
        self.insightface_dir = os.path.join(folder_paths.models_dir, "insightface")
        self.face_analysis = FaceAnalysis(name=name, root=self.insightface_dir, providers=[provider + 'ExecutionProvider',])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def get_face(self, image):
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_analysis.det_model.input_size = size
            faces = self.face_analysis.get(np.array(image))
            if len(faces) > 0:
                return sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
        return None
    
    def get_bbox(self, image, padding=0, padding_percent=0):
        faces = self.get_face(image)
        img = []
        x = []
        y = []
        w = []
        h = []
        if faces is None or len(faces) == 0:
            return (img, x, y, w, h)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            width = x2 - x1
            height = y2 - y1
            x1 = int(max(0, x1 - int(width * padding_percent) - padding))
            y1 = int(max(0, y1 - int(height * padding_percent) - padding))
            x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
            y2 = int(min(image.height, y2 + int(height * padding_percent) + padding))
            crop = image.crop((x1, y1, x2, y2))
            img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
            x.append(x1)
            y.append(y1)
            w.append(x2 - x1)
            h.append(y2 - y1)
        return (img, x, y, w, h)

class YCFaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], {"default": "CPU"}),
            "model": (["buffalo_l", "buffalo_m", "buffalo_s"], {"default": "buffalo_l"}),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "YCNode/Face"

    def load_models(self, provider, model):
        return (InsightFace(provider=provider, name=model), )


class YCFaceAlignToCanvas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "canvas_width": ("INT", { "default": 512, "min": 64, "max": 4096, "step": 1 }),
                "canvas_height": ("INT", { "default": 512, "min": 64, "max": 4096, "step": 1 }),
                "target_face_x": ("INT", { "default": 128, "min": 0, "max": 4096, "step": 1 }),
                "target_face_y": ("INT", { "default": 128, "min": 0, "max": 4096, "step": 1 }),
                "target_face_width": ("INT", { "default": 256, "min": 1, "max": 4096, "step": 1 }),
                "target_face_height": ("INT", { "default": 256, "min": 1, "max": 4096, "step": 1 }),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "padding_percent": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05 }),
                "keep_aspect_ratio": ("BOOLEAN", { "default": True }),
                "face_index": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align_to_canvas"
    CATEGORY = "YCNode/Face"

    def align_to_canvas(self, analysis_models, image, canvas_width, canvas_height, 
                        target_face_x, target_face_y, target_face_width, target_face_height,
                        padding, padding_percent, keep_aspect_ratio, face_index=0):
        # 处理输入图像
        input_image = image[0]  # 取第一帧
        pil_image = T.ToPILImage()(input_image.permute(2, 0, 1)).convert('RGB')
        
        # 检测脸部
        img, x, y, w, h = analysis_models.get_bbox(pil_image, padding, padding_percent)
        
        if not img:
            raise Exception('No face detected in image.')
        
        # 确保face_index在有效范围内
        if face_index >= len(img):
            face_index = 0
        
        # 获取检测到的脸部坐标和尺寸
        face_x = x[face_index]
        face_y = y[face_index]
        face_width = w[face_index]
        face_height = h[face_index]
        
        # 计算缩放比例
        scale_x = target_face_width / face_width
        scale_y = target_face_height / face_height
        
        # 如果需要保持宽高比，选择较小的缩放比例
        if keep_aspect_ratio:
            scale = min(scale_x, scale_y)
            scale_x = scale
            scale_y = scale
        
        # 计算调整后的图像尺寸
        new_width = int(pil_image.width * scale_x)
        new_height = int(pil_image.height * scale_y)
        
        # 计算偏移量，使脸部对齐到目标位置
        offset_x = target_face_x - int(face_x * scale_x)
        offset_y = target_face_y - int(face_y * scale_y)
        
        # 调整图像大小
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建新的画布
        result_image = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        
        # 将调整后的图像粘贴到画布上
        result_image.paste(resized_image, (offset_x, offset_y))
        
        # 转换回tensor
        result_tensor = T.ToTensor()(result_image).permute(1, 2, 0).unsqueeze(0)
        
        return (result_tensor,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "YCFaceAnalysisModels": YCFaceAnalysisModels,
    "YCFaceAlignToCanvas": YCFaceAlignToCanvas
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YCFaceAnalysisModels": "YC Face Analysis Models",
    "YCFaceAlignToCanvas": "YC Face Align To Canvas"
}
