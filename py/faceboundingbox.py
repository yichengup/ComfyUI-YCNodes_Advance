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
                "mode": (["contain", "cover", "stretch"], {"default": "contain"}),
                "contain_mode": (["auto", "by_width", "by_height"], {"default": "auto"}),
                "detect_face": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "uncovered_mask")
    FUNCTION = "align_to_canvas"
    CATEGORY = "YCNode/Face"

    def align_to_canvas(self, analysis_models, image, canvas_width, canvas_height, 
                        target_face_x, target_face_y, target_face_width, target_face_height,
                        padding, padding_percent, keep_aspect_ratio, face_index=0, mode="contain", contain_mode="auto", detect_face=True):
        input_image = image[0]  # 取第一帧
        pil_image = T.ToPILImage()(input_image.permute(2, 0, 1)).convert('RGB')
        img_w, img_h = pil_image.width, pil_image.height
        canvas_w, canvas_h = canvas_width, canvas_height

        # 是否检测人脸
        do_face_detect = detect_face
        faces_found = False
        if do_face_detect:
            img, x, y, w, h = analysis_models.get_bbox(pil_image, padding, padding_percent)
            if img and len(img) > 0:
                faces_found = True
        else:
            img, x, y, w, h = [], [], [], [], []

        if not faces_found:
            # 没有人脸时，按mode处理
            if mode == "contain":
                if contain_mode == "auto":
                    scale = min(canvas_w / img_w, canvas_h / img_h)
                elif contain_mode == "by_width":
                    scale = canvas_w / img_w
                elif contain_mode == "by_height":
                    scale = canvas_h / img_h
                else:
                    scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                offset_x = (canvas_w - new_w) // 2
                offset_y = (canvas_h - new_h) // 2
            elif mode == "cover":
                scale = max(canvas_w / img_w, canvas_h / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                offset_x = (canvas_w - new_w) // 2
                offset_y = (canvas_h - new_h) // 2
            elif mode == "stretch":
                new_w = canvas_w
                new_h = canvas_h
                resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                offset_x = 0
                offset_y = 0
            else:
                # 默认contain auto
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                offset_x = (canvas_w - new_w) // 2
                offset_y = (canvas_h - new_h) // 2
            # 创建画布
            result_image = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
            mask_image = Image.new('L', (canvas_w, canvas_h), 255)
            # 粘贴图片
            result_image.paste(resized_image, (offset_x, offset_y))
            # 粘贴遮罩
            mask_draw = Image.new('L', (new_w, new_h), 0)
            mask_image.paste(mask_draw, (offset_x, offset_y))
            # 转tensor
            result_tensor = T.ToTensor()(result_image).permute(1, 2, 0).unsqueeze(0)
            mask_np = np.array(mask_image).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np)
            mask_tensor = mask_tensor.unsqueeze(0)
            return (result_tensor, mask_tensor)

        # 有人脸时，按原逻辑
        if face_index >= len(img):
            face_index = 0
        face_x = x[face_index]
        face_y = y[face_index]
        face_width = w[face_index]
        face_height = h[face_index]
        scale_x = target_face_width / face_width
        scale_y = target_face_height / face_height
        if keep_aspect_ratio:
            scale = min(scale_x, scale_y)
            scale_x = scale
            scale_y = scale
        new_width = int(pil_image.width * scale_x)
        new_height = int(pil_image.height * scale_y)
        offset_x = target_face_x - int(face_x * scale_x)
        offset_y = target_face_y - int(face_y * scale_y)
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        result_image = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
        mask_image = Image.new('L', (canvas_w, canvas_h), 255)
        mask_draw = Image.new('L', (new_width, new_height), 0)
        result_image.paste(resized_image, (offset_x, offset_y))
        mask_image.paste(mask_draw, (offset_x, offset_y))
        result_tensor = T.ToTensor()(result_image).permute(1, 2, 0).unsqueeze(0)
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)
        mask_tensor = mask_tensor.unsqueeze(0)
        return (result_tensor, mask_tensor)

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
