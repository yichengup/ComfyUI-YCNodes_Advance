import torch
import numpy as np
import cv2
import os
import sys
import comfy.model_management as model_management

# 检查是否已安装insightface
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAVE_FACE_LIBRARY = True
except ImportError:
    HAVE_FACE_LIBRARY = False
    print("请安装insightface库以使用人脸检测功能: pip install insightface onnxruntime")

class FaceDetectorSelector:
    """人脸检测选择器 - 检测第一个图像是否有人脸，有则输出第一个图像，没有则输出第二个图像"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detect_image": ("IMAGE", ),  # 用于检测人脸的图像
                "alternative_image": ("IMAGE", ),  # 如果没有检测到人脸，将输出的备选图像
                "min_face_size": ("INT", {"default": 30, "min": 10, "max": 1000, "step": 1}),  # 最小人脸尺寸
                "det_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),  # 检测阈值
                "use_cpu": (["否", "是"], {"default": "否"}),  # 是否使用CPU进行推理
                "detection_size": (["大(640)", "中(320)", "小(160)"], {"default": "中(320)"}),  # 检测尺寸
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "face_count", "face_message")
    FUNCTION = "select_by_face"
    CATEGORY = "YCNode/Face"

    def __init__(self):
        self.face_analyzer = None
        self.device = model_management.get_torch_device()
        self.initialized = False
        self.current_config = None  # 存储当前配置，用于检测配置变化
        
    def initialize_face_detector(self, use_cpu="否", detection_size="中(320)"):
        """初始化人脸检测器"""
        if not HAVE_FACE_LIBRARY:
            return False
            
        # 解析检测尺寸
        if detection_size == "大(640)":
            det_size = (640, 640)
        elif detection_size == "中(320)":
            det_size = (320, 320)
        else:  # 小(160)
            det_size = (160, 160)
            
        # 检查配置是否变化，如果变化则重新初始化
        current_config = (use_cpu, det_size)
        if self.initialized and self.current_config == current_config:
            return True
            
        # 如果之前初始化过不同配置，先清理
        if self.initialized:
            self.face_analyzer = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        try:
            providers = []
            if use_cpu == "否" and self.device.type == 'cuda':
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            # 选择较小的模型以减少资源占用
            model_name = "buffalo_l"  # 统一使用buffalo_l模型，无论GPU还是CPU
            self.face_analyzer = FaceAnalysis(providers=providers, name=model_name)
            self.face_analyzer.prepare(ctx_id=0, det_size=det_size)
            
            self.initialized = True
            self.current_config = current_config
            return True
        except Exception as e:
            print(f"初始化人脸检测器时出错: {str(e)}")
            return False

    def select_by_face(self, detect_image, alternative_image, min_face_size=30, det_threshold=0.5, use_cpu="否", detection_size="中(320)"):
        # 确保输入图像尺寸一致
        if detect_image.shape[1:] != alternative_image.shape[1:]:
            print("警告：两个输入图像的尺寸不一致，可能导致输出不符合预期")
        
        # 初始化人脸检测器
        if not self.initialize_face_detector(use_cpu, detection_size):
            return (detect_image, 0, "未安装insightface库，无法进行人脸检测")
        
        try:
            # 转换为OpenCV格式处理
            img = detect_image[0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            # 检测人脸
            faces = self.face_analyzer.get(img)
            
            # 筛选足够大的人脸
            valid_faces = []
            for face in faces:
                bbox = face.bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                
                # 只保留足够大的人脸，并且置信度高于阈值
                if face_width >= min_face_size and face_height >= min_face_size and face.det_score >= det_threshold:
                    valid_faces.append(face)
            
            face_count = len(valid_faces)
            
            # 根据检测结果选择输出图像
            if face_count > 0:
                message = f"检测到{face_count}张人脸"
                if use_cpu == "是":
                    message += " (CPU模式)"
                return (detect_image, face_count, message)
            else:
                message = "未检测到人脸，输出备选图像"
                if use_cpu == "是":
                    message += " (CPU模式)"
                return (alternative_image, 0, message)
                
        except Exception as e:
            print(f"人脸检测过程中出错: {str(e)}")
            return (detect_image, 0, f"检测错误: {str(e)}")
            
    def __del__(self):
        """析构函数，确保资源释放"""
        if self.initialized:
            self.face_analyzer = None
            self.initialized = False
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# 节点映射
NODE_CLASS_MAPPINGS = {
    "FaceDetectorSelector": FaceDetectorSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectorSelector": "Face Detector Selector"
} 
