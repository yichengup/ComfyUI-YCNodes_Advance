import torch
import numpy as np
import cv2
import os
import sys

class FaceDetectorSelector:
    """人脸检测选择器 - 检测第一个图像是否有人脸，有则输出第一个图像，没有则输出第二个图像
    需要连接YCFaceAnalysisModels节点提供的ANALYSIS_MODELS"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),  # 外部人脸分析模型
                "detect_image": ("IMAGE", ),  # 用于检测人脸的图像
                "alternative_image": ("IMAGE", ),  # 如果没有检测到人脸，将输出的备选图像
                "min_face_size": ("INT", {"default": 30, "min": 10, "max": 1000, "step": 1}),  # 最小人脸尺寸
                "det_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),  # 检测阈值
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "face_count", "face_message")
    FUNCTION = "select_by_face"
    CATEGORY = "YCNode/Face"

    def select_by_face(self, analysis_models, detect_image, alternative_image, min_face_size=30, det_threshold=0.5):
        # 确保输入图像尺寸一致
        if detect_image.shape[1:] != alternative_image.shape[1:]:
            print("警告：两个输入图像的尺寸不一致，可能导致输出不符合预期")
        
        try:
            # 转换为OpenCV格式处理
            img = detect_image[0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            # 使用外部模型检测人脸
            face_analyzer = analysis_models.face_analysis
            faces = face_analyzer.get(img)
            
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
                return (detect_image, face_count, message)
            else:
                message = "未检测到人脸，输出备选图像"
                return (alternative_image, 0, message)
                
        except Exception as e:
            print(f"人脸检测过程中出错: {str(e)}")
            return (detect_image, 0, f"检测错误: {str(e)}")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "FaceDetectorSelector": FaceDetectorSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectorSelector": "Face Detector Selector"
} 
