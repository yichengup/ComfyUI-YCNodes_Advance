# ComfyUI-YCNodes_Advance 

本文档提供ComfyUI-YCNodes_Advance节点包中各个节点所需模型（一般都会自动下载）的下载链接和安装方法。

## 目录

1. [人脸检测模型 (YCFaceAnalysisModels)](#人脸检测模型-ycfaceanalysismodels)
2. [人体分割模型 (HumanPartsUltra)](#人体分割模型-humanpartsultra)


## 人脸检测模型 (YCFaceAnalysisModels)

YCFaceAnalysisModels节点使用InsightFace库进行人脸检测和分析。

### 模型下载

InsightFace模型会在首次运行时自动下载，但您也可以手动下载并放置到正确位置：

1. 创建目录：`ComfyUI/models/insightface/`
2. 下载对应模型文件：
   - buffalo_l: [下载链接](https://github.com/deepinsight/insightface/tree/master/model_zoo) 一般只用这个模型
   - buffalo_m: [下载链接](https://github.com/deepinsight/insightface/tree/master/model_zoo)
   - buffalo_s: [下载链接](https://github.com/deepinsight/insightface/tree/master/model_zoo)

### 依赖安装

```bash
pip install insightface onnxruntime
```

## 人体分割模型 (HumanPartsUltra)

HumanPartsUltra节点使用DeepLabV3+模型进行人体部位分割。

### 模型下载

1. 创建目录：`ComfyUI/models/onnx/human-parts/`
2. 下载模型文件：[deeplabv3p-resnet50-human.onnx](https://huggingface.co/Metal3d/deeplabv3p-resnet50-human/resolve/main/deeplabv3p-resnet50-human.onnx)
3. 将下载的模型文件放入上述目录

### 依赖安装

```bash
pip install onnxruntime scipy
```


## 常见问题

### 模型无法下载或加载

如果遇到模型无法下载或加载的问题，请尝试以下解决方案：

1. 检查网络连接
2. 手动下载模型并放置到正确位置
3. 检查是否安装了所有必要的依赖
4. 确认您有足够的磁盘空间

### 模型版本兼容性

如果节点更新后模型不兼容，请删除旧模型并重新下载最新版本。

## 其他资源

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [DeepLabV3+ 模型信息](https://huggingface.co/Metal3d/deeplabv3p-resnet50-human)
