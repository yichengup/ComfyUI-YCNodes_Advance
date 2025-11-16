// author.yichengup.Loadimage_brushmask 2025.01.XX
import { app } from "../../scripts/app.js";

class ycimagebrushmask
{
    constructor(node)
    {
        this.node = node;
        // 初始化属性
        this.node.properties = { 
            brushPaths: [], // 存储画笔路径，每个路径是一个对象 {points: [], mode: 'brush'/'erase', size: 20, opacity: 1.0}
            isDrawing: false, // 是否正在绘制
            currentPath: [], // 当前绘制的路径
            brushSize: 20, // 当前画笔大小（用于新绘制）
            brushOpacity: 1.0, // 当前画笔透明度（用于新绘制）(0.0 - 1.0)
            brushMode: 'brush', // 当前模式：'brush' 或 'erase'
            backgroundImage: null, // 背景图
            imageWidth: 512, // 图像宽度
            imageHeight: 512, // 图像高度
        };

        // 设置节点大小（初始值，会根据背景图自动调整）
        this.node.size = [500, 500];
        const fontsize = LiteGraph.NODE_SUBTEXT_SIZE;
        const shiftLeft = 10;
        const shiftRight = 80;
        const panelHeight = 70; // 顶部面板高度（两行：按钮行+滑动条行）

        // 隐藏默认小部件
        // widget查找：brush_data, brush_size, image_base64
        const brushDataWidget = this.node.widgets.find(w => w.name === "brush_data");
        if (brushDataWidget) {
            brushDataWidget.hidden = true;
        }
        
        // 确保brush_size widget可见
        const brushSizeWidget = this.node.widgets.find(w => w.name === "brush_size");
        if (brushSizeWidget) {
            brushSizeWidget.hidden = false;
        }
        
        
        // 添加隐藏的尺寸widget（用于存储从后端获取的尺寸）
        // 这些widget会在节点执行后通过某种方式更新
        let widthWidget = this.node.widgets.find(w => w.name === "image_width");
        let heightWidget = this.node.widgets.find(w => w.name === "image_height");
        let imageBase64Widget = this.node.widgets.find(w => w.name === "image_base64");
        
        if (!widthWidget) {
            widthWidget = this.node.addWidget("number", "image_width", 512, () => {}, {min: 64, max: 4096});
            widthWidget.hidden = true;
        }
        if (!heightWidget) {
            heightWidget = this.node.addWidget("number", "image_height", 512, () => {}, {min: 64, max: 4096});
            heightWidget.hidden = true;
        }
        if (!imageBase64Widget) {
            imageBase64Widget = this.node.addWidget("text", "image_base64", "", () => {});
            imageBase64Widget.hidden = true;
        } else {
            // 如果已存在，隐藏它
            imageBase64Widget.hidden = true;
        }
        
        // 存储背景图像对象和base64数据
        this.node.properties.backgroundImageObj = null;
        this.node.properties.imageBase64Data = "";

        // 初始化按钮函数
        this.node.initButtons = function() {
            // 设置输出名称和类型（确保顺序正确：image, mask, width, height）
            if (this.outputs && this.outputs.length >= 4) {
                // 第一个输出：image (IMAGE类型)
                this.outputs[0].name = this.outputs[0].localized_name = "image";
                this.outputs[0].type = "IMAGE";
                
                // 第二个输出：mask (MASK类型)
                this.outputs[1].name = this.outputs[1].localized_name = "mask";
                this.outputs[1].type = "MASK";
                
                // 第三个输出：width (INT类型)
                this.outputs[2].name = this.outputs[2].localized_name = "width";
                this.outputs[2].type = "INT";
                
                // 第四个输出：height (INT类型)
                this.outputs[3].name = this.outputs[3].localized_name = "height";
                this.outputs[3].type = "INT";
            }

            this.widgets_start_y = -4.8e8 * LiteGraph.NODE_SLOT_HEIGHT;

            // 初始化画笔大小
            if (!this.widgets[1] || !this.widgets[1].value) {
                // 如果widget不存在，会在onConfigure中处理
            } else {
                this.properties.brushSize = this.widgets[1].value || 20;
            }

            // 定义顶部面板按钮和滑动条
            const buttonY = 5; // 面板内按钮Y位置（第一行）
            const buttonHeight = 20;
            const sliderHeight = 12; // 滑动条高度
            const sliderY = buttonY + buttonHeight + 8; // 滑动条Y位置（第二行）
            let buttonX = 10;
            
            this.properties.buttons = [
                {
                    text: "Load Image",
                    x: buttonX,
                    y: buttonY,
                    width: 70,
                    height: buttonHeight,
                    action: () => {
                        this.loadImageFromFile();
                    }
                },
                {
                    text: "Clear",
                    x: buttonX += 75,
                    y: buttonY,
                    width: 50,
                    height: buttonHeight,
                    action: () => {
                        this.properties.brushPaths = [];
                        this.properties.currentPath = [];
                        this.updateThisNodeGraph?.();
                    }
                },
                {
                    text: "Undo",
                    x: buttonX += 55,
                    y: buttonY,
                    width: 50,
                    height: buttonHeight,
                    action: () => {
                        if (this.properties.brushPaths.length > 0) {
                            this.properties.brushPaths.pop();
                            this.updateThisNodeGraph?.();
                        }
                    }
                },
                {
                    text: "Eraser",
                    x: buttonX += 55,
                    y: buttonY,
                    width: 60,
                    height: buttonHeight,
                    isToggle: true, // 标记为切换按钮
                    action: () => {
                        // 切换模式
                        this.properties.brushMode = this.properties.brushMode === 'brush' ? 'erase' : 'brush';
                        this.updateThisNodeGraph?.();
                    }
                }
            ];
            
            // 定义滑动条（第二行，从左侧开始）
            this.properties.sliders = [
                {
                    label: "Size",
                    x: 10, // 从左侧开始
                    y: sliderY,
                    width: 150,
                    height: sliderHeight,
                    min: 1,
                    max: 200,
                    value: this.properties.brushSize,
                    type: "size",
                    dragging: false
                },
                {
                    label: "Opacity",
                    x: 170, // Size滑动条右侧
                    y: sliderY,
                    width: 150,
                    height: sliderHeight,
                    min: 0.1,
                    max: 1.0,
                    value: this.properties.brushOpacity,
                    type: "opacity",
                    dragging: false
                }
            ];
        };

        // 节点初始化
        this.node.onAdded = function () {
            this.initButtons();
        };

        // 节点配置
        this.node.onConfigure = function () {
            // 从隐藏的widget中获取图像尺寸
            const widthWidget = this.widgets.find(w => w.name === "image_width");
            const heightWidget = this.widgets.find(w => w.name === "image_height");
            
            if (widthWidget && heightWidget && widthWidget.value && heightWidget.value) {
                this.updateImageSize(widthWidget.value, heightWidget.value);
            } else {
                // 默认尺寸
                this.updateImageSize(512, 512);
            }

            // 获取画笔大小（从brush_size widget）
            const brushSizeWidget = this.widgets.find(w => w.name === "brush_size");
            if (brushSizeWidget && brushSizeWidget.value !== undefined) {
                this.properties.brushSize = brushSizeWidget.value || 20;
            } else {
                this.properties.brushSize = 20;
            }
            
            // 确保透明度已初始化
            if (this.properties.brushOpacity === undefined) {
                this.properties.brushOpacity = 1.0;
            }
            
            // 初始化滑动条值
            if (this.properties.sliders) {
                for (let slider of this.properties.sliders) {
                    if (slider.type === "size") {
                        slider.value = this.properties.brushSize;
                    } else if (slider.type === "opacity") {
                        slider.value = this.properties.brushOpacity;
                    }
                }
            }
            
            // 加载背景图（从image_base64 widget）
            const imageBase64Widget = this.widgets.find(w => w.name === "image_base64");
            if (imageBase64Widget && imageBase64Widget.value) {
                this.properties.imageBase64Data = imageBase64Widget.value;
                this.loadBackgroundImageFromBase64(imageBase64Widget.value);
            } else if (this.properties.imageBase64Data) {
                // 如果widget没有值但properties中有，恢复
                this.loadBackgroundImageFromBase64(this.properties.imageBase64Data);
            }

            // 从字符串中解析画笔数据
            this.properties.brushPaths = [];
            if (this.widgets[2] && this.widgets[2].value) {
                try {
                    const brushData = this.widgets[2].value;
                    if (brushData && brushData.trim()) {
                        // 格式: "x1,y1;x2,y2;...|x1,y1;x2,y2;..."
                        const strokes = brushData.split('|');
                        for (const stroke of strokes) {
                            if (stroke.trim()) {
                                let mode = 'brush';
                                let size = 20; // 默认值
                                let opacity = 1.0; // 默认值
                                let pointsStr = stroke;
                                
                                // 解析格式：支持多种格式
                                // 新格式：mode:size:opacity:points
                                // 旧格式1：mode:points
                                // 旧格式2：points（无模式）
                                if (stroke.includes(':')) {
                                    const parts = stroke.split(':');
                                    if (parts.length >= 2) {
                                        // 检查第一部分是否是模式
                                        if (parts[0] === 'brush' || parts[0] === 'erase') {
                                            mode = parts[0];
                                            
                                            // 检查是否有size和opacity（新格式）
                                            if (parts.length >= 4) {
                                                // 新格式：mode:size:opacity:points
                                                size = parseFloat(parts[1]) || 20;
                                                opacity = parseFloat(parts[2]) || 1.0;
                                                pointsStr = parts.slice(3).join(':'); // 处理points中可能包含冒号的情况
                                            } else {
                                                // 旧格式1：mode:points
                                                pointsStr = parts.slice(1).join(':');
                                            }
                                        } else {
                                            // 可能是旧格式，尝试解析
                                            pointsStr = stroke;
                                        }
                                    }
                                }
                                
                                const points = pointsStr.split(';');
                                const path = [];
                                for (const point of points) {
                                    if (point.trim()) {
                                        const coords = point.split(',');
                                        if (coords.length === 2) {
                                            path.push({
                                                x: parseFloat(coords[0]),
                                                y: parseFloat(coords[1])
                                            });
                                        }
                                    }
                                }
                                if (path.length > 0) {
                                    // 保存路径时包含模式、size和opacity信息
                                    this.properties.brushPaths.push({
                                        points: path,
                                        mode: mode,
                                        size: size,
                                        opacity: opacity
                                    });
                                }
                            }
                        }
                    }
                } catch (e) {
                    console.error("Error parsing brush data:", e);
                    this.properties.brushPaths = [];
                }
            }

            // 确保按钮已初始化
            this.initButtons();
        };

        // 更新图像尺寸（从widget或执行结果）
        this.node.updateImageSize = function(width, height) {
            if (width && height && width > 0 && height > 0) {
                this.properties.imageWidth = width;
                this.properties.imageHeight = height;
                
                // 更新隐藏的widget
                const widthWidget = this.widgets.find(w => w.name === "image_width");
                const heightWidget = this.widgets.find(w => w.name === "image_height");
                if (widthWidget) widthWidget.value = width;
                if (heightWidget) heightWidget.value = height;
                
                // 根据图像尺寸调整节点大小（保持合适的显示比例）
                const maxDisplaySize = 500;
                const scale = Math.min(
                    maxDisplaySize / width,
                    maxDisplaySize / height,
                    1.0
                );
                
                const displayWidth = Math.max(300, Math.min(width * scale + shiftRight + shiftLeft, 800));
                const displayHeight = Math.max(300, Math.min(height * scale + shiftLeft * 2 + panelHeight, 800));
                
                this.size = [displayWidth, displayHeight];
                
                this.updateThisNodeGraph?.();
            }
        };

        // 绘制前景
        this.node.onDrawForeground = function(ctx) {
            if (this.flags.collapsed) return false;

            const canvasWidth = this.properties.imageWidth || 512;
            const canvasHeight = this.properties.imageHeight || 512;

            // 绘制顶部控制面板
            const panelY = shiftLeft;
            ctx.fillStyle = "rgba(40,40,40,0.9)";
            ctx.beginPath();
            ctx.roundRect(shiftLeft - 4, panelY - 4, this.size[0] - shiftRight - shiftLeft + 8, panelHeight, 4);
            ctx.fill();
            
            ctx.strokeStyle = "rgba(100,100,100,0.5)";
            ctx.lineWidth = 1;
            ctx.strokeRect(shiftLeft - 4, panelY - 4, this.size[0] - shiftRight - shiftLeft + 8, panelHeight);

            // 计算实际画布区域（在面板下方）
            let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
            let canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

            // 计算缩放比例
            const scaleX = canvasAreaWidth / canvasWidth;
            const scaleY = canvasAreaHeight / canvasHeight;
            const scale = Math.min(scaleX, scaleY);

            // 计算居中偏移（画布在面板下方）
            const scaledWidth = canvasWidth * scale;
            const scaledHeight = canvasHeight * scale;
            const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
            const offsetY = shiftLeft + panelHeight + (canvasAreaHeight - scaledHeight) / 2;

            // 画布背景
            ctx.fillStyle = "rgba(20,20,20,0.8)";
            ctx.beginPath();
            ctx.roundRect(offsetX - 4, offsetY - 4, scaledWidth + 8, scaledHeight + 8, 4);
            ctx.fill();

            // 绘制背景图（如果有）
            if (this.properties.backgroundImageObj && this.properties.backgroundImageObj.complete) {
                try {
                    ctx.drawImage(
                        this.properties.backgroundImageObj,
                        offsetX,
                        offsetY,
                        scaledWidth,
                        scaledHeight
                    );
                } catch (e) {
                    console.error("Error drawing background image:", e);
                    // 如果绘制失败，显示占位符
                    ctx.fillStyle = "rgba(100,100,100,0.3)";
                    ctx.fillRect(offsetX, offsetY, scaledWidth, scaledHeight);
                }
            } else {
                // 绘制背景图占位符（显示图像区域）
                ctx.fillStyle = "rgba(100,100,100,0.3)";
                ctx.fillRect(offsetX, offsetY, scaledWidth, scaledHeight);
                
                // 绘制网格背景（帮助对齐）
                ctx.strokeStyle = "rgba(150,150,150,0.2)";
                ctx.lineWidth = 1;
                const gridSize = 32;
                const gridScale = gridSize * scale;
                
                for (let x = offsetX; x <= offsetX + scaledWidth; x += gridScale) {
                    ctx.beginPath();
                    ctx.moveTo(x, offsetY);
                    ctx.lineTo(x, offsetY + scaledHeight);
                    ctx.stroke();
                }
                
                for (let y = offsetY; y <= offsetY + scaledHeight; y += gridScale) {
                    ctx.beginPath();
                    ctx.moveTo(offsetX, y);
                    ctx.lineTo(offsetX + scaledWidth, y);
                    ctx.stroke();
                }
            }
            
            // 移除Background文字显示（用户要求）

            // 绘制所有画笔路径（先绘制画笔，再绘制橡皮擦）
            ctx.lineCap = "round";
            ctx.lineJoin = "round";

            // 第一步：绘制所有画笔路径（白色）
            for (const pathObj of this.properties.brushPaths) {
                // 兼容旧格式（直接是数组）和新格式（对象）
                const path = pathObj.points || pathObj;
                const mode = pathObj.mode || 'brush';
                // 使用路径保存的size和opacity，如果没有则使用默认值
                const pathSize = pathObj.size !== undefined ? pathObj.size : this.properties.brushSize;
                const pathOpacity = pathObj.opacity !== undefined ? pathObj.opacity : this.properties.brushOpacity;
                
                // 只绘制画笔模式，跳过橡皮擦
                if (mode === 'erase' || path.length < 2) continue;

                ctx.lineWidth = pathSize * scale;
                ctx.strokeStyle = `rgba(255,255,255,${pathOpacity})`;
                ctx.beginPath();
                for (let i = 0; i < path.length; i++) {
                    const x = offsetX + path[i].x * scale;
                    const y = offsetY + path[i].y * scale;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // 第二步：绘制橡皮擦路径（用醒目的红色显示预览）
            ctx.globalCompositeOperation = 'source-over'; // 确保使用正常混合模式
            for (const pathObj of this.properties.brushPaths) {
                const path = pathObj.points || pathObj;
                const mode = pathObj.mode || 'brush';
                // 使用路径保存的size和opacity
                const pathSize = pathObj.size !== undefined ? pathObj.size : this.properties.brushSize;
                const pathOpacity = pathObj.opacity !== undefined ? pathObj.opacity : this.properties.brushOpacity;
                
                // 只绘制橡皮擦模式
                if (mode !== 'erase' || path.length < 2) continue;

                ctx.lineWidth = pathSize * scale;
                // 用醒目的红色显示橡皮擦路径（完全使用opacity滑动条的值）
                ctx.strokeStyle = `rgba(255,50,50,${pathOpacity})`;
                ctx.beginPath();
                for (let i = 0; i < path.length; i++) {
                    const x = offsetX + path[i].x * scale;
                    const y = offsetY + path[i].y * scale;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // 第三步：绘制当前正在绘制的路径（使用当前的size和opacity）
            if (this.properties.currentPath.length > 1) {
                ctx.globalCompositeOperation = 'source-over'; // 确保使用正常混合模式
                ctx.lineWidth = this.properties.brushSize * scale;
                
                if (this.properties.brushMode === 'erase') {
                    // 橡皮擦模式：用醒目的红色显示预览（完全使用opacity滑动条的值）
                    ctx.strokeStyle = `rgba(255,50,50,${this.properties.brushOpacity})`;
                } else {
                    // 画笔模式：正常绘制白色
                    ctx.strokeStyle = `rgba(255,255,255,${this.properties.brushOpacity})`;
                }
                
                ctx.beginPath();
                for (let i = 0; i < this.properties.currentPath.length; i++) {
                    const x = offsetX + this.properties.currentPath[i].x * scale;
                    const y = offsetY + this.properties.currentPath[i].y * scale;
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // 显示图像尺寸（在画布下方）
            ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
            ctx.font = (fontsize) + "px Arial";
            ctx.textAlign = "center";
            ctx.fillText(`${canvasWidth}×${canvasHeight}`, this.size[0]/2, offsetY + scaledHeight + 15);

            // 绘制顶部面板按钮
            for (let button of this.properties.buttons) {
                // 按钮背景（切换按钮特殊处理）
                if (button.isToggle && button.text === 'Eraser') {
                    // 橡皮擦按钮：激活时显示红色，未激活时显示灰色
                    if (this.properties.brushMode === 'erase') {
                        ctx.fillStyle = "rgba(255,100,100,0.8)";
                    } else {
                        ctx.fillStyle = "rgba(60,60,60,0.7)";
                    }
                } else {
                    ctx.fillStyle = "rgba(60,60,60,0.7)";
                }
                ctx.fillRect(button.x, button.y, button.width, button.height);

                // 按钮边框
                ctx.strokeStyle = "rgba(150,150,150,0.6)";
                ctx.lineWidth = 1;
                ctx.strokeRect(button.x, button.y, button.width, button.height);

                // 按钮文本
                ctx.fillStyle = "rgba(220,220,220,0.9)";
                ctx.font = "11px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(button.text, button.x + button.width/2, button.y + button.height/2);
            }
            
            // 绘制滑动条
            for (let slider of this.properties.sliders) {
                // 更新滑动条当前值
                if (slider.type === "size") {
                    slider.value = this.properties.brushSize;
                } else if (slider.type === "opacity") {
                    slider.value = this.properties.brushOpacity;
                }
                
                // 计算滑块位置（0-1范围）
                const ratio = (slider.value - slider.min) / (slider.max - slider.min);
                const sliderX = slider.x;
                const sliderY = slider.y;
                const sliderWidth = slider.width;
                const sliderHeight = slider.height;
                const thumbSize = 14; // 滑块大小
                const thumbX = sliderX + ratio * (sliderWidth - thumbSize);
                const thumbY = sliderY - 1;
                
                // 绘制滑动条背景
                ctx.fillStyle = "rgba(50,50,50,0.8)";
                ctx.fillRect(sliderX, sliderY, sliderWidth, sliderHeight);
                
                // 绘制滑动条轨道
                ctx.strokeStyle = "rgba(100,100,100,0.6)";
                ctx.lineWidth = 1;
                ctx.strokeRect(sliderX, sliderY, sliderWidth, sliderHeight);
                
                // 绘制已填充部分（可选，显示进度）
                ctx.fillStyle = "rgba(100,150,255,0.4)";
                ctx.fillRect(sliderX, sliderY, ratio * sliderWidth, sliderHeight);
                
                // 绘制滑块
                ctx.fillStyle = "rgba(120,170,255,0.9)";
                ctx.beginPath();
                ctx.roundRect(thumbX, thumbY, thumbSize, thumbSize + 2, 2);
                ctx.fill();
                
                ctx.strokeStyle = "rgba(150,200,255,1.0)";
                ctx.lineWidth = 1;
                ctx.strokeRect(thumbX, thumbY, thumbSize, thumbSize + 2);
                
                // 绘制标签和值
                ctx.fillStyle = "rgba(200,200,200,0.9)";
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                let valueText = "";
                if (slider.type === "size") {
                    valueText = `${slider.label}: ${Math.round(slider.value)}`;
                } else if (slider.type === "opacity") {
                    valueText = `${slider.label}: ${Math.round(slider.value * 100)}%`;
                }
                ctx.fillText(valueText, sliderX, sliderY + sliderHeight + 2);
            }

            // 更新画笔数据字符串到widget（包含模式、size和opacity信息）
            // 格式：mode:size:opacity:x1,y1;x2,y2;...|mode:size:opacity:x1,y1;x2,y2;...
            const brushDataStrings = this.properties.brushPaths.map(pathObj => {
                // 兼容旧格式
                const path = pathObj.points || pathObj;
                const mode = pathObj.mode || 'brush';
                const size = pathObj.size !== undefined ? pathObj.size : this.properties.brushSize;
                const opacity = pathObj.opacity !== undefined ? pathObj.opacity : this.properties.brushOpacity;
                const pointsStr = path.map(p => `${p.x},${p.y}`).join(';');
                // 格式：mode:size:opacity:points
                return `${mode}:${size}:${opacity}:${pointsStr}`;
            });
            const brushDataWidget = this.widgets.find(w => w.name === "brush_data");
            if (brushDataWidget) {
                brushDataWidget.value = brushDataStrings.join('|');
            } else if (this.widgets[2]) {
                // 备用：如果找不到widget，使用索引
                this.widgets[2].value = brushDataStrings.join('|');
            }
        };

        // 鼠标按下事件
        this.node.onMouseDown = function(e) {
            if (e.canvasY - this.pos[1] < 0) return false;

            // 检查按钮点击（在顶部面板区域）
            const mouseX = e.canvasX - this.pos[0];
            const mouseY = e.canvasY - this.pos[1];
            
            // 检查是否在面板区域内
            if (mouseY >= shiftLeft && mouseY <= shiftLeft + panelHeight) {
                // 检查滑动条
                for (let slider of this.properties.sliders) {
                    const sliderX = slider.x;
                    const sliderY = slider.y;
                    const sliderWidth = slider.width;
                    const sliderHeight = slider.height;
                    const thumbSize = 14;
                    
                    // 检查是否点击在滑动条区域（包括标签区域）
                    if (mouseX >= sliderX && mouseX <= sliderX + sliderWidth &&
                        mouseY >= sliderY - 5 && mouseY <= sliderY + sliderHeight + 15) {
                        // 开始拖拽
                        slider.dragging = true;
                        this.updateSliderValue(slider, mouseX);
                        this.capture = true;
                        this.captureInput(true);
                        return true;
                    }
                }
                
                // 检查按钮
                for (let button of this.properties.buttons) {
                    if (button.action && mouseX >= button.x && mouseX <= button.x + button.width &&
                        mouseY >= button.y && mouseY <= button.y + button.height) {
                        button.action();
                        return true;
                    }
                }
                // 点击在面板上但不是按钮或滑动条，不处理
                return false;
            }

            // 检查是否点击在画布区域（排除面板区域）
            const canvasWidth = this.properties.imageWidth || 512;
            const canvasHeight = this.properties.imageHeight || 512;
            let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
            let canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

            const scaleX = canvasAreaWidth / canvasWidth;
            const scaleY = canvasAreaHeight / canvasHeight;
            const scale = Math.min(scaleX, scaleY);

            const scaledWidth = canvasWidth * scale;
            const scaledHeight = canvasHeight * scale;
            const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
            const offsetY = shiftLeft + panelHeight + (canvasAreaHeight - scaledHeight) / 2;

            if (e.canvasX < this.pos[0] + offsetX || 
                e.canvasX > this.pos[0] + offsetX + scaledWidth) return false;
            if (e.canvasY < this.pos[1] + offsetY || 
                e.canvasY > this.pos[1] + offsetY + scaledHeight) return false;

            // 左键开始绘制
            if (e.button === 0) {
                let mouseX = e.canvasX - this.pos[0] - offsetX;
                let mouseY = e.canvasY - this.pos[1] - offsetY;

                // 转换为实际坐标
                let realX = mouseX / scale;
                let realY = mouseY / scale;

                // 确保坐标在有效范围内
                realX = Math.max(0, Math.min(realX, canvasWidth - 1));
                realY = Math.max(0, Math.min(realY, canvasHeight - 1));

                this.properties.isDrawing = true;
                this.properties.currentPath = [{x: realX, y: realY}];
                
                this.capture = true;
                this.captureInput(true);
                return true;
            }

            return false;
        };

        // 鼠标移动事件
        this.node.onMouseMove = function(e, pos, canvas) {
            if (!this.capture) return;
            
            // 检查是否在拖拽滑动条
            let isDraggingSlider = false;
            for (let slider of this.properties.sliders) {
                if (slider.dragging) {
                    const mouseX = e.canvasX - this.pos[0];
                    this.updateSliderValue(slider, mouseX);
                    isDraggingSlider = true;
                    break;
                }
            }
            
            if (isDraggingSlider) {
                return;
            }
            
            // 处理画笔绘制
            if (!this.properties.isDrawing) return;
            if (canvas.pointer.isDown === false) { 
                this.onMouseUp(e); 
                return; 
            }
            this.valueUpdate(e);
        };
        
        // 更新滑动条值
        this.node.updateSliderValue = function(slider, mouseX) {
            const sliderX = slider.x;
            const sliderWidth = slider.width;
            const thumbSize = 14;
            
            // 计算鼠标在滑动条上的位置比例
            let ratio = (mouseX - sliderX - thumbSize / 2) / (sliderWidth - thumbSize);
            ratio = Math.max(0, Math.min(1, ratio)); // 限制在0-1范围
            
            // 计算新值
            const newValue = slider.min + ratio * (slider.max - slider.min);
            
            // 更新值
            if (slider.type === "size") {
                this.properties.brushSize = Math.round(newValue);
                slider.value = this.properties.brushSize;
                // 更新widget
                const brushSizeWidget = this.widgets.find(w => w.name === "brush_size");
                if (brushSizeWidget) {
                    brushSizeWidget.value = this.properties.brushSize;
                }
            } else if (slider.type === "opacity") {
                this.properties.brushOpacity = Math.round(newValue * 100) / 100; // 保留2位小数
                slider.value = this.properties.brushOpacity;
            }
            
            // 触发重绘，确保size和opacity变化立即反映在绘制中
            this.updateThisNodeGraph?.();
        };

        // 更新值
        this.node.valueUpdate = function(e) {
            if (!this.properties.isDrawing) return;

            const canvasWidth = this.properties.imageWidth || 512;
            const canvasHeight = this.properties.imageHeight || 512;
            let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
            let canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

            const scaleX = canvasAreaWidth / canvasWidth;
            const scaleY = canvasAreaHeight / canvasHeight;
            const scale = Math.min(scaleX, scaleY);

            const scaledWidth = canvasWidth * scale;
            const scaledHeight = canvasHeight * scale;
            const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
            const offsetY = shiftLeft + panelHeight + (canvasAreaHeight - scaledHeight) / 2;

            let mouseX = e.canvasX - this.pos[0] - offsetX;
            let mouseY = e.canvasY - this.pos[1] - offsetY;

            // 转换为实际坐标
            let realX = mouseX / scale;
            let realY = mouseY / scale;

            // 确保坐标在有效范围内
            realX = Math.max(0, Math.min(realX, canvasWidth - 1));
            realY = Math.max(0, Math.min(realY, canvasHeight - 1));

            // 添加到当前路径
            const lastPoint = this.properties.currentPath[this.properties.currentPath.length - 1];
            const dist = Math.sqrt(
                Math.pow(realX - lastPoint.x, 2) + 
                Math.pow(realY - lastPoint.y, 2)
            );

            // 只有当距离足够大时才添加点（优化性能）
            if (dist > 1) {
                this.properties.currentPath.push({x: realX, y: realY});
                this.updateThisNodeGraph?.();
            }
        };

        // 鼠标释放事件
        this.node.onMouseUp = function() {
            if (!this.capture) return;
            
            // 停止滑动条拖拽
            for (let slider of this.properties.sliders) {
                if (slider.dragging) {
                    slider.dragging = false;
                    this.capture = false;
                    this.captureInput(false);
                    return;
                }
            }
            
            if (this.properties.isDrawing && this.properties.currentPath.length > 0) {
                // 保存当前路径（包含模式、size和opacity信息）
                this.properties.brushPaths.push({
                    points: [...this.properties.currentPath],
                    mode: this.properties.brushMode,
                    size: this.properties.brushSize, // 保存绘制时的size
                    opacity: this.properties.brushOpacity // 保存绘制时的opacity
                });
                this.properties.currentPath = [];
                
                // 立即更新画笔数据到widget（包含size和opacity）
                const brushDataStrings = this.properties.brushPaths.map(pathObj => {
                    const path = pathObj.points || pathObj;
                    const mode = pathObj.mode || 'brush';
                    const size = pathObj.size !== undefined ? pathObj.size : this.properties.brushSize;
                    const opacity = pathObj.opacity !== undefined ? pathObj.opacity : this.properties.brushOpacity;
                    const pointsStr = path.map(p => `${p.x},${p.y}`).join(';');
                    // 格式：mode:size:opacity:points
                    return `${mode}:${size}:${opacity}:${pointsStr}`;
                });
                const brushDataWidget = this.widgets.find(w => w.name === "brush_data");
                if (brushDataWidget) {
                    brushDataWidget.value = brushDataStrings.join('|');
                    console.log("Updated brush_data widget:", brushDataWidget.value.substring(0, 100) + "...");
                }
            }
            
            this.properties.isDrawing = false;
            this.capture = false;
            this.captureInput(false);
            this.updateThisNodeGraph?.();
        };

        // 节点选中事件
        this.node.onSelected = function() {
            this.onMouseUp();
        };

        // 监听节点执行完成，更新图像尺寸
        // 注意：由于ComfyUI的限制，前端无法直接访问执行结果
        // 我们通过以下方式获取尺寸：
        // 1. 如果width和height输出被连接，可以通过监听输出值变化
        // 2. 或者通过其他机制获取
        // 目前采用：前端在节点执行后，通过读取输出的width和height值来更新
        // 但由于限制，我们采用更实用的方法：在节点配置时从输入图像获取尺寸
        
        // 监听输入连接变化，当背景图连接时，尝试获取尺寸
        const originalOnConnectionsChange = this.node.onConnectionsChange;
        this.node.onConnectionsChange = function(type, slot, isInput, link, info) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
            
            // 当背景图输入连接时，延迟更新尺寸
            if (isInput && slot === 0 && type === 1) {
                // 连接建立，等待执行后更新尺寸
                setTimeout(() => {
                    // 尝试从输出获取尺寸（如果已执行）
                    // 由于无法直接访问，我们使用默认值或等待用户执行
                }, 100);
            }
        };
        
        // 监听节点执行完成（通过hook机制）
        // 注意：这需要ComfyUI支持，如果不行，可以通过其他方式
        const originalOnAfterExecuteNode = this.node.onAfterExecuteNode;
        this.node.onAfterExecuteNode = function(message) {
            if (originalOnAfterExecuteNode) {
                originalOnAfterExecuteNode.apply(this, arguments);
            }
            
            // 尝试从执行结果中获取base64图像数据
            // 注意：ComfyUI的执行结果可能不直接可用，这里尝试通过其他方式获取
            // 如果image_base64输出被连接，可以通过监听输出值变化来获取
        };
        
        // 监听widget值变化
        const originalOnWidgetChange = this.node.onWidgetChange;
        this.node.onWidgetChange = function(widget) {
            if (originalOnWidgetChange) {
                originalOnWidgetChange.apply(this, arguments);
            }
            
            // 如果画笔大小widget值变化，更新画笔大小
            if (widget && widget.name === "brush_size") {
                this.properties.brushSize = widget.value || 20;
                this.updateThisNodeGraph?.();
            }
            
            // 如果width或height widget值变化，更新显示
            if (widget && (widget.name === "image_width" || widget.name === "image_height")) {
                const widthWidget = this.widgets.find(w => w.name === "image_width");
                const heightWidget = this.widgets.find(w => w.name === "image_height");
                
                if (widthWidget && heightWidget && widthWidget.value && heightWidget.value) {
                    this.updateImageSize(widthWidget.value, heightWidget.value);
                }
            }
            
            // 如果image_base64 widget值变化，加载背景图
            if (widget && widget.name === "image_base64") {
                if (widget.value) {
                    this.properties.imageBase64Data = widget.value;
                    this.loadBackgroundImageFromBase64(widget.value);
                } else {
                    this.properties.backgroundImageObj = null;
                    this.properties.imageBase64Data = "";
                    this.updateThisNodeGraph?.();
                }
            }
        };
        
        // 从文件加载图片
        this.node.loadImageFromFile = function() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (!file) return;
                
                const reader = new FileReader();
                reader.onload = (event) => {
                    try {
                        const dataURL = event.target.result;
                        // 获取base64数据（去掉data:image/xxx;base64,前缀）
                        let base64String = dataURL;
                        if (dataURL.includes(',')) {
                            base64String = dataURL.split(',')[1];
                        }
                        
                        // 存储base64数据（纯base64，不带前缀）
                        this.properties.imageBase64Data = base64String;
                        
                        // 更新widget
                        const imageBase64Widget = this.widgets.find(w => w.name === "image_base64");
                        if (imageBase64Widget) {
                            imageBase64Widget.value = base64String;
                        }
                        
                        // 加载图片显示（使用完整data URL）
                        this.loadBackgroundImageFromBase64(dataURL);
                        
                        // 清除之前的画笔路径（因为图片尺寸可能变化）
                        this.properties.brushPaths = [];
                        this.properties.currentPath = [];
                        
                        console.log("Image loaded successfully, size:", base64String.length, "bytes");
                    } catch (e) {
                        console.error("Error processing image file:", e);
                        alert("加载图片失败: " + e.message);
                    }
                };
                reader.onerror = (e) => {
                    console.error("Error reading file:", e);
                    alert("读取文件失败");
                };
                reader.readAsDataURL(file);
            };
            input.click();
        };
        
        // 从base64加载背景图
        this.node.loadBackgroundImageFromBase64 = function(base64String) {
            if (!base64String || base64String.trim() === "") {
                this.properties.backgroundImageObj = null;
                this.updateThisNodeGraph?.();
                return;
            }
            
            try {
                const img = new Image();
                img.onload = () => {
                    this.properties.backgroundImageObj = img;
                    // 更新图像尺寸
                    this.updateImageSize(img.width, img.height);
                    this.updateThisNodeGraph?.();
                };
                img.onerror = (e) => {
                    console.error("Error loading background image from base64:", e);
                    this.properties.backgroundImageObj = null;
                };
                // 如果base64字符串不包含前缀，添加它（假设是PNG格式）
                if (base64String.startsWith('data:')) {
                    img.src = base64String;
                } else {
                    // 纯base64字符串，添加data URL前缀
                    img.src = "data:image/png;base64," + base64String;
                }
            } catch (e) {
                console.error("Error creating image from base64:", e);
                this.properties.backgroundImageObj = null;
            }
        };
    }
}

// author.yichengup.Loadimage_brushmask 2025.01.XX
app.registerExtension({
    name: "ycimagebrushmask",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "ycimagebrushmask") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.ycimagebrushmask = new ycimagebrushmask(this);
                // 确保输出名称和类型正确设置
                if (this.initButtons) {
                    this.initButtons();
                }
            }
        }
    }
});

// author.yichengup.Loadimage_brushmask 2025.01.XX

