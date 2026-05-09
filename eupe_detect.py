import lightly_train
import urllib.request
import cv2
import numpy as np

def visualize_detection(image_path, results, model, output_path="detection_result.jpg"):
    # 读取图片
    image = cv2.imread(image_path)
    
    # 定义颜色
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    # 绘制每个检测目标
    for idx, (label, score, box) in enumerate(zip(results["labels"], results["scores"], results["bboxes"])):
        # 转换label
        label_id = int(label) if hasattr(label, 'item') else label
        
        # 获取坐标
        if len(box) == 4:
            if hasattr(box[0], 'item'):
                x1, y1, x2, y2 = [int(b.item()) for b in box]
            else:
                x1, y1, x2, y2 = [int(b) for b in box]
        else:
            if hasattr(box[0], 'item'):
                cx, cy, w, h = [int(b.item()) for b in box]
            else:
                cx, cy, w, h = [int(b) for b in box]
            x1, y1, x2, y2 = cx - w//2, cy - h//2, cx + w//2, cy + h//2
        
        # 绘制边界框
        color = colors[idx % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 添加标签
        class_name = model.classes[label_id]
        confidence = float(score) if hasattr(score, 'item') else score
        label_text = f"{class_name}: {confidence:.2%}"
        
        # 绘制标签背景和文字
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 添加统计信息
    info_text = f"Total: {len(results['labels'])} objects"
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite(output_path, image)
    print(f"可视化结果已保存至: {output_path}")

def detect():
    # 下载图片
    urllib.request.urlretrieve(
        "http://images.cocodataset.org/val2017/000000577932.jpg", "test.jpg")
    # 加载模型并预测
    model = lightly_train.load_model("dinov3/convnext-tiny-ltdetr-coco")
    results = model.predict("test.jpg")
    # 打印结果
    print(f"检测到 {len(results['labels'])} 个目标：")
    for label, score in zip(results["labels"], results["scores"]):
        label_id = int(label) if hasattr(label, 'item') else label
        print(f"  {model.classes[label_id]}: {float(score):.2%}")
    # 可视化
    visualize_detection("test.jpg", results, model, "detection_result.jpg")

if __name__ == "__main__":
    detect()