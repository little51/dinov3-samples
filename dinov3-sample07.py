import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

DINOV3_LOCATION = "./dinov3"
MODEL_DINOV3_VITL = "dinov3_vit7b16_de"
MODEL_NAME = MODEL_DINOV3_VITL

detector = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local",
    weights='./weights/dinov3_vit7b16_coco_detr_head-b0235ff7.pth',
    backbone_weights='./weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
)
# detector.cuda()


def detector_img(img_filename):
    pil_img = Image.open(img_filename).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((768, 768)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.inference_mode():
        detections = detector(input_tensor)[0]

    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for score, label, box in zip(detections['scores'], detections['labels'], detections['boxes']):
        if score.item() > 0.75:  # threshold
            x0, y0, x1, y1 = box.cpu().numpy()
            class_idx = int(label.item())
            class_name = COCO_CLASSES[class_idx] if 0 <= class_idx < len(
                COCO_CLASSES) else str(class_idx)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, color='red', linewidth=2))
            ax.text(x0, y0, f'{class_name}: {score:.2f}', fontsize=12,
                    color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    # plt.show()
    plt.savefig('_' + img_filename, bbox_inches='tight', dpi=100)
    plt.close()


if __name__ == "__main__":
    detector_img('image06.png')
    detector_img('image07.png')
    detector_img('image08.png')
