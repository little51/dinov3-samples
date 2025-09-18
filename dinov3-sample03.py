from transformers import pipeline
from transformers.image_utils import load_image
import clip
import torch
import torch.nn as nn
import torch

url = "./image04.png"

image = load_image(url)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = pipeline(
    model="./models/facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction",
    device=device
)
features = feature_extractor(image)
# shape: [1, 768]
image_features = torch.tensor(features[0][0]).unsqueeze(0)
# 创建投影层将768维映射到512维
projection_layer = nn.Linear(768, 512).to("cpu")
# 装载ViT-B模型
model, preprocess = clip.load("ViT-B/32", device="cpu")
# 使用CLIP处理文本
class_names = ["cat", "dog", "car", "bird", "person"]
text_inputs = torch.cat(
    [clip.tokenize(f"a photo of a {c}") for c in class_names]).to("cpu")
with torch.no_grad():
    # 投影图像特征到512维
    projected_image_features = projection_layer(image_features.to("cpu"))
    projected_image_features = projected_image_features / \
        projected_image_features.norm(dim=1, keepdim=True)
    # 获取文本特征
    text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 计算相似度
    similarity = (100.0 * projected_image_features @
                  text_features.T).softmax(dim=-1)
# 输出结果
values, indices = similarity[0].topk(len(class_names))
print("零样本分类结果:")
for value, index in zip(values, indices):
    print(f"{class_names[index]}: {value.item():.3f}")