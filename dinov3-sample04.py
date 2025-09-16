import urllib.request
from PIL import Image
import torch
from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l


def load_image_from_url(url):
    with urllib.request.urlopen(url) as f:
        img_pil = Image.open(f).convert("RGB")
    return img_pil


def load_model_from_local():
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
    return model, tokenizer


def main():
    # 加载模型和分词器
    model, tokenizer = load_model_from_local()
    # 加载示例图像
    url = "./image01.jpeg"
    img_pil = load_image_from_url(url)
    # 图像预处理
    image_preprocess = make_classification_eval_transform()
    image_tensor = torch.stack([image_preprocess(img_pil)], dim=0)
    # 文本准备
    texts = ["photo of dogs", "photo of a cat",
             "photo of a person", "photo of a bird"]
    class_names = ["dog", "cat", "person", "bird"]
    # 将数据移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    # 文本编码
    tokenized_texts_tensor = tokenizer(
        texts, return_tensors="pt", padding=True).to(device)
    # 模型推理
    with torch.autocast(device.type, dtype=torch.float):
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_texts_tensor)
    # 特征归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # 计算相似度
    similarity = (
        text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
    )
    # 打印结果
    print("图像与文本的相似度:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {similarity[i][0]:.4f}")
    # 找出最匹配的类别
    best_match_idx = similarity.argmax()
    print(
        f"\n最匹配的类别: {class_names[best_match_idx]} (相似度: {similarity[best_match_idx][0]:.4f})")


if __name__ == "__main__":
    main()
