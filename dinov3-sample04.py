from PIL import Image
import torch
from dinov3.data.transforms import make_classification_eval_transform
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l


def load_image_from_file(file_path):
    img_pil = Image.open(file_path).convert("RGB")
    return img_pil


def main():
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
    img_pil = load_image_from_file("./image05.jpg")
    image_preprocess = make_classification_eval_transform()
    image_tensor = torch.stack([image_preprocess(img_pil)], dim=0).cuda()
    texts = ["photo of cat", "photo of a dog",
             "photo of a bird", "photo of a person"]
    class_names = ["cat", "dog", "bird", "person"]
    tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()
    model = model.cuda()
    with torch.autocast('cuda', dtype=torch.float):
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(tokenized_texts_tensor)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (
        text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
    )
    for i, class_name in enumerate(class_names):
        score = similarity[i][0]
        print(f"{class_name}: {score:.3f}")


if __name__ == "__main__":
    main()
