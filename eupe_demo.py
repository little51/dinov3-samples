import torch
import torchvision
from torchvision.transforms import v2
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import requests

# ============================================================
# 源码：git clone https://github.com/facebookresearch/eupe
# 模型：https://hf-mirror.com/facebook/EUPE-ViT-T/blob/main/EUPE-ViT-T.pt
# ============================================================
REPO_DIR = "eupe"
CHECKPOINT_PATH = "EUPE-ViT-T.pt"

def get_img(url=None):
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def make_transform(resize_size=256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def pca_visualize(patchtokens, output_path="eupe_pca.png"):
    """将 patch tokens 用 PCA 降到 3 维，映射到 RGB 并保存"""
    # patchtokens: [1, N, D]
    features = patchtokens[0].cpu().float().numpy()
    N, D = features.shape

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(features)

    # 归一化到 [0, 1]
    rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)

    h = w = int(np.sqrt(N))
    vis = rgb.reshape(h, w, 3)

    # 放大以便观看
    img = Image.fromarray(vis).resize((256, 256), Image.NEAREST)
    img.save(output_path)
    print(f"PCA 可视化已保存到 {output_path}")
    print(f"  方差解释率: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
          f"PC3={pca.explained_variance_ratio_[2]:.3f}")

def run_with_hub(image_path=None):
    if REPO_DIR is None or CHECKPOINT_PATH is None:
        print("请设置环境变量 EUPE_REPO 和 EUPE_CKPT")
        return

    model = torch.hub.load(REPO_DIR, 'eupe_vitt16', source='local',
                           weights=CHECKPOINT_PATH)
    model.eval()

    img = get_img(image_path)
    transform = make_transform(256)

    with torch.inference_mode():
        batch_img = transform(img)[None]
        outputs = model.forward_features(batch_img)

    clstoken = outputs["x_norm_clstoken"]
    patchtokens = outputs["x_norm_patchtokens"]

    print(f"class token shape:  {clstoken.shape}")   # [1, 192]
    print(f"patch tokens shape: {patchtokens.shape}") # [1, 256, 192]

    pca_visualize(patchtokens)



if __name__ == "__main__":
    run_with_hub("http://images.cocodataset.org/val2017/000000039769.jpg")
