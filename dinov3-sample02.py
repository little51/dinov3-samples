from transformers import pipeline
from transformers.image_utils import load_image
from sklearn.metrics.pairwise import cosine_similarity

url1 = "./image01.jpeg"
url2 = "./image02.jpg"
url3 = "./image03.png"

image1 = load_image(url1)
image2 = load_image(url2)
image3 = load_image(url3)


feature_extractor = pipeline(
    model="./models/facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features1 = feature_extractor(image1)[0][0]
features2 = feature_extractor(image2)[0][0]
features3 = feature_extractor(image3)[0][0]

similarity = cosine_similarity([features1], [features2])
print(f"图1与图2相似度: {similarity[0][0]}")

similarity = cosine_similarity([features1], [features3])
print(f"图1与图3相似度: {similarity[0][0]}")