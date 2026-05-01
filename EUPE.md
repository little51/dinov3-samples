# Efficient Universal Perception Encoder (EUPE) Demo

## 一、克隆源码

```shell
git clone https://gitclone.com/github.com/facebookresearch/eupe --depth=1
```

## 二、安装基础环境

```shell
# 创建虚拟环境
conda create -n eupe python=3.12 -y
# 激活虚拟环境
conda activate eupe
# 安装依赖库
pip install -r eupe/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装requests库
pip install requests -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 三、下载模型权重

```shell
wget https://hf-mirror.com/facebook/EUPE-ViT-T/blob/main/EUPE-ViT-T.pt
```

## 四、运行Demo

```shell
python eupe_demo.py
```

## 五、训练

```shell
# 激活虚拟环境
conda activate eupe
# 安装timm库
pip install timm -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装PyTorch（仅Windows）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 训练
python eupe_train.py
```

