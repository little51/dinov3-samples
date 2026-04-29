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
# 
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

