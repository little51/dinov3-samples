# DINOv3-Samples

Meta开源DINOv3视觉大模型，采用了无需标注的自学习技术，开启了人工智能图像处理的先河，是这个领域未来的方向。本项目是本地部署DINOv3模型的方法说明和一些样例介绍。

关于DINOv3的训练另开了[训练示例](https://github.com/little51/dinov3-train)项目。

## 一、基础条件

NVIDIA显卡（4G或以上），CUDA12.4或以上，如没有GPU资源，在CPU上也可以运行。

## 二、Python虚拟化工具

本文中，前3个例子依赖环境较少，采用uv管理Python虚拟化。后面的例子由于要从源码安装dinov3库，依赖环境较复杂，所以使用了Miniconda管理Python虚拟化。

1、uv从https://aliendao.cn/uv/uv-x86_64-pc-windows-msvc.zip 下载，解压后将目录加到操作系统环境变量的Path中，其实这里只用到uv.exe。

2、Miniconda的安装方法见：https://docs.anaconda.net.cn/miniconda/install/

## 三、创建Python虚拟环境

```shell
# 1、Clone代码
git clone https://github.com/little51/dinov3-samples
# 2、切换工作目录
cd dinov3-samples
# 3、设置Python镜像
set UV_PYTHON_INSTALL_MIRROR=https://aliendao.cn/uv
# 4、设置库镜像
set UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# 5、创建虚拟环境
uv init -p=3.12
# 6、验证虚拟环境
uv run python --version
```

## 四、安装依赖库

```shell
# 1、安装transformers
uv add transformers==4.56.1
# 2、安装scikit-learn，相似度比对要用到
uv add  scikit-learn==1.7.2
# 3、重装PyTorch（因为Windows从pypi.org自动安装的是CPU版的，要从PyTorch官网重装）
set UV_DEFAULT_INDEX=
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 4、验证PyTorch（如果显示True则为正常）
uv run --no-project python -c "import torch; print(torch.cuda.is_available())"
```

## 五、下载模型

```shell
uv run python model_download2.py --repo_id facebook/dinov3-convnext-tiny-pretrain-lvd1689m
```

## 六、测试例程

### 例1：获取图片的features

```shell
uv run --no-project python dinov3-sample01.py
## 结果
Device set to use cuda:0
[[[-3.0382916927337646, -0.3411354124546051, 2.226456642150879 ... ...
```

### 例2：比对图片

```shell
uv run --no-project python dinov3-sample02.py
## 结果
Device set to use cuda:0
图1与图2相似度: 0.3949535340831939
图1与图3相似度: 0.936580648208369
```

### 例3：零样本分类（与clip配合）

```shell
uv add openai-clip==1.0.1
uv pip install --upgrade setuptools pip
set HF_ENDPOINT=https://hf-mirror.com
uv run --no-project python dinov3-sample03.py
## 结果
零样本分类结果:
dog: 0.513
cat: 0.211
bird: 0.141
person: 0.116
car: 0.020
```

### 例4：零样本分类（dinotxt）

```shell
# 1、安装dinov3
git clone https://github.com/facebookresearch/dinov3
cd dinov3
git checkout 1e358a2
# 2、创建虚拟环境
conda create -n dinov3 python=3.12 -y
# 3、激活虚拟环境
conda activate dinov3
# 4、安装dinov3及其他依赖库
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple
pip install transformers==4.56.1 -i https://pypi.mirrors.ustc.edu.cn/simple
# 5、验证是否安装成功
python -c "import torch; print(torch.cuda.is_available())"
# 如不成功，重装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 6、下载权重
# 从以下地址下载两个权重文件，放到C:\Users\用户名\.cache\torch\hub\checkpoints目录下
https://aliendao.cn/models/facebook/dinov3_vitl16_dinotxt_tet1280d20h24l
# 7、运行实例
cd ..
python dinov3-sample04.py
# 运行结果
cat: 0.076
dog: 0.120
bird: 0.055
person: 0.062
```

### 例5：训练前景分割工具

```shell
conda activate dinov3
pip install matplotlib==3.10.6
python dinov3-sample05.py
```

### 例6：彩色语义分割图

```shell
conda activate dinov3
python dinov3-sample06.py
```

### 例7：目标检测

```shell
# 1、克隆本项目源码
git clone https://github.com/little51/dinov3-samples
cd dinov3-samples
# 2、安装dinov3（在本项目目录下）
git clone https://github.com/facebookresearch/dinov3
cd dinov3
git checkout 1e358a2
# 3、创建虚拟环境
conda create -n dinov3 python=3.12 -y
# 4、激活虚拟环境
conda activate dinov3
# 5、安装dinov3及其他依赖库
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple
pip install transformers==4.56.1 -i https://pypi.mirrors.ustc.edu.cn/simple
pip install matplotlib==3.10.6 -i https://pypi.mirrors.ustc.edu.cn/simple
# 6、验证是否安装成功
python -c "import torch; print(torch.cuda.is_available())"
# 如不成功，重装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 7、下载权重
# 从以下地址下载两个权重文件，放到项目的weights目录下
https://aliendao.cn/models/facebook/dinov3_vit7b16_pretrain_lvd1689m#/
# 8、运行实例
cd ..
python dinov3-sample07.py
```

## 作者新书《大模型项目实战：多领域智能应用》和《大模型项目实战：Agent开发与应用》技术交流群

![交流群](https://gitclone.com/download1/aliendao/aliendao20251117.jpg)
![图书](https://gitclone.com/download1/llm-dev/llm-dev.png)
![图书](https://gitclone.com/download1/ai-agent/agent-dev1.png)