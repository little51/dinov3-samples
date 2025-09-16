# DINOv3-Samples

Meta开源DINOv3视觉大模型，采用了无需标注的自学习技术，开启了人工智能图像处理的先河，是这个领域未来的方向。本项目是本地部署DINOv3模型的方法说明和一些样例介绍。

## 一、基础条件

NVIDIA显卡（4G或以上），CUDA12.4或以上。

## 二、Python虚拟化工具UV安装

官网上介绍的是用[MinoConda](https://zhida.zhihu.com/search?content_id=261875970&content_type=Article&match_order=1&q=MinoConda&zhida_source=entity)，这个有点重量级，本文采用uv管理Python虚拟化。uv从

```shell
https://aliendao.cn/uv/uv-x86_64-pc-windows-msvc.zip
```

 下载，解压后将目录加到操作系统环境变量的Path中，其实这里只用到uv.exe。

## 三、创建Python虚拟环境

```shell
# 1、Clone代码
https://github.com/little51/dinov3-samples
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

![](https://github.com/little51/dinov3-samples/blob/main/image01.jpeg)
![](https://github.com/little51/dinov3-samples/blob/main/image02.jpg)
![](https://github.com/little51/dinov3-samples/blob/main/image03.png)

```shell
# 例1，获取图片的features
uv run --no-project python dinov3-sample01.py
## 结果
Device set to use cuda:0
[[[-3.0382916927337646, -0.3411354124546051, 2.226456642150879 ... ...
# 例2，比对图片
uv run --no-project python dinov3-sample02.py
## 结果
Device set to use cuda:0
图1与图2相似度: 0.3949535340831939
图1与图3相似度: 0.936580648208369
```

