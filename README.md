贪吃蛇大师，尝试使用智能体漂亮通关贪吃蛇游戏


# SnakeAI

[简体中文] | [English](README-EN.md) | [日本語](README_JA.md)

贪吃蛇大师，尝试使用智能体通关贪吃蛇游戏
本项目包含一经典游戏贪吃蛇，一次意外训练生成的通关解法，走汉密尔顿环（有些无赖虽然也能通关），还有就是之前cnn训练的结果，和mlp结果。训练数据也会一并公开并打包上传。
这里提供一个本人训练的模型，下载地址  

### 文件结构

```bash
├───gpu.py
├───PyTorch.py
├───cnn.py
├───ceshi.py
├───Start.ps1
├───requirement.txt
└───best_model.pth
```

项目主要包括用于检测GPU的脚本（gpu.py和PyTorch.py），卷积神经网络的实现（cnn.py），模型测试脚本（ceshi.py），后续的自动化启动脚本（Start.ps1），以及训练好的模型文件（best_model.pth）。


## 运行指南

本项目基于 Python 编程语言，主要使用了 PyTorch、Pillow (PIL)、NumPy 等外部代码库进行图像处理和深度学习模型的训练与推理。程序运行使用的 Python 版本为 3.8.18，建议使用 Anaconda 配置 Python 环境。以下配置过程已在 Windows 11系统上测试通过。以下为Windows Terminal指令。


### 环境配置

```bash
# 创建 conda 环境，将其命名为 GallaryAI并激活环境
conda create -n SnakeAI python=3.8.18
conda activate SnakeAI
```


Windows:
```bash 
# 前往官网下载对应版本的PyTorch。使用 GPU 训练需要手动安装完整版 PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 运行程序脚本测试 PyTorch 是否能成功调用 GPU
python gpu.py
python PyTorch.py

# 安装外部代码库
pip install -r requirements.txt
```


### 运行训练和测试

项目文件夹下可以直接运行以下指令进行游戏：

```bash
cd "所在目录"
# 运行卷积神经网络模型训练脚本
python train_cnn.py
python train_mlp.py

# 运行模型测试脚本
python test_cnn.py
python test_mlp.py
```

模型权重文件存储在项目cnn.py所在目录下的 best_model.pth。测试脚本 ceshi.py 默认调用文件所在目录下的模型文件也就是训练完成后的模型。如果需要观察模型在不同训练阶段的表现，可以在 cnn.py 中修改模型保存路径。

如果需要重新训练模型，可以在cnn.py所在目录下运行此文件


### 查看曲线

项目中包含了训练过程的 Tensorboard 曲线图，可以使用 Tensorboard 查看其中的详细数据。推荐使用 VSCode 集成的 Tensorboard 插件直接查看，也可以使用传统方法：

```bash
cd "所在目录"
tensorboard --logdir=[上级目录]\snakeai\main\logs --bind_all --reload_interval 60
```

此命令会将端口发布到公网方便随时随地查看，且每60s自动刷新一次图像。可在浏览器中打开 Tensorboard 服务默认地址 `http://localhost:6006/`，即可查看训练过程的交互式曲线图。
