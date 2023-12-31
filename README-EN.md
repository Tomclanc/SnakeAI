Snake Assistant, try to pass the game of Snake beautifully using the smart body


# SnakeAI

[简体中文](README-CN.md) | [English] | [日本語](README_JA.md)

Snake Assistant, an attempt to use intelligences to pass the game Snake
This project contains a classic game of Snake, a pass solution generated by an accidental training, walking the Hamilton ring (some rogues can pass the game though), and the results of previous cnn training, and mlp results. The training data will also be disclosed and packaged and uploaded.
Here is a model I trained, download address
https://dlink.host/1drv/aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBckpVdXJnZURwTHNpNUI3Vkxacm0yYU9fQU9aLXc_ZT1YVk01TUo.zip

### File Structure

```bash
├───main
│   ├───logs
│   ├───trained_models_cnn
│   ├───sound
│   ├───20231217
│   ├───20231219
│   ├───20231220
│   ├───20231222
│   ├───requirements.txt
│   └───scripts
├───utils
│   └───scripts
```

The project mainly consists of scripts (gpu.py and PyTorch.py) for checking GPUs, training data for several times (folder named numeric), logs folder for terminal text and data curves containing the training process (viewed using Tensorboard); requirements.txt is the anaconda configuration file, and
check_gpu_status/ for checking if the GPU can be called by PyTorch

## Running Guide

This project is based on the Python programming language, and uses external code libraries such as Pygame, OpenAI Gym, Stable-Baselines3, and so on. The version of Python used to run the program is 3.8.16. It is recommended to use Anaconda to configure the Python environment. The following are Windows Terminal commands.


### Environment Setup

```bash
# Create a conda environment, name it SnakeAI and activate the environment
conda create -n SnakeAI python=3.8.18
conda activate SnakeAI
```


Windows:

```bash 
# Go to the official website to download the corresponding version of PyTorch. Manual installation of the full version of PyTorch is required for GPU training
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run script programs to test if PyTorch can successfully call GPU
python gpu.py
python PyTorch.py

# Install external libraries
pip install -r requirements.txt
```


### Run training and testing

The following commands can be directly run in the project folder:

```bash
cd "Directory"
# Run the Convolutional Neural Network model training script
python snake_game.py
```

The following commands can be run directly from the project folder to play the game:

```bash
cd "directory"
# Run the convolutional neural network model training script
python train_cnn.py

# Run the model testing script
python test_cnn.py
```

The model weights file is stored in main/trained_models_cnn/

If you need to retrain the model, you can run this file in the directory where train_cnn.py is located. The test scripts all default to the trained model such as ppo_snake_final.zip. If you need to observe the performance of the AI at different stages of training, you can change the MODEL_PATH variable in the test scripts to the path of other model files.


### Viewing Curves

The project includes Tensorboard curve graphs of the training process, which can be viewed using Tensorboard. It is recommended to use the integrated Tensorboard plugin in VSCode for direct viewing, or you can use the traditional method:

```bash
cd "Directory"
tensorboard --logdir=[parent directory]\snakeai\main\logs --bind_all --reload_interval 60
```

This command binds the TensorBoard to all available network interfaces (including public IPs) for access on the extranet for easy viewing anytime, anywhere, and automatically refreshes the image every 60s. Interactive graphs of the training process can be viewed on any device in the network by opening the Tensorboard service default address `http://[your public IP]:6006/` in a browser.
