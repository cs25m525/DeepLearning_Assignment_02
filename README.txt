****************************************Compression on MObileNetV2 for CIFAR-10***************************************************************
|                         |
|Name:Prateek Maheshwari  |
|Roll No: CS25M525        |
|_________________________|

1. Project Description
Compression on MobileNetV2

2. Environment Details
    Operating System:           Windows 10/11 (64-bit)
    Programming Language:       Python
    Deep Learning Framework:    PyTorch
    Dataset:                    CIFAR-10
    Hardware:                   CPU and GPU(if available)

3. Dependencies
    The project requires the following Python packages are torch, torchvision, numpy, Pillow, os, psutil, shutil, json, csv, threading, warnings and wandb.

4. Environment Setup
    4.1 Create and activate a virtual environment by command: python -m venv .venv
    4.2 Upgrade pip by command: python -m pip install --upgrade pip
    4.3 Install the compatible PyTorch and torchvision versions required by the project.

5. Random Seed Configuration is SEED = 42

6. Running the Project by command:
	Execute Command: python main.py --weight_quant_bits <WEIGHT_QUANT_BITS> --activation_quant_bits <ACTIVATION_QUANT_BITS>
	Ex: python main.py --weight_quant_bits 8 --activation_quant_bits 8

7. Dataset: cifar-10-batches-py dataset to be present at same location as of main.py
