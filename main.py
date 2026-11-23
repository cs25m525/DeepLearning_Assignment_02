import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

import wandb
import psutil
import shutil

class MobileNetV2CIFAR10:

    def __init__(
            self,
            batch_size=128,
            epochs=150,
            backbone_lr=0.005,
            classifier_lr=0.05,
            momentum=0.9,
            weight_decay=0.0005,
            num_workers=4,
            seed=42,
            data_dir="./data",
            model_dir="./Models",

            # Architecture configuration
            width_mult=1.0,
            dropout=0.2,

            # Batch Normalization configuration
            bn_eps=1e-5,
            bn_momentum=0.1,
            bn_affine=True,
            bn_track_running_stats=True,

            # Training configuration
            lr_scheduler="cosine",
            min_lr=0.00001,
            label_smoothing=0.1,

            # W&B configuration
            wandb_project="MobileNetV2_CIFAR10",
            wandb_run_name=None

    ):
        # Configuration
        self.batch_size = batch_size
        self.epochs = epochs
        self.backbone_lr = backbone_lr
        self.classifier_lr = classifier_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.seed = seed

        self.data_dir = data_dir
        self.model_dir = model_dir

        # MobileNetV2 architecture configuration
        self.width_mult = width_mult
        self.dropout = dropout

        # Batch Normalization configuration
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        # Training configuration
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr
        self.label_smoothing = label_smoothing

        # W&B configuration
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.wandb_run = None

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

        # Model paths
        self.best_model_path = os.path.join(
            self.model_dir,
            f"{self.wandb_run_name}_best.pth"
        )

        self.final_model_path = os.path.join(
            self.model_dir,
            f"{self.wandb_run_name}_final.pth"
        )

        self.checkpoint_path = os.path.join(
            self.model_dir,
            f"{self.wandb_run_name}_checkpoint.pth"
        )

        # Automatic CPU / GPU configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("\nDevice Configuration")
        print("----------------------------")
        print("Device:", self.device)

        if self.device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA:", torch.version.cuda)
        else:
            print("Running on CPU")

        print("PyTorch:", torch.__version__)

        self.set_seed()

        # Initialize components
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.model = None

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
        )

        self.optimizer = None
        self.scheduler = None

        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    # ========================================================
    # Reproducibility and FP32 configuration
    # ========================================================

    def set_seed(self):

        random.seed(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Faster convolution algorithms
        torch.backends.cudnn.benchmark = True

        # Strict FP32 computation: disable TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        print("\nPrecision Configuration")
        print("TF32 MatMul:", torch.backends.cuda.matmul.allow_tf32)
        print("TF32 cuDNN:", torch.backends.cudnn.allow_tf32)
        print("Training Precision: FP32")

    # ========================================================
    # Dataset loading
    # ========================================================

    def load_dataset(self):
        mean = (0.485, 0.456, 0.406)  #Calculated in previous Assignment used Directly here
        std = (0.229, 0.224, 0.225)   #Calculated in previous Assignment used Directly here
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                32, padding=4, padding_mode="reflect"
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=2, magnitude=7
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Training dataset
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        # Validation dataset
        val_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=test_transform
        )

        # 90% training, 10% validation
        indices = torch.randperm(
            len(train_dataset),
            generator=torch.Generator().manual_seed(self.seed)
        ).tolist()

        train_indices = indices[:45000]
        val_indices = indices[45000:]

        train_subset = Subset(
            train_dataset, train_indices
        )

        val_subset = Subset(
            val_dataset, val_indices
        )

        # Test dataset
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )

        #only for CPU
        # loader_args = {
        #     "batch_size": self.batch_size,
        #     "num_workers": self.num_workers,
        #     "pin_memory": True,
        #     "persistent_workers": self.num_workers > 0
        # }
        #For CPU and GPU both
        loader_args = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": self.num_workers > 0
        }

        self.train_loader = DataLoader(
            train_subset,
            shuffle=True,
            **loader_args
        )

        self.val_loader = DataLoader(
            val_subset,
            shuffle=False,
            **loader_args
        )

        self.test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_args
        )
        print("\nCIFAR-10 datasets loaded successfully.")
        print("Training samples:", len(train_subset))
        print("Validation samples:", len(val_subset))
        print("Testing samples:", len(test_dataset))
    # ========================================================
    # Model initialization
    # ========================================================
    def build_model(self):

        norm_layer = lambda num_features: nn.BatchNorm2d(
            num_features,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=self.bn_affine,
            track_running_stats=self.bn_track_running_stats
        )

        self.model = mobilenet_v2(
            weights=None,
            width_mult=self.width_mult,
            dropout=self.dropout,
            norm_layer=norm_layer
        )

        # Modify initial convolution for CIFAR-10
        self.model.features[0][0] = nn.Conv2d(
            3,
            self.model.features[0][0].out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Replace final classification layer
        self.model.classifier[1] = nn.Linear(
            self.model.last_channel, 10
        )

        self.model = self.model.to(self.device)

        print("\nMobileNetV2 initialized from scratch.")
        print("Model Precision:", next(self.model.parameters()).dtype)

        return self.model
    # def build_model(self):
    #     # Initialize MobileNetV2 from scratch
    #     self.model = mobilenet_v2(
    #         weights=None,
    #         width_mult=self.width_mult,     #width_mult=2,
    #         dropout=self.dropout            #dropout=0.15
    #     )
    #     # Modify initial convolution for CIFAR-10
    #     self.model.features[0][0] = nn.Conv2d(
    #         3,
    #         self.model.features[0][0].out_channels,#32,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #         bias=False
    #     )
    #
    #     # Replace final classification layer
    #     self.model.classifier[1] = nn.Linear(
    #         self.model.last_channel, 10
    #     )
    #     self.model = self.model.to(self.device)
    #     print("\nMobileNetV2 initialized from scratch.")
    #     print("Model Precision:", next(self.model.parameters()).dtype)
    #     return self.model

    # ========================================================
    # Optimizer and scheduler
    # ========================================================

    def configure_training(self):
        self.optimizer = optim.SGD(
            [
                {
                    "params": self.model.features.parameters(),
                    "lr": self.backbone_lr
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.classifier_lr
                }
            ],
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )

        if self.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr
            )
        else:
            raise ValueError(
                f"Unsupported scheduler: {self.lr_scheduler}"
            )

    # ========================================================
    # Training
    # ========================================================

    def train_one_epoch(self):
        self.model.train()
        running_loss = torch.zeros(
            1, device=self.device
        )
        correct = torch.zeros(
            1, device=self.device
        )
        total = 0
        for images, labels in self.train_loader:
            images = images.to(
                self.device, non_blocking=True
            )
            labels = labels.to(
                self.device, non_blocking=True
            )
            self.optimizer.zero_grad(
                set_to_none=True
            )
            # FP32 forward pass
            outputs = self.model(images)
            loss = self.criterion(
                outputs, labels
            )
            # FP32 backward pass
            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach()
            predicted = outputs.argmax(dim=1)
            correct += (
                predicted == labels
            ).sum()
            total += labels.size(0)
        avg_loss = (
            running_loss / len(self.train_loader)
        ).item()
        accuracy = (
            100.0 * correct / total
        ).item()
        return avg_loss, accuracy

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(self, data_loader):

        self.model.eval()

        running_loss = torch.zeros(
            1, device=self.device
        )

        correct = torch.zeros(
            1, device=self.device
        )

        total = 0

        with torch.no_grad():

            for images, labels in data_loader:

                images = images.to(
                    self.device, non_blocking=True
                )

                labels = labels.to(
                    self.device, non_blocking=True
                )

                # FP32 forward pass
                outputs = self.model(images)

                loss = self.criterion(
                    outputs, labels
                )

                running_loss += loss

                predicted = outputs.argmax(dim=1)

                correct += (
                    predicted == labels
                ).sum()

                total += labels.size(0)

        avg_loss = (
            running_loss / len(data_loader)
        ).item()

        accuracy = (
            100.0 * correct / total
        ).item()

        return avg_loss, accuracy

    # ========================================================
    # Save best model
    # ========================================================

    def save_best_model(self, accuracy, epoch):
        self.best_val_accuracy = accuracy
        self.best_epoch = epoch
        torch.save(
            self.model.state_dict(),
            self.best_model_path
        )

        print(
            f"\nBest model saved! "
            f"Validation Accuracy: {accuracy:.2f}%"
        )

    # ========================================================
    # Save training checkpoint
    # ========================================================

    def save_checkpoint(self, epoch):

        torch.save({

            "epoch": epoch,

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "scheduler_state_dict": self.scheduler.state_dict(),

            "best_val_accuracy": self.best_val_accuracy

        }, self.checkpoint_path)

    # ========================================================
    # Load best model
    # ========================================================

    def load_best_model(self):

        self.model.load_state_dict(
            torch.load(
                self.best_model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        print("\nBest model loaded successfully.")

    # ========================================================
    # W&B Initialization
    # ========================================================

    def initialize_wandb(self):
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=self.wandb_run_name,
            mode="offline",
            config={
                # Training configuration
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "backbone_lr": self.backbone_lr,
                "classifier_lr": self.classifier_lr,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
                "num_workers": self.num_workers,

                # Model configuration
                "architecture": "MobileNetV2",
                "width_multiplier": self.width_mult,
                "dropout": self.dropout,
                "dataset": "CIFAR-10",
                "pretrained": False,

                # Batch Normalization configuration
                "bn_eps": self.bn_eps,
                "bn_momentum": self.bn_momentum,
                "bn_affine": self.bn_affine,
                "bn_track_running_stats": self.bn_track_running_stats,

                # Additional training configuration
                "label_smoothing": self.label_smoothing,
                "min_lr": self.min_lr,
                "lr_scheduler": self.lr_scheduler,

                # Optimizer configuration
                "optimizer": "SGD",
                "nesterov": True,

                # Scheduler configuration
                "scheduler": "CosineAnnealingLR",
                "scheduler_T_max": self.epochs,
                "scheduler_eta_min": 0.00001,

                # Precision configuration
                "precision": "FP32",
                "TF32_matmul": False,
                "TF32_cudnn": False,

                # Device configuration
                "device": str(self.device),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda
            }
        )
        print("\nW&B Offline Run Initialized")
        print("Project:", self.wandb_project)
        print("Run Name:", self.wandb_run_name)

    def log_model_parameters(self):

        parameter_logs = {}

        for name, param in self.model.named_parameters():

            if param.requires_grad:
                parameter_logs[f"Weights_Biases/{name}"] = wandb.Histogram(
                    param.detach().cpu().numpy()
                )

        wandb.log(parameter_logs)

    # ========================================================
    # Main training function
    # ========================================================

    def train(self):

        self.load_dataset()
        self.build_model()
        self.configure_training()
        self.initialize_wandb()

        start_time = time.time()
        print("\nStarting MobileNetV2 FP32 Training...\n")

        for epoch in range(self.epochs):
            print("Starting Epoch " + str(epoch + 1))
            epoch_start = time.time()
            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.evaluate(
                self.val_loader
            )

            # Log training and validation metrics
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "validation/loss": val_loss,
                "validation/accuracy": val_acc
            })
            self.scheduler.step()
            self.log_model_parameters()
            #lr = self.optimizer.param_groups[0]["lr"]
            # Get learning rates
            backbone_lr = self.optimizer.param_groups[0]["lr"]
            classifier_lr = self.optimizer.param_groups[1]["lr"]

            # Log learning rates
            wandb.log({
                "learning_rate/backbone": backbone_lr,
                "learning_rate/classifier": classifier_lr
            })

            lr = backbone_lr

            # Log RAM, disk and GPU memory usage
            self.log_resource_usage()

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            if val_acc > self.best_val_accuracy:
                self.save_best_model(
                    val_acc, epoch + 1
                )

            self.save_checkpoint(epoch + 1)
            # Log saved model and checkpoint sizes
            self.log_model_file_sizes()
        # Load best validation model
        self.load_best_model()

        # Evaluate test dataset only after training
        test_loss, test_acc = self.evaluate(
            self.test_loader
        )

        print("\nFinal Test Results")
        print("----------------------------")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        # Log final test results
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "best_epoch": self.best_epoch,
            "best_validation_accuracy": self.best_val_accuracy
        })

        # Save final FP32 model
        self.save_model()
        self.log_model_file_sizes()

        print("\nTraining completed.")
        print("Best Epoch:", self.best_epoch)
        # Finish W&B offline run
        wandb.finish()
        print("\nW&B offline logging completed.")

        print(
            "Best Validation Accuracy:",
            f"{self.best_val_accuracy:.2f}%"
        )
        print(
            f"Total Time: {(time.time() - start_time) / 60:.2f} minutes"
        )

    # ========================================================
    # Log Model File Sizes
    # ========================================================

    def log_model_file_sizes(self):
        model_files = {
            "model_size/best_model_mb": self.best_model_path,
            "model_size/final_model_mb": self.final_model_path,
            "model_size/checkpoint_mb": self.checkpoint_path
        }
        size_logs = {}
        for metric_name, file_path in model_files.items():
            if os.path.exists(file_path):
                file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
                size_logs[metric_name] = file_size_mb
        if size_logs:
            wandb.log(size_logs)

    # ========================================================
    # System Resource Monitoring
    # ========================================================

    def log_resource_usage(self):

        process = psutil.Process(os.getpid())

        # CPU RAM usage (current Python process)
        process_ram = process.memory_info().rss / (1024 ** 3)

        # Total system RAM usage
        system_ram = psutil.virtual_memory()

        ram_used = system_ram.used / (1024 ** 3)
        ram_percent = system_ram.percent

        # Disk usage (drive containing model directory)
        disk = shutil.disk_usage(self.model_dir)

        disk_used = disk.used / (1024 ** 3)
        disk_free = disk.free / (1024 ** 3)
        disk_total = disk.total / (1024 ** 3)

        resource_logs = {

            "resources/process_ram_gb": process_ram,
            "resources/system_ram_used_gb": ram_used,
            "resources/system_ram_percent": ram_percent,

            "resources/disk_used_gb": disk_used,
            "resources/disk_free_gb": disk_free,
            "resources/disk_total_gb": disk_total
        }

        # GPU memory usage
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            gpu_max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

            resource_logs.update({

                "resources/gpu_memory_allocated_gb": gpu_allocated,
                "resources/gpu_memory_reserved_gb": gpu_reserved,
                "resources/gpu_max_memory_allocated_gb": gpu_max_allocated

            })

        wandb.log(resource_logs)


    # ========================================================
    # Save model
    # ========================================================

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            self.final_model_path
        )
        print(
            "\nFP32 model saved:",
            self.final_model_path
        )

    # ========================================================
    # Load model for compression / quantization
    # ========================================================

    def load_model(self, model_path=None):

        if model_path is None:
            model_path = self.best_model_path

        self.build_model()

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        self.model.eval()

        print("\nFP32 model loaded for further experiments.")

        return self.model

def generate_run_name(config):
    return (
        f"MobileNetV2"
        f"_W{config['width_mult']}"
        f"_D{config['dropout']}"
        f"_BN{config['bn_momentum']}"
        f"_BS{config['batch_size']}"
        f"_BLR{config['backbone_lr']}"
        f"_CLR{config['classifier_lr']}"
        f"_WD{config['weight_decay']}"
        f"_E{config['epochs']}"
    )

# ============================================================
# MAIN
# ============================================================
import argparse
from Compression import PostTrainingQuantizer
if __name__ == "__main__":
    # weight_bits = 4 #8
    # activation_bits = 8
    model_path = "BestModel"
    file_name = (
        "C10_MobileNetV2_W1.0_D0.1_BN0.1_"
        "BS128_BLR0.01_CLR0.05_WD0.001_E200_best.pth"
    )

    # Initialize MobileNetV2 with the same training configuration
    mobilenet = MobileNetV2CIFAR10(
        batch_size=128,
        width_mult=1.0,
        dropout=0.1,
        bn_eps=1e-5,
        bn_momentum=0.1,
        bn_affine=True,
        bn_track_running_stats=True,
        data_dir="./data",
        model_dir=model_path
    )
    # Load CIFAR-10 datasets
    mobilenet.load_dataset()
    # Initialize the model architecture
    mobilenet.build_model()


    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="MobileNetV2 CIFAR-10 Quantization"
        )

        parser.add_argument(
            "--weight_quant_bits",
            type=int,
            default=8,
            choices=[2, 3, 4, 5, 6, 7, 8],
            help="Weight quantization bitwidth"
        )

        parser.add_argument(
            "--activation_quant_bits",
            type=int,
            default=8,
            choices=[2, 3, 4, 5, 6, 7, 8],
            help="Activation quantization bitwidth"
        )
        return parser.parse_args()


    args = parse_arguments()
    weight_bits = args.weight_quant_bits
    activation_bits = args.activation_quant_bits
    name = f"W{weight_bits}A{activation_bits}_calibration_40"
    quantizer = PostTrainingQuantizer(
        model=mobilenet.model,
        model_path=model_path,
        file_name=file_name,
        weight_quant_bits=weight_bits,
        activation_quant_bits=activation_bits,
        calibration_loader=mobilenet.val_loader,
        test_loader=mobilenet.test_loader,
        calibration_batches=40,
        wandb_project="MobileNetV2_Compression",
        run_name=name
    )

    quantized_model = quantizer.quantize_model()
    print("\nQuantization completed.")
    exit()



    #################################

    weightsActivationConfigurations = [
        (8, 8),
        (4, 8),
        (8, 4),
        (4, 4),
        (6, 8),
        (8, 6),
        (6, 6),
        (2, 8),
        (8, 2),
        (2, 2)
    ]
    for weight_bits, activation_bits in weightsActivationConfigurations:
        name = f"W{weight_bits}A{activation_bits}_calibration_40"
        quantizer = PostTrainingQuantizer(
            model=mobilenet.model,
            model_path=model_path,
            file_name=file_name,
            weight_quant_bits=weight_bits,
            activation_quant_bits=activation_bits,
            calibration_loader=mobilenet.val_loader,
            test_loader=mobilenet.test_loader,
            calibration_batches=40,
            wandb_project="MobileNetV2_Compression",
            run_name=name
        )
        quantized_model = quantizer.quantize_model()
    print("\nQuantization completed.")
    exit()
    #Below is the code for model trainig
    #Some comman Configuration
    common_config = {
        "momentum": 0.9,
        "num_workers": 4,
        "seed": 42,
        "data_dir": "./data",
        "model_dir": "./Models",
        "lr_scheduler": "cosine",
        "min_lr": 1e-5,
        "label_smoothing": 0.1,
        "wandb_project": "MobileNetV2_CIFAR10"
    }

    configurations = [
        # 1. Small model, conservative learning rate
        {"width_mult": 0.5, "dropout": 0.5, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.0001, "classifier_lr": 0.001, "weight_decay": 0.005,
         "batch_size": 128, "epochs": 100},

        # 2. Small model, moderate learning rate
        {"width_mult": 0.75, "dropout": 0.4, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.0005, "classifier_lr": 0.005, "weight_decay": 0.002,
         "batch_size": 128, "epochs": 150},

        # 3. Medium model, reduced dropout
        {"width_mult": 0.75, "dropout": 0.3, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.001, "classifier_lr": 0.01, "weight_decay": 0.001,
         "batch_size": 128, "epochs": 150},

        # 4. Standard width, moderate dropout
        {"width_mult": 1.0, "dropout": 0.4, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.001, "classifier_lr": 0.01, "weight_decay": 0.001,
         "batch_size": 128, "epochs": 150},

        # 5. Standard width, moderate learning rate
        {"width_mult": 1.0, "dropout": 0.2, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.003, "classifier_lr": 0.03, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 150},

        # 6. Standard width, higher learning rate
        {"width_mult": 1.0, "dropout": 0.2, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.005, "classifier_lr": 0.05, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 200},

        # 7. Wider model
        {"width_mult": 1.25, "dropout": 0.2, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.005, "classifier_lr": 0.03, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 200},

        # 8. Wider model, lower dropout
        {"width_mult": 1.25, "dropout": 0.1, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.005, "classifier_lr": 0.05, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 200},

        # 9. Standard width, modified BN momentum
        {"width_mult": 1.0, "dropout": 0.1, "bn_eps": 1e-5, "bn_momentum": 0.05,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.01, "classifier_lr": 0.05, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 200},

        # 10. Standard width, modified weight decay
        {"width_mult": 1.0, "dropout": 0.1, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.01, "classifier_lr": 0.05, "weight_decay": 0.001,
         "batch_size": 128, "epochs": 200},

        # 11. Paper-informed CIFAR-10 baseline
        {"width_mult": 1.0, "dropout": 0.2, "bn_eps": 1e-5, "bn_momentum": 0.1,
         "bn_affine": True, "bn_track_running_stats": True,
         "backbone_lr": 0.05, "classifier_lr": 0.05, "weight_decay": 0.0005,
         "batch_size": 128, "epochs": 200}
    ]

    start_index = 0
    end_index = 5

    for i in range(start_index, end_index):
        print("Starting Config number: ", i + 1)
        config = configurations[i]
        run_name = f"C{i + 1:02d}_" + generate_run_name(config)
        mobilenet = MobileNetV2CIFAR10(
            **config,
            **common_config,
            wandb_run_name=run_name
        )
        mobilenet.train()

    print("Ended.")