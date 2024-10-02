import os
import torch
from Models import tinyimagenet_resnet

# sparsities = [0.99]
compression_ratios = [3]
model = tinyimagenet_resnet.resnet34((3, 32, 32), 10)
print(model)
for cr in compression_ratios:
    # compression_ratio = 1 / (1 - sparsity)
    command = f"python main.py " \
        f"--dataset cifar10 " \
        f"--model resnet34 " \
        f"--model-class tinyimagenet " \
        f"--pruner synflow " \
        f"--compression {cr} " \
        f"--prune-epochs 100 " \
        f"--post-epochs 160 " \
        f"--lr 0.1 " \
        f"--lr-drops 60 120 " \
        f"--optimizer sgd " \
        f"--weight-decay 1e-4 " \
        f"--train-batch-size 128 " \
        f"--test-batch-size 256 " \
        f"--dense-classifier True " \
        f"--prune-dataset-ratio 10 " \
        f"--prune-batch-size 256 " \
        f"--prune-bias False " \
        f"--prune-batchnorm False " \
        f"--prune-residual False " \
        f"--prune-train-mode False " \
        f"--verbose " \
        f"--experiment singleshot " \
        f"--seed 42 " \
        f"--expid synflow_resnet34_cifar10_sparsity_{cr}" \
        f"--gpu 1" 
# print(f"Running experiment for sparsity {sparsity}")
print(f"Compression ratio: {cr}")
os.system(command)