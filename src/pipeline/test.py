import torch

checkpoint = torch.load('../models/classifiers/notebooks/best_model_optuna.pth', map_location='cpu')
print(checkpoint.keys())
