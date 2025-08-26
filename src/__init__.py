import torch
import numpy as np
import matplotlib.pyplot as plt
import math


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())  # if True: GPU works
