import torch
import os

# --- 1. Configuration and Setup ---
# Create a directory to save the final animation frames
OUTPUT_DIR = 'mnist_final_animation'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
# How often to save a frame (in iterations)
VISUALIZATION_FREQUENCY = 20

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
