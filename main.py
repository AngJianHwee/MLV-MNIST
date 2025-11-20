import torch
import torch.nn as nn
import torch.optim as optim
import os
import subprocess

from src.config import DEVICE, EPOCHS, LEARNING_RATE, VISUALIZATION_FREQUENCY, OUTPUT_DIR
from src.model import MLP
from src.data_loader import get_data_loaders
from src.visualization import visualize_dashboard_final

def main():
    """
    Main function to run the MNIST training and visualization.
    """
    # --- 1. Setup ---
    print(f"Using device: {DEVICE}")

    # --- 2. Load Data ---
    train_loader, test_loader = get_data_loaders()

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    iteration_count = 0
    loss_history = []
    iteration_history = []
    last_vis_iter = -1
    initial_loss = 2.4 # Approximate starting loss

    print("Visualizing initial state...")
    visualize_dashboard_final(model, test_loader, DEVICE, 0, 0, [initial_loss], [0], initial_loss) 

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            iteration_count += 1
            loss_history.append(loss.item())
            iteration_history.append(iteration_count)
            
            # Visualize progress
            if (iteration_count % VISUALIZATION_FREQUENCY == 0):
                print(f'Epoch [{epoch}/{EPOCHS}], Iteration {iteration_count}, Loss: {loss.item():.4f}')
                visualize_dashboard_final(model, test_loader, DEVICE, iteration_count, epoch, loss_history, iteration_history, loss.item())
                last_vis_iter = iteration_count

        # Ensure a frame is saved at the very end of each epoch
        if iteration_count != last_vis_iter:
            print(f"End of Epoch [{epoch}/{EPOCHS}], generating frame...")
            visualize_dashboard_final(model, test_loader, DEVICE, iteration_count, epoch, loss_history, iteration_history, loss.item())

    print(f"Training complete. All frames saved to the '{OUTPUT_DIR}' directory.")

    # --- 5. Generate Video ---
    print("\nAttempting to generate video from frames...")
    try:
        # Note: ffmpeg must be installed and in the system's PATH.
        video_command = [
            'ffmpeg',
            '-framerate', '15',
            '-pattern_type', 'glob',
            '-i', f'{OUTPUT_DIR}/frame_*.png',
            '-c:v', 'libx264',
            '-b:v', '3M',
            '-r', '30',
            f'{OUTPUT_DIR}/mnist_dashboard_final.mp4'
        ]
        subprocess.run(video_command, check=True)
        print(f"Successfully created video: {OUTPUT_DIR}/mnist_dashboard_final.mp4")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n---")
        print("VIDEO CREATION FAILED. Could not run the 'ffmpeg' command.")
        print("Please ensure ffmpeg is installed and accessible in your system's PATH.")
        print("You can manually create the video by running the following command from your terminal:")
        print(f"cd {OUTPUT_DIR} && ffmpeg -framerate 15 -pattern_type glob -i 'frame_*.png' -c:v libx264 -b:v 3M -r 30 mnist_dashboard_final.mp4")
        print("---")

if __name__ == '__main__':
    main()
