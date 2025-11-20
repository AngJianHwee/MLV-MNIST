# MLV-MNIST: MLP Latent Visualization for MNIST


https://github.com/user-attachments/assets/49b9dc75-5f76-48df-8680-3b0b17ba7b39


## Project Overview

This project, **MLV-MNIST (MLP Latent Visualization for MNIST)**, trains a simple **PyTorch MLP** on the MNIST dataset and generates a **real-time animated dashboard** of the training process. It visualizes the model's loss, per-class accuracy, and the evolution of its 2D latent space, compiling the results into a final MP4 video.

The project is structured to demonstrate best practices in ML engineering, emphasizing a modular, readable, and reproducible codebase.

## Key Features

- **Real-Time Multi-Plot Dashboard**: Generates a frame-by-frame visualization showing the **Training Loss Curve**, **Per-Class Test Accuracy**, and the **2D Latent Space**.
- **Deep Learning with PyTorch**: Implements a standard MLP with a 2-dimensional latent space to facilitate visualization.
- **Automated Video Generation**: Uses `ffmpeg` to automatically compile all saved frames into a final MP4 video.
- **Modular Codebase**: The project is logically separated into modules for configuration, data loading, model architecture, and visualization.

## Tech Stack

- **Languages & Frameworks**: Python, PyTorch
- **Libraries**: Matplotlib, NumPy
- **Tools**: FFmpeg, Git

## Core Competencies Demonstrated

-   **Deep Learning with PyTorch**: Proficiently implemented a complete training pipeline, including a custom **`nn.Module` (MLP)**, **Cross-Entropy loss**, **Adam optimizer**, and **latent space projection**.
-   **Data Visualization**: Created a real-time, multi-plot dashboard with **Matplotlib** to interpret model behavior and learning dynamics.
-   **ML Engineering**: Designed and built a **modular, readable, and reproducible codebase** with a clear separation of concerns (config, data, model, visualization).
-   **Automation**: Scripted the end-to-end workflow, from training and frame generation to automated video compilation using **`ffmpeg`**.

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AngJianHwee/MLV-MNIST.git
    cd MLV-MNIST
    ```

2.  **Prerequisites**
    - Ensure you have **FFmpeg** installed and accessible in your system's PATH. This is required for video generation. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

3.  **Set Up a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Training Script**
    ```bash
    python main.py
    ```

## Expected Output

-   The script will create a directory named `mnist_final_animation/`.
-   Inside this directory, it will save hundreds of PNG frames, each representing a snapshot of the dashboard at a specific training iteration.
-   Once training is complete, the script will automatically attempt to create a video file named `mnist_dashboard_final.mp4` in the same directory.
