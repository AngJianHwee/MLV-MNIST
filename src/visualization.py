import torch
import matplotlib.pyplot as plt
import numpy as np
from .config import OUTPUT_DIR

plt.style.use('ggplot')

def visualize_dashboard_final(model, loader, device, iteration, epoch, loss_history, iteration_history, current_loss):
    """
    Generates the final, polished 3-subplot dashboard.
    """
    model.eval()
    
    all_latents, all_labels = [], []
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            latents = model.get_latent_features(data)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # --- Create the Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 9), dpi=120)
    fig.suptitle(f'MNIST Training Dashboard | Epoch: {epoch} | Iteration: {iteration}', fontsize=28, weight='bold')

    # Plot 1: Loss Curve (with transparency)
    ax1.plot(iteration_history, loss_history, linewidth=2.5, alpha=0.6) # MODIFIED: Increased transparency
    ax1.scatter(iteration_history[-1], loss_history[-1], s=100, c='red', zorder=5, label='Current')
    ax1.set_title('Training Loss', fontsize=22, weight='bold')
    ax1.set_xlabel('Iteration', fontsize=18)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    if len(loss_history) > 1:
        ax1.set_xlim(0, max(iteration_history) * 1.1)
        ax1.set_ylim(0, max(loss_history) * 1.1)
    ax1.legend(fontsize=14)

    # Plot 2: Per-Class Accuracy (light cyan and transparent)
    per_class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    # MODIFIED: Changed color to light cyan ('c') and set alpha
    bars = ax2.bar(range(10), per_class_acc, color='c', alpha=0.7, edgecolor='black', linewidth=1.5, zorder=3)
    ax2.set_title('Per-Class Test Accuracy', fontsize=22, weight='bold')
    ax2.set_xlabel('Digit Class', fontsize=18)
    ax2.set_ylabel('Accuracy (%)', fontsize=18)
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=12, weight='bold')

    # Plot 3: Latent Space (legend moved to bottom right)
    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    scatter = ax3.scatter(all_latents[:, 0], all_latents[:, 1], c=all_labels, cmap='tab10', s=8, alpha=0.6)
    legend_properties = {'weight':'bold', 'size':14}
    # MODIFIED: Changed legend location with loc='lower right'
    legend = ax3.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(10)],
                        title="Digits", prop=legend_properties, title_fontsize=16, loc='lower right')
    ax3.set_title('2D Latent Space Representation', fontsize=22, weight='bold')
    ax3.set_xlabel('Latent Dimension 1', fontsize=18)
    ax3.set_ylabel('Latent Dimension 2', fontsize=18)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    # Final adjustments and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    filename = f'{OUTPUT_DIR}/frame_{iteration:05d}_epoch_{epoch}_loss_{current_loss:.4f}.png'
    plt.savefig(filename)
    plt.close(fig)
