import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def extract_and_plot_everything(log_base_dir, output_dir):
    log_base = Path(log_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    event_files = list(log_base.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_base}")
        return

    print(f"Found {len(event_files)} event files. Extracting all available metrics...")

    all_data = {}

    for event_file in event_files:
        acc = EventAccumulator(str(event_file.parent))
        acc.Reload()
        
        # Get every single scalar tag available in this file
        tags = acc.Tags().get('scalars', [])
        for tag in tags:
            values = acc.Scalars(tag)
            if tag not in all_data:
                all_data[tag] = {'steps': [], 'values': []}
            
            all_data[tag]['steps'].extend([v.step for v in values])
            all_data[tag]['values'].extend([v.value for v in values])
            print(f"  -> Extracted: {tag}")

    if not all_data:
        print("No metrics found in any files.")
        return

    # Create the Plots
    # 1. Accuracy Curves
    plt.figure(figsize=(10, 6))
    acc_found = False
    for tag, data in all_data.items():
        if 'acc' in tag.lower():
            # Sort by step to ensure a clean line
            sorted_points = sorted(zip(data['steps'], data['values']))
            steps, vals = zip(*sorted_points)
            plt.plot(steps, vals, label=tag, linewidth=2)
            acc_found = True
    
    if acc_found:
        plt.title('Training History: Accuracy', fontsize=14)
        plt.xlabel('Steps/Epochs'); plt.ylabel('Value'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
        print(f"Saved: accuracy_curve.png")

    # 2. Loss Curves
    plt.figure(figsize=(10, 6))
    loss_found = False
    for tag, data in all_data.items():
        if 'loss' in tag.lower():
            sorted_points = sorted(zip(data['steps'], data['values']))
            steps, vals = zip(*sorted_points)
            plt.plot(steps, vals, label=tag, linewidth=2)
            loss_found = True

    if loss_found:
        plt.title('Training History: Loss', fontsize=14)
        plt.xlabel('Steps/Epochs'); plt.ylabel('Value'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "loss_curve.png", dpi=300)
        print(f"Saved: loss_curve.png")

if __name__ == "__main__":
    extract_and_plot_everything("logs", Path("results/plots"))
    print("\nCheck results/plots/ for your files.")