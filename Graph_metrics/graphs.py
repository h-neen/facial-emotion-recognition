import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "extracted_training_metrics_clean.csv"
SAVE_DIR = "."

df = pd.read_csv(CSV_PATH)

df["train_acc"] = pd.to_numeric(df.get("train_acc"), errors="coerce")
df["val_acc"] = pd.to_numeric(df.get("val_acc"), errors="coerce")
df = df.dropna(subset=["train_acc", "val_acc"], how="all")

df = df.reset_index(drop=True)
df["step"] = df.index

WINDOW = 10
df["train_acc_smooth"] = df["train_acc"].rolling(WINDOW, min_periods=1).mean()
df["val_acc_smooth"] = df["val_acc"].rolling(WINDOW, min_periods=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(df["step"], df["train_acc_smooth"], label="Train Accuracy")
plt.plot(df["step"], df["val_acc_smooth"], label="Validation Accuracy")

try:
    max_step = df["step"].max()
    plt.text(max_step * 0.05, 10,
             "Start: Low accuracy (~1–15%)",
             fontsize=9)

    plt.text(max_step * 0.25, 25,
             "Resume → jump (~20%)",
             fontsize=9)

    plt.text(max_step * 0.5, 45,
             "Unfreeze → improvement (~40%)",
             fontsize=9)

    plt.text(max_step * 0.8, 70,
             "Final → peak (~70%+)",
             fontsize=9)
except Exception:
    pass

try:
    if "phase" in df.columns:
        for p in df["phase"].dropna().unique():
            subset = df[df["phase"] == p]
            if not subset.empty:
                plt.axvline(subset["step"].min(), linestyle="--", alpha=0.3)
except Exception:
    pass

plt.xlabel("Training Progress")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy (with Training Stages)")
plt.legend()
plt.grid()

save_path = os.path.join(SAVE_DIR, "accuracy_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("Graph saved at:", save_path)

plt.show()