import os
import yaml
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datetime import datetime

from utils import set_seed
from train import Trainer
from models import build_inn
from data import RandomRigDataset
from loss_functions import LittleNIKILoss



if __name__ == "__main__":

    # Define output directory for results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", f"{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Open configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Armature segment lengths
    lengths_value = config["LENGTHS"]
    num_segments = config["NUM_SEGMENTS"]

    assert isinstance(lengths_value, (str, int, float)), \
        f"Invalid type for LENGTHS: expected str, int, or float, got {type(lengths_value).__name__}"

    if isinstance(lengths_value, str):
        if lengths_value.lower() == "random":
            print("[INFO] Generating random segment lengths from 0.01 to 0.1!")
            lengths = np.random.uniform(0.01, 0.1, size=num_segments)
        else:
            print(f"[WARNING] Unrecognized LENGTHS string '{lengths_value}'. Defaulting to uniform lengths of 0.1.")
            lengths = np.full(num_segments, 0.1)
    else:
        # lengths_value is a number (int or float)
        lengths = np.full(num_segments, lengths_value)

    # Add lengths to config for downstream use
    config["LENGTHS_ARRAY"] = lengths

    # Set the seed
    set_seed(config["SEED"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = build_inn(config["NUM_SEGMENTS"]).to(device)
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"]
    )

    # Define scheduler
    scheduler = None

    # Define loss function
    loss_fn = LittleNIKILoss(config)

    # Instantiate trainer
    trainer = Trainer(
        config=config,
        loss_function=loss_fn,
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Training loop
    for epoch in range(1, config["NUM_EPOCHS"] + 1):

        print(f"| ---------------- | EPOCH {epoch:d} | ------------------ |")

        # Generate datasets
        train_dataset = RandomRigDataset(
            num_samples=config["TRAIN_SIZE"],
            num_segments=config["NUM_SEGMENTS"],
            lengths=lengths
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=True,
            num_workers=0
        )

        val_dataset = RandomRigDataset(
            num_samples=config["TEST_SIZE"],
            num_segments=config["NUM_SEGMENTS"],
            lengths=lengths
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            num_workers=0
        )

        # Training step
        trainer.train_epoch(dataloader=train_loader)

        # Validation step
        trainer.test_epoch(dataloader=val_loader)

        # Analysis
        if epoch % config["ANALYSIS_INTERVAL"] == 0:
            print("[INFO] Analyzing...")

            trainer.plot_losses(save_dir=results_dir)

            trainer.plot_scatter(save_dir=os.path.join(results_dir, f"{epoch:04d}", "pred_label_scatter"))

            trainer.plot_analysis(dataloader=val_loader, lengths=lengths, save_dir=os.path.join(results_dir, f"{epoch:04d}", "rigs"))