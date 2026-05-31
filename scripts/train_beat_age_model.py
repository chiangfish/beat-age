import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--device', type=str, required=True, help='GPU device id')
ap.add_argument('-n', '--name', type=str, required=True, help='Version name for this experiment')
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
ap.add_argument('-wd', '--weight_decay', type=float, default=1e-1, help='Weight decay')
args = ap.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torchmetrics
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import swanlab
import numpy as np
from tqdm import tqdm
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from beat_age.models import Net1D
from beat_age.datasets import ECGDataset

# ===== Configures =====
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
WARMUP_EPOCHS = 0.2      # number of epochs for linear warmup
LR_MIN = 1e-5            # eta_min for cosine annealing
RANDOM_SEED = 3407
CPU_WORKERS = min(os.cpu_count(), 8)

VER_NAME = args.name
MODEL_CHECKPOINT_DIR = 'ckpts'

os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

swanlab.init(
    project="ecg-beat-age",
    experiment_name=VER_NAME + "_" + timestamp,
    config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "lr_min": LR_MIN,
        "random_seed": RANDOM_SEED,
    }
)

# ===== Prepare =====
def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
_seed_all(RANDOM_SEED)

def train_model(model, train_loader, valid_loader, device, 
                num_epochs=50, lr=0.001):
    # Loss for regression
    criterion = nn.MSELoss()

    # AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Warmup + CosineAnnealingLR (per-step schedule)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)

    # Linear warmup to base lr
    if warmup_steps > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: float(step + 1) / float(warmup_steps) if step < warmup_steps else 1.0,
        )
    else:
        warmup_scheduler = None

    # Cosine annealing from base lr to LR_MIN after warmup
    cosine_total_steps = max(1, total_steps - warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_total_steps, eta_min=LR_MIN)

    # Metrics for regression
    train_mae = torchmetrics.MeanAbsoluteError().to(device)
    train_pearson = torchmetrics.PearsonCorrCoef(num_outputs=1).to(device)
    val_mae = torchmetrics.MeanAbsoluteError().to(device)
    val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=1).to(device)

    global_step = 0
    best_val_mae = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
    
        train_mae.reset()
        train_pearson.reset()
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for i, (inputs, targets) in enumerate(train_loop):
            # (N, L, C) -> (N, C, L) and move to device
            inputs = inputs.to(device, non_blocking=True) 
            targets = targets.float().to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # LR scheduling: warmup first, then cosine
            if warmup_scheduler is not None and global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            # Update metrics
            train_mae.update(outputs, targets)
            train_pearson.update(outputs, targets)
            
            train_loss += loss.item()
            
            current_mae = train_mae.compute().item()
            train_loop.set_postfix(loss=loss.item(), mae=current_mae)
            
            if global_step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                swanlab.log({
                    'Batch/Train Loss': loss.item(),
                    'Batch/Train MAE': current_mae,
                    'Batch/LR': current_lr
                }, step=global_step)
            
            global_step += 1
        
        train_loss /= len(train_loader)
        train_mae_score = train_mae.compute().item()
        train_pearson_score = train_pearson.compute().item()
        
        model.eval()
        val_loss = 0
        val_mae.reset()
        val_pearson.reset()
        
        with torch.no_grad():
            val_loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for inputs, targets in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.float().to(device, non_blocking=True)

                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets)
                
                val_mae.update(outputs, targets)
                val_pearson.update(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss /= len(valid_loader)
        val_mae_score = val_mae.compute().item()
        val_pearson_score = val_pearson.compute().item()
        
        swanlab.log({
            'Epoch/Train Loss': train_loss,
            'Epoch/Train MAE': train_mae_score,
            'Epoch/Train Pearson': train_pearson_score,
            'Epoch/Validation Loss': val_loss,
            'Epoch/Validation MAE': val_mae_score,
            'Epoch/Validation Pearson': val_pearson_score,
            'Epoch/LR': optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Save best model
        if val_mae_score < best_val_mae:
            best_val_mae = val_mae_score
            best_ckpt_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{VER_NAME}_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> New best model saved (MAE={val_mae_score:.4f})")

def get_model():
    global DEVICE
    model = Net1D(
        in_channels=12,
        base_filters=24,
        ratio=1.0,
        filter_list = [24, 48, 96, 192],
        m_blocks_list = [2, 2, 2, 2],
        kernel_size=13,
        stride=1,
        groups_width=12,
        verbose=False,
        n_classes=1,
    )

    model.to(DEVICE)
    return model

def collate_fn(batch):
    """
    与之前保持一致：将 beats 列表展平并 Padding。
    """
    all_beats = []
    all_labels = []

    for beats, labels in batch:
        all_beats.extend(beats)
        all_labels.extend(labels)

    if not all_beats:
        return torch.tensor([]), torch.tensor([])

    # 动态 Padding: list of (12, L) -> (B, 12, Max_L)
    # 1. Permute to (L, 12) for pad_sequence
    beats_transposed = [b.permute(1, 0) for b in all_beats]
    
    # 2. Pad (batch_first=True -> B, Max_L, 12)
    padded_beats = rnn_utils.pad_sequence(beats_transposed, batch_first=True, padding_value=0.0)
    
    # 3. Permute back to (B, 12, Max_L)
    x = padded_beats.permute(0, 2, 1)
    
    if isinstance(all_labels[0], torch.Tensor):
        y = torch.stack(all_labels)
    else:
        y = torch.tensor(all_labels, dtype=torch.float32)

    return x, y

train_dataset = ECGDataset(split='train')
valid_dataset = ECGDataset(split='val')

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=CPU_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=CPU_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

model = get_model()

train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
)

# Save final checkpoint with timestamp
final_ckpt_path = os.path.join(MODEL_CHECKPOINT_DIR, f"{VER_NAME}_{timestamp}.pth")
torch.save(model.state_dict(), final_ckpt_path)

swanlab.finish()
print('Done')