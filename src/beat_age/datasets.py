import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import random

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ECGDataset(Dataset):
    def __init__(self, split='train'):
        """
        加载预处理好的 beat 数据集，使用 splits.csv 进行分割。
        """
        self.processed_dir = "datasets/processed_beats"
        splits_csv = "datasets/splits.csv"

        df = pd.read_csv(splits_csv)
        df = df[df['split'] == split]

        self.file_names = df['FileName'].tolist()
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        pt_path = os.path.join(self.processed_dir, f"{file_name}.pt")
        
        try:
            # 极速加载: torch.load 读取 .pt 比 np.load 读取 .npz 更快，且直接就在 tensor 格式
            data_dict = torch.load(pt_path)
            
            beat_tensors = data_dict['beats'] # List[Tensor(12, L)]
            age = data_dict['age']
            
            # 构造标签列表
            labels = torch.tensor([age] * len(beat_tensors), dtype=torch.float32)
            
            return beat_tensors, labels
            
        except FileNotFoundError:
            # 如果某个文件预处理失败不存在，随机换一个
            print(f"Warning: {pt_path} not found. Replacing...")
            return self.__getitem__(random.randint(0, len(self)-1))
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))


class ECGRecordingDataset(Dataset):
    def __init__(
        self,
        split='train',
        raw_dir="datasets/raw_data/ecg_data",
        splits_csv="datasets/splits.csv",
    ):
        """
        Load full 10-second ECG recordings for ECG-age model training/evaluation.
        """
        self.raw_dir = raw_dir

        df = pd.read_csv(splits_csv)
        df = df[df['split'] == split].copy()
        df = df[df['FileName'].apply(lambda x: os.path.exists(os.path.join(raw_dir, f"{x}.npz")))]

        self.file_names = df['FileName'].tolist()
        self.ages = df['RecordAge'].astype(float).tolist()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        age = self.ages[idx]
        npz_path = os.path.join(self.raw_dir, f"{file_name}.npz")

        try:
            raw = np.load(npz_path)
            signal = np.stack([raw[lead] for lead in LEAD_ORDER], axis=0)
            signal = torch.tensor(signal, dtype=torch.float32)
            return signal, torch.tensor(age, dtype=torch.float32), file_name
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

def ecg_beat_collate_fn(batch):
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


def ecg_recording_collate_fn(batch):
    signals, labels, file_names = zip(*batch)
    lengths = [signal.shape[-1] for signal in signals]
    max_len = max(lengths)

    padded = torch.zeros(len(signals), 12, max_len, dtype=torch.float32)
    for i, signal in enumerate(signals):
        padded[i, :, :signal.shape[-1]] = signal

    return padded, torch.stack(labels), list(file_names)

if __name__ == "__main__":
    # 测试读取速度
    import time
    dataset = ECGDataset(split='train')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        t0 = time.time()
        loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=ecg_beat_collate_fn, num_workers=4)
        
        for i, (x, y) in enumerate(loader):
            print(f"Batch {i}: X={x.shape}, Y={y.shape}")
            if i >= 2: break
        
        print(f"Load check finished in {time.time()-t0:.4f}s")
