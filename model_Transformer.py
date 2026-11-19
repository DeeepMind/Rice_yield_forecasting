import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_loader import load_tensor, load_split, get_index
from tensorboardX import SummaryWriter
from datetime import datetime

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
log_dir = f"log/train/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


class TimeSeriesDataset(Dataset):
    """
    时序数据 Dataset，支持三种数据增强（仅训练集启用）

      ① 随机丢弃时相   —— 模拟无人机因天气漏拍
      ② 幅度扰动       —— 模拟光照/曝光差异
      ③ 时相插值       —— 模拟采样时刻微小偏移（本项目默认启用）"""

    def __init__(
        self,
        X,
        y,
        augment=False,
        p_interp=0.10,
        p_drop=0.0,
        p_scale=0.0,
        scale_range=(0.95, 1.05),
    ):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.augment = augment
        self.p_interp = p_interp
        self.p_drop = p_drop
        self.p_scale = p_scale
        self.scale_range = scale_range

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()  # (T, D) 深拷贝，不污染原始数据
        y = self.y[i]

        if self.augment:
            T, D = x.shape
            rng = np.random

            # ① 时相插值（默认启用）
            if T > 2 and rng.random() < self.p_interp:
                t = rng.randint(0, T - 1)
                alpha = rng.uniform(0.2, 0.8)
                x[t] = alpha * x[t] + (1 - alpha) * x[t + 1]

            # ② 随机丢弃某一期（默认关闭）
            if self.p_drop > 0 and rng.random() < self.p_drop:
                x[rng.randint(T)] = 0.0

            # ③ 幅度扰动（默认关闭）
            if self.p_scale > 0 and rng.random() < self.p_scale:
                x = x * rng.uniform(*self.scale_range, size=(1, D))

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class LearnablePE(nn.Module):
    """可学习位置编码（Learnable Positional Encoding）"""

    def __init__(self, d_model, max_len=64, std=0.02):
        super().__init__()
        # (1, max_len, d_model)：每个时相一个可学习向量
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * std)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class AttentionModel(nn.Module):
    """注意力模型：(B,T,D) → (B,)"""

    def __init__(self, n_feat, hidden=64, nhead=4, nlayers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.pos = LearnablePE(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, return_attn=False):
        h = self.pos(self.proj(x))
        h = self.enc(h)
        w = torch.softmax(self.attn_pool(h), dim=1)
        pooled = (w * h).sum(dim=1)
        out = self.head(pooled).squeeze(-1)
        if return_attn:
            return out, w
        return out


def metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nrmse": float(rmse / np.mean(y_true)),
        "pred_mean": float(y_pred.mean()),
        "pred_std": float(y_pred.std()),
        "true_mean": float(y_true.mean()),
        "true_std": float(y_true.std()),
    }


def huber(pred, true, delta=1.0):
    err = (pred - true).abs()
    return torch.where(err <= delta, 0.5 * err**2, delta * (err - 0.5 * delta))


def train_deep(
    X,
    y,
    tr,
    te,
    hidden=64,
    epochs=350,
    lr=3e-3,
    batch_size=16,
    patience=40,
    verbose=False,
):
    """
    X          : (N, T, D) 张量
    y          : (N,) 产量
    tr, te     : 训练 / 测试索引
    返回"""
    # 特征标准化
    mu = X[tr].reshape(-1, X.shape[2]).mean(axis=0)  # axis=0以行为标准，顺着列进行压缩
    sd = X[tr].reshape(-1, X.shape[2]).std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    # 标签标准化
    ym, ys = y[tr].mean(), y[tr].std() + 1e-6
    yz = (y - ym) / ys
    # 内部分割验证集
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(tr)
    n_val = max(8, int(len(perm) * 0.2))
    va, tr2 = perm[:n_val], perm[n_val:]
    # ---------- 3. 增强参数 ----------
    # 数据增强（仅用于训练集）
    train_ds = TimeSeriesDataset(Xn[tr2], yz[tr2], augment=True)
    val_ds = TimeSeriesDataset(Xn[va], yz[va], augment=False)
    test_ds = TimeSeriesDataset(Xn[te], yz[te], augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # 每轮自动打乱
        num_workers=0,  # 进程数
        pin_memory=False,  # CPU 训练无意义
        drop_last=False,  # 样本少，不丢最后一批
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=len(val_ds),  # 验证集一次算完
        shuffle=False,  # 不打乱
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(Xn[te], dtype=torch.float32)),
        batch_size=len(te),
        shuffle=False,
        num_workers=0,
    )

    model = AttentionModel(X.shape[2], hidden)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=epochs
    )
    best_val, best_state, wait = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_seen = 0.0, 0
        for data, target in train_loader:
            loss = huber(model(data), target).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(data)
            n_seen += len(data)
        scheduler.step()
        train_loss = total / n_seen
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                val_loss = huber(model(data), target).mean().item()
        hist["train"].append(train_loss)
        hist["val"].append(val_loss)
        # 早停机制
        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"早停于第 {epoch} 轮（最佳验证损失 {best_val:.4f}）")
                break
    if best_state:
        model.load_state_dict(best_state)
    # 测试集
    model.eval()
    pred_parts = []
    with torch.no_grad():
        for (data,) in test_loader:
            pred_parts.append(model(data))
    pred = torch.cat(pred_parts).numpy() * ys + ym
    get_attention(model, Xn[te])
    return pred, n_params, hist


def extract_temporal_attention(model, Xn_te):
    Xt = torch.tensor(Xn_te, dtype=torch.float32)  # (73, 11, 62)
    with torch.no_grad():  # 只要推理，不算梯度
        _, w = model(Xt, return_attn=True)
    return w.squeeze(-1).numpy()  # (73, 11)


def get_attention(model, Xn_te):
    temporal_w = extract_temporal_attention(model, Xn_te)  # (73, 11)
    mean_w = temporal_w.mean(axis=0)  # (11,)  平均权重
    std_w = temporal_w.std(axis=0)  # (11,)  标准差
    uniform = 1.0 / len(mean_w)  # 0.0909
    dates = [
        "2021-06-14",
        "2021-06-23",
        "2021-06-30",
        "2021-07-14",
        "2021-07-28",
        "2021-08-03",
        "2021-08-10",
        "2021-08-17",
        "2021-08-27",
        "2021-09-03",
        "2021-09-10",
    ]
    stages = [
        "分蘖初",
        "分蘖中",
        "分蘖盛",
        "分蘖末",
        "拔节末",
        "孕穗初",
        "孕穗",
        "孕穗后",
        "抽穗",
        "灌浆初",
        "灌浆中",
    ]
    print(f"{'时相':<6}{'日期':<13}{'阶段':<8}{'权重':>8}{'标准差':>8}{'相对':>8}")
    for i in range(len(mean_w)):
        print(
            f"t{i:<5}{dates[i]:<13}{stages[i]:<8}"
            f"{mean_w[i]:>8.4f}{std_w[i]:>8.4f}"
            f"{mean_w[i]/uniform:>7.2f}x"
        )
    print(f"{'合计':<27}{mean_w.sum():>8.4f}")


def main():
    X, y, plot_ids, feats, dates = load_tensor()
    fold = load_split("Datasets/splits/loro_2021.json")
    tr, te = get_index(plot_ids, fold)  # tr,te为训练集和测试集小区号分别对应的索引
    pred, n_par, hist = train_deep(X, y, tr, te, verbose=True)
    r = metrics(y[te], pred)
    print(f"R²    = {r['r2']:.4f}")
    print(f"RMSE  = {r['rmse']:.4f} t/ha")
    print(f"MAE   = {r['mae']:.4f} t/ha")
    print(f"NRMSE = {r['nrmse']*100:.2f}%")
    print(f"真实产量: 均值 {r['true_mean']:.2f}  标准差 {r['true_std']:.2f}")
    print(f"预测产量: 均值 {r['pred_mean']:.2f}  标准差 {r['pred_std']:.2f}")


if __name__ == "__main__":
    main()
