import json
import re

import numpy as np
import pandas as pd

PAT = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")

PANEL_PATH = "output/panel_wide_2021_weather.csv"


def load_tensor(panel_path=PANEL_PATH):
    """
    读取宽表，返回：
        X       : (N, T, D) 时序张量
        y       : (N,) 产量
        plot_ids: (N,) 小区编号
        feats   : list，D 个特征名
        dates   : list，T 个日期
    """
    panel = pd.read_csv(panel_path)
    print(f"  宽表: {panel.shape}")
    # ---- 解析列名 NDVI_2021-06-14 → ('NDVI', '2021-06-14') ----
    parsed = {}
    for c in panel.columns:
        m = PAT.match(c)
        if m:
            parsed[c] = (m.group(1), m.group(2))
        # 把每个时序列名拆成 (特征, 日期)
    if not parsed:
        raise ValueError(f"{panel_path} 中没有 '特征名_日期' 格式的时序列")

    dates = sorted({d for _, d in parsed.values()})  # 日期组
    feats = sorted({f for f, _ in parsed.values()})  # 特征组
    T, D, N = len(dates), len(feats), len(panel)
    print(f"  时相 T={T}  特征 D={D}  小区 N={N}")

    # ---- 填张量 ----
    X = np.zeros((N, T, D), dtype=np.float32)
    di = {d: i for i, d in enumerate(dates)}  # [1:2021-xx-xx,2:2021-xx-xx]
    fi = {f: i for i, f in enumerate(feats)}
    for c, (f, d) in parsed.items():
        X[:, di[d], fi[f]] = panel[c].values  # 核心

    X = np.nan_to_num(X, nan=0.0)
    y = panel["Yield"].values.astype(np.float32)
    return X, y, panel["Plot_Number"].values, feats, dates


def load_split(split_path):
    """读取划分 JSON"""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_index(plot_ids, fold):
    """把划分的 Plot_Number 列表转成数组索引"""
    is_test = np.isin(plot_ids, fold["test_plots"])
    return np.where(~is_test)[0], np.where(is_test)[0]
