import json
import os
import numpy as np
import pandas as pd

META_COLS = [
    "Plot_Number",
    "Date",
    "Rice_Cultivar",
    "Experiment_Name",
    "Bay_Length",
    "Nitrogen_Rate",
    "Replicate",
]

# 标签 / 性状列：不参与建模
LABEL_COLS = [
    "Yield",
    "Heading_25",
    "Heading_50",
    "Heading_100",
    "SPAD",
    "Percent_Stand",
    "Percent_Lodge",
    "Final_Lodge",
    "Leaf1_Temp",
    "Leaf2_Temp",
    "Vegetation_Fraction",
    # 2022 独有列，必须排除
    "Measurement_Stage",
    "DOY",
    "LAI",
    "LCI",
    "BNDVI",
]

NON_FEATURE_COLS = META_COLS + LABEL_COLS


def get_feature_cols(df, extra_exclude=None):
    """
    从宽表取特征列
    排除：标识列、标签列、Year、非数值列（如 Bay_Length 的字符串）
    """
    exclude = set(NON_FEATURE_COLS) | {"Year"}
    if extra_exclude:
        exclude |= set(extra_exclude)

    numeric = set(df.select_dtypes(include="number").columns)
    feats = [c for c in df.columns if c not in exclude and c in numeric]

    dropped = [c for c in df.columns if c not in exclude and c not in numeric]
    if dropped:
        print(f"  [提示] 自动排除非数值列: {dropped}")
    return feats


def load_csv(year, raw_csv_dir="data/csv"):
    path = os.path.join(raw_csv_dir, f"Dataset_{year}.csv")
    df = pd.read_csv(path)
    assert "Date" in df.columns, f"{path} 中没有 Date 列"
    return df


def clean_long(df):
    """
    清洗长表

    1) 统一日期格式
    2) 去无产量小区（Mini_Core 28 个）
    3) 去掉 2021-09-16（只有 Santee_Gold 72 个小区，样本不齐）
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df[df["Yield"].notna()]
    df = df[df["Date"] != "2021-09-16"]
    return df.reset_index(drop=True)


# =====================================================================
# 长表 → 宽表
# =====================================================================
def build_wide(df):

    # ---- 1) 标签表 ----
    label_cols_available = [c for c in LABEL_COLS if c in df.columns]
    labels = df.drop_duplicates("Plot_Number")[
        ["Plot_Number"]
        + label_cols_available
        + [c for c in META_COLS[1:] if c != "Date"]
    ].copy()

    # ---- 2) 特征列（只保留数值型）----
    feat_cols = get_feature_cols(df)

    long = df.melt(
        id_vars=["Plot_Number", "Date"],
        value_vars=feat_cols,
        var_name="feature",
        value_name="value",
    )
    long["feat_date"] = long["feature"] + "_" + long["Date"]

    wide = long.pivot_table(
        index="Plot_Number",
        columns="feat_date",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # ---- 3) 合并 ----
    return labels.merge(wide, on="Plot_Number", how="inner")


def make_loro(df_plot, test_frac=1 / 3, seed=42, strat_cols=None):
    # LORO：按 (实验, 品种, 氮肥) 分层，每层随机留 1/n 重复进测试集
    if strat_cols is None:
        strat_cols = ["Experiment_Name", "Rice_Cultivar", "Nitrogen_Rate"]

    assert df_plot["Plot_Number"].is_unique, "请先 drop_duplicates('Plot_Number')"

    rng = np.random.default_rng(seed)
    test_plots = []
    for _, g in df_plot.groupby(strat_cols):
        reps = g["Replicate"].unique()
        n_test = max(1, int(round(len(reps) * test_frac)))
        chosen = rng.choice(reps, size=n_test, replace=False)
        test_plots += g[g["Replicate"].isin(chosen)]["Plot_Number"].tolist()
    test_plots = sorted(set(test_plots))
    all_plots = set(df_plot["Plot_Number"])
    return [
        {
            "name": "loro",
            "train_plots": sorted(all_plots - set(test_plots)),
            "test_plots": test_plots,
        }
    ]


def make_loeo(df_plot):
    # LOEO：留一实验
    all_plots = set(df_plot["Plot_Number"])
    return [
        {
            "name": f"loeo_{exp}",
            "train_plots": sorted(all_plots - set(g["Plot_Number"])),
            "test_plots": sorted(set(g["Plot_Number"])),
        }
        for exp, g in df_plot.groupby("Experiment_Name")
    ]


def check_fold(fold, all_plots):
    """断言：训练/测试无交集，且并集为全集"""
    tr, te = set(fold["train_plots"]), set(fold["test_plots"])
    assert not (tr & te), f"{fold['name']}: {len(tr & te)} 个小区重叠！"
    assert tr | te == all_plots, f"{fold['name']}: 有小区未分配"
    return len(te)


def main(raw_csv_dir="raw/csv", out_dir="output", split_dir="data/splits"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # ---- 读 + 清洗 + 宽表 ----
    panels = {}
    for year in [2021]:
        print(f"\n[{year}] 读取与清洗")
        df = load_csv(year, raw_csv_dir)
        print(f"  原始长表: {df.shape}")
        df = clean_long(df)
        print(f"  清洗后: {df.shape}  小区 {df['Plot_Number'].nunique()}")

        panel = build_wide(df)
        panel["Year"] = year
        print(f"  宽表: {panel.shape}")

        fp = os.path.join(out_dir, f"panel_wide_{year}.csv")
        panel.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"  [√] {fp}")
        panels[year] = panel
        # loro
    print("\n" + "=" * 60)
    print("生成 LORO 划分")
    print("=" * 60)
    for year, panel in panels.items():
        df_plot = panel.drop_duplicates("Plot_Number")
        for f in make_loro(df_plot):
            n = check_fold(f, set(df_plot["Plot_Number"]))
            fp = os.path.join(split_dir, f"loro_{year}.json")
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump(f, fh, indent=2)
            print(f"  {year}: 训练 {len(f['train_plots'])} / 测试 {n}  → {fp}")
    # loeo
    print("\n" + "=" * 60)
    print("生成 LOEO 划分")
    print("=" * 60)
    for year, panel in panels.items():
        df_plot = panel.drop_duplicates("Plot_Number")
        for f in make_loeo(df_plot):
            n = check_fold(f, set(df_plot["Plot_Number"]))
            fp = os.path.join(split_dir, f"loeo_{year}_{f['name']}.json")
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump(f, fh, indent=2)
            print(f"  {year} {f['name']}: 测试 {n}")

    print("\n完成")


if __name__ == "__main__":
    main()
