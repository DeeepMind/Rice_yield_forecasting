import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import sys

sys.path.insert(0, "src")  # 如果 make_splits.py 在 src/
from make_splits import get_feature_cols


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
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


def run_baselines(data: pd.DataFrame, fold: dict, feature_cols: list) -> pd.DataFrame:
    is_test = data["Plot_Number"].isin(fold["test_plots"])
    X_tr, y_tr = data.loc[~is_test, feature_cols], data.loc[~is_test, "Yield"]
    X_te, y_te = data.loc[is_test, feature_cols], data.loc[is_test, "Yield"]
    rows = []
    # 均值预测
    y_pred = np.full(len(y_te), y_tr.mean())
    rows.append(
        {
            "model": "mean",
            **metrics(
                y_te, y_pred
            ),  # **代表字典解包，即把这个字典里的所有键值对平铺到当前字典中。
        }
    )
    # Ridge 岭回归和lasso 套索回归
    med = X_tr.median()
    X_tr_filled = X_tr.fillna(med)
    X_te_filled = X_te.fillna(med)

    sc = StandardScaler().fit(X_tr_filled)
    Xtr_s = sc.transform(X_tr_filled)
    Xte_S = sc.transform(X_te_filled)
    for name, mdl in [
        ("ridge", Ridge(alpha=0.1)),
        ("Lasso", Lasso(alpha=0.1, max_iter=1000, tol=1e-3)),
    ]:
        mdl.fit(Xtr_s, y_tr)
        rows.append({"model": name, **metrics(y_te, mdl.predict(Xte_S))})
    # XGBoost 梯度提升树（GBDT）
    xgbm = xgb.XGBRegressor(
        n_estimators=500,  # 树的数量（迭代轮数）
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,  # 随机抽 80% 的训练样本
        colsample_bytree=0.8,  # 随机抽 80% 的特征
        random_state=42,
        n_jobs=1,
    )
    xgbm.fit(X_tr, y_tr)
    rows.append({"model": "xgboost", **metrics(y_te, xgbm.predict(X_te))})
    return pd.DataFrame(rows)


# Experiment_Name_2021 = [
#     "Hybrid_XP753",
#     "Inbred_Jeff + NIL",
#     "Mini_Core",
#     "Santee_Gold",
#     "Tiara",
# ]
# for exp in Experiment_Name_2021:
#     data = pd.read_csv("Output/panel_wide_2022.csv")
#     fold = json.load(open(f"Datasets/splits/loeo_2022_loeo_{exp}.json"))
#     feat = get_feature_cols(data)
#     res = run_baselines(data, fold, feat)
#     print(f'name:{fold["name"]};year:2022')
#     print(res.to_string(index=False))
#     print(
#         f"\n测试集产量: 均值={data.loc[data.Plot_Number.isin(fold['test_plots']), 'Yield'].mean():.2f}"
#         f"  标准差={data.loc[data.Plot_Number.isin(fold['test_plots']), 'Yield'].std():.2f}"
#     )

data = pd.read_csv("output/panel_wide_2021_weather.csv")
fold = json.load(open("data/splits/loro_2021.json"))
feat = get_feature_cols(data)
print(f"数据形状: {data.shape}", flush=True)
print(f"特征数: {len(feat)}", flush=True)
res = run_baselines(data, fold, feat)
print(fold["name"])
print(res)
