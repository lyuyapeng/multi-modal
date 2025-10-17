#    xgboost                    xgboost                          xgboost                             xgboost                   xgboost
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from plyer import notification

# 路径与超参数设置
FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5


def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 文件不存在：{path}")


def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None: sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)


def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values


def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]


start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()
try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数（需 1/2/3）：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)})与标签行数({len(y_all)})不一致。")

# 首先划分训练集/测试集 (防止数据泄露)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

# XGBoost要求标签从0开始，计算偏移量
label_offset = int(np.min(y_train_raw))
y_train0 = (y_train_raw - label_offset).astype(int)
y_test0 = (y_test_raw - label_offset).astype(int)
num_class = int(len(np.unique(y_train0)))


def cv_objective(learning_rate, n_estimators, max_depth, min_child_weight, gamma,
                 subsample, colsample_bytree, max_delta_step, colsample_bylevel,
                 reg_lambda, reg_alpha):
    params = dict(
        learning_rate=float(learning_rate),
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_child_weight=int(min_child_weight),
        gamma=float(gamma),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        max_delta_step=int(max_delta_step),
        colsample_bylevel=float(colsample_bylevel),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        objective='multi:softmax',
        num_class=num_class,
        nthread=-1,
        seed=RANDOM_STATE,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train0):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr0, y_val0 = y_train0[tr_idx], y_train0[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res0 = smote.fit_resample(X_tr_raw, y_tr0)

        # 标准化：在 SMOTE 重采样后的折训练子集上拟合
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr_res0)

        y_val_pred0 = model.predict(X_val)
        scores.append(accuracy_score(y_val0, y_val_pred0))

    return float(np.mean(scores))


pbounds = {
    'learning_rate': (0.01, 1.0),
    'n_estimators': (500, 3000),
    'max_depth': (3, 15),
    'min_child_weight': (1, 10),
    'gamma': (0.1, 0.8),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.01, 1.0),
    'max_delta_step': (0, 20),
    'colsample_bylevel': (0.01, 1.0),
    'reg_lambda': (1e-9, 1000),
    'reg_alpha': (1e-9, 10)
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    random_state=RANDOM_STATE,
)
optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best:", optimizer.max)

best = optimizer.max['params']
best_int = {
    'learning_rate': float(best['learning_rate']),
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_child_weight': int(best['min_child_weight']),
    'gamma': float(best['gamma']),
    'subsample': float(best['subsample']),
    'colsample_bytree': float(best['colsample_bytree']),
    'max_delta_step': int(best['max_delta_step']),
    'colsample_bylevel': float(best['colsample_bylevel']),
    'reg_lambda': float(best['reg_lambda']),
    'reg_alpha': float(best['reg_alpha']),
}
print("[Best Params]", best_int)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final0 = smote_final.fit_resample(X_train_raw, y_train0)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_class,
    nthread=-1,
    seed=RANDOM_STATE,
    eval_metric='mlogloss',
    use_label_encoder=False,
    **best_int
)
xgb_model.fit(X_train_scaled, y_train_res_final0)

inference_start_time = time.time()
y_pred0 = xgb_model.predict(X_test_scaled)
y_proba = xgb_model.predict_proba(X_test_scaled)
inference_end_time = time.time()

# 将预测标签加回偏移量，以匹配原始标签
y_pred = (y_pred0 + label_offset).astype(int)

print("\n--- Final Evaluation on Test Set ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_raw, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_raw, y_pred, digits=4))

# 计算并打印AUC值 (使用0-offset的真实标签 y_test0)
auc_macro = roc_auc_score(y_test0, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test0, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

total_duration_minutes = (time.time() - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters:", best_int)
print(f"Best cross-validation accuracy: {optimizer.max['target']:.4f}")
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

try:
    notification.notify(
        title='Python Script (XGBoost)',
        message=f"Finished! Best CV ACC: {optimizer.max['target']:.4f}\nTotal Time: {total_duration_minutes:.2f} min.",
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass









# dt                 dt                 dt                  dt                          dt                           dt
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from bayes_opt import BayesianOptimization
from plyer import notification

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 文件不存在：{path}")

def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None: sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)

def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values

def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]

start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()
try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数（需 1/2/3）：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)}) 与 标签行数({len(y_all)}) 不一致。")

# 首先划分训练集/测试集 (防止数据泄露)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

def cv_objective(max_depth, min_samples_leaf, min_samples_split):
    max_depth = int(max_depth)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # 仅在折内训练集上应用SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_raw, y_tr)

        # 仅在SMOTE后的折内训练集上拟合标准化器
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr_res)

        scores.append(model.score(X_val, y_val))

    return np.mean(scores)

pbounds = {
    'max_depth': (1, 50),
    'min_samples_leaf': (1, 10),
    'min_samples_split': (2, 10),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    verbose=2,
    random_state=RANDOM_STATE,
)

optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best CV Score:", optimizer.max['target'])

best_params_raw = optimizer.max['params']
best_params = {
    'max_depth': int(best_params_raw['max_depth']),
    'min_samples_leaf': int(best_params_raw['min_samples_leaf']),
    'min_samples_split': int(best_params_raw['min_samples_split']),
}
print("[Best Params]", best_params)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train_raw, y_train)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

final_model = DecisionTreeClassifier(**best_params, random_state=RANDOM_STATE)
final_model.fit(X_train_scaled, y_train_res_final)

inference_start_time = time.time()
y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)
inference_end_time = time.time()

print("\n--- Final Evaluation on Test Set ---")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Decision Tree)')
plt.colorbar()
unique_labels = np.unique(y_all).astype(int)
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, unique_labels)
plt.yticks(tick_marks, unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

total_duration_minutes = (time.time() - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters found: ", best_params)
print("Best cross-validation accuracy: {:.4f}".format(optimizer.max['target']))
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

try:
    notification.notify(
        title='Python Script (Decision Tree)',
        message=f'Finished! Best CV ACC: {optimizer.max["target"]:.4f}\nTotal Time: {total_duration_minutes:.2f} min.',
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass









#lightgbm                          lightgbm                                        lightgbm                       lightgbm
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from plyer import notification

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 文件不存在：{path}")

def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None: sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)

def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values

def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]

start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()
try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数（需 1/2/3）：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)})与标签行数({len(y_all)})不一致。")

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

label_offset = int(np.min(y_train_raw))
y_train0 = (y_train_raw - label_offset).astype(int)
y_test0 = (y_test_raw - label_offset).astype(int)
num_class = len(np.unique(y_train0))

def cv_objective(n_estimators, num_leaves, max_depth, learning_rate,
                 feature_fraction, bagging_fraction, lambda_l1, lambda_l2, min_child_samples):
    params = {
        'n_estimators': int(n_estimators),
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'learning_rate': float(learning_rate),
        'feature_fraction': float(feature_fraction),
        'bagging_fraction': float(bagging_fraction),
        'lambda_l1': float(lambda_l1),
        'lambda_l2': float(lambda_l2),
        'min_child_samples': int(min_child_samples),
        'objective': 'multiclass',
        'num_class': num_class,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train0):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr0, y_val0 = y_train0[tr_idx], y_train0[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res0 = smote.fit_resample(X_tr_raw, y_tr0)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr_res0)

        y_val_pred0 = model.predict(X_val)
        scores.append(accuracy_score(y_val0, y_val_pred0))

    return np.mean(scores)

pbounds = {
    'n_estimators': (50, 300),
    'num_leaves': (5, 64),
    'max_depth': (3, 25),
    'learning_rate': (0.01, 0.2),
    'feature_fraction': (0.5, 0.95),
    'bagging_fraction': (0.5, 0.95),
    'lambda_l1': (0.0, 0.6),
    'lambda_l2': (0.0, 0.6),
    'min_child_samples': (5, 50),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    random_state=RANDOM_STATE,
    verbose=2
)
optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best CV Score:", optimizer.max['target'])

best_params_raw = optimizer.max['params']
best_params = {
    'n_estimators': int(best_params_raw['n_estimators']),
    'num_leaves': int(best_params_raw['num_leaves']),
    'max_depth': int(best_params_raw['max_depth']),
    'learning_rate': float(best_params_raw['learning_rate']),
    'feature_fraction': float(best_params_raw['feature_fraction']),
    'bagging_fraction': float(best_params_raw['bagging_fraction']),
    'lambda_l1': float(best_params_raw['lambda_l1']),
    'lambda_l2': float(best_params_raw['lambda_l2']),
    'min_child_samples': int(best_params_raw['min_child_samples']),
}
print("[Best Params]", best_params)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final0 = smote_final.fit_resample(X_train_raw, y_train0)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

final_model = LGBMClassifier(
    objective='multiclass',
    num_class=num_class,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=-1,
    **best_params
)
final_model.fit(X_train_scaled, y_train_res_final0)

inference_start_time = time.time()
y_pred0 = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)
inference_end_time = time.time()

y_pred = (y_pred0 + label_offset).astype(int)

print("\n--- Final Evaluation on Test Set ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_raw, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_raw, y_pred, digits=4))

auc_macro = roc_auc_score(y_test0, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test0, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

total_duration_minutes = (time.time() - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters found: ", best_params)
print("Best cross-validation accuracy: {:.4f}".format(optimizer.max['target']))
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

try:
    notification.notify(
        title='Python Script (LightGBM)',
        message=f'Finished! Best CV ACC: {optimizer.max["target"]:.4f}\nTotal Time: {total_duration_minutes:.2f} min.',
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass









#adaboost                               adaboost                                    adaboost                                       adaboost
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from bayes_opt import BayesianOptimization
from plyer import notification

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5


def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] 文件不存在：{path}\n"
            f"请检查：\n"
            f"  1) 路径是否正确（盘符/文件夹/文件名/后缀）\n"
            f"  2) 是否另存为了 CSV 或 Excel（UTF-8）\n"
            f"  3) 路径中是否含有空格或全角字符导致粘贴出错\n"
        )


def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None:
            sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df, dict):
            first_key = list(df.keys())[0]
            df = df[first_key]
        return df
    else:
        return pd.read_csv(path)


def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce')
    df_num = df_num.fillna(df_num.median(numeric_only=True))
    return df_num.astype(np.float32).values


def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if isinstance(df, dict):
        df = list(df.values())[0]
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower:
            return df[cols_lower[k]]
    return df.iloc[:, 0]


start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()

try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数，请检查内容是否为 1/2/3：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)}) 与 标签行数({len(y_all)}) 不一致，请确认两文件行顺序严格对应。")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)


def cv_objective(n_estimators, max_depth, min_samples_leaf, min_samples_split, learning_rate):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)
    learning_rate = float(learning_rate)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_raw, y_tr)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        base = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE
        )
        clf = AdaBoostClassifier(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm='SAMME',
            random_state=RANDOM_STATE
        )
        clf.fit(X_tr, y_tr_res)
        scores.append(clf.score(X_val, y_val))

    return float(np.mean(scores))


pbounds = {
    'n_estimators': (50, 300),
    'max_depth': (5, 50),
    'min_samples_leaf': (1, 6),
    'min_samples_split': (10, 30),
    'learning_rate': (0.1, 1.0),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    verbose=2,
    random_state=RANDOM_STATE,
)

optimizer.maximize(init_points=3, n_iter=30)
print("\n[BayesOpt] Best:", optimizer.max)

best = optimizer.max['params']
best_int = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'min_samples_split': int(best['min_samples_split']),
    'learning_rate': float(best['learning_rate']),
}
print("[Best Params]", best_int)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train_raw, y_train)

scaler_final = StandardScaler()
X_train = scaler_final.fit_transform(X_train_res_final)
X_test = scaler_final.transform(X_test_raw)

tree = DecisionTreeClassifier(
    max_depth=best_int['max_depth'],
    min_samples_leaf=best_int['min_samples_leaf'],
    min_samples_split=best_int['min_samples_split'],
    random_state=RANDOM_STATE
)

model = AdaBoostClassifier(
    estimator=tree,
    n_estimators=best_int['n_estimators'],
    learning_rate=best_int['learning_rate'],
    algorithm='SAMME',
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train_res_final)

t0 = time.time()
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
t1 = time.time()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')

print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

print("\nBest parameters:", best_int)
print(f"Inference time on the test set: {t1 - t0:.6f} seconds")
print(f"Total runtime of the script: {(time.time() - start_time) / 60:.4f} minutes")

try:
    notification.notify(
        title='Python Script Finished',
        message=f'Runtime: {((time.time() - start_time) / 60):.2f} min. Best CV score: {optimizer.max["target"]:.4f}',
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass










#ann                       ann                             ann                               ann                           ann
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from bayes_opt import BayesianOptimization
from plyer import notification

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5


def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 文件不存在：{path}")


def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None: sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)


def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values


def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]


start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()
try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数（需 1/2/3）：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)})与标签行数({len(y_all)})不一致。")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)


def cv_objective(hidden_layer_sizes, alpha):
    hidden_layer_sizes = int(hidden_layer_sizes)
    alpha = float(alpha)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_raw, y_tr)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_sizes,),
            alpha=alpha,
            max_iter=500,
            random_state=RANDOM_STATE,
            solver='adam',
            activation='relu',
            early_stopping=True,
            n_iter_no_change=15
        )
        mlp.fit(X_tr, y_tr_res)

        y_val_pred = mlp.predict(X_val)
        scores.append(accuracy_score(y_val, y_val_pred))

    return float(np.mean(scores))


pbounds = {
    'hidden_layer_sizes': (50, 300),
    'alpha': (1e-4, 1e-1)
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    random_state=RANDOM_STATE,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best CV Score:", optimizer.max['target'])

best_params_raw = optimizer.max['params']
best_params = {
    'hidden_layer_sizes': int(best_params_raw['hidden_layer_sizes']),
    'alpha': float(best_params_raw['alpha'])
}
print("[Best Params]", best_params)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train_raw, y_train)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

mlp_final = MLPClassifier(
    hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
    alpha=best_params['alpha'],
    max_iter=500,
    random_state=RANDOM_STATE,
    solver='adam',
    activation='relu',
    early_stopping=True,
    n_iter_no_change=15
)
mlp_final.fit(X_train_scaled, y_train_res_final)

inference_start_time = time.time()
y_pred = mlp_final.predict(X_test_scaled)
y_proba = mlp_final.predict_proba(X_test_scaled)
inference_end_time = time.time()

print("\n--- Final Evaluation on Test Set ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

total_duration_minutes = (time.time() - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters found: ", best_params)
print("Best cross-validation accuracy: {:.4f}".format(optimizer.max['target']))
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

try:
    notification.notify(
        title='Python Script (MLP)',
        message=f'Finished! Best CV ACC: {optimizer.max["target"]:.4f}\nTotal Time: {total_duration_minutes:.2f} min.',
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass
















# knn               knn               knn               knn               knn               knn               knn
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from bayes_opt import BayesianOptimization
from plyer import notification
import winsound

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n[ERROR] 文件不存在：{path}")

def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)

def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values

def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]

start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df)
y_all = y_series.astype(int).values
X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)})与标签行数({len(y_all)})不一致。")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)

def cv_objective(n_neighbors, p):
    n_neighbors = int(n_neighbors)
    p = int(p)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_raw, y_tr)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, n_jobs=-1)
        model.fit(X_tr, y_tr_res)

        scores.append(model.score(X_val, y_val))

    return np.mean(scores)

pbounds = {
    'n_neighbors': (1, 20),
    'p': (1, 2),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    verbose=2,
    random_state=RANDOM_STATE,
)

optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best CV Score:", optimizer.max['target'])

best_params = optimizer.max['params']
best_int_params = {
    'n_neighbors': int(best_params['n_neighbors']),
    'p': int(best_params['p']),
}
print("[Best Params]", best_int_params)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train_raw, y_train)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

final_model = KNeighborsClassifier(**best_int_params, n_jobs=-1)
final_model.fit(X_train_scaled, y_train_res_final)

inference_start_time = time.time()
y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)
inference_end_time = time.time()

print("\n--- Final Evaluation on Test Set ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

end_time = time.time()
total_duration_minutes = (end_time - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters found: ", best_int_params)
print("Best cross-validation accuracy: {:.4f}".format(optimizer.max['target']))
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

notification.notify(
    title='Python Script Finished',
    message=f'KNN script finished.\nCV Score: {optimizer.max["target"]:.4f}\nTotal Time: {total_duration_minutes:.2f} min.',
    app_icon=None,
    timeout=10,
)

winsound.Beep(1500, 1000)














#svm                 svm               svm               svm               svm               svm               svm
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from bayes_opt import BayesianOptimization
from plyer import notification

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 文件不存在：{path}")

def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None: sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)

def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values

def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]

start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()
try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数（需 1/2/3）：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)}) 与 标签行数({len(y_all)}) 不一致。")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

def cv_objective(C, gamma):
    C = float(C)
    gamma = float(gamma)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_raw, y_tr)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr_res)

        scores.append(model.score(X_val, y_val))

    return np.mean(scores)

pbounds = {
    'C': (0.1, 2000.0),
    'gamma': (0.01, 10.0),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    verbose=2,
    random_state=RANDOM_STATE,
)

optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best CV Score:", optimizer.max['target'])

best_params = optimizer.max['params']
print("[Best Params]", best_params)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train_raw, y_train)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

final_model = SVC(**best_params, kernel='rbf', probability=True, random_state=RANDOM_STATE)
final_model.fit(X_train_scaled, y_train_res_final)

inference_start_time = time.time()
y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)
inference_end_time = time.time()

print("\n--- Final Evaluation on Test Set ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

end_time = time.time()
total_duration_minutes = (end_time - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters found: ", best_params)
print("Best cross-validation accuracy: {:.4f}".format(optimizer.max['target']))
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

try:
    notification.notify(
        title='Python Script (SVM)',
        message=f'Finished! Best CV ACC: {optimizer.max["target"]:.4f}\nTotal Time: {total_duration_minutes:.2f} min.',
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass











#rf                rf                rf                rf                rf                rf                rf                rf
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from bayes_opt import BayesianOptimization
from plyer import notification

FEATURE_PATH = r"特征"
LABEL_PATH = r"标签"
FEATURE_SHEET = None
LABEL_SHEET = None

TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 文件不存在：{path}")

def load_table(path: str, sheet=None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None: sheet = 0
        df = pd.read_excel(path, sheet_name=sheet)
        return list(df.values())[0] if isinstance(df, dict) else df
    return pd.read_csv(path)

def to_float32_matrix(df: pd.DataFrame) -> np.ndarray:
    df_num = df.apply(pd.to_numeric, errors='coerce').fillna(df.median(numeric_only=True))
    return df_num.astype(np.float32).values

def pick_label_series(df: pd.DataFrame) -> pd.Series:
    if df.shape[1] == 1: return df.iloc[:, 0]
    candidates = ["label", "labels", "y", "target", "fatigue_level", "标签", "类别", "class"]
    cols_lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols_lower: return df[cols_lower[k]]
    return df.iloc[:, 0]

start_time = time.time()

assert_file_exists(FEATURE_PATH)
assert_file_exists(LABEL_PATH)

X_df = load_table(FEATURE_PATH, FEATURE_SHEET)
y_df = load_table(LABEL_PATH, LABEL_SHEET)

y_series = pick_label_series(y_df).astype(str).str.strip()
try:
    y_all = y_series.astype(int).values
except Exception as e:
    raise ValueError(f"[ERROR] 标签列无法转换为整数（需 1/2/3）：\n{e}")

X_all = to_float32_matrix(X_df)

if len(X_all) != len(y_all):
    raise ValueError(f"[ERROR] 特征行数({len(X_all)}) 与 标签行数({len(y_all)}) 不一致。")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

def cv_objective(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)

    max_features_map = {0: 'sqrt', 1: 'log2', 2: None}
    max_features = max_features_map[int(max_features)]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw[tr_idx], X_train_raw[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_raw, y_tr)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_res)
        X_val = scaler.transform(X_val_raw)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr_res)

        scores.append(model.score(X_val, y_val))

    return np.mean(scores)

pbounds = {
    'n_estimators': (5, 300),
    'max_depth': (1, 50),
    'min_samples_leaf': (1, 10),
    'min_samples_split': (2, 10),
    'max_features': (0, 2.99),
}

optimizer = BayesianOptimization(
    f=cv_objective,
    pbounds=pbounds,
    verbose=2,
    random_state=RANDOM_STATE,
)

optimizer.maximize(init_points=5, n_iter=35)
print("\n[BayesOpt] Best CV Score:", optimizer.max['target'])

best_params_raw = optimizer.max['params']
max_features_map = {0: 'sqrt', 1: 'log2', 2: None}
best_params = {
    'n_estimators': int(best_params_raw['n_estimators']),
    'max_depth': int(best_params_raw['max_depth']),
    'min_samples_leaf': int(best_params_raw['min_samples_leaf']),
    'min_samples_split': int(best_params_raw['min_samples_split']),
    'max_features': max_features_map[int(best_params_raw['max_features'])],
}
print("[Best Params]", best_params)

smote_final = SMOTE(random_state=RANDOM_STATE)
X_train_res_final, y_train_res_final = smote_final.fit_resample(X_train_raw, y_train)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_res_final)
X_test_scaled = scaler_final.transform(X_test_raw)

final_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
final_model.fit(X_train_scaled, y_train_res_final)

inference_start_time = time.time()
y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)
inference_end_time = time.time()

print("\n--- Final Evaluation on Test Set ---")
test_acc = (y_pred == y_test).mean()
print(f"Test set accuracy: {test_acc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
print(f"ROC AUC (Macro): {auc_macro:.4f}")
print(f"ROC AUC (Micro): {auc_micro:.4f}")

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (RandomForest)')
plt.colorbar()
unique_labels = np.unique(y_all).astype(int)
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, unique_labels)
plt.yticks(tick_marks, unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

total_duration_minutes = (time.time() - start_time) / 60
inference_duration_seconds = inference_end_time - inference_start_time

print("\n--- Summary ---")
print("Best parameters found: ", best_params)
print("Best cross-validation accuracy: {:.4f}".format(optimizer.max['target']))
print(f"Inference time on the test set: {inference_duration_seconds:.6f} seconds")
print(f"Total runtime of the script: {total_duration_minutes:.4f} minutes")

try:
    notification.notify(
        title='Python Script (RandomForest)',
        message=f'Finished! Best CV ACC: {optimizer.max["target"]:.4f}\nTotal Time: {total_duration_minutes:.2f} min.',
        app_icon=None,
        timeout=10,
    )
except Exception:
    pass















