
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# ── Imbalanced-learn (SMOTE + ENN  →  SMOTEND) ───────────────────────────────
from imblearn.combine import SMOTEENN

# ── Scipy ─────────────────────────────────────────────────────────────────────
from scipy.stats import kendalltau

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_DIR    = r"D:\projects\Ml-case-study\data"
RESULTS_DIR = r"D:\projects\Ml-case-study\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.30
BATCH_SIZE   = 16
EPOCHS       = 100

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── 20 object-oriented metrics (paper Table II) ───────────────────────────────
ALL_METRICS = [
    'wmc', 'dit', 'noc', 'cbo',          # group 1
    'rfc', 'lcom', 'ca', 'ce',            # group 2
    'npm', 'lcom3', 'loc', 'dam',         # group 3
    'moa', 'mfa', 'cam', 'ic',            # group 4
    'cbm', 'amc', 'max_cc', 'avg_cc',     # group 5
]

FEATURE_GROUPS = {
    'group1': ['wmc', 'dit', 'noc', 'cbo'],
    'group2': ['rfc', 'lcom', 'ca', 'ce'],
    'group3': ['npm', 'lcom3', 'loc', 'dam'],
    'group4': ['moa', 'mfa', 'cam', 'ic'],
    'group5': ['cbm', 'amc', 'max_cc', 'avg_cc'],
}

TARGET = 'bug'

# ── Alternative column name aliases found in PROMISE CSV files ────────────────
COL_ALIASES = {
    'bugs': 'bug', 'defects': 'bug', 'fault': 'bug', 'faults': 'bug',
    'lcom*': 'lcom3', 'lcom_3': 'lcom3',
    'max_cc': 'max_cc', 'maxcc': 'max_cc',
    'avg_cc': 'avg_cc', 'avgcc': 'avg_cc',
}

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ normalise all the colums for a pattern"""
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns=COL_ALIASES, inplace=True)
    return df


def load_promise_data(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, '**', '*.csv')
    files   = glob.glob(pattern, recursive=True)

    if not files:
        # Fallback: look for .arff files converted to CSV with same pattern
        pattern = os.path.join(data_dir, '*.csv')
        files   = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No CSV files found under '{data_dir}'. "
            "Please place the PROMISE project CSV files there."
        )

    frames = []
    for fp in sorted(files):
        try:
            df = pd.read_csv(fp, low_memory=False)
            df = normalise_columns(df)
            frames.append(df)
            print(f"Loaded {os.path.basename(fp):40s}  → {len(df):>6,} rows")
        except Exception as exc:
            print(f"Skipped {os.path.basename(fp)}: {exc}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows loaded: {len(combined):,}")
    return combined


def preprocess(df: pd.DataFrame):
    """
    Steps (following the paper):
      1. Keep only the 20 metrics + target column.
      2. Drop rows with all-NaN metrics.
      3. Drop duplicates.
      4. Impute remaining NaNs with column median.
      5. Clip negative fault counts to 0.
    Returns X (DataFrame), y (Series).
    """
    # ── Keep relevant columns ──────────────────────────────────────────────
    available_metrics = [c for c in ALL_METRICS if c in df.columns]
    missing_metrics   = [c for c in ALL_METRICS if c not in df.columns]

    if TARGET not in df.columns:
        raise KeyError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if missing_metrics:
        print(f"\n Missing metrics (will be zero-filled): {missing_metrics}")
        for m in missing_metrics:
            df[m] = 0.0

    df = df[ALL_METRICS + [TARGET]].copy()

    # ── Drop rows where ALL metric columns are NaN ─────────────────────────
    df.dropna(subset=ALL_METRICS, how='all', inplace=True)

    # ── Drop duplicates ────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Dropped {before - len(df):,} duplicate rows → {len(df):,} remaining")

    # ── Impute NaN with median ─────────────────────────────────────────────
    df[ALL_METRICS] = df[ALL_METRICS].fillna(df[ALL_METRICS].median())
    df[TARGET]      = df[TARGET].fillna(0)

    # ── Clip negative fault counts ─────────────────────────────────────────
    df[TARGET] = df[TARGET].clip(lower=0)

    X = df[ALL_METRICS].astype(float)
    y = df[TARGET].astype(float)

    print(f"  Faulty classes  : {(y > 0).sum():,}  ({(y > 0).mean()*100:.1f}%)")
    print(f"  Non-faulty      : {(y == 0).sum():,}  ({(y == 0).mean()*100:.1f}%)")
    return X, y


def apply_smotend(X: np.ndarray, y: np.ndarray):
    """
    SMOTEND  ≈  SMOTE + Edited Nearest Neighbours (SMOTEENN).
    Converts continuous target to binary (faulty / non-faulty) for
    over-sampling, then maps synthetic samples back.
    Strategy: oversample minority (faulty, label=1).
    """
    y_bin = (y > 0).astype(int)
    smote_enn = SMOTEENN(random_state=RANDOM_STATE)
    X_res, y_bin_res = smote_enn.fit_resample(X, y_bin)

    # For the synthetic 'faulty' rows, assign a fault count drawn from
    # a Poisson distribution with mean = original faulty mean.
    faulty_mean = y[y > 0].mean() if (y > 0).any() else 1.0
    y_res = np.where(y_bin_res == 0, 0.0,
                     np.random.poisson(faulty_mean, size=len(y_bin_res)).clip(1))
    print(f"  After SMOTEND   : {len(X_res):,} rows  "
          f"(faulty {(y_bin_res==1).sum():,} / "
          f"non-faulty {(y_bin_res==0).sum():,})")
    return X_res.astype(np.float32), y_res.astype(np.float32)


def log_transform_and_scale(X_train, X_test, y_train, y_test):
    """Log-transform features + target, then standardise features."""
    # Log1p on features (handles zeros)
    X_train_log = np.log1p(np.abs(X_train))
    X_test_log  = np.log1p(np.abs(X_test))

    # Standardise
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_log).astype(np.float32)
    X_test_sc  = scaler.transform(X_test_log).astype(np.float32)

    # Log-transform target
    y_train_log = np.log1p(y_train).astype(np.float32)
    y_test_log  = np.log1p(y_test).astype(np.float32)

    return X_train_sc, X_test_sc, y_train_log, y_test_log, scaler


# =============================================================================
# 2. MODEL DEFINITIONS
# =============================================================================

GROUP_INDICES = {
    'group1': [0, 1, 2, 3],
    'group2': [4, 5, 6, 7],
    'group3': [8, 9, 10, 11],
    'group4': [12, 13, 14, 15],
    'group5': [16, 17, 18, 19],
}


class FeatureSlice(layers.Layer):
    """Keras layer that selects a fixed subset of feature columns."""
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, x):
        import keras.ops as kops
        return kops.take(x, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.indices))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'indices': self.indices})
        return cfg


def build_cnn_model(dropout_rate: float = 0.3) -> Model:
    """
    CNN model – Figure 7 in the paper.
    5 separate Input branches (one per feature group of 4 features each):
        Input(4) → Reshape(4,1) → Conv1D(32,k=2) → MaxPool1D → Dropout → Flatten
    Merged → Dense(128) → Dropout → Dense(1, linear)
    """
    inputs   = []
    branches = []

    for item in GROUP_INDICES:
        inp = Input(shape=(4,), name=f'input_{item}')
        inputs.append(inp)

        x = layers.Reshape((4, 1), name=f'reshape_{item}')(inp)
        x = layers.Conv1D(32, kernel_size=2, activation='relu',
                          kernel_initializer='glorot_uniform',
                          name=f'conv_{item}')(x)
        x = layers.MaxPool1D(pool_size=1, name=f'pool_{item}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_{item}')(x)
        x = layers.Flatten(name=f'flat_{item}')(x)
        branches.append(x)

    merged = layers.Concatenate(name='merge')(branches)
    x      = layers.Dense(128, activation='relu',
                          kernel_initializer='glorot_uniform',
                          name='dense_1')(merged)
    x      = layers.Dropout(dropout_rate, name='drop_merged')(x)
    output = layers.Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=output, name='CNN_SFP')
    model.compile(optimizer='adam', loss='mse')
    return model


def build_mlp_model(dropout_rate: float = 0.0) -> Model:
    """
    MLP model – Figure 8 in the paper.
    5 separate Input branches:
        Input(4) → Dense(32) → Flatten
    Merged → Dense(128) → Dense(1, linear)
    """
    inputs   = []
    branches = []

    for gname in GROUP_INDICES:
        inp = Input(shape=(4,), name=f'input_{gname}')
        inputs.append(inp)

        x = layers.Dense(32, activation='relu',
                         kernel_initializer='glorot_uniform',
                         name=f'dense_{gname}')(inp)
        x = layers.Flatten(name=f'flat_{gname}')(x)
        branches.append(x)

    merged = layers.Concatenate(name='merge')(branches)
    x      = layers.Dense(128, activation='relu',
                          kernel_initializer='glorot_uniform',
                          name='dense_1')(merged)
    output = layers.Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=output, name='MLP_SFP')
    model.compile(optimizer='adam', loss='mse')
    return model


def split_to_groups(X: np.ndarray) -> list:
    """Split a (N, 20) array into 5 arrays of shape (N, 4), one per group."""
    return [X[:, idxs] for idxs in GROUP_INDICES.values()]


# =============================================================================
# 3. TRAINING UTILITIES
# =============================================================================

def train_model(model: Model, X_train, y_train, X_test, y_test,
                label: str = ''):
    """Train a Keras model with early stopping; return history."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, verbose=0, min_lr=1e-6),
    ]
    print(f"\n  Training {label} …")
    history = model.fit(
        split_to_groups(X_train), y_train,
        validation_data=(split_to_groups(X_test), y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    print(f"  Stopped at epoch {len(history.history['loss'])}")
    return history


def evaluate_model(model: Model, X_train, y_train, X_test, y_test):
    """Return (train_kendall, train_mse, test_kendall, test_mse)."""
    y_train_pred = model.predict(split_to_groups(X_train), verbose=0).ravel()
    y_test_pred  = model.predict(split_to_groups(X_test),  verbose=0).ravel()

    train_mse  = mean_squared_error(y_train, y_train_pred)
    test_mse   = mean_squared_error(y_test,  y_test_pred)
    train_ken  = kendalltau(y_train, y_train_pred).correlation
    test_ken   = kendalltau(y_test,  y_test_pred).correlation

    return train_ken, train_mse, test_ken, test_mse


def train_ml_model(model, X_train, y_train, X_test, y_test, label=''):
    """Train an sklearn regressor and return metrics."""
    print(f"  Training {label} …")
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse  = mean_squared_error(y_test,  y_test_pred)
    train_ken = kendalltau(y_train, y_train_pred).correlation
    test_ken  = kendalltau(y_test,  y_test_pred).correlation
    return train_ken, train_mse, test_ken, test_mse


# =============================================================================
# 4. VISUALISATION
# =============================================================================

PALETTE = {
    'CNN'  : '#1f77b4',
    'MLP'  : '#ff7f0e',
    'DTR'  : '#2ca02c',
    'SVR'  : '#d62728',
    'train': '#2196F3',
    'val'  : '#FF5722',
}


def _savefig(fig, name: str):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾  Saved → {path}")
    plt.close(fig)


# ── Figure 1: distribution of faults ─────────────────────────────────────────
def plot_fault_distribution(y_raw: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    counts  = y_raw.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values,
           color='#1f77b4', edgecolor='white', linewidth=0.4)
    ax.set_xlim(-0.5, min(30, len(counts)) - 0.5)
    ax.set_xlabel('Number of Faults', fontsize=12)
    ax.set_ylabel('Number of Classes', fontsize=12)
    ax.set_title('Distribution: Number of Classes vs Number of Faults', fontsize=13)
    ax.set_xticks(range(min(30, len(counts))))
    ax.set_xticklabels(counts.index[:30].astype(int), rotation=45, fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.tight_layout()
    _savefig(fig, '01_fault_distribution.png')


# ── Figure 2: correlation matrix ─────────────────────────────────────────────
def plot_correlation_matrix(X: pd.DataFrame, y: pd.Series):
    df_corr = X.copy()
    df_corr['bug'] = y.values
    corr = df_corr.corr()

    # Keep only |r| ≥ 0.45 (same threshold as paper)
    mask_arr = (np.abs(corr.values) < 0.45).copy()
    np.fill_diagonal(mask_arr, False)
    mask = pd.DataFrame(mask_arr, index=corr.index, columns=corr.columns)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlBu_r', center=0, linewidths=0.5,
                annot_kws={'size': 7}, ax=ax,
                cbar_kws={'shrink': 0.7})
    ax.set_title('Correlation Matrix – Strongly Correlated Features (|r| ≥ 0.45)',
                 fontsize=12)
    fig.tight_layout()
    _savefig(fig, '02_correlation_matrix.png')


# ── Figure 3 & 4: target distribution before/after SMOTEND ───────────────────
def plot_target_distributions(y_orig: np.ndarray, y_balanced: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, y, title in zip(
        axes,
        [y_orig, y_balanced],
        ['Original Data (imbalanced)', 'After SMOTEND (balanced)']
    ):
        ax.hist(y[y > 0], bins=40, color='#1f77b4',
                edgecolor='white', linewidth=0.4, density=True)
        ax.set_xlabel('Number of Faults', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(left=0)

    fig.suptitle('Fault Distribution Before and After SMOTEND', fontsize=13)
    fig.tight_layout()
    _savefig(fig, '03_target_distribution_comparison.png')


# ── Training / validation loss curves ────────────────────────────────────────
def plot_loss_curves(histories: dict, suffix: str = ''):
    """
    histories: {'CNN': history_obj, 'MLP': history_obj}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    titles = {'CNN': '(a) CNN', 'MLP': '(b) MLP'}

    for ax, (model_name, hist) in zip(axes, histories.items()):
        h = hist.history
        ep = range(1, len(h['loss']) + 1)
        ax.plot(ep, h['loss'],     label='Training Loss',
                color=PALETTE['train'], linewidth=1.8)
        ax.plot(ep, h['val_loss'], label='Validation Loss',
                color=PALETTE['val'],   linewidth=1.8, linestyle='--')
        ax.set_xlabel('Epochs', fontsize=11)
        ax.set_ylabel('MSE Loss', fontsize=11)
        ax.set_title(f'{titles[model_name]} {suffix}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Training & Validation Loss {suffix}', fontsize=13)
    fig.tight_layout()
    tag = suffix.replace(' ', '_').replace('(', '').replace(')', '')
    _savefig(fig, f'04_loss_curves_{tag}.png')


# ── Results comparison bar chart ──────────────────────────────────────────────
def plot_results_comparison(results: dict):
    """
    results: {
        'model_name': {'Exp1_Kendall': .., 'Exp1_MSE': ..,
                       'Exp2_Kendall': .., 'Exp2_MSE': ..}
    }
    """
    models    = list(results.keys())
    exp1_ken  = [results[m]['Exp1_Kendall'] for m in models]
    exp2_ken  = [results[m]['Exp2_Kendall'] for m in models]
    exp1_mse  = [results[m]['Exp1_MSE']     for m in models]
    exp2_mse  = [results[m]['Exp2_MSE']     for m in models]

    x   = np.arange(len(models))
    w   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Kendall
    ax = axes[0]
    bars1 = ax.bar(x - w/2, exp1_ken, w, label='Exp 1 – Without SMOTEND',
                   color='#90CAF9', edgecolor='#1565C0', linewidth=0.8)
    bars2 = ax.bar(x + w/2, exp2_ken, w, label='Exp 2 – With SMOTEND',
                   color='#1f77b4', edgecolor='#0D47A1', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Kendall's τ", fontsize=11)
    ax.set_title("Kendall Coefficient (higher = better)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    # MSE
    ax = axes[1]
    bars3 = ax.bar(x - w/2, exp1_mse, w, label='Exp 1 – Without SMOTEND',
                   color='#FFAB91', edgecolor='#BF360C', linewidth=0.8)
    bars4 = ax.bar(x + w/2, exp2_mse, w, label='Exp 2 – With SMOTEND',
                   color='#ff7f0e', edgecolor='#E65100', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Mean Squared Error (lower = better)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar in [*bars3, *bars4]:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Model Performance: Test Set – Experiment 1 vs Experiment 2',
                 fontsize=13)
    fig.tight_layout()
    _savefig(fig, '05_results_comparison_bar.png')


# ── Summary table as figure ───────────────────────────────────────────────────
def plot_summary_table(results_exp1: dict, results_exp2: dict):
    rows    = []
    for model, (r1, r2) in zip(results_exp1.keys(),
                                zip(results_exp1.values(),
                                    results_exp2.values())):
        rows.append([
            model,
            f"{r1['train_kendall']:.3f}", f"{r1['train_mse']:.3f}",
            f"{r1['test_kendall']:.3f}",  f"{r1['test_mse']:.3f}",
            f"{r2['train_kendall']:.3f}", f"{r2['train_mse']:.3f}",
            f"{r2['test_kendall']:.3f}",  f"{r2['test_mse']:.3f}",
        ])

    col_labels = [
        'Model',
        'Tr Ken\n(no SMO)', 'Tr MSE\n(no SMO)',
        'Te Ken\n(no SMO)', 'Te MSE\n(no SMO)',
        'Tr Ken\n(SMO)',    'Tr MSE\n(SMO)',
        'Te Ken\n(SMO)',    'Te MSE\n(SMO)',
    ]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    # Style header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#E3F2FD')

    fig.suptitle('Summary – Training & Test Metrics (Experiment 1 vs 2)',
                 fontsize=12, y=1.02)
    _savefig(fig, '06_summary_table.png')


# ── Predicted vs Actual scatter ───────────────────────────────────────────────
def plot_pred_vs_actual(model, X_test, y_test, model_name: str, suffix: str):
    y_pred = model.predict(split_to_groups(X_test), verbose=0).ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10,
               color=PALETTE.get(model_name, '#1f77b4'))
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Faults (log scale)', fontsize=11)
    ax.set_ylabel('Predicted Faults (log scale)', fontsize=11)
    ax.set_title(f'{model_name} – Predicted vs Actual {suffix}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    tag = f"{model_name}_{suffix.replace(' ','_')}"
    _savefig(fig, f'07_pred_vs_actual_{tag}.png')


# ── DTR scatter (Figure 11 in paper) ─────────────────────────────────────────
def plot_dtr_actual_vs_pred(y_test, y_pred):
    idx = np.arange(len(y_test))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(idx, y_test, s=4,  color='red',  alpha=0.5, label='Actual')
    ax.scatter(idx, y_pred, s=4,  color='blue', alpha=0.5, label='Predicted')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Faults (log scale)', fontsize=11)
    ax.set_title('DTR with SMOTEND – Actual vs Predicted', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, '08_dtr_actual_vs_predicted.png')


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def run():
    print("=" * 65)
    print("  SOFTWARE FAULT PREDICTION – Deep Learning (CNN & MLP)")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    df_raw = load_promise_data(DATA_DIR)

    print("\n[2/7] Preprocessing …")
    X_df, y_raw = preprocess(df_raw)
    X_arr = X_df.values.astype(np.float32)
    y_arr = y_raw.values.astype(np.float32)

    # ── EDA plots ─────────────────────────────────────────────────────────
    print("\n[3/7] Generating EDA plots …")
    plot_fault_distribution(y_raw)
    plot_correlation_matrix(X_df, y_raw)

    # ── EXPERIMENT 1 – Without SMOTEND ────────────────────────────────────
    print("\n[4/7] EXPERIMENT 1 – without SMOTEND …")
    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X_arr, y_arr, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=(y_arr > 0).astype(int)
    )
    X_tr1_sc, X_te1_sc, y_tr1_log, y_te1_log, _ = \
        log_transform_and_scale(X_tr1, X_te1, y_tr1, y_te1)

    cnn1 = build_cnn_model()
    mlp1 = build_mlp_model()

    hist_cnn1 = train_model(cnn1, X_tr1_sc, y_tr1_log,
                             X_te1_sc, y_te1_log, 'CNN (Exp 1)')
    hist_mlp1 = train_model(mlp1, X_tr1_sc, y_tr1_log,
                             X_te1_sc, y_te1_log, 'MLP (Exp 1)')

    cnn1_metrics = evaluate_model(cnn1, X_tr1_sc, y_tr1_log,
                                   X_te1_sc, y_te1_log)
    mlp1_metrics = evaluate_model(mlp1, X_tr1_sc, y_tr1_log,
                                   X_te1_sc, y_te1_log)

    # ML baselines – Exp 1
    dtr1 = DecisionTreeRegressor(random_state=RANDOM_STATE)
    svr1 = SVR()
    dtr1_m = train_ml_model(dtr1, X_tr1_sc, y_tr1_log,
                             X_te1_sc, y_te1_log, 'DTR (Exp 1)')
    try:
        svr1_m = train_ml_model(svr1, X_tr1_sc, y_tr1_log,
                                 X_te1_sc, y_te1_log, 'SVR (Exp 1)')
    except Exception:
        svr1_m = (np.nan, np.nan, np.nan, np.nan)
        print("  ⚠  SVR skipped (Exp 1) – dataset too large")

    plot_loss_curves({'CNN': hist_cnn1, 'MLP': hist_mlp1},
                     suffix='(Exp 1 – without SMOTEND)')

    # ── EXPERIMENT 2 – With SMOTEND ───────────────────────────────────────
    print("\n[5/7] EXPERIMENT 2 – with SMOTEND …")

    # Apply SMOTEND on full dataset, then split
    print("  Applying SMOTEND …")
    X_balanced, y_balanced = apply_smotend(X_arr, y_arr)
    plot_target_distributions(y_arr, y_balanced)

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_balanced, y_balanced, test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=(y_balanced > 0).astype(int)
    )
    X_tr2_sc, X_te2_sc, y_tr2_log, y_te2_log, _ = \
        log_transform_and_scale(X_tr2, X_te2, y_tr2, y_te2)

    cnn2 = build_cnn_model()
    mlp2 = build_mlp_model()

    hist_cnn2 = train_model(cnn2, X_tr2_sc, y_tr2_log,
                             X_te2_sc, y_te2_log, 'CNN (Exp 2)')
    hist_mlp2 = train_model(mlp2, X_tr2_sc, y_tr2_log,
                             X_te2_sc, y_te2_log, 'MLP (Exp 2)')

    cnn2_metrics = evaluate_model(cnn2, X_tr2_sc, y_tr2_log,
                                   X_te2_sc, y_te2_log)
    mlp2_metrics = evaluate_model(mlp2, X_tr2_sc, y_tr2_log,
                                   X_te2_sc, y_te2_log)

    # ML baselines – Exp 2
    dtr2 = DecisionTreeRegressor(random_state=RANDOM_STATE)
    svr2 = SVR()
    dtr2_m = train_ml_model(dtr2, X_tr2_sc, y_tr2_log,
                             X_te2_sc, y_te2_log, 'DTR (Exp 2)')
    try:
        svr2_m = train_ml_model(svr2, X_tr2_sc, y_tr2_log,
                                 X_te2_sc, y_te2_log, 'SVR (Exp 2)')
    except Exception:
        svr2_m = (np.nan, np.nan, np.nan, np.nan)
        print("  ⚠  SVR skipped (Exp 2) – dataset too large")

    plot_loss_curves({'CNN': hist_cnn2, 'MLP': hist_mlp2},
                     suffix='(Exp 2 – with SMOTEND)')

    # ── Scatter plots ─────────────────────────────────────────────────────
    print("\n[6/7] Generating result plots …")
    plot_pred_vs_actual(mlp2, X_te2_sc, y_te2_log, 'MLP', 'with SMOTEND')
    plot_pred_vs_actual(cnn2, X_te2_sc, y_te2_log, 'CNN', 'with SMOTEND')
    plot_dtr_actual_vs_pred(y_te2_log, dtr2.predict(X_te2_sc))

    # ── Collect results ───────────────────────────────────────────────────
    print("\n[7/7] Saving summary …")

    def _metrics_dict(m):
        return {
            'train_kendall': m[0], 'train_mse': m[1],
            'test_kendall' : m[2], 'test_mse' : m[3],
        }

    results_exp1 = {
        'CNN': _metrics_dict(cnn1_metrics),
        'MLP': _metrics_dict(mlp1_metrics),
        'DTR': _metrics_dict(dtr1_m),
        'SVR': _metrics_dict(svr1_m),
    }
    results_exp2 = {
        'CNN': _metrics_dict(cnn2_metrics),
        'MLP': _metrics_dict(mlp2_metrics),
        'DTR': _metrics_dict(dtr2_m),
        'SVR': _metrics_dict(svr2_m),
    }

    # Bar chart comparison
    bar_data = {}
    for model in ['CNN', 'MLP', 'DTR', 'SVR']:
        bar_data[model] = {
            'Exp1_Kendall': results_exp1[model]['test_kendall'],
            'Exp1_MSE'    : results_exp1[model]['test_mse'],
            'Exp2_Kendall': results_exp2[model]['test_kendall'],
            'Exp2_MSE'    : results_exp2[model]['test_mse'],
        }
    plot_results_comparison(bar_data)
    plot_summary_table(results_exp1, results_exp2)

    # ── Print console table ────────────────────────────────────────────────
    header = f"\n{'Model':<6} {'Exp':^4} {'Split':<6} {'Kendall':>8} {'MSE':>8}"
    print(header)
    print('-' * len(header))
    for exp_num, (r_exp, label) in enumerate(
        [(results_exp1, 'no SMO'), (results_exp2, 'SMO   ')], start=1
    ):
        for model_name, metrics in r_exp.items():
            for split in ['train', 'test']:
                k = metrics[f'{split}_kendall']
                m = metrics[f'{split}_mse']
                k_str = f'{k:.3f}' if not (isinstance(k, float) and np.isnan(k)) else '  -  '
                m_str = f'{m:.3f}' if not (isinstance(m, float) and np.isnan(m)) else '  -  '
                print(f'{model_name:<6} {exp_num:^4} {split:<6} {k_str:>8} {m_str:>8}')

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_rows = []
    for exp_num, r_exp in [(1, results_exp1), (2, results_exp2)]:
        for model_name, metrics in r_exp.items():
            for split in ['train', 'test']:
                csv_rows.append({
                    'Experiment': exp_num,
                    'Model'     : model_name,
                    'Split'     : split,
                    'Kendall'   : metrics[f'{split}_kendall'],
                    'MSE'       : metrics[f'{split}_mse'],
                })
    csv_path = os.path.join(RESULTS_DIR, 'results_summary.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\n  💾  Results CSV saved → {csv_path}")
    print("\n✅  Pipeline complete.  All outputs → " + RESULTS_DIR)


# =============================================================================
if __name__ == '__main__':
    run()