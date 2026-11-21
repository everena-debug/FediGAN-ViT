# FediGAN-ViT engine — written by the notebook; do not edit here, edit the notebook settings instead.
# Imported by the notebook (%run) to define everything, and launched as background GPU workers:
#   FEDIGAN_ROLE=prep                          build cache, pre-train all client cGANs, download backbones
#   FEDIGAN_ROLE=shard FEDIGAN_SHARD=i FEDIGAN_NSHARDS=n   run experiments i, i+n, ... on one GPU
import os, matplotlib
if os.environ.get("FEDIGAN_ROLE"):
    matplotlib.use("Agg")
# ============================== CORE IMPORTS ==============================
import os, re, gc, json, time, math, copy, glob, random, hashlib, shutil, warnings
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score, matthews_corrcoef, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

warnings.filterwarnings("ignore")
NOTEBOOK_START = float(os.environ.get("FEDIGAN_T0", time.time()))   # session start, shared with GPU workers
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ============================== CONFIGURATION ==============================
PROFILE = os.environ.get("FEDIGAN_PROFILE", "paper")   # "smoke" = every code path, 1 seed, 1 round; "paper" = full study

# ---- paths ----
WORK = Path(os.environ.get("FEDIGAN_WORK", "/kaggle/working" if os.path.isdir("/kaggle/working") else "./work"))
OUT = WORK / "fedigan_outputs"
SPEED = os.environ.get("FEDIGAN_SPEED", "fast")   # fast | standard | full  (set in the notebook's Settings cell)
RUN_TAG = f"{PROFILE}_{SPEED}"                     # results of different model sizes are never mixed
RUNS_DIR = OUT / f"runs_{RUN_TAG}"
for d in ["tables", "figures", "cache", "synthetic", "gan", "models", f"runs_{RUN_TAG}"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ---- data ----
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
SHORT = ["Cataract", "DR", "Glaucoma", "Normal"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE, GAN_SIZE = 224, 64
SPLIT_SEED = 42            # fixed test/val split for every run (comparability)
TEST_FRAC, VAL_FRAC = 0.15, 0.15
GROUP_SPLIT = True         # keep both eyes of a patient / near-duplicates in the same split and client

# ---- federated setup ----
NUM_CLIENTS = 3
ALPHA = 0.5                # Dirichlet concentration (non-IID)
MIN_CLIENT_FRAC = 0.10     # resample partition until every client holds >=10% of training data
AGG_WEIGHT = "real"        # FedAvg weights = number of REAL images per client (synthetic does not inflate weight)

# ---- classifier ----
BACKBONE = "vit_small_patch16_224" if SPEED == "fast" else "vit_base_patch16_224"   # ViT-S/16 or ViT-B/16 (IN-21k pretrained)
PRETRAINED = os.environ.get("FEDIGAN_PRETRAINED", "1") == "1"
BATCH = 32
LR = {"vit": 1e-4, "cnn": 3e-4}
WEIGHT_DECAY = 0.05
DROP_RATE = 0.3
ROUNDS, LOCAL_EPOCHS = (15 if SPEED == "fast" else 20), 1
PATIENCE, MIN_ROUNDS = (4 if SPEED == "fast" else 5), 5   # early stopping on validation macro-AUC
MU = 0.01                            # FedProx proximal coefficient
FOCAL_GAMMA, CB_BETA = 2.0, 0.9999   # class-balanced focal loss
GRAD_CLIP = 1.0

# ---- cGAN ----
LATENT, EMBED = 100, 50
GAN_EPOCHS, GAN_BATCH, GAN_LR, GAN_BETA1 = 100, 32, 2e-4, 0.5
SYNTH_CAP = 100            # max synthetic images per class per client (controls real:synthetic mixing ratio λ)
MIN_REAL_FOR_SYNTH = 1     # never synthesise a class the client has no real images of
FEW_SHOT_WARN = 20         # flag classes with fewer real images than this (GAN unreliable)

# ---- robustness / statistics ----
PARTICIPATION = 0.67       # partial-participation run: each client joins a round with this probability
SEEDS_MAIN = [42, 43, 44, 45, 46]   # proposed model
SEEDS_OTHER = [42, 43, 44]          # every other configuration (paired with proposed on these seeds)
SENS_ALPHAS = [0.1, 1.0]
SENS_MUS = [0.001, 0.1]
BOOTSTRAP_B = 1000
TIME_BUDGET_H = 10.8       # stop launching runs after this (Kaggle hard limit is 12 h)
NUM_WORKERS = int(os.environ.get("FEDIGAN_WORKERS", min(4, os.cpu_count() or 2)))
PLAN_MODE = {"fast": "minimal", "standard": "reduced", "full": "full"}.get(SPEED, "reduced")   # 19 / 32 / 56 experiments

if PROFILE == "smoke":
    ROUNDS, MIN_ROUNDS, PATIENCE = 1, 1, 1
    GAN_EPOCHS = 2
    SEEDS_MAIN, SEEDS_OTHER = [42], [42]
    BOOTSTRAP_B = 100
    TIME_BUDGET_H = 2.0
TIME_BUDGET_H = float(os.environ.get("FEDIGAN_BUDGET_H", TIME_BUDGET_H))
if PLAN_MODE in ("reduced", "minimal") and PROFILE != "smoke":
    SEEDS_MAIN = SEEDS_OTHER = [42, 43, 44]

print(f"PROFILE={PROFILE} | SPEED={SPEED} ({BACKBONE}, {PLAN_MODE} plan) | rounds={ROUNDS} | seeds main={SEEDS_MAIN} other={SEEDS_OTHER} | output={OUT}")
_SHARD = os.environ.get("FEDIGAN_SHARD")
_TAG = "" if _SHARD is None else f"_shard{_SHARD}"
TBL = OUT / f"tables{_TAG}"; FIG = OUT / f"figures{_TAG}"
TBL.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

# ============================== TORCH SETUP ==============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
USE_AMP = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

print("torch", torch.__version__, "| timm", timm.__version__, "| device", DEVICE, "| GPUs", N_GPU,
      [torch.cuda.get_device_name(i) for i in range(N_GPU)])

def find_data_root():
    env = os.environ.get("FEDIGAN_DATA")
    if env:
        return env
    for c in sorted(glob.glob("/kaggle/input/**/cataract", recursive=True)):
        p = os.path.dirname(c)
        if all(os.path.isdir(os.path.join(p, k)) for k in CLASS_NAMES):
            return p
    raise FileNotFoundError("EDC dataset not found under /kaggle/input — add 'eye-diseases-classification'.")

def crop_to_fundus(bgr, thr=10):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g > thr)
    if len(xs) < 100:
        return bgr
    return bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def preprocess_fundus(bgr):
    """Illumination correction -> non-local-means denoising -> CLAHE (LAB L-channel) -> background mask."""
    mask = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    blur = cv2.GaussianBlur(bgr, (0, 0), bgr.shape[0] / 30)
    ill = cv2.addWeighted(bgr, 4, blur, -4, 128)
    den = cv2.fastNlMeansDenoisingColored(ill, None, 5, 5, 7, 21)
    l, a, b = cv2.split(cv2.cvtColor(den, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return (out * mask[..., None]).astype(np.uint8)

def dhash(bgr, size=8):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)

def _load_one(path):
    cv2.setNumThreads(0)
    with open(path, "rb") as f:
        data = f.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    arr = crop_to_fundus(arr)
    raw = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    proc = preprocess_fundus(raw)
    g64 = cv2.resize(proc, (GAN_SIZE, GAN_SIZE), interpolation=cv2.INTER_AREA)
    return (hashlib.md5(data).hexdigest(), dhash(raw),
            np.ascontiguousarray(raw[..., ::-1]), np.ascontiguousarray(proc[..., ::-1]),
            np.ascontiguousarray(g64[..., ::-1]))

DATA_ROOT = find_data_root()
CACHE = OUT / "cache"
cache_files = [CACHE / f for f in ["meta.csv", "raw.npy", "proc.npy", "g64.npy"]]
prev_cache = [Path(p).parent for p in glob.glob("/kaggle/input/**/fedigan_outputs/cache/meta.csv", recursive=True)]

if not all(f.exists() for f in cache_files) and prev_cache:
    for f in ["meta.csv", "raw.npy", "proc.npy", "g64.npy"]:
        shutil.copy(prev_cache[0] / f, CACHE / f)

if all(f.exists() for f in cache_files):
    META = pd.read_csv(CACHE / "meta.csv")
    RAW, PROC, G64 = (np.load(CACHE / f) for f in ["raw.npy", "proc.npy", "g64.npy"])
    print("Loaded preprocessing cache:", RAW.shape)
else:
    recs = []
    for c in CLASS_NAMES:
        for fn in sorted(os.listdir(os.path.join(DATA_ROOT, c))):
            recs.append((os.path.join(DATA_ROOT, c, fn), fn, c, CLASS_NAMES.index(c)))
    t0 = time.time()
    with Pool(NUM_WORKERS) as pool:
        outs = pool.map(_load_one, [r[0] for r in recs], chunksize=16)
    keep = [i for i, o in enumerate(outs) if o is not None]
    META = pd.DataFrame([dict(path=recs[i][0], file=recs[i][1], label_name=recs[i][2], label=recs[i][3],
                              md5=outs[i][0], dhash=outs[i][1]) for i in keep])
    RAW = np.stack([outs[i][2] for i in keep]); PROC = np.stack([outs[i][3] for i in keep])
    G64 = np.stack([outs[i][4] for i in keep])
    del outs; gc.collect()
    META.to_csv(CACHE / "meta.csv", index=False)
    for name, arr in [("raw", RAW), ("proc", PROC), ("g64", G64)]:
        np.save(CACHE / f"{name}.npy", arr)
    print(f"Preprocessed {len(META)} images in {time.time() - t0:.0f}s ({len(recs) - len(keep)} unreadable)")

META["arr_idx"] = np.arange(len(META))
LABELS_ALL = META["label"].to_numpy()
print(META["label_name"].value_counts().reindex(CLASS_NAMES))

# ---- exact duplicates (md5) ----
dup_report = []
drop = set()
for md5, g in META.groupby("md5"):
    if len(g) > 1:
        conflict = g["label"].nunique() > 1
        dup_report.append(dict(md5=md5, n=len(g), labels="|".join(sorted(g["label_name"].unique())), conflict=conflict))
        drop.update(g.index if conflict else g.index[1:])
DUP_DF = pd.DataFrame(dup_report)
DF = META.drop(index=sorted(drop)).reset_index(drop=True)

# ---- grouping: patient id (both eyes) + perceptual near-duplicates ----
parent = {}
def _find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def _union(a, b):
    ra, rb = _find(a), _find(b)
    if ra != rb: parent[rb] = ra

pid_re = re.compile(r"^(\d+)_(left|right)\.(jpe?g|png)$", re.I)
for i, fn in enumerate(DF["file"]):
    node = f"img:{i}"; _find(node)
    m = pid_re.match(fn)
    if GROUP_SPLIT and m:
        _union(node, f"pid:{m.group(3).lower()}:{m.group(1)}")   # .jpg (ODIR) and .jpeg (EyePACS) ids are separate namespaces
if GROUP_SPLIT:
    for h, g in DF.groupby("dhash"):
        if 1 < len(g) <= 5:
            for j in g.index[1:]:
                _union(f"img:{g.index[0]}", f"img:{j}")
DF["group"] = [_find(f"img:{i}") for i in range(len(DF))]
n_near = int(sum(1 for _, g in DF.groupby("dhash") if 1 < len(g) <= 5))
print(f"Exact duplicate sets: {len(DUP_DF)} (label conflicts: {int(DUP_DF['conflict'].sum()) if len(DUP_DF) else 0}); "
      f"dropped {len(drop)} images | near-duplicate hash sets: {n_near} | groups: {DF['group'].nunique()} for {len(DF)} images")

def stratified_group_split(df, frac, seed):
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    a, b = next(sgkf.split(df, df["label"], df["group"]))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)

_trval, TEST_DF = stratified_group_split(DF, TEST_FRAC, SPLIT_SEED)
TRAIN_DF, VAL_DF = stratified_group_split(_trval, VAL_FRAC / (1 - TEST_FRAC), SPLIT_SEED)
assert not (set(TRAIN_DF.group) & set(TEST_DF.group)) and not (set(VAL_DF.group) & set(TEST_DF.group))
TRAIN_IDX, VAL_IDX, TEST_IDX = (d["arr_idx"].to_numpy() for d in (TRAIN_DF, VAL_DF, TEST_DF))

t01 = pd.DataFrame({
    "Class": SHORT,
    "Images (raw)": META["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values,
    "After de-dup": DF["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values,
    "Train": TRAIN_DF["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values,
    "Val": VAL_DF["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values,
    "Test": TEST_DF["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values})
t01.loc[len(t01)] = ["Total"] + t01.iloc[:, 1:].sum().tolist()
t01.to_csv(TBL / "T01_dataset.csv", index=False)
if len(DUP_DF): DUP_DF.to_csv(TBL / "T01b_duplicates.csv", index=False)
print(t01.to_string(index=False))

_PARTITIONS = {}
def get_partition(alpha):
    """Dirichlet split of TRAIN groups over clients; all images of a patient go to the same client."""
    if alpha in _PARTITIONS:
        return _PARTITIONS[alpha]
    gl = TRAIN_DF.groupby("group")["label"].agg(lambda s: s.value_counts().idxmax())
    for t in range(500):
        rng = np.random.default_rng(SPLIT_SEED + t)
        assign = {}
        for c in range(NUM_CLASSES):
            groups = gl[gl == c].index.to_numpy().copy(); rng.shuffle(groups)
            cuts = (np.cumsum(rng.dirichlet([alpha] * NUM_CLIENTS)) * len(groups)).astype(int)[:-1]
            for k, part in enumerate(np.split(groups, cuts)):
                for g in part: assign[g] = k
        client = TRAIN_DF["group"].map(assign).to_numpy()
        sizes = np.bincount(client, minlength=NUM_CLIENTS)
        if sizes.min() >= MIN_CLIENT_FRAC * len(TRAIN_DF):
            _PARTITIONS[alpha] = {k: TRAIN_IDX[client == k] for k in range(NUM_CLIENTS)}
            return _PARTITIONS[alpha]
    raise RuntimeError("Could not find a partition satisfying MIN_CLIENT_FRAC")

def synth_plan(counts):
    m = counts.max()
    return np.array([min(m - n, SYNTH_CAP) if n >= MIN_REAL_FOR_SYNTH else 0 for n in counts], dtype=int)

def partition_table(alpha):
    rows = []
    for k, idx in get_partition(alpha).items():
        cnt = np.bincount(LABELS_ALL[idx], minlength=NUM_CLASSES); plan = synth_plan(cnt)
        for c in range(NUM_CLASSES):
            rows.append(dict(alpha=alpha, client=k, cls=SHORT[c], real=int(cnt[c]), synthetic=int(plan[c]),
                             after=int(cnt[c] + plan[c]), few_shot_flag=bool(0 < cnt[c] < FEW_SHOT_WARN)))
    return pd.DataFrame(rows)

T02 = partition_table(ALPHA)
T02.to_csv(TBL / "T02_partition.csv", index=False)
print(T02.pivot(index="cls", columns="client", values=["real", "after"]).reindex(SHORT))
if T02.few_shot_flag.any():
    print("WARNING — classes with very few real images at a client (GAN output unreliable):\n",
          T02[T02.few_shot_flag][["client", "cls", "real"]].to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
w = 0.8 / NUM_CLIENTS
for k in range(NUM_CLIENTS):
    d = T02[T02.client == k]
    ax[0].bar(np.arange(NUM_CLASSES) + k * w, d.real, w, label=f"Client {k}")
    ax[1].bar(np.arange(NUM_CLASSES) + k * w, d.real, w, color=f"C{k}")
    ax[1].bar(np.arange(NUM_CLASSES) + k * w, d.synthetic, w, bottom=d.real, color=f"C{k}", alpha=0.35, hatch="//")
for a, t in zip(ax, [f"Real images per client (Dirichlet α={ALPHA})", "After cGAN augmentation (hatched = synthetic)"]):
    a.set_xticks(np.arange(NUM_CLASSES) + w * (NUM_CLIENTS - 1) / 2); a.set_xticklabels(SHORT); a.set_title(t); a.set_ylabel("Images")
top = (T02.real + T02.synthetic).max() * 1.1
for a in ax: a.set_ylim(0, top)
ax[0].legend(); plt.tight_layout(); plt.savefig(FIG / "F04_partition_and_augmentation.png", bbox_inches="tight"); plt.show()

def expected_calibration_error(y, prob, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)

def compute_metrics(y, prob):
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    pred = prob.argmax(1); n = len(y)
    present = sorted(set(y.tolist()))
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=present, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, labels=present, average="weighted", zero_division=0)
    out = dict(n=n, accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
               precision_macro=p, recall_macro=r, f1_macro=f, precision_weighted=pw, recall_weighted=rw, f1_weighted=fw,
               kappa=cohen_kappa_score(y, pred) if len(present) > 1 else float("nan"),
               mcc=matthews_corrcoef(y, pred) if len(present) > 1 else float("nan"),
               ece=expected_calibration_error(y, prob))
    per = {}
    aucs = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = n - tp - fn - fp
        yb = (y == c).astype(int)
        auc_c = roc_auc_score(yb, prob[:, c]) if 0 < yb.sum() < n else float("nan")
        if not np.isnan(auc_c): aucs.append(auc_c)
        prec = tp / (tp + fp) if tp + fp else float("nan"); sens = tp / (tp + fn) if tp + fn else float("nan")
        per[SHORT[c]] = dict(support=int(tp + fn), precision=prec, recall_sensitivity=sens,
                             specificity=tn / (tn + fp) if tn + fp else float("nan"),
                             f1=2 * prec * sens / (prec + sens) if (tp and prec + sens) else 0.0,
                             ovr_accuracy=(tp + tn) / n, auc=auc_c,
                             ap=average_precision_score(yb, prob[:, c]) if 0 < yb.sum() < n else float("nan"))
    out["auc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    out = {k: float(v) for k, v in out.items()}
    out["per_class"] = per; out["cm"] = cm.tolist()
    return out

SCALARS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
           "precision_weighted", "recall_weighted", "f1_weighted", "auc_macro", "kappa", "mcc", "ece"]

def bootstrap_ci(y, prob, metrics=("accuracy", "f1_macro", "auc_macro", "kappa"), B=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); prob = np.asarray(prob); vals = defaultdict(list)
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < NUM_CLASSES: continue
        m = compute_metrics(y[i], prob[i])
        for k in metrics: vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}

def mean_sd_ci(v):
    v = np.asarray([x for x in v if not np.isnan(x)], float)
    if len(v) == 0: return (np.nan, np.nan, np.nan, np.nan, 0)
    if len(v) == 1: return (v[0], np.nan, np.nan, np.nan, 1)
    m, s = v.mean(), v.std(ddof=1); h = stats.t.ppf(0.975, len(v) - 1) * s / np.sqrt(len(v))
    return (m, s, m - h, m + h, len(v))

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(np.where(np.isnan(p), np.inf, p)); adj = np.full_like(p, np.nan)
    m = int((~np.isnan(p)).sum()); running = 0.0
    for rank, i in enumerate(order):
        if np.isnan(p[i]): continue
        running = max(running, min(1.0, (m - rank) * p[i])); adj[i] = running
    return adj

def mcnemar_exact(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(p)

def fmt(m, s, pct=True, d=2):
    if np.isnan(m): return "—"
    k = 100 if pct else 1
    return f"{m * k:.{d}f}" + ("" if np.isnan(s) else f" ± {s * k:.{d}f}")

class ArrayDS(Dataset):
    def __init__(self, src, idx, labels, tf):
        self.src, self.idx, self.labels, self.tf = src, np.asarray(idx), np.asarray(labels), tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(self.src[self.idx[i]])).permute(2, 0, 1))
        return self.tf(x), int(self.labels[i])

def make_tfs(mean, std):
    train = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.15, 0.15, 0.15),
                       T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std), T.ToPureTensor()])
    test = T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std), T.ToPureTensor()])
    return train, test

def train_loader(parts, tf, seed):
    ds = ConcatDataset([ArrayDS(src, idx, lab, tf) for src, idx, lab in parts if len(idx)])
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=len(ds) > BATCH, num_workers=NUM_WORKERS,
                      pin_memory=True, generator=g, persistent_workers=NUM_WORKERS > 0)

def eval_loader(idx, tf):
    return DataLoader(ArrayDS(PROC, idx, LABELS_ALL[idx], tf), batch_size=64, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)

_INIT = {}
def new_model(backbone, seed=None):
    if backbone not in _INIT:
        m = timm.create_model(backbone, pretrained=PRETRAINED, num_classes=NUM_CLASSES, drop_rate=DROP_RATE)
        _INIT[backbone] = ({k: v.detach().clone() for k, v in m.state_dict().items()},
                           timm.data.resolve_model_data_config(m))
        del m
    m = timm.create_model(backbone, pretrained=False, num_classes=NUM_CLASSES, drop_rate=DROP_RATE)
    m.load_state_dict(_INIT[backbone][0])
    if seed is not None:
        torch.manual_seed(seed); m.reset_classifier(NUM_CLASSES)   # seed-dependent head initialisation
    return m.to(DEVICE)

def lr_for(backbone): return LR["vit"] if "vit" in backbone else LR["cnn"]

class ClassBalancedFocalLoss(nn.Module):
    """Cui et al. (2019) class-balanced weights + focal term (Lin et al., 2017). Weights from LOCAL counts."""
    def __init__(self, counts, beta=CB_BETA, gamma=FOCAL_GAMMA):
        super().__init__()
        counts = np.asarray(counts, float); present = counts > 0
        w = np.zeros_like(counts)
        w[present] = (1 - beta) / (1 - np.power(beta, counts[present]))
        w[present] = w[present] / w[present].sum() * present.sum()
        w[~present] = 1.0
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32)); self.gamma = gamma
    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
        return (self.w[y] * (1 - logp.exp()) ** self.gamma * (-logp)).mean()

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=LATENT, num_classes=NUM_CLASSES, embed_dim=EMBED):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(nn.Linear(latent_dim + embed_dim, 256 * 8 * 8), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),   # 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),     # 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),      # 64
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())
    def forward(self, z, y):
        return self.net(self.fc(torch.cat([z, self.label_emb(y)], 1)).view(-1, 256, 8, 8))

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=EMBED):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.class_proj = nn.Linear(embed_dim, 64)
        self.final = nn.Sequential(nn.Conv2d(257, 128, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Flatten(),
                                   nn.Linear(128 * 8 * 8, 1), nn.Sigmoid())
    def forward(self, x, y):
        return self.final(torch.cat([self.net(x), self.class_proj(self.label_emb(y)).view(-1, 1, 8, 8)], 1))

def train_cgan(x64, labels, seed, epochs=GAN_EPOCHS):
    set_seed(seed)
    X = torch.from_numpy(x64).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEVICE)
    Y = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)
    G, D = ConditionalGenerator().to(DEVICE), ConditionalDiscriminator().to(DEVICE)
    oG = torch.optim.Adam(G.parameters(), GAN_LR, betas=(GAN_BETA1, 0.999))
    oD = torch.optim.Adam(D.parameters(), GAN_LR, betas=(GAN_BETA1, 0.999))
    bce = nn.BCELoss(); n = len(X); bs = min(GAN_BATCH, n); hist = {"g": [], "d": []}
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE); gs, ds_, nb = 0.0, 0.0, 0
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]; real, y = X[b], Y[b]
            flip = torch.rand(bs, device=DEVICE) < 0.5
            real = torch.where(flip[:, None, None, None], real.flip(3), real)
            fake = G(torch.randn(bs, LATENT, device=DEVICE), y)
            dl = 0.5 * (bce(D(real, y), torch.full((bs, 1), 0.9, device=DEVICE)) +
                        bce(D(fake.detach(), y), torch.zeros(bs, 1, device=DEVICE)))
            oD.zero_grad(); dl.backward(); oD.step()
            gl = bce(D(fake, y), torch.ones(bs, 1, device=DEVICE))
            oG.zero_grad(); gl.backward(); oG.step()
            gs += gl.item(); ds_ += dl.item(); nb += 1
        hist["g"].append(gs / max(nb, 1)); hist["d"].append(ds_ / max(nb, 1))
    return G.eval(), hist

@torch.no_grad()
def generate(G, plan, seed):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    ys = np.concatenate([np.full(n, c) for c, n in enumerate(plan) if n > 0]).astype(np.int64) if plan.sum() else np.zeros(0, np.int64)
    imgs = []
    for i in range(0, len(ys), 128):
        y = torch.from_numpy(ys[i:i + 128]).to(DEVICE)
        x = G(torch.randn(len(y), LATENT, device=DEVICE, generator=g), y)
        imgs.append(((x * 0.5 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return (np.concatenate(imgs) if imgs else np.zeros((0, GAN_SIZE, GAN_SIZE, 3), np.uint8)), ys

_SYN = {}
def get_synthetic(seed, alpha=ALPHA):
    """Per-client cGAN trained only on that client's real images. Returns {k: (imgs224, labels, imgs64)}."""
    key = (seed, alpha)
    if key in _SYN: return _SYN[key]
    path = OUT / "synthetic" / f"syn_a{alpha}_s{seed}_{PROFILE}.npz"
    out = {}
    if path.exists():
        z = np.load(path)
        for k in range(NUM_CLIENTS): out[k] = (z[f"x224_{k}"], z[f"y_{k}"], z[f"x64_{k}"])
    else:
        save, hists = {}, {}
        for k, idx in get_partition(alpha).items():
            lab = LABELS_ALL[idx]
            G, hist = train_cgan(G64[idx], lab, seed * 10 + k)
            torch.save(G.state_dict(), OUT / "gan" / f"G_a{alpha}_s{seed}_c{k}_{PROFILE}.pt")
            x64, ys = generate(G, synth_plan(np.bincount(lab, minlength=NUM_CLASSES)), seed * 10 + k)
            x224 = np.stack([cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC) for im in x64]) \
                if len(x64) else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
            out[k] = (x224, ys, x64); hists[k] = hist
            save.update({f"x224_{k}": x224, f"y_{k}": ys, f"x64_{k}": x64})
            del G; torch.cuda.empty_cache()
        np.savez_compressed(path, **save)
        json.dump(hists, open(OUT / "gan" / f"loss_a{alpha}_s{seed}_{PROFILE}.json", "w"))
    _SYN[key] = out
    return out

def client_units(seed, aug, alpha=ALPHA):
    part = get_partition(alpha)
    syn = get_synthetic(seed, alpha) if aug == "gan" else None
    units = []
    for k in range(NUM_CLIENTS):
        idx = part[k]; lab = LABELS_ALL[idx]; cnt = np.bincount(lab, minlength=NUM_CLASSES)
        plan = synth_plan(cnt); parts = [(PROC, idx, lab)]
        if aug == "gan":
            x224, ys, _ = syn[k]
            parts.append((x224, np.arange(len(ys)), ys)); cnt = cnt + np.bincount(ys, minlength=NUM_CLASSES)
        elif aug == "oversample":
            rng = np.random.default_rng(seed * 100 + k)
            extra = np.concatenate([rng.choice(idx[lab == c], n, replace=True) for c, n in enumerate(plan) if n > 0] or [np.zeros(0, int)])
            parts.append((PROC, extra, LABELS_ALL[extra])); cnt = cnt + plan
        units.append(dict(client=k, parts=parts, counts=cnt, n_real=len(idx), real_idx=idx))
    return units

def merge_units(units):
    return dict(client="all", parts=[p for u in units for p in u["parts"]], counts=sum(u["counts"] for u in units),
                n_real=sum(u["n_real"] for u in units), real_idx=np.concatenate([u["real_idx"] for u in units]))

def _dp(model): return nn.DataParallel(model) if N_GPU > 1 else model

def train_local(model, loader, crit, lr, epochs, prox_ref=None, mu=0.0):
    net = _dp(model); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    tot, n, correct = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                out = net(x)
            loss = crit(out.float(), y)
            if prox_ref is not None and mu > 0:
                loss = loss + 0.5 * mu * sum(((p - q) ** 2).sum() for p, q in zip(params, prox_ref))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y); correct += (out.argmax(1) == y).sum().item()
    return tot / max(n, 1), correct / max(n, 1)

@torch.no_grad()
def predict(model, loader):
    net = _dp(model); model.eval(); P, Y = [], []
    for x, y in loader:
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            out = net(x.to(DEVICE, non_blocking=True))
        P.append(F.softmax(out.float(), 1).cpu()); Y.append(y)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()

def nll(y, p): return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean())

def aggregate(states, weights):
    w = np.asarray(weights, float); w = w / w.sum(); new = {}
    for k, v0 in states[0].items():
        new[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states)).to(v0.dtype) if v0.is_floating_point() else v0.clone()
    return new

def fl_train(units, backbone, seed, agg, mu, participation, tf_train, val_ld, test_ld, tag=""):
    set_seed(seed)
    gm = new_model(backbone, seed); local = new_model(backbone)
    lr0 = lr_for(backbone); rng = np.random.default_rng(seed)
    crits = [ClassBalancedFocalLoss(u["counts"]).to(DEVICE) for u in units]
    loaders = [train_loader(u["parts"], tf_train, seed * 1000 + i) for i, u in enumerate(units)]
    n_params = sum(p.numel() for p in gm.parameters()); comm = 0.0
    hist, best, best_state, best_round, bad = [], -1.0, None, 0, 0
    for r in range(1, ROUNDS + 1):
        lr = lr0 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (r - 1) / max(ROUNDS, 1))))
        chosen = [i for i in range(len(units)) if rng.random() < participation] if participation < 1 else list(range(len(units)))
        if not chosen: chosen = [int(rng.integers(len(units)))]
        gstate = {k: v.detach().clone() for k, v in gm.state_dict().items()}
        prox_ref = [p.detach().clone() for p in gm.parameters() if p.requires_grad] if agg == "fedprox" else None
        states, weights, tl, ta = [], [], [], []
        for i in chosen:
            local.load_state_dict(gstate)
            l, a = train_local(local, loaders[i], crits[i], lr, LOCAL_EPOCHS, prox_ref, mu)
            states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
            weights.append(units[i]["n_real"] if AGG_WEIGHT == "real" else len(loaders[i].dataset))
            tl.append(l); ta.append(a)
        gm.load_state_dict(aggregate(states, weights) if len(states) > 1 else states[0])
        comm += 2 * len(chosen) * n_params * 4 / 1e6
        del states; torch.cuda.empty_cache()
        yv, pv = predict(gm, val_ld); yt, pt = predict(gm, test_ld)
        mv, mt = compute_metrics(yv, pv), compute_metrics(yt, pt)
        hist.append(dict(round=r, lr=lr, participants=len(chosen), train_loss=float(np.mean(tl)), train_acc=float(np.mean(ta)),
                         val_loss=nll(yv, pv), val_acc=mv["accuracy"], val_f1=mv["f1_macro"], val_auc=mv["auc_macro"],
                         test_loss=nll(yt, pt), test_acc=mt["accuracy"], test_f1=mt["f1_macro"], test_auc=mt["auc_macro"]))
        print(f"  {tag} r{r:02d} | clients {chosen} | train loss {hist[-1]['train_loss']:.3f} acc {hist[-1]['train_acc']:.3f}"
              f" | val acc {mv['accuracy']:.4f} auc {mv['auc_macro']:.4f} | test acc {mt['accuracy']:.4f}")
        score = float(np.nan_to_num(mv["auc_macro"], nan=0.0)) + 1e-3 * mv["accuracy"]
        if score > best:
            best, best_round, bad = score, r, 0
            best_state = {k: v.detach().cpu().clone() for k, v in gm.state_dict().items()}
        else:
            bad += 1
            if r >= MIN_ROUNDS and bad >= PATIENCE: break
    gm.load_state_dict(best_state); del local; torch.cuda.empty_cache()
    return gm, hist, best_round, comm

def run_experiment(exp, seed):
    t0 = time.time(); backbone = exp.get("backbone", BACKBONE); alpha = exp.get("alpha", ALPHA)
    mu = exp.get("mu", MU) if exp["agg"] == "fedprox" else 0.0
    new_model(backbone)  # make sure init weights / data config are cached
    tf_train, tf_test = make_tfs(_INIT[backbone][1]["mean"], _INIT[backbone][1]["std"])
    val_ld, test_ld = eval_loader(VAL_IDX, tf_test), eval_loader(TEST_IDX, tf_test)
    units = client_units(seed, exp["aug"], alpha)
    holdout = exp.get("holdout")
    res = dict(key=exp["key"], name=exp["name"], group=exp["group"], seed=seed, mode=exp["mode"], backbone=backbone,
               agg=exp["agg"], aug=exp["aug"], mu=mu, alpha=alpha, participation=exp.get("participation", 1.0),
               holdout=holdout, profile=PROFILE)
    if exp["mode"] == "local":
        per, probs = [], []
        for u in units:
            m, hist, br, comm = fl_train([u], backbone, seed, "fedavg", 0.0, 1.0, tf_train, val_ld, test_ld, f"[{exp['key']} c{u['client']}]")
            y, p = predict(m, test_ld); per.append(dict(client=u["client"], **{k: v for k, v in compute_metrics(y, p).items() if k in SCALARS}))
            probs.append(p); del m
        test = {k: float(np.mean([d[k] for d in per])) for k in SCALARS}
        res.update(test=test, per_client=per, history=hist, best_round=br, comm_MB=0.0)
        np.savez_compressed(RUNS_DIR / f"{exp['key']}__s{seed}.npz", y=y, prob=np.stack(probs))
    else:
        if exp["mode"] == "central":
            train_units = [merge_units(units)]
        else:
            train_units = [u for u in units if u["client"] != holdout]
        m, hist, br, comm = fl_train(train_units, backbone, seed, exp["agg"], mu, exp.get("participation", 1.0),
                                     tf_train, val_ld, test_ld, f"[{exp['key']} s{seed}]")
        y, p = predict(m, test_ld)
        res.update(test=compute_metrics(y, p), history=hist, best_round=br,
                   comm_MB=0.0 if exp["mode"] == "central" else comm)
        np.savez_compressed(RUNS_DIR / f"{exp['key']}__s{seed}.npz", y=y, prob=p)
        if holdout is not None:
            hidx = units[holdout]["real_idx"]; yu, pu = predict(m, eval_loader(hidx, tf_test))
            res["unseen_client"] = compute_metrics(yu, pu)
            np.savez_compressed(RUNS_DIR / f"{exp['key']}__s{seed}__unseen.npz", y=yu, prob=pu)
        if exp["key"] == "proposed" and seed == SEEDS_MAIN[0]:
            torch.save(m.state_dict(), OUT / "models" / f"proposed_s{seed}_{RUN_TAG}.pth")
        del m
    res["train_time_s"] = time.time() - t0
    json.dump(res, open(RUNS_DIR / f"{exp['key']}__s{seed}.json", "w"), default=float)
    gc.collect(); torch.cuda.empty_cache()
    return res

def E(key, name, group, mode="fed", agg="fedprox", aug="gan", seeds="other", **kw):
    return dict(key=key, name=name, group=group, mode=mode, agg=agg, aug=aug, seeds=seeds, **kw)

EXPERIMENTS = [
    E("proposed", "FediGAN-ViT (FedProx + cGAN + ViT)", "main", seeds="main"),
    # --- ablation: components ---
    E("fedavg_gan", "FedAvg + cGAN + ViT", "ablation", agg="fedavg"),
    E("fedprox_none", "FedProx + ViT (no augmentation)", "ablation", aug="none"),
    E("fedavg_none", "FedAvg + ViT (no augmentation)", "ablation", agg="fedavg", aug="none"),
    E("fedprox_oversample", "FedProx + ViT + random oversampling (same counts as cGAN)", "ablation", aug="oversample"),
    E("local_gan", "Local-only ViT + cGAN (no collaboration, mean of clients)", "ablation", mode="local", agg="fedavg"),
    E("central_none", "Centralised ViT (pooled data, upper bound)", "ablation", mode="central", agg="fedavg", aug="none"),
    E("central_gan", "Centralised ViT + cGAN", "ablation", mode="central", agg="fedavg"),
    # --- backbones ---
    E("fedprox_gan_resnet50", "FedProx + cGAN + ResNet-50", "backbone", backbone="resnet50"),
    E("fedprox_gan_effb0", "FedProx + cGAN + EfficientNet-B0", "backbone", backbone="efficientnet_b0"),
    # --- robustness ---
    E("partial", f"Partial participation (p={PARTICIPATION})", "robustness", participation=PARTICIPATION),
    *[E(f"loco_c{k}", f"Leave client {k} out (unseen institution)", "robustness", holdout=k) for k in range(NUM_CLIENTS)],
    # --- sensitivity (lowest priority) ---
    *[E(f"alpha_{a}", f"Dirichlet α = {a}", "sensitivity", alpha=a) for a in SENS_ALPHAS],
    *[E(f"mu_{m}", f"FedProx μ = {m}", "sensitivity", mu=m) for m in SENS_MUS],
]
EXP_BY_KEY = {e["key"]: e for e in EXPERIMENTS}

PLAN = []
for s in SEEDS_OTHER:                                   # every core config on each shared seed first
    PLAN += [(e, s) for e in EXPERIMENTS if e["group"] != "sensitivity"]
PLAN += [(EXP_BY_KEY["proposed"], s) for s in SEEDS_MAIN if s not in SEEDS_OTHER]
for s in SEEDS_OTHER:
    PLAN += [(e, s) for e in EXPERIMENTS if e["group"] == "sensitivity"]
if PLAN_MODE == "minimal":
    KEY3 = ["proposed", "fedprox_none", "fedprox_oversample", "local_gan", "central_none"]   # 3 seeds
    ONE = ["fedavg_gan", "partial", "loco_c0", "fedprox_gan_resnet50"]                           # 1 seed
    PLAN = [(EXP_BY_KEY[k], s) for s in SEEDS_OTHER for k in KEY3] + [(EXP_BY_KEY[k], SEEDS_OTHER[0]) for k in ONE]
elif PLAN_MODE == "reduced":
    KEY3 = ["proposed", "fedprox_none", "fedprox_oversample", "local_gan", "central_none",
            "partial", "loco_c0", "loco_c1", "loco_c2"]          # 3 seeds: needed for paired statistics
    ONE = ["fedavg_gan", "fedavg_none", "central_gan", "fedprox_gan_resnet50", "fedprox_gan_effb0"]  # 1 seed
    PLAN = [(EXP_BY_KEY[k], s) for s in SEEDS_OTHER for k in KEY3] + [(EXP_BY_KEY[k], SEEDS_OTHER[0]) for k in ONE]
print(f"{len(PLAN)} runs planned ({PLAN_MODE} plan)")

# ============================== BACKGROUND WORKER ROLES ==============================
_ROLE = os.environ.get("FEDIGAN_ROLE")
if _ROLE == "prep":
    for bb in sorted({e.get("backbone", BACKBONE) for e, _ in PLAN}):
        new_model(bb); print("backbone ready:", bb, flush=True)
    need = sorted({(s, e.get("alpha", ALPHA)) for e, s in PLAN if e["aug"] == "gan"})
    for s, a in need:
        t = time.time(); get_synthetic(s, a); print(f"cGANs ready: seed {s}, alpha {a} ({time.time() - t:.0f}s)", flush=True)
    print("PREP DONE", flush=True)
elif _ROLE == "shard":
    todo = [(e, s) for e, s in PLAN if not (RUNS_DIR / f"{e['key']}__s{s}.json").exists()]
    shard, nshards = int(_SHARD), int(os.environ.get("FEDIGAN_NSHARDS", 1))
    mine = todo[shard::nshards]
    print(f"[GPU{shard}] {len(mine)} of {len(todo)} remaining experiments", flush=True)
    durations, skipped = [], []
    for i, (exp, seed) in enumerate(mine):
        if (RUNS_DIR / f"{exp['key']}__s{seed}.json").exists():
            continue
        est = (np.median(durations) * 1.3) if durations else 1500
        if time.time() - NOTEBOOK_START + est > TIME_BUDGET_H * 3600:
            skipped.append(f"{exp['key']}__s{seed}"); continue
        print(f"\n[GPU{shard}] [{i + 1}/{len(mine)}] {exp['name']} | seed {seed} | "
              f"session time {(time.time() - NOTEBOOK_START) / 3600:.2f} h", flush=True)
        r = run_experiment(exp, seed); durations.append(r["train_time_s"])
        print(f"[GPU{shard}]  -> test acc {r['test']['accuracy']:.4f} | macro-F1 {r['test']['f1_macro']:.4f} | "
              f"macro-AUC {r['test']['auc_macro']:.4f} | {r['train_time_s'] / 60:.1f} min", flush=True)
    json.dump(dict(skipped=skipped), open(OUT / f"skipped_{RUN_TAG}_shard{shard}.json", "w"))
    print(f"[GPU{shard}] DONE. skipped for time: {skipped}", flush=True)
