from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import MinMaxScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_PATH = ROOT / "lab_7_dane_i_skrypty" / "dane_z.mat"
OUT_DIR = ROOT / "wyniki_lab7"
REPORT_TEX = ROOT / "raport_lab7_Igor_Jozefowicz.tex"
REPORT_PDF = ROOT / "raport_lab7_Igor_Jozefowicz.pdf"

N_OBJECTS = 500
CLASS_NAMES = {
    1: "muł",
    2: "muł/piasek/odpady",
    3: "piasek drobnoziarnisty",
    4: "piasek gruboziarnisty",
}
CLASS_COLORS = {
    1: "#1f77b4",
    2: "#2ca02c",
    3: "#ffbf00",
    4: "#d62728",
}
BATH_FEATURES = [
    "bath_std_41_80",
    "bath_sixph_41_80",
    "bath_mad_41_80",
    "bath_diff_std_41_80",
    "bath_p90_p10_41_80",
    "bath_range_41_80",
]
BSS_FEATURES = [
    "mean_80_100",
    "drop_80_100",
    "std_80_100",
    "left_right_diff",
    "edge_center_diff",
    "glcm_entropy_91_100",
    "glcm_homogeneity_91_100",
]


def beam_slice(first: int, last: int) -> slice:
    return slice(first - 1, last)


def load_bathymetry() -> np.ndarray:
    data = loadmat(DATA_PATH)
    return np.asarray(data["vz"], dtype=float)


def detrend_line(values: np.ndarray) -> np.ndarray:
    x = np.arange(1, len(values) + 1, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    return values - (slope * x + intercept)


def prepare_sector(vz: np.ndarray, sector: slice, detrending: bool) -> np.ndarray:
    raw = vz[:, :N_OBJECTS, sector]
    result = np.empty_like(raw)
    for class_idx in range(raw.shape[0]):
        for object_idx in range(raw.shape[1]):
            line = raw[class_idx, object_idx].copy()
            result[class_idx, object_idx] = detrend_line(line) if detrending else line
    return result


def six_point_height(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, axis=2)
    return sorted_values[:, :, -3:].mean(axis=2) - sorted_values[:, :, :3].mean(axis=2)


def compute_bathymetry_features(vz: np.ndarray, detrending: bool = True) -> pd.DataFrame:
    sector = beam_slice(41, 80)
    values = prepare_sector(vz, sector, detrending=detrending)
    rows = []

    features = {
        "bath_std_41_80": values.std(axis=2),
        "bath_sixph_41_80": six_point_height(values),
        "bath_mad_41_80": np.mean(np.abs(values - values.mean(axis=2, keepdims=True)), axis=2),
        "bath_diff_std_41_80": np.diff(values, axis=2).std(axis=2),
        "bath_p90_p10_41_80": np.percentile(values, 90, axis=2) - np.percentile(values, 10, axis=2),
        "bath_range_41_80": values.max(axis=2) - values.min(axis=2),
    }

    for class_id in range(1, 5):
        for object_id in range(N_OBJECTS):
            row = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "object_id": object_id + 1,
                "detrending": detrending,
            }
            row.update({name: data[class_id - 1, object_id] for name, data in features.items()})
            rows.append(row)

    return pd.DataFrame(rows)


def import_lab6_solution():
    module_path = REPO_ROOT / "lab6" / "lab6_solution.py"
    spec = importlib.util.spec_from_file_location("lab6_solution_for_lab7", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import lab 6 solution from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_bss_features() -> pd.DataFrame:
    lab6 = import_lab6_solution()
    bss, _ = lab6.compute_features(lab6.load_bss())
    return bss[["class_id", "object_id", *BSS_FEATURES]].copy()


def split_by_class(y: np.ndarray, train_pct: int) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    test_indices: list[int] = []
    train_count = int(N_OBJECTS * train_pct / 100)
    for class_id in range(1, 5):
        indices = np.where(y == class_id)[0]
        train_indices.extend(indices[:train_count])
        test_indices.extend(indices[train_count:])
    return np.asarray(train_indices), np.asarray(test_indices)


def run_classifier(
    df: pd.DataFrame,
    feature_names: list[str],
    train_pct: int,
    classifier_name: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    x = MinMaxScaler().fit_transform(df[feature_names].to_numpy())
    y = df["class_id"].to_numpy()
    train_idx, test_idx = split_by_class(y, train_pct)

    if classifier_name == "nearest_centroid":
        classifier = NearestCentroid()
    elif classifier_name == "knn_5":
        classifier = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    classifier.fit(x[train_idx], y[train_idx])
    predictions = classifier.predict(x[test_idx])
    cm = confusion_matrix(y[test_idx], predictions, labels=[1, 2, 3, 4])
    accuracy = accuracy_score(y[test_idx], predictions) * 100
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    return accuracy, cm, cm_pct


def classify_bathymetry(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    pair_sets = {
        "std+sixph": ["bath_std_41_80", "bath_sixph_41_80"],
        "std+diff_std": ["bath_std_41_80", "bath_diff_std_41_80"],
        "std+p90_p10": ["bath_std_41_80", "bath_p90_p10_41_80"],
        "mad+range": ["bath_mad_41_80", "bath_range_41_80"],
    }
    rows = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for feature_set, features in pair_sets.items():
        for train_pct in [20, 40, 60]:
            accuracy, cm, cm_pct = run_classifier(df, features, train_pct, "nearest_centroid")
            rows.append(
                {
                    "model": "nearest_centroid",
                    "feature_set": feature_set,
                    "features": ", ".join(features),
                    "train_pct": train_pct,
                    "accuracy_pct": accuracy,
                }
            )
            if train_pct == 20:
                matrices[f"nearest_centroid_{feature_set}_20pct"] = (cm, cm_pct)

    for train_pct in [20, 40, 60]:
        for model in ["nearest_centroid", "knn_5"]:
            accuracy, cm, cm_pct = run_classifier(df, BATH_FEATURES, train_pct, model)
            rows.append(
                {
                    "model": model,
                    "feature_set": "bath_all",
                    "features": ", ".join(BATH_FEATURES),
                    "train_pct": train_pct,
                    "accuracy_pct": accuracy,
                }
            )
            if train_pct == 20:
                matrices[f"{model}_bath_all_20pct"] = (cm, cm_pct)

    results = pd.DataFrame(rows).sort_values(["accuracy_pct", "train_pct"], ascending=[False, True])
    return results, matrices


def classify_combined(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    feature_sets = {
        "bathymetry_only": BATH_FEATURES,
        "bss_only": BSS_FEATURES,
        "combined_bath_bss": BATH_FEATURES + BSS_FEATURES,
    }
    rows = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for feature_set, features in feature_sets.items():
        for train_pct in [20, 40, 60]:
            for model in ["nearest_centroid", "knn_5"]:
                accuracy, cm, cm_pct = run_classifier(df, features, train_pct, model)
                rows.append(
                    {
                        "model": model,
                        "feature_set": feature_set,
                        "features": ", ".join(features),
                        "train_pct": train_pct,
                        "accuracy_pct": accuracy,
                    }
                )
                if train_pct == 20:
                    matrices[f"{model}_{feature_set}_20pct"] = (cm, cm_pct)

    results = pd.DataFrame(rows).sort_values(["accuracy_pct", "train_pct"], ascending=[False, True])
    return results, matrices


def plot_bathymetry_surfaces(vz: np.ndarray) -> Path:
    path = OUT_DIR / "01_bathymetry_surfaces.png"
    swath_sector = slice(50, 80)
    beam_sector = slice(40, 80)
    beam_grid, swath_grid = np.meshgrid(np.arange(41, 81), np.arange(51, 81))

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    for class_id in range(1, 5):
        ax = fig.add_subplot(2, 2, class_id, projection="3d")
        sector = vz[class_id - 1, swath_sector, beam_sector].copy()
        for row_idx in range(sector.shape[0]):
            sector[row_idx] = detrend_line(sector[row_idx])
        ax.plot_surface(beam_grid, swath_grid, sector, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"{class_id}. {CLASS_NAMES[class_id]}")
        ax.set_xlabel("Wiązka")
        ax.set_ylabel("Sondowanie")
        ax.set_zlabel("z po detrendingu [m]")
        ax.view_init(elev=24, azim=-135)
    fig.suptitle("Powierzchnie batymetryczne po detrendingu")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, filename: str, title: str) -> Path:
    path = OUT_DIR / filename
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for class_id in range(1, 5):
        subset = df[df["class_id"] == class_id]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=12,
            alpha=0.55,
            c=CLASS_COLORS[class_id],
            label=f"{class_id}. {CLASS_NAMES[class_id]}",
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_boxplots(df: pd.DataFrame) -> Path:
    path = OUT_DIR / "04_boxplots.png"
    columns = ["bath_std_41_80", "bath_sixph_41_80", "bath_diff_std_41_80"]
    fig, axes = plt.subplots(1, len(columns), figsize=(12, 4), constrained_layout=True)
    for ax, column in zip(axes, columns):
        data = [df[df["class_id"] == class_id][column].to_numpy() for class_id in range(1, 5)]
        ax.boxplot(data, tick_labels=[str(i) for i in range(1, 5)], showfliers=False)
        ax.set_title(column)
        ax.set_xlabel("Typ dna")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Rozkłady wybranych parametrów batymetrycznych")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame) -> Path:
    path = OUT_DIR / "05_correlation_heatmap.png"
    corr = df[BATH_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [feature.replace("bath_", "").replace("_41_80", "") for feature in BATH_FEATURES]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Korelacje parametrów batymetrycznych")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scatter_3d(df: pd.DataFrame) -> Path:
    path = OUT_DIR / "06_scatter_3d.png"
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for class_id in range(1, 5):
        subset = df[df["class_id"] == class_id]
        ax.scatter(
            subset["bath_std_41_80"],
            subset["bath_sixph_41_80"],
            subset["bath_diff_std_41_80"],
            s=10,
            alpha=0.55,
            c=CLASS_COLORS[class_id],
            label=f"{class_id}. {CLASS_NAMES[class_id]}",
        )
    ax.set_title("Rozkład 3D parametrów batymetrycznych")
    ax.set_xlabel("std")
    ax.set_ylabel("sixph")
    ax.set_zlabel("diff_std")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    index = [f"{i}. {CLASS_NAMES[i]}" for i in range(1, 5)]
    columns = [f"pred_{i}" for i in range(1, 5)]
    pd.DataFrame(matrix, index=index, columns=columns).to_csv(path, encoding="utf-8-sig")


FEATURE_LABELS = {
    "bath_std_41_80": "std",
    "bath_sixph_41_80": "sixph",
    "bath_mad_41_80": "MAD",
    "bath_diff_std_41_80": "diff std",
    "bath_p90_p10_41_80": "p90-p10",
    "bath_range_41_80": "range",
}


def compact_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["class_id", "class_name"])[BATH_FEATURES].mean().round(4).reset_index()
    summary["typ dna"] = summary["class_id"].astype(str) + ". " + summary["class_name"]
    summary = summary.drop(columns=["class_id", "class_name"])
    return summary[["typ dna", *BATH_FEATURES]].rename(columns=FEATURE_LABELS)


def compact_results(results: pd.DataFrame) -> pd.DataFrame:
    table = results.copy()
    table["model"] = table["model"].replace({"nearest_centroid": "min. odl.", "knn_5": "kNN"})
    table["feature_set"] = table["feature_set"].replace(
        {
            "std+sixph": "std + sixph",
            "std+diff_std": "std + diff std",
            "std+p90_p10": "std + p90-p10",
            "mad+range": "MAD + range",
            "bath_all": "bath all",
            "bathymetry_only": "batymetria",
            "bss_only": "BSS",
            "combined_bath_bss": "batymetria + BSS",
        }
    )
    table["accuracy_pct"] = table["accuracy_pct"].round(2)
    table = table[["model", "feature_set", "train_pct", "accuracy_pct"]]
    return table.rename(
        columns={
            "feature_set": "parametry",
            "train_pct": "trening [%]",
            "accuracy_pct": "accuracy [%]",
        }
    )


def compact_matrix(matrix: np.ndarray, decimals: int = 0) -> pd.DataFrame:
    table = pd.DataFrame(
        matrix,
        index=[f"rzecz. {i}" for i in range(1, 5)],
        columns=[f"kl. {i}" for i in range(1, 5)],
    )
    if decimals == 0:
        table = table.astype(int)
    else:
        table = table.round(decimals)
    return table.reset_index(names="")


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def format_latex_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if abs(float(value) - round(float(value))) < 1e-9:
            return f"{float(value):.0f}"
        return f"{float(value):.3f}" if abs(float(value)) < 1 else f"{float(value):.2f}"
    return latex_escape(value)


def latex_table(df: pd.DataFrame, column_spec: str, font_size: str = r"\small") -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        font_size,
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(format_latex_value(row[col]) for col in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def latex_matrix_rows(table: pd.DataFrame, decimals: int) -> str:
    rows = []
    for _, row in table.iterrows():
        values = []
        for col in table.columns:
            value = row[col]
            if col == "":
                values.append(latex_escape(value))
            elif decimals:
                values.append(f"{float(value):.2f}")
            else:
                values.append(str(int(value)))
        rows.append(" & ".join(values) + r" \\")
    return "\n".join(rows)


def latex_matrix_table(title: str, counts: np.ndarray, percentages: np.ndarray) -> str:
    counts_df = compact_matrix(counts)
    pct_df = compact_matrix(percentages, decimals=2)
    return rf"""
\begin{{table}}[H]
\centering
\small
\caption{{{latex_escape(title)}}}
\begin{{minipage}}{{0.47\textwidth}}
\centering
\textbf{{Liczba obiektów}}\\[0.35em]
\begin{{tabular}}{{lrrrr}}
\toprule
 & kl. 1 & kl. 2 & kl. 3 & kl. 4 \\
\midrule
{latex_matrix_rows(counts_df, decimals=0)}
\bottomrule
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}{{0.47\textwidth}}
\centering
\textbf{{Udział procentowy [\%]}}\\[0.35em]
\begin{{tabular}}{{lrrrr}}
\toprule
 & kl. 1 & kl. 2 & kl. 3 & kl. 4 \\
\midrule
{latex_matrix_rows(pct_df, decimals=2)}
\bottomrule
\end{{tabular}}
\end{{minipage}}
\end{{table}}
"""


def image_figure(path: Path, caption: str, width: str = r"0.88\textwidth") -> str:
    rel = path.relative_to(ROOT).as_posix()
    return rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width={width}]{{\detokenize{{{rel}}}}}
\caption{{{latex_escape(caption)}}}
\end{{figure}}
"""


def latex_code_block(code: str, caption: str) -> str:
    return rf"""
\begin{{lstlisting}}[caption={{{latex_escape(caption)}}}]
{code.strip()}
\end{{lstlisting}}
"""


def build_latex_report(
    vz: np.ndarray,
    bath_df: pd.DataFrame,
    no_detrend_df: pd.DataFrame,
    bath_results: pd.DataFrame,
    combined_results: pd.DataFrame,
    bath_matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    combined_matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    figure_paths: list[Path],
) -> str:
    summary = compact_feature_summary(bath_df)
    no_detrend_summary = no_detrend_df.groupby("class_id")[["bath_std_41_80", "bath_sixph_41_80", "bath_range_41_80"]].mean().round(4)
    detrend_summary = bath_df.groupby("class_id")[["bath_std_41_80", "bath_sixph_41_80", "bath_range_41_80"]].mean().round(4)
    detrend_compare = detrend_summary.join(no_detrend_summary, lsuffix=" det.", rsuffix=" bez det.").reset_index()
    detrend_compare = detrend_compare.rename(
        columns={
            "class_id": "klasa",
            "bath_std_41_80 det.": "std det.",
            "bath_sixph_41_80 det.": "sixph det.",
            "bath_range_41_80 det.": "range det.",
            "bath_std_41_80 bez det.": "std bez det.",
            "bath_sixph_41_80 bez det.": "sixph bez det.",
            "bath_range_41_80 bez det.": "range bez det.",
        }
    )
    bath_table = compact_results(bath_results)
    combined_table = compact_results(combined_results)

    top_pair = bath_results[
        (bath_results["model"] == "nearest_centroid")
        & (bath_results["feature_set"].isin(["std+sixph", "std+diff_std", "std+p90_p10", "mad+range"]))
        & (bath_results["train_pct"] == 20)
    ].sort_values("accuracy_pct", ascending=False).iloc[0]
    top_bath = bath_results.iloc[0]
    top_combined = combined_results.iloc[0]

    pair_key = f"nearest_centroid_{top_pair['feature_set']}_20pct"
    pair_cm, pair_pct = bath_matrices[pair_key]
    bath_cm, bath_pct = bath_matrices["nearest_centroid_bath_all_20pct"]
    combined_cm, combined_pct = combined_matrices["knn_5_combined_bath_bss_20pct"]

    code_features = """
def detrend_line(values):
    x = np.arange(1, len(values) + 1, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    return values - (slope * x + intercept)

features = {
    "bath_std_41_80": values.std(axis=2),
    "bath_sixph_41_80": six_point_height(values),
    "bath_mad_41_80": np.mean(np.abs(values - values.mean(axis=2, keepdims=True)), axis=2),
    "bath_diff_std_41_80": np.diff(values, axis=2).std(axis=2),
}
"""
    code_classifier = """
def run_classifier(df, feature_names, train_pct, classifier_name):
    x = MinMaxScaler().fit_transform(df[feature_names].to_numpy())
    y = df["class_id"].to_numpy()
    train_idx, test_idx = split_by_class(y, train_pct)
    classifier = NearestCentroid() if classifier_name == "nearest_centroid" else KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x[train_idx], y[train_idx])
    predictions = classifier.predict(x[test_idx])
    return accuracy_score(y[test_idx], predictions) * 100, confusion_matrix(y[test_idx], predictions)
"""
    code_merge = """
lab6 = import_lab6_solution()
bss_features, _ = lab6.compute_features(lab6.load_bss())
combined = bath_features.merge(
    bss_features[["class_id", "object_id", *BSS_FEATURES]],
    on=["class_id", "object_id"],
)
"""

    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{fontspec}}
\usepackage{{polyglossia}}
\setmainlanguage{{polish}}
\setmainfont{{Latin Modern Roman}}
\setsansfont{{Latin Modern Sans}}
\setmonofont{{Latin Modern Mono}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{listings}}
\lstdefinestyle{{pythonstyle}}{{
    language=Python,
    basicstyle=\small\ttfamily,
    breaklines=true,
    columns=fullflexible,
    keepspaces=true,
    frame=single,
    rulecolor=\color{{black}},
    showstringspaces=false
}}
\lstset{{style=pythonstyle}}
\geometry{{margin=2.2cm}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}
\renewcommand{{\arraystretch}}{{1.15}}
\sloppy

\title{{Laboratorium 7\\Dane batymetryczne i klasyfikacja dna morskiego}}
\author{{Igor Józefowicz, indeks 193257}}
\date{{}}

\begin{{document}}
\maketitle

\section*{{1. Dane i cel}}
Raport dotyczy batymetrycznych danych sonarowych z pliku \texttt{{dane\_z.mat}}. Zmienna \texttt{{vz}} ma rozmiar \(4 \times 600 \times 160\): cztery typy dna, 600 sondowań i 160 wiązek. Celem było obliczenie parametrów opisujących małoskalową rzeźbę dna, wizualizacja ich rozkładów oraz klasyfikacja czterech typów dna.

Typy dna są takie same jak w laboratorium 6: 1 -- muł, 2 -- mieszanina mułu, piasku i odpadów antropogenicznych, 3 -- piasek drobnoziarnisty, 4 -- piasek gruboziarnisty. Do obliczeń użyto pierwszych 500 sondowań z każdej klasy.

\section*{{2. Detrending i parametry}}
Główne wyniki policzono dla sektora wiązek 41--80 po detrendingu liniowym każdego sondowania. Jest to zgodne z instrukcją, ponieważ bez detrendingu parametry łatwo opisują głębokość i nachylenie większej skali, a nie samą lokalną rzeźbę dna.

{latex_table(summary, column_spec="lrrrrrr", font_size=r"\scriptsize")}

Dla kontroli policzono także wariant bez detrendingu. Porównanie pokazuje, że wartości parametrów bez detrendingu są dużo większe, bo zawierają trend głębokości.

{latex_table(detrend_compare, column_spec="lrrrrrr", font_size=r"\scriptsize")}

\section*{{3. Wizualizacja parametrów}}
Wizualizacje pokazują, że parametry batymetryczne po detrendingu słabiej separują klasy niż parametry BSS z laboratorium 6. Klasy 2 i 3 częściowo się nakładają, a klasa 4 jest bardziej podobna do klasy 1 pod względem części parametrów wysokościowych.

{image_figure(figure_paths[0], "Powierzchnie batymetryczne po detrendingu.", r"0.92\textwidth")}
{image_figure(figure_paths[1], "Rozkład 2D: odchylenie standardowe i six-point height.", r"0.78\textwidth")}
{image_figure(figure_paths[2], "Rozkład 2D: odchylenie standardowe i percentyl p90-p10.", r"0.78\textwidth")}
{image_figure(figure_paths[3], "Boxploty wybranych parametrów batymetrycznych.", r"0.92\textwidth")}
{image_figure(figure_paths[4], "Korelacje parametrów batymetrycznych.", r"0.78\textwidth")}
{image_figure(figure_paths[5], "Dodatkowy rozkład 3D parametrów.", r"0.82\textwidth")}

\section*{{4. Klasyfikacja batymetryczna}}
Klasyfikację minimalnoodległościową odtworzono przez \texttt{{NearestCentroid}}. Dane normalizowano metodą min-max, a podział uczący/testowy wykonano per klasa.

{latex_table(bath_table, column_spec="llrr", font_size=r"\small")}

Najlepsza para parametrów przy 20\% treningu to \texttt{{{latex_escape(top_pair['feature_set'])}}}, accuracy={top_pair['accuracy_pct']:.2f}\%. Najlepszy wynik dla samej batymetrii w całym zestawieniu to \texttt{{{latex_escape(top_bath['model'])}}}, \texttt{{{latex_escape(top_bath['feature_set'])}}}, train\_pct={int(top_bath['train_pct'])}\%, accuracy={top_bath['accuracy_pct']:.2f}\%.

{latex_matrix_table("Najlepsza para parametrów batymetrycznych, 20% treningu", pair_cm, pair_pct)}
{latex_matrix_table("NearestCentroid, wszystkie parametry batymetryczne, 20% treningu", bath_cm, bath_pct)}

\section*{{5. Wariant łączony z BSS}}
Instrukcja dopuszcza łączne użycie parametrów z bieżącego ćwiczenia i parametrów BSS z laboratorium 6. Cechy BSS policzono ponownie przez funkcje z \texttt{{lab6\_solution.py}} i połączono z batymetrią po \texttt{{class\_id}} oraz \texttt{{object\_id}}.

{latex_table(combined_table, column_spec="llrr", font_size=r"\small")}

Najlepszy wynik wariantu łączonego uzyskał model \texttt{{{latex_escape(top_combined['model'])}}} dla zestawu \texttt{{{latex_escape(top_combined['feature_set'])}}}, train\_pct={int(top_combined['train_pct'])}\%, accuracy={top_combined['accuracy_pct']:.2f}\%.

{latex_matrix_table("kNN, batymetria + BSS, 20% treningu", combined_cm, combined_pct)}

\section*{{6. Dodany kod własny}}
Pliki startowe do laboratorium zawierały skrypty MATLAB/Octave: \texttt{{v\_z.m}}, \texttt{{obl\_stdh.m}}, \texttt{{obl\_sixph.m}}, \texttt{{obl\_sredniah.m}}, \texttt{{rozkl2D.m}} oraz \texttt{{kl\_min\_dist\_2D.m}}. Nowy kod dodany w ramach rozwiązania znajduje się w \texttt{{lab7\_solution.py}}. Poniżej pokazano tylko najważniejsze fragmenty.

{latex_code_block(code_features, "Fragment własnego kodu: detrending i parametry batymetryczne.")}
{latex_code_block(code_classifier, "Fragment własnego kodu: klasyfikacja i macierz niezgodności.")}
{latex_code_block(code_merge, "Fragment własnego kodu: połączenie batymetrii z cechami BSS.")}

\section*{{7. Wnioski}}
\begin{{enumerate}}
    \item Po detrendingu parametry batymetryczne opisują małoskalową rzeźbę dna, ale separacja klas jest umiarkowana. Najlepszy wynik samej batymetrii wyniósł {top_bath['accuracy_pct']:.2f}\%.
    \item Wariant bez detrendingu daje większe wartości parametrów, lecz taki wynik łatwo interpretować jako wpływ głębokości lub trendu większej skali, a nie rodzaju dna.
    \item Największe pomyłki dla samej batymetrii występują między klasami o podobnej lokalnej zmienności powierzchni.
    \item Połączenie batymetrii z BSS znacząco poprawiło wynik: najlepszy wariant łączony osiągnął accuracy={top_combined['accuracy_pct']:.2f}\%.
    \item W praktyce parametry BSS są w tych danych silniejsze klasyfikacyjnie, ale batymetria dostarcza dodatkowej informacji o lokalnym kształcie powierzchni dna.
\end{{enumerate}}

\end{{document}}
"""


def compile_latex() -> bool:
    engines = ["xelatex", "lualatex", "pdflatex"]
    engine = next((cmd for cmd in engines if shutil.which(cmd)), None)
    if engine is None:
        miktex_bin = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
        for name in engines:
            candidate = miktex_bin / f"{name}.exe"
            if candidate.exists():
                engine = str(candidate)
                break
    if engine is None:
        return False

    command = [engine, "-interaction=nonstopmode", "-halt-on-error", REPORT_TEX.name]
    for _ in range(2):
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return REPORT_PDF.exists()


def cleanup_latex_auxiliary_files() -> None:
    for suffix in [".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz"]:
        path = REPORT_TEX.with_suffix(suffix)
        if path.exists():
            path.unlink()


def validate_outputs(
    vz: np.ndarray,
    bath_df: pd.DataFrame,
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    figure_paths: list[Path],
) -> None:
    if vz.shape != (4, 600, 160):
        raise RuntimeError(f"Unexpected vz shape: {vz.shape}")
    if bath_df.groupby("class_id").size().tolist() != [N_OBJECTS] * 4:
        raise RuntimeError("Unexpected number of objects per class.")
    for path in [OUT_DIR / "bathymetry_features.csv", OUT_DIR / "bathymetry_classification.csv", OUT_DIR / "combined_classification.csv", REPORT_TEX, REPORT_PDF, *figure_paths]:
        if not path.exists():
            raise RuntimeError(f"Missing output file: {path}")
    for name, (cm, cm_pct) in matrices.items():
        if cm.shape != (4, 4):
            raise RuntimeError(f"Unexpected confusion matrix shape for {name}: {cm.shape}")
        if not np.allclose(cm_pct.sum(axis=1), 100.0):
            raise RuntimeError(f"Percent rows do not sum to 100 for {name}")
    for path in [REPORT_TEX, Path(__file__)]:
        if "\ufffd" in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Possible broken text encoding in {path}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    vz = load_bathymetry()
    bath_df = compute_bathymetry_features(vz, detrending=True)
    no_detrend_df = compute_bathymetry_features(vz, detrending=False)
    bss_df = compute_bss_features()
    combined_df = bath_df.merge(bss_df, on=["class_id", "object_id"], validate="one_to_one")

    bath_df.to_csv(OUT_DIR / "bathymetry_features.csv", index=False, encoding="utf-8-sig")
    no_detrend_df.to_csv(OUT_DIR / "bathymetry_features_no_detrending.csv", index=False, encoding="utf-8-sig")
    combined_df.to_csv(OUT_DIR / "combined_bathymetry_bss_features.csv", index=False, encoding="utf-8-sig")
    bath_df.groupby(["class_id", "class_name"])[BATH_FEATURES].agg(["mean", "std"]).round(6).to_csv(
        OUT_DIR / "bathymetry_feature_summary.csv",
        encoding="utf-8-sig",
    )

    figure_paths = [
        plot_bathymetry_surfaces(vz),
        plot_scatter_2d(
            bath_df,
            "bath_std_41_80",
            "bath_sixph_41_80",
            "02_scatter_std_sixph.png",
            "Odchylenie standardowe vs six-point height",
        ),
        plot_scatter_2d(
            bath_df,
            "bath_std_41_80",
            "bath_p90_p10_41_80",
            "03_scatter_std_p90_p10.png",
            "Odchylenie standardowe vs p90-p10",
        ),
        plot_boxplots(bath_df),
        plot_correlation(bath_df),
        plot_scatter_3d(bath_df),
    ]

    bath_results, bath_matrices = classify_bathymetry(bath_df)
    combined_results, combined_matrices = classify_combined(combined_df)
    bath_results.to_csv(OUT_DIR / "bathymetry_classification.csv", index=False, encoding="utf-8-sig")
    combined_results.to_csv(OUT_DIR / "combined_classification.csv", index=False, encoding="utf-8-sig")

    all_matrices = {**bath_matrices, **combined_matrices}
    for name, (cm, cm_pct) in all_matrices.items():
        save_matrix(cm, OUT_DIR / f"confusion_{name}.csv")
        save_matrix(cm_pct, OUT_DIR / f"confusion_pct_{name}.csv")

    REPORT_TEX.write_text(
        build_latex_report(vz, bath_df, no_detrend_df, bath_results, combined_results, bath_matrices, combined_matrices, figure_paths),
        encoding="utf-8",
    )
    if REPORT_PDF.exists():
        REPORT_PDF.unlink()
    compiled = compile_latex()
    if compiled:
        cleanup_latex_auxiliary_files()
    validate_outputs(vz, bath_df, all_matrices, figure_paths)

    print("Generated:")
    print(f"- {REPORT_TEX}")
    print(f"- {REPORT_PDF}")
    print(f"- {OUT_DIR}")
    print()
    print("Best bathymetry results:")
    print(bath_results.head(5).round({"accuracy_pct": 2}).to_string(index=False))
    print()
    print("Best combined results:")
    print(combined_results.head(5).round({"accuracy_pct": 2}).to_string(index=False))


if __name__ == "__main__":
    main()
