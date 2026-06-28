from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "lab_6_dane_i_skrypty" / "dane_bss.mat"
OUT_DIR = ROOT / "wyniki_lab6"
REPORT_TEX = ROOT / "raport_lab6_Igor_Jozefowicz.tex"
REPORT_PDF = ROOT / "raport_lab6_Igor_Jozefowicz.pdf"

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


def beam_slice(first: int, last: int) -> slice:
    """Convert inclusive 1-based beam numbers to a Python slice."""
    return slice(first - 1, last)


def load_bss() -> np.ndarray:
    data = loadmat(DATA_PATH)
    return np.asarray(data["vbss"], dtype=float)


def replace_zero_with_min(vbss: np.ndarray) -> np.ndarray:
    values = vbss.copy()
    non_zero_min = values[values != 0].min()
    values[values == 0] = non_zero_min
    return values


def glcm_matrix(matrix: np.ndarray, di: int, dj: int, levels: int, total_min: float, total_max: float) -> np.ndarray:
    if total_min < total_max:
        quantized = np.floor(levels * (matrix - total_min) / (total_max - total_min)).astype(int)
        quantized[quantized == levels] = levels - 1
        quantized = np.clip(quantized, 0, levels - 1)
    else:
        quantized = np.zeros_like(matrix, dtype=int)

    rows, cols = quantized.shape
    result = np.zeros((levels, levels), dtype=float)
    row_start = max(0, -di)
    row_stop = rows - max(0, di)
    col_start = 0
    col_stop = cols - dj

    for i in range(row_start, row_stop):
        for j in range(col_start, col_stop):
            a = quantized[i, j]
            b = quantized[i + di, j + dj]
            lo, hi = min(a, b), max(a, b)
            result[lo, hi] += 1

    result = result + result.T
    total = result.sum()
    if total > 0:
        result /= total
    return result


def glcm_entropy_homogeneity(matrix: np.ndarray, levels: int = 8, distance: int = 1, total_min: float = 0.0, total_max: float = 1.0) -> tuple[float, float]:
    offsets = [(distance, 0), (0, distance), (distance, distance), (-distance, distance)]
    glcm = sum(glcm_matrix(matrix, di, dj, levels, total_min, total_max) for di, dj in offsets) / len(offsets)
    positive = glcm > 0
    entropy = float(-(glcm[positive] * np.log10(glcm[positive])).sum())

    rows, cols = np.indices(glcm.shape)
    homogeneity = float((glcm / (1 + (rows - cols) ** 2)).sum())
    return entropy, homogeneity


def compute_features(vbss: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    values = replace_zero_with_min(vbss)
    total_min = float(values.min())
    total_max = float(values.max())

    central = beam_slice(80, 100)
    right = beam_slice(91, 100)
    left = beam_slice(61, 80)
    edge = beam_slice(1, 20)

    arrays: dict[str, np.ndarray] = {
        "mean_80_100": values[:, :N_OBJECTS, central].mean(axis=2),
        "drop_80_100": values[:, :N_OBJECTS, 79] - values[:, :N_OBJECTS, 99],
        "std_80_100": values[:, :N_OBJECTS, central].std(axis=2),
        "left_right_diff": values[:, :N_OBJECTS, left].mean(axis=2) - values[:, :N_OBJECTS, right].mean(axis=2),
        "edge_center_diff": values[:, :N_OBJECTS, edge].mean(axis=2) - values[:, :N_OBJECTS, central].mean(axis=2),
    }

    glcm_entropy = np.zeros((4, N_OBJECTS), dtype=float)
    glcm_homogeneity = np.zeros((4, N_OBJECTS), dtype=float)
    for class_idx in range(4):
        for object_idx in range(N_OBJECTS):
            window = values[class_idx, object_idx : object_idx + 30, right]
            entropy, homogeneity = glcm_entropy_homogeneity(
                window,
                levels=8,
                distance=1,
                total_min=total_min,
                total_max=total_max,
            )
            glcm_entropy[class_idx, object_idx] = entropy
            glcm_homogeneity[class_idx, object_idx] = homogeneity

    arrays["glcm_entropy_91_100"] = glcm_entropy
    arrays["glcm_homogeneity_91_100"] = glcm_homogeneity

    rows = []
    for class_id in range(1, 5):
        for object_id in range(N_OBJECTS):
            row = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "object_id": object_id + 1,
            }
            row.update({name: arr[class_id - 1, object_id] for name, arr in arrays.items()})
            rows.append(row)
    return pd.DataFrame(rows), arrays


def split_by_class(train_pct: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    x = df[feature_names].to_numpy()
    y = df["class_id"].to_numpy()
    x_scaled = MinMaxScaler().fit_transform(x)
    train_idx, test_idx = split_by_class(train_pct, y)

    if classifier_name == "nearest_centroid":
        classifier = NearestCentroid()
    elif classifier_name == "knn_5":
        classifier = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    classifier.fit(x_scaled[train_idx], y[train_idx])
    predictions = classifier.predict(x_scaled[test_idx])
    cm = confusion_matrix(y[test_idx], predictions, labels=[1, 2, 3, 4])
    accuracy = accuracy_score(y[test_idx], predictions) * 100
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    return accuracy, cm, cm_pct


def plot_bss_images(vbss: np.ndarray) -> Path:
    path = OUT_DIR / "01_bss_images.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for class_id, ax in zip(range(1, 5), axes.ravel()):
        image = vbss[class_id - 1]
        im = ax.imshow(image, aspect="auto", cmap="viridis")
        ax.set_title(f"{class_id}. {CLASS_NAMES[class_id]}")
        ax.set_xlabel("Numer wiązki")
        ax.set_ylabel("Numer sondowania")
        fig.colorbar(im, ax=ax, shrink=0.85, label="BSS [dB]")
    fig.suptitle("Obrazy wartości BSS dla czterech typów dna")
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
    columns = ["mean_80_100", "drop_80_100", "glcm_homogeneity_91_100"]
    fig, axes = plt.subplots(1, len(columns), figsize=(12, 4), constrained_layout=True)
    for ax, column in zip(axes, columns):
        data = [df[df["class_id"] == class_id][column].to_numpy() for class_id in range(1, 5)]
        ax.boxplot(data, tick_labels=[str(i) for i in range(1, 5)], showfliers=False)
        ax.set_title(column)
        ax.set_xlabel("Typ dna")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Rozkłady wybranych parametrów według typu dna")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame, feature_names: list[str]) -> Path:
    path = OUT_DIR / "05_correlation_heatmap.png"
    corr = df[feature_names].corr()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_names)), feature_names)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Korelacje między parametrami")
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
            subset["mean_80_100"],
            subset["drop_80_100"],
            subset["glcm_homogeneity_91_100"],
            s=10,
            alpha=0.55,
            c=CLASS_COLORS[class_id],
            label=f"{class_id}. {CLASS_NAMES[class_id]}",
        )
    ax.set_title("Rozkład 3D wybranych parametrów")
    ax.set_xlabel("mean_80_100")
    ax.set_ylabel("drop_80_100")
    ax.set_zlabel("glcm_homogeneity_91_100")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    index = [f"{i}. {CLASS_NAMES[i]}" for i in range(1, 5)]
    columns = [f"pred_{i}" for i in range(1, 5)]
    pd.DataFrame(matrix, index=index, columns=columns).to_csv(path, encoding="utf-8-sig")


def classify_all(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pair_sets = {
        "mean+homogeneity": ["mean_80_100", "glcm_homogeneity_91_100"],
        "mean+drop": ["mean_80_100", "drop_80_100"],
        "mean+edge_center": ["mean_80_100", "edge_center_diff"],
    }

    for name, features in pair_sets.items():
        for train_pct in [20, 40, 60]:
            accuracy, cm, cm_pct = run_classifier(df, features, train_pct, "nearest_centroid")
            rows.append(
                {
                    "model": "nearest_centroid",
                    "feature_set": name,
                    "features": ", ".join(features),
                    "train_pct": train_pct,
                    "accuracy_pct": accuracy,
                }
            )
            if train_pct == 20:
                matrices[f"nearest_centroid_{name}_20pct"] = (cm, cm_pct)

    all_sets = {
        "all_features": feature_names,
    }
    for name, features in all_sets.items():
        for train_pct in [20, 40, 60]:
            for model in ["nearest_centroid", "knn_5"]:
                accuracy, cm, cm_pct = run_classifier(df, features, train_pct, model)
                rows.append(
                    {
                        "model": model,
                        "feature_set": name,
                        "features": ", ".join(features),
                        "train_pct": train_pct,
                        "accuracy_pct": accuracy,
                    }
                )
                if train_pct == 20:
                    matrices[f"{model}_{name}_20pct"] = (cm, cm_pct)

    results = pd.DataFrame(rows).sort_values(["accuracy_pct", "train_pct"], ascending=[False, True])
    return results, matrices


FEATURE_LABELS = {
    "mean_80_100": "mean",
    "drop_80_100": "drop",
    "glcm_entropy_91_100": "GLCM E",
    "glcm_homogeneity_91_100": "GLCM LH",
    "std_80_100": "std",
    "left_right_diff": "L-R",
    "edge_center_diff": "edge-center",
}


def compact_feature_summary(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    summary = df.groupby(["class_id", "class_name"])[feature_names].mean().round(3).reset_index()
    summary["typ dna"] = summary["class_id"].astype(str) + ". " + summary["class_name"]
    summary = summary.drop(columns=["class_id", "class_name"])
    summary = summary[["typ dna", *feature_names]].rename(columns=FEATURE_LABELS)
    return summary


def compact_classification_table(classification: pd.DataFrame) -> pd.DataFrame:
    table = classification.copy()
    table["model"] = table["model"].replace(
        {
            "nearest_centroid": "min. odl.",
            "knn_5": "kNN",
        }
    )
    table["feature_set"] = table["feature_set"].replace(
        {
            "mean+homogeneity": "mean + GLCM LH",
            "mean+drop": "mean + drop",
            "mean+edge_center": "mean + edge-center",
            "all_features": "wszystkie",
        }
    )
    table["accuracy_pct"] = table["accuracy_pct"].round(2)
    table = table[["model", "feature_set", "train_pct", "accuracy_pct"]]
    return table.rename(
        columns={
            "model": "model",
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
        return f"{float(value):.2f}"
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
    df: pd.DataFrame,
    classification: pd.DataFrame,
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    figure_paths: list[Path],
    feature_names: list[str],
) -> str:
    summary = compact_feature_summary(df, feature_names)
    classification_table = compact_classification_table(classification)
    pair_table = classification_table[classification_table["parametry"] != "wszystkie"].reset_index(drop=True)
    all_table = classification_table[classification_table["parametry"] == "wszystkie"].reset_index(drop=True)
    top_pair = classification[
        (classification["model"] == "nearest_centroid")
        & (classification["feature_set"].isin(["mean+homogeneity", "mean+drop", "mean+edge_center"]))
        & (classification["train_pct"] == 20)
    ].sort_values("accuracy_pct", ascending=False).iloc[0]
    top_all = classification.iloc[0]
    best_by_train = classification[classification["feature_set"] == "all_features"].pivot_table(
        index="train_pct", columns="model", values="accuracy_pct"
    ).round(2)
    best_by_train = best_by_train.rename(columns={"nearest_centroid": "min. odl.", "knn_5": "kNN"}).reset_index()
    main_key = f"nearest_centroid_{top_pair['feature_set']}_20pct"
    main_cm, main_pct = matrices[main_key]
    centroid_cm, centroid_pct = matrices["nearest_centroid_all_features_20pct"]
    knn_cm, knn_pct = matrices["knn_5_all_features_20pct"]

    code_features = """
def compute_features(vbss):
    values = replace_zero_with_min(vbss)
    arrays = {
        "mean_80_100": values[:, :N_OBJECTS, beam_slice(80, 100)].mean(axis=2),
        "drop_80_100": values[:, :N_OBJECTS, 79] - values[:, :N_OBJECTS, 99],
        "std_80_100": values[:, :N_OBJECTS, beam_slice(80, 100)].std(axis=2),
        "left_right_diff": values[:, :N_OBJECTS, beam_slice(61, 80)].mean(axis=2)
            - values[:, :N_OBJECTS, beam_slice(91, 100)].mean(axis=2),
        "edge_center_diff": values[:, :N_OBJECTS, beam_slice(1, 20)].mean(axis=2)
            - values[:, :N_OBJECTS, beam_slice(80, 100)].mean(axis=2),
    }
"""
    code_glcm = """
for class_idx in range(4):
    for object_idx in range(N_OBJECTS):
        window = values[class_idx, object_idx : object_idx + 30, beam_slice(91, 100)]
        entropy, homogeneity = glcm_entropy_homogeneity(
            window, levels=8, distance=1, total_min=total_min, total_max=total_max
        )
        glcm_entropy[class_idx, object_idx] = entropy
        glcm_homogeneity[class_idx, object_idx] = homogeneity
"""
    code_classifier = """
def run_classifier(df, feature_names, train_pct, classifier_name):
    x = MinMaxScaler().fit_transform(df[feature_names].to_numpy())
    y = df["class_id"].to_numpy()
    train_idx, test_idx = split_by_class(train_pct, y)
    classifier = NearestCentroid() if classifier_name == "nearest_centroid" else KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x[train_idx], y[train_idx])
    predictions = classifier.predict(x[test_idx])
    return accuracy_score(y[test_idx], predictions) * 100, confusion_matrix(y[test_idx], predictions)
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

\title{{Laboratorium 6\\Dane sonarowe BSS i klasyfikacja dna morskiego}}
\author{{Igor Józefowicz, indeks 193257}}
\date{{}}

\begin{{document}}
\maketitle

\section*{{1. Dane i cel}}
Raport dotyczy ćwiczenia z przedmiotu \emph{{Uczenie maszynowe w badaniach Ziemi}}. Celem było obliczenie parametrów opisujących sondowania BSS z sonaru wielowiązkowego, pokazanie ich rozkładów oraz sprawdzenie skuteczności prostych klasyfikatorów dla czterech typów dna.

Wykorzystano plik \texttt{{dane\_bss.mat}}, który zawiera tablicę \texttt{{vbss}} o rozmiarze \(4 \times 600 \times 160\): cztery typy dna, 600 sondowań dla każdego typu i 160 wiązek sonaru w jednym sondowaniu. Do obliczeń użyto pierwszych 500 sondowań z każdej klasy, zgodnie z dostarczonymi skryptami.

Typy dna: 1 -- muł, 2 -- mieszanina mułu, piasku i odpadów antropogenicznych, 3 -- piasek drobnoziarnisty, 4 -- piasek gruboziarnisty.

Najlepszy wynik klasyfikacji w wykonanych próbach uzyskał model \texttt{{{latex_escape(top_all['model'])}}} dla wszystkich parametrów i {int(top_all['train_pct'])}\% danych uczących: accuracy={top_all['accuracy_pct']:.2f}\%.

\section*{{2. Obliczone parametry}}
Obliczono cztery parametry wymagane w instrukcji oraz trzy lekkie rozszerzenia. W tabeli zastosowano skrócone nagłówki: GLCM E oznacza entropię, GLCM LH spójność lokalną, L-R różnicę sektorów lewego i prawego.

{latex_table(summary, column_spec="lrrrrrrr", font_size=r"\scriptsize")}

\section*{{3. Wizualizacja parametrów}}
Wykresy pokazują, że klasa 1 jest najlepiej oddzielona od pozostałych przez poziom BSS. Klasy 2, 3 i 4 częściowo nakładają się w przestrzeni parametrów, dlatego sama średnia BSS nie wystarcza do stabilnej klasyfikacji. Dodatkowe parametry opisują kształt zmian sygnału wraz z kątem sondowania i lokalną teksturę obrazu sonarowego.

{image_figure(figure_paths[0], "Obrazy BSS dla czterech typów dna.", r"0.86\textwidth")}
{image_figure(figure_paths[1], "Rozkład 2D: średnia BSS i spójność lokalna GLCM.", r"0.78\textwidth")}
{image_figure(figure_paths[2], "Rozkład 2D: średnia BSS i różnica sektorów.", r"0.78\textwidth")}
{image_figure(figure_paths[3], "Rozkłady wybranych parametrów według typu dna.", r"0.92\textwidth")}
{image_figure(figure_paths[4], "Korelacje między obliczonymi parametrami.", r"0.78\textwidth")}
{image_figure(figure_paths[5], "Dodatkowy rozkład 3D dla trzech parametrów.", r"0.82\textwidth")}

\section*{{4. Klasyfikacja}}
Klasyfikację minimalnoodległościową odtworzono przez \texttt{{NearestCentroid}}. Dane normalizowano metodą min-max, a podział uczący/testowy wykonano per klasa: pierwsze 20\%, 40\% albo 60\% obiektów jako zbiór uczący, reszta jako testowy.

\subsection*{{Pary parametrów}}
{latex_table(pair_table, column_spec="llrr", font_size=r"\small")}

\subsection*{{Wszystkie parametry}}
{latex_table(all_table, column_spec="llrr", font_size=r"\small")}

Najlepsza para dla klasyfikatora minimalnoodległościowego przy 20\% treningu to \texttt{{{latex_escape(top_pair['feature_set'])}}}, accuracy={top_pair['accuracy_pct']:.2f}\%.

{latex_matrix_table("Najlepsza para parametrów, 20% treningu", main_cm, main_pct)}
{latex_matrix_table("NearestCentroid, wszystkie parametry, 20% treningu", centroid_cm, centroid_pct)}
{latex_matrix_table("kNN, wszystkie parametry, 20% treningu", knn_cm, knn_pct)}

\section*{{5. Dodany kod własny}}
Pliki startowe do laboratorium zawierały skrypty MATLAB/Octave, m.in. \texttt{{vbss.m}}, \texttt{{obl\_srednia.m}}, \texttt{{obl\_spadek\_dB.m}}, \texttt{{obl\_GLCM\_entr\_spoj.m}}, \texttt{{rozkl2D.m}} oraz \texttt{{kl\_min\_dist\_2D.m}}. Nowy kod dodany w ramach rozwiązania znajduje się w pliku \texttt{{lab6\_solution.py}}. Poniżej pokazano tylko najważniejsze fragmenty: obliczenie parametrów, obliczenie parametrów GLCM oraz wspólną funkcję klasyfikacji.

{latex_code_block(code_features, "Fragment własnego kodu: obliczanie parametrów BSS.")}

{latex_code_block(code_glcm, "Fragment własnego kodu: obliczanie entropii i spójności lokalnej GLCM.")}

{latex_code_block(code_classifier, "Fragment własnego kodu: wspólna funkcja klasyfikacji i ewaluacji.")}

\section*{{6. Wnioski}}
\begin{{enumerate}}
    \item Klasa 1, czyli muł, jest najłatwiejsza do rozpoznania. W głównych macierzach niezgodności wszystkie lub prawie wszystkie obiekty tej klasy trafiają do poprawnej klasy.
    \item Największe pomyłki występują między klasami 2, 3 i 4, co zgadza się z wykresami parametrów, gdzie rozkłady tych klas częściowo się nakładają.
    \item Najlepsza para parametrów przy 20\% treningu uzyskała accuracy={top_pair['accuracy_pct']:.2f}\%, więc klasyfikacja 2D jest użyteczna, ale ograniczona.
    \item Najlepszy wynik całego zestawienia to \texttt{{{latex_escape(top_all['model'])}}} dla wszystkich parametrów, train\_pct={int(top_all['train_pct'])}\%, accuracy={top_all['accuracy_pct']:.2f}\%.
    \item Dla \texttt{{NearestCentroid}} na wszystkich parametrach accuracy wzrosło od {best_by_train.loc[best_by_train['train_pct'] == 20, 'min. odl.'].iloc[0]:.2f}\% do {best_by_train.loc[best_by_train['train_pct'] == 60, 'min. odl.'].iloc[0]:.2f}\%, więc większy zbiór uczący poprawił stabilność klasyfikacji.
    \item Dodatkowe parametry własne są przydatne, bo opisują nie tylko średni poziom BSS, lecz także zmianę sygnału wraz z kątem sondowania i lokalną teksturę obrazu.
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
    runs = 2

    for _ in range(runs):
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return REPORT_PDF.exists()


def cleanup_latex_auxiliary_files() -> None:
    for suffix in [".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz"]:
        path = REPORT_TEX.with_suffix(suffix)
        if path.exists():
            path.unlink()


def validate_outputs(classification: pd.DataFrame, matrices: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    expected = [
        OUT_DIR / "features.csv",
        OUT_DIR / "feature_summary.csv",
        OUT_DIR / "classification_results.csv",
        REPORT_TEX,
        REPORT_PDF,
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing expected output files: {missing}")

    if classification.empty:
        raise RuntimeError("Classification table is empty.")

    for name, (cm, cm_pct) in matrices.items():
        if cm.shape != (4, 4):
            raise RuntimeError(f"Unexpected confusion matrix shape for {name}: {cm.shape}")
        row_sums = cm_pct.sum(axis=1)
        if not np.allclose(row_sums, 100.0):
            raise RuntimeError(f"Percent confusion matrix rows do not sum to 100 for {name}: {row_sums}")

    for path in [REPORT_TEX, Path(__file__)]:
        text = path.read_text(encoding="utf-8")
        suspicious = [line for line in text.splitlines() if "\ufffd" in line]
        if suspicious:
            raise RuntimeError(f"Possible broken Polish characters in {path}: {suspicious[:3]}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    vbss = load_bss()
    df, _ = compute_features(vbss)

    feature_names = [
        "mean_80_100",
        "drop_80_100",
        "glcm_entropy_91_100",
        "glcm_homogeneity_91_100",
        "std_80_100",
        "left_right_diff",
        "edge_center_diff",
    ]

    df.to_csv(OUT_DIR / "features.csv", index=False, encoding="utf-8-sig")
    df.groupby(["class_id", "class_name"])[feature_names].agg(["mean", "std"]).round(6).to_csv(
        OUT_DIR / "feature_summary.csv",
        encoding="utf-8-sig",
    )

    figure_paths = [
        plot_bss_images(vbss),
        plot_scatter_2d(
            df,
            "mean_80_100",
            "glcm_homogeneity_91_100",
            "02_scatter_mean_homogeneity.png",
            "Średnia BSS vs spójność lokalna GLCM",
        ),
        plot_scatter_2d(
            df,
            "mean_80_100",
            "edge_center_diff",
            "03_scatter_mean_edge_center.png",
            "Średnia BSS vs różnica sektor skrajny-centrum",
        ),
        plot_boxplots(df),
        plot_correlation(df, feature_names),
        plot_scatter_3d(df),
    ]

    classification, matrices = classify_all(df, feature_names)
    classification.to_csv(OUT_DIR / "classification_results.csv", index=False, encoding="utf-8-sig")
    for name, (cm, cm_pct) in matrices.items():
        save_matrix(cm, OUT_DIR / f"confusion_{name}.csv")
        save_matrix(cm_pct, OUT_DIR / f"confusion_pct_{name}.csv")

    latex_report = build_latex_report(df, classification, matrices, figure_paths, feature_names)
    REPORT_TEX.write_text(latex_report, encoding="utf-8")
    if REPORT_PDF.exists():
        REPORT_PDF.unlink()
    compiled = compile_latex()
    if compiled:
        cleanup_latex_auxiliary_files()
    validate_outputs(classification, matrices)

    print("Generated:")
    print(f"- {REPORT_TEX}")
    print(f"- {REPORT_PDF}")
    print(f"- {OUT_DIR}")
    print()
    print("Best classification results:")
    print(classification.head(5).round({"accuracy_pct": 2}).to_string(index=False))


if __name__ == "__main__":
    main()

