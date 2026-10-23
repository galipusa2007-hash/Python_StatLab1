"""Единый стиль графиков Lab01 и типовые рисунки.

Цвета: типы проб — первые три слота категориальной палитры, проверенные на различимость
при дальтонизме; упорядоченные уровни (концентрация, цикл) — один тон от светлого к тёмному.
Помимо цвета тип пробы кодируется формой маркера, поэтому графики читаются и в ч/б печати.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TYPE_COLORS = {"control": SERIES[0], "free": SERIES[1], "immobilized": SERIES[2]}
TYPE_MARKERS = {"control": "s", "free": "o", "immobilized": "^"}
TYPE_LABELS = {"control": "контроль", "free": "свободный", "immobilized": "иммобилизованный"}
ORDINAL5 = ["#a3a29b", "#86857f", "#696863", "#4c4b47", "#2b2a28"]
ORDINAL3 = ["#a3a29b", "#696863", "#2b2a28"]
DIST_STYLE = {"norm": (SERIES[0], "-"), "laplace": (SERIES[1], "--"), "t": (SERIES[2], "-.")}
HIST_FILL = "#c9c8bf"

AXIS_LABELS = {
    "A": "Оптическая плотность A₅₈₀, отн. ед.",
    "D": "D = ln(A₁/A₂), безразм.",
    "C": "Концентрация крахмала C, % (г/100 мл)",
    "t": "Время реакции t, мин",
    "T": "Температура T, °C",
    "pH": "pH",
    "X": "Степень гидролиза X, %",
    "n": "Объём выборки n",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.labelcolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "figure.constrained_layout.use": True, "axes.formatter.use_locale": False,
    })


def _grid_for(x, pad: float = 0.15) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    span = hi - lo
    return np.linspace(lo - pad * span, hi + pad * span, 400)


def hist_with_fits(ax, x, fits=None, bins="fd", xlabel: str = "", title: str = "") -> None:
    """Гистограмма плотности + плотности подогнанных распределений."""
    ax.hist(x, bins=bins, density=True, color=HIST_FILL, edgecolor=SURFACE, linewidth=1.0, label="данные")
    if fits is not None:
        grid = _grid_for(x)
        for row in fits.itertuples():
            color, ls = DIST_STYLE[row.код]
            ax.plot(grid, getattr(stats, row.код).pdf(grid, *row.параметры), color=color, ls=ls, label=row.распределение)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Плотность вероятности")
    ax.set_title(title)
    ax.legend()


def ecdf_with_fits(ax, x, fits=None, xlabel: str = "", title: str = "") -> None:
    """Эмпирическая функция распределения против теоретических CDF."""
    ax.ecdf(x, color=INK, linewidth=1.4, label="ECDF (данные)")
    if fits is not None:
        grid = _grid_for(x)
        for row in fits.itertuples():
            color, ls = DIST_STYLE[row.код]
            ax.plot(grid, getattr(stats, row.код).cdf(grid, *row.параметры), color=color, ls=ls, label=row.распределение)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F(x) = P(X ≤ x)")
    ax.set_title(title)
    ax.legend(loc="lower right")


def qq_plot(ax, x, name: str = "norm", params=None, title: str = "", color: str = INK) -> None:
    """Q-Q-график: квантили выборки против квантилей теоретического закона.

    Если закон подходит, точки лежат на прямой y = x (для стандартизованных данных).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    dist = getattr(stats, name)
    params = params if params is not None else dist.fit(x)
    theo = dist.ppf(probs, *params)
    ax.plot(theo, x, ls="none", marker="o", markersize=4.5, color=color, alpha=0.85, label="данные")
    lim = [min(theo.min(), x.min()), max(theo.max(), x.max())]
    ax.plot(lim, lim, color=MUTED, linewidth=1.2, label="y = x")
    ax.set_xlabel("Теоретические квантили")
    ax.set_ylabel("Выборочные квантили")
    ax.set_title(title)
    ax.legend(loc="upper left")


def strip_by_group(ax, groups: dict[str, np.ndarray], colors: dict[str, str], markers: dict[str, str] | None = None,
                   ylabel: str = "", title: str = "", seed: int = 0) -> None:
    """Точки каждой группы + ящик с усами (медиана, квартили, 1,5·IQR)."""
    rng = np.random.default_rng(seed)
    names = list(groups)
    ax.boxplot([groups[k] for k in names], positions=range(len(names)), widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 1.4}, boxprops={"color": MUTED},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for i, k in enumerate(names):
        jitter = rng.uniform(-0.13, 0.13, len(groups[k]))
        ax.plot(i + jitter, groups[k], ls="none", marker=(markers or {}).get(k, "o"), markersize=4.5,
                color=colors[k], alpha=0.75)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def consistency_figure(summary, estimators: list[str], labels: dict[str, str], title: str):
    """Три панели на каждую оценку: коридор оценок, SD оценок (лог-лог), P(|θ̂ − θ| > ε)."""
    fig, axes = plt.subplots(len(estimators), 3, figsize=(11.5, 3.1 * len(estimators)), squeeze=False)
    for r, est in enumerate(estimators):
        part = summary[summary["оценка"] == est].sort_values("n")
        n = part["n"].to_numpy()
        ax = axes[r, 0]
        ax.fill_between(n, part["Q 2,5 %"], part["Q 97,5 %"], color=SERIES[0], alpha=0.18, linewidth=0,
                        label="95 % оценок")
        ax.plot(n, part["среднее оценок"], color=SERIES[0], marker="o", markersize=4, label="среднее оценок")
        ax.axhline(part["истинное"].iloc[0], color=INK, linewidth=1.1, label="истинное значение")
        ax.set_xscale("log")
        ax.set_xlabel(AXIS_LABELS["n"])
        ax.set_ylabel(labels[est])
        ax.set_title(f"{labels[est]}: коридор сужается")
        ax.legend(loc="upper right")
        ax = axes[r, 1]
        ax.loglog(n, part["SD оценок"], color=SERIES[1], marker="o", markersize=4, label="SD оценок (опыт)")
        ref = part["SD оценок"].iloc[0] * np.sqrt(n[0] / n)
        ax.loglog(n, ref, color=MUTED, ls="--", linewidth=1.2, label="∝ 1/√n")
        ax.set_xlabel(AXIS_LABELS["n"])
        ax.set_ylabel("SD оценки")
        ax.set_title("Разброс оценки ∝ 1/√n")
        ax.legend(loc="upper right")
        ax = axes[r, 2]
        ax.plot(n, part["P(|θ̂ − θ| > ε)"], color=SERIES[2], marker="o", markersize=4)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(AXIS_LABELS["n"])
        ax.set_ylabel("P(|θ̂ − θ| > ε)")
        ax.set_title(f"Вероятность ошибки > ε = {part['ε'].iloc[0]:.3g}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    return fig


def sampling_figure(sampling_table, population_label: str, title: str):
    """Дисперсия выборочного среднего: опыт и теория, с возвращением и без."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    n = sampling_table["n"].to_numpy()
    ax.plot(n, sampling_table["Var с возвр. (теория)"], color=SERIES[0], linewidth=1.4, label="с возвращением: σ²/n")
    ax.plot(n, sampling_table["Var с возвр. (опыт)"], color=SERIES[0], ls="none", marker="o", label="с возвращением: опыт")
    ax.plot(n, sampling_table["Var без возвр. (теория)"], color=SERIES[1], linewidth=1.4,
            label="без возвращения: σ²/n·(N−n)/(N−1)")
    ax.plot(n, sampling_table["Var без возвр. (опыт)"], color=SERIES[1], ls="none", marker="s", label="без возвращения: опыт")
    ax.set_yscale("log")
    ax.set_xlabel(AXIS_LABELS["n"])
    ax.set_ylabel(f"Дисперсия выборочного среднего, ({population_label})²")
    ax.set_title(title)
    ax.legend()
    return fig
