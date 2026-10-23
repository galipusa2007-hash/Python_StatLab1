"""Проверка данных: досчитанная третья повторность и сверка датасета 2 с датасетом 1."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .data import CONDITION_KEYS, FACTORS

KIND_LABELS = {
    "mean": "A₃ = (A₁ + A₂)/2",
    "mirror": "A₃ = 2·A₁ − A₂",
    "mirror_2": "A₃ = 2·A₂ − A₁",
    "unlinked": "не связано формулой",
}


def classify_triplets(od: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Для каждой тройки проверяет, не вычисляется ли третье значение из первых двух.

    tol = 1e-9 — это проверка на точное равенство: значения записаны с 5–6 знаками,
    поэтому совпадение до 10⁻⁹ возможно только при вычислении по формуле.
    """
    rows = []
    for triplet_id, group in od.sort_values("row_id").groupby("triplet_id"):
        a1, a2, a3 = group["absorbance"].to_numpy()
        if abs(a3 - (a1 + a2) / 2) <= tol:
            kind = "mean"
        elif abs(a3 - (2 * a1 - a2)) <= tol:
            kind = "mirror"
        elif abs(a3 - (2 * a2 - a1)) <= tol:
            kind = "mirror_2"
        else:
            kind = "unlinked"
        rows.append({"triplet_id": triplet_id, **group.iloc[0][CONDITION_KEYS].to_dict(),
                     "row_first": int(group["row_id"].min()), "a1": a1, "a2": a2, "a3": a3, "third_kind": kind})
    return pd.DataFrame(rows)


def expected_chance_coincidences(sd_per_triplet: np.ndarray, tol: float = 6e-6,
                                 n_sim: int = 20_000, seed: int = 0) -> float:
    """Сколько троек «одно значение = среднее двух других» дала бы чистая случайность.

    Для каждой тройки моделируем n_sim троек независимых N(0, σ²) с σ этой тройки и считаем
    долю случаев, когда |A_k − (сумма двух других)/2| ≤ tol хотя бы для одного k.
    Сумма долей по тройкам = ожидаемое число случайных совпадений во всём файле.
    """
    rng = np.random.default_rng(seed)
    expected = 0.0
    for sd in sd_per_triplet:
        x = rng.normal(0.0, sd, size=(n_sim, 3))
        others_mean = (x.sum(axis=1, keepdims=True) - x) / 2
        expected += (np.abs(x - others_mean) <= tol).any(axis=1).mean()
    return expected


def _subsets(values: np.ndarray) -> dict[str, np.ndarray]:
    return {"all3": values, "p12": values[[0, 1]], "p13": values[[0, 2]], "p23": values[[1, 2]]}


def _avedev_rel(values: np.ndarray) -> float:
    """Формула столбца «Погрешность»: =СРОТКЛ(A)/A_ср*100 (СРОТКЛ = AVEDEV)."""
    return 100 * np.mean(np.abs(values - values.mean())) / values.mean()


def audit_table(table: pd.DataFrame, triplets: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Сверяет каждую строку «Табличек» с ОП.

    status:
        exact             — X получается из текущих троек целиком;
        two_of_three      — X получается, только если из тройки пробы или контроля убрать одно значение;
        not_reproducible  — ни один вариант не даёт X.
    err_basis — из каких значений пробы получается «Погрешность» по формуле СРОТКЛ/среднее·100.
    """
    vals = {r.triplet_id: np.array([r.a1, r.a2, r.a3]) for r in triplets.itertuples()}
    out = []
    for row in table.itertuples():
        same = (triplets["temperature_c"] == row.temperature_c) & (triplets["ph"] == row.ph) & \
               (triplets["substrate_pct"] == row.substrate_pct) & (triplets["time_min"] == row.time_min)
        samples = triplets[same & (triplets["sample_type"] == row.sample_type) & (triplets["cycle"] == row.cycle)]
        best = None  # (rank, sample triplet, control triplet)
        for s, c in itertools.product(samples.itertuples(), triplets[same & (triplets["sample_type"] == "control")].itertuples()):
            if s.experiment != c.experiment:
                continue
            for (sn, sv), (cn, cv) in itertools.product(_subsets(vals[s.triplet_id]).items(), _subsets(vals[c.triplet_id]).items()):
                if abs(100 * (1 - sv.mean() / cv.mean()) - row.x_pct) <= tol:
                    rank = 0 if (sn == "all3" and cn == "all3") else 1
                    if best is None or rank < best[0]:
                        best = (rank, s, c)
        if best is not None:
            status = "exact" if best[0] == 0 else "two_of_three"
            s = best[1]
        else:
            status = "not_reproducible"
            in_block = samples[samples["experiment"] == row.table_block]
            s = (in_block if len(in_block) else samples).iloc[0]
        sample_vals = vals[s.triplet_id]
        basis = next((name for name, v in _subsets(sample_vals).items() if abs(_avedev_rel(v) - row.err_pct) <= tol), "none")
        out.append({"file_line": row.file_line, "table_block": row.table_block, "source_experiment": s.experiment,
                    "sample_triplet": s.triplet_id, "x_pct": row.x_pct, "err_pct": row.err_pct,
                    "status": status, "err_basis": basis})
    return pd.DataFrame(out)


def table_duplicates(table: pd.DataFrame) -> pd.DataFrame:
    """Строки с одинаковыми X и «Погрешностью» — одно измерение, записанное дважды."""
    dup = table[table.duplicated(subset=["x_pct", "err_pct"], keep=False)]
    return dup.sort_values(["x_pct", "file_line"])


def _matched_free_immobilized(triplets: pd.DataFrame) -> list[tuple]:
    key = ["experiment", *FACTORS]
    free = triplets[triplets["sample_type"] == "free"]
    imm = triplets[(triplets["sample_type"] == "immobilized") & (triplets["cycle"] == 1)]
    merged = free.merge(imm, on=key, suffixes=("_f", "_i"))
    return [(np.array([r.a1_f, r.a2_f, r.a3_f]), np.array([r.a1_i, r.a2_i, r.a3_i])) for r in merged.itertuples()]


def impact_of_third_value(triplets: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Насколько меняются результаты, если считать третье значение независимым сосудом."""
    v3 = triplets[["a1", "a2", "a3"]].to_numpy()
    log3, log2 = np.log(v3), np.log(v3[:, :2])
    pooled = pd.DataFrame({
        "как считаем": ["тройка целиком (n = 3)", "два измеренных сосуда (n = 2)"],
        "пулированное SD ln A": [np.sqrt(log3.var(axis=1, ddof=1).mean()), np.sqrt(log2.var(axis=1, ddof=1).mean())],
        "степеней свободы": [2 * len(v3), len(v3)],
    })
    ratio_rows = []
    for kind in ["mean", "mirror"]:
        part = triplets[triplets["third_kind"] == kind][["a1", "a2", "a3"]].to_numpy()
        sd_ratio = part.std(axis=1, ddof=1) / part[:, :2].std(axis=1, ddof=1)
        err_ratio = np.array([_avedev_rel(v) / _avedev_rel(v[:2]) for v in part])
        ratio_rows.append({"тип третьего значения": KIND_LABELS[kind], "троек": len(part),
                           "SD тройки / SD пары": sd_ratio.mean(),
                           "«Погрешность» по 3 / по 2": f"{err_ratio.min():.3f}–{err_ratio.max():.3f}"})
    pairs = _matched_free_immobilized(triplets)
    p3 = np.array([stats.ttest_ind(f, i, equal_var=False).pvalue for f, i in pairs])
    p2 = np.array([stats.ttest_ind(f[:2], i[:2], equal_var=False).pvalue for f, i in pairs])
    tests = pd.DataFrame({
        "как считаем": ["тройка целиком (n = 3)", "два измеренных сосуда (n = 2)"],
        "условий": [len(pairs)] * 2,
        f"значимых различий (p < {alpha})": [int((p3 < alpha).sum()), int((p2 < alpha).sum())],
    })
    flips = int(((p3 < alpha) != (p2 < alpha)).sum())
    return {"pooled_sd": pooled, "ratios": pd.DataFrame(ratio_rows), "welch": tests, "welch_flips": flips}
