"""Загрузка и разметка данных Lab01.

Исходные таблицы выгружены из Google Sheets: десятичный разделитель — запятая,
заголовки на русском. Здесь они приводятся к числам и к англоязычным именам столбцов,
а каждая строка получает метки эксперимента, тройки и позиции в тройке.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OD_RAW = DATA_DIR / "raw" / "od_statistika_list1.csv"            # «Статистика - Лист1.csv»
TABLE_RAW = DATA_DIR / "raw" / "hydrolysis_tablichki_list2.csv"  # «Таблички - Лист2.csv»

SAMPLE_TYPES = {
    "контрольная проба": "control",
    "свободная проба": "free",
    "иммобилизированная проба": "immobilized",
}
TYPE_LABELS = {"control": "контроль", "free": "свободный", "immobilized": "иммобилизованный"}

# Пять экспериментов в файле ОП: диапазоны номеров строк «№» (включительно).
# В каждом эксперименте меняли один фактор, остальные держали постоянными.
EXPERIMENTS = {
    "conc": (1, 330, "Концентрация крахмала 0,25–1,25 %; 30 °C; pH 4,7; t = 5–30 мин"),
    "kin90": (331, 390, "Кинетика при 90 °C; C = 1 %; pH 4,7; t = 5–30 мин"),
    "kin60": (391, 480, "Кинетика при 60 °C; C = 1 %; pH 4,7; t = 5–30 мин"),
    "temp": (481, 534, "Температура 25–90 °C; C = 1 %; pH 4,7; t = 30 мин"),
    "ph": (535, 588, "pH 3,8–5,8; 30 °C; C = 1 %; t = 30 мин"),
}
# Те же блоки в «Табличках»: номера строк файла (строка 1 — заголовок).
TABLE_BLOCKS = {"conc": (2, 81), "kin60": (82, 105), "kin90": (106, 119), "temp": (120, 137), "ph": (138, 149)}

FACTORS = ["temperature_c", "ph", "substrate_pct", "time_min"]
CONDITION_KEYS = ["experiment", *FACTORS, "sample_type", "cycle"]

OD_COLUMNS = ["№", "Оптическая плотность", "температуры", "pH", "концентрации субстрата",
              "время реакции, мин", "тип образца", "номер цикла"]
TABLE_COLUMNS = ["№", "Степень гидролиза", "Погрешность", "температуры", "pH", "концентрации субстрата",
                 "время реакции, мин", "тип образца", "номер цикла"]


def _to_number(values: pd.Series) -> pd.Series:
    """Строки с десятичной запятой → числа: '0,32904' → 0.32904. Ошибка, если что-то не число."""
    return pd.to_numeric(values.astype(str).str.strip().str.replace(",", ".", regex=False), errors="raise")


def _experiment_of_row(row_id: int) -> str:
    for name, (first, last, _) in EXPERIMENTS.items():
        if first <= row_id <= last:
            return name
    raise ValueError(f"строка {row_id} не попала ни в один эксперимент")


def load_od(path: Path = OD_RAW) -> pd.DataFrame:
    """Датасет 1: оптическая плотность, одна строка = одно значение ОП.

    Добавляет:
        experiment  — код эксперимента (conc, kin90, kin60, temp, ph);
        triplet_id  — номер тройки (условие опыта), 0…195;
        position    — позиция значения в тройке (1, 2, 3);
        is_measured — True для позиций 1–2 (измеренные сосуды), False для позиции 3.
    """
    raw = pd.read_csv(path, dtype=str)
    if list(raw.columns) != OD_COLUMNS:
        raise ValueError(f"неожиданные столбцы: {list(raw.columns)}")
    od = pd.DataFrame({
        "row_id": _to_number(raw["№"]).astype(int),
        "absorbance": _to_number(raw["Оптическая плотность"]),
        "temperature_c": _to_number(raw["температуры"]).astype(int),
        "ph": _to_number(raw["pH"]),
        "substrate_pct": _to_number(raw["концентрации субстрата"]),
        "time_min": _to_number(raw["время реакции, мин"]).astype(int),
        "sample_type": raw["тип образца"].str.strip().map(SAMPLE_TYPES),
        "cycle": _to_number(raw["номер цикла"]).astype(int),
    })
    if od["sample_type"].isna().any():
        raise ValueError("в столбце «тип образца» есть неизвестное значение")
    if not od["row_id"].equals(pd.Series(np.arange(1, len(od) + 1), name="row_id")):
        raise ValueError("номера строк «№» должны идти подряд с 1")
    od["experiment"] = od["row_id"].map(_experiment_of_row)
    od["triplet_id"] = (od["row_id"] - 1) // 3
    od["position"] = (od["row_id"] - 1) % 3 + 1
    # внутри тройки все условия обязаны совпадать — иначе разметка по три строки неверна
    varying = od.groupby("triplet_id")[CONDITION_KEYS].nunique().gt(1).any(axis=1)
    if varying.any():
        raise ValueError(f"в тройках {list(varying[varying].index)} меняются условия")
    od["is_measured"] = od["position"] <= 2
    return od


def load_table(path: Path = TABLE_RAW) -> pd.DataFrame:
    """Датасет 2 в исходном виде: степень гидролиза X, % и столбец «Погрешность», %."""
    raw = pd.read_csv(path, dtype=str)
    if list(raw.columns) != TABLE_COLUMNS:
        raise ValueError(f"неожиданные столбцы: {list(raw.columns)}")
    table = pd.DataFrame({
        "file_line": np.arange(len(raw)) + 2,
        "x_pct": _to_number(raw["Степень гидролиза"]),
        "err_pct": _to_number(raw["Погрешность"]),
        "temperature_c": _to_number(raw["температуры"]).astype(int),
        "ph": _to_number(raw["pH"]),
        "substrate_pct": _to_number(raw["концентрации субстрата"]),
        "time_min": _to_number(raw["время реакции, мин"]).astype(int),
        "sample_type": raw["тип образца"].str.strip().map(SAMPLE_TYPES),
        "cycle": _to_number(raw["номер цикла"]).astype(int),
    })
    table["table_block"] = table["file_line"].map(
        lambda line: next(name for name, (a, b) in TABLE_BLOCKS.items() if a <= line <= b))
    return table


def experiments_overview(od: pd.DataFrame) -> pd.DataFrame:
    """Сводка по экспериментам: что меняли, сколько условий и значений."""
    rows = []
    for name, (first, last, description) in EXPERIMENTS.items():
        part = od[od["experiment"] == name]
        rows.append({
            "эксперимент": name,
            "описание": description,
            "строки №": f"{first}–{last}",
            "условий (троек)": part["triplet_id"].nunique(),
            "контроль": part.loc[part.sample_type == "control", "triplet_id"].nunique(),
            "свободный": part.loc[part.sample_type == "free", "triplet_id"].nunique(),
            "иммобилизованный": part.loc[part.sample_type == "immobilized", "triplet_id"].nunique(),
            "циклы": ", ".join(map(str, sorted(part["cycle"].unique()))),
        })
    return pd.DataFrame(rows)


def add_series(df: pd.DataFrame) -> pd.DataFrame:
    """Серия = одна постановка опыта с общим приготовлением растворов.

    В концентрационном эксперименте каждая концентрация готовилась отдельно, поэтому это
    пять серий («conc 0.25 %» … «conc 1.25 %»); остальные эксперименты — по одной серии.
    Итого 9 серий. Серия важна как единица анализа: сосуды одной серии делят общие
    ошибки приготовления (см. раздел Independence в СВ-1).
    """
    out = df.copy()
    out["series"] = [f"conc {c:g} %" if e == "conc" else e for e, c in zip(out["experiment"], out["substrate_pct"])]
    return out


def measured_pairs(od: pd.DataFrame) -> pd.DataFrame:
    """Одна строка = одно условие: два измеренных сосуда (позиции 1 и 2).

    Столбцы:
        a1, a2     — ОП первого и второго сосуда;
        mean       — среднее двух сосудов;
        sd         — выборочное SD по двум значениям, |a1 − a2|/√2;
        log_ratio  — D = ln(a1/a2), случайная составляющая (см. ноутбук, СВ-1).
    """
    measured = od[od["is_measured"]]
    wide = measured.pivot(index="triplet_id", columns="position", values="absorbance").rename(columns={1: "a1", 2: "a2"})
    info = od.groupby("triplet_id")[CONDITION_KEYS].first()
    first_row = od.groupby("triplet_id")["row_id"].min().rename("row_first")
    pairs = info.join(first_row).join(wide).reset_index()
    pairs["mean"] = (pairs["a1"] + pairs["a2"]) / 2
    pairs["sd"] = (pairs["a1"] - pairs["a2"]).abs() / np.sqrt(2)
    pairs["log_ratio"] = np.log(pairs["a1"] / pairs["a2"])
    return pairs


def recompute_hydrolysis(pairs: pd.DataFrame) -> pd.DataFrame:
    """Пересчёт датасета 2 по измеренным сосудам.

    X = 100 % · (1 − Ā_п / Ā_к), где Ā — среднее двух сосудов, контроль — из того же
    эксперимента при тех же T, pH, C, t. Сосуды контроля и пробы не связаны, поэтому
    для каждого сосуда пробы используется средний контроль: X_j = 100 % · (1 − A_п,j / Ā_к).

    Погрешность — закон распространения ошибок (первый порядок, независимые Ā_п и Ā_к):
        u(X) = 100 % · (Ā_п/Ā_к) · √[(u_п/Ā_п)² + (u_к/Ā_к)²],  u = SD/√2 — стандартная ошибка среднего двух.
    """
    key = ["experiment", *FACTORS]
    control = pairs[pairs["sample_type"] == "control"][key + ["triplet_id", "a1", "a2", "mean", "sd"]]
    samples = pairs[pairs["sample_type"] != "control"]
    merged = samples.merge(control, on=key, suffixes=("", "_ctrl"), validate="many_to_one")
    ratio = merged["mean"] / merged["mean_ctrl"]
    u_sample = merged["sd"] / np.sqrt(2)
    u_ctrl = merged["sd_ctrl"] / np.sqrt(2)
    merged["x_pct"] = 100 * (1 - ratio)
    merged["u_x_pct"] = 100 * ratio * np.sqrt((u_sample / merged["mean"]) ** 2 + (u_ctrl / merged["mean_ctrl"]) ** 2)
    merged["x_vessel1"] = 100 * (1 - merged["a1"] / merged["mean_ctrl"])
    merged["x_vessel2"] = 100 * (1 - merged["a2"] / merged["mean_ctrl"])
    columns = ["triplet_id", "experiment", *FACTORS, "sample_type", "cycle", "mean", "mean_ctrl",
               "x_pct", "u_x_pct", "x_vessel1", "x_vessel2"]
    return merged[columns].rename(columns={"mean": "a_sample", "mean_ctrl": "a_control"})


def all_triplets(od: pd.DataFrame) -> pd.DataFrame:
    """Все три записи каждой тройки, включая значения, полученные формулой."""
    wide = od.pivot(index="triplet_id", columns="position", values="absorbance").rename(columns={1: "a1", 2: "a2", 3: "a3"})
    info = od.groupby("triplet_id")[CONDITION_KEYS + ["series"]].first()
    out = info.join(wide).reset_index()
    out["mean"] = out[["a1", "a2", "a3"]].mean(axis=1)
    out["sd"] = out[["a1", "a2", "a3"]].std(axis=1, ddof=1)
    return out

def hydrolysis_all(triplets: pd.DataFrame) -> pd.DataFrame:
    """Гидролиз X=100(1-Aпр/Aконтр) из средних ВСЕХ ТРЁХ записей.

    Распространение ошибок дано как описательная условная оценка: вычисленные
    компоненты троек не являются независимыми измерениями, поэтому SE по n=3
    нельзя считать корректной физической неопределённостью.
    """
    key = ["experiment", *FACTORS]
    controls = triplets.loc[triplets.sample_type == "control", key + ["mean", "sd"]].rename(columns={"mean":"a_control", "sd":"sd_control"})
    samples = triplets.loc[triplets.sample_type != "control"].copy()
    out = samples.merge(controls, on=key, validate="many_to_one")
    out["a_sample"] = out["mean"]
    out["x_pct"] = 100 * (1 - out["a_sample"] / out["a_control"])
    out["descriptive_spread_pct"] = 100 * np.sqrt((out["sd"] / out["a_control"])**2 + (out["a_sample"]*out["sd_control"] / out["a_control"]**2)**2)
    return out
