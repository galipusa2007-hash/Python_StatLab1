"""Статистические инструменты Lab01: описательная статистика, распределения, тесты,
состоятельность оценок, выборки из конечной совокупности."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multitest import multipletests

DIST_LABELS = {"norm": "нормальное", "laplace": "Лапласа", "t": "Стьюдента (t)", "lognorm": "логнормальное"}


# ----------------------------------------------------------------------------- описательная статистика
def describe(x, cv: bool = True) -> pd.Series:
    """Описательные статистики по ТЗ. Дисперсия и SD — несмещённые (ddof = 1),
    асимметрия и эксцесс — скорректированные на смещение; эксцесс избыточный (у нормального 0)."""
    x = np.asarray(x, dtype=float)
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    mean, sd = x.mean(), x.std(ddof=1)
    return pd.Series({
        "n": len(x), "среднее": mean, "медиана": median, "дисперсия": x.var(ddof=1), "SD": sd,
        "минимум": x.min(), "максимум": x.max(), "Q1": q1, "Q3": q3, "IQR": q3 - q1,
        "CV, %": 100 * sd / abs(mean) if cv else np.nan,
        "асимметрия": stats.skew(x, bias=False), "эксцесс": stats.kurtosis(x, bias=False),
    })


def describe_groups(groups: dict[str, np.ndarray], cv: bool = True) -> pd.DataFrame:
    return pd.DataFrame({name: describe(values, cv=cv) for name, values in groups.items()})


# ----------------------------------------------------------------------------- нормальность
def normality_tests(x) -> pd.DataFrame:
    """Четыре проверки H0: «выборка из нормального распределения» (параметры неизвестны)."""
    x = np.asarray(x, dtype=float)
    sw = stats.shapiro(x)
    dp = stats.normaltest(x)
    try:  # SciPy ≥ 1.17: p-value интерполируется по таблицам
        ad = stats.anderson(x, dist="norm", method="interpolate")
        ad_p = ad.pvalue
    except TypeError:  # старые SciPy: только критические значения → p-value недоступен
        ad = stats.anderson(x, dist="norm")
        ad_p = np.nan
    lf_stat, lf_p = lilliefors(x, dist="norm", pvalmethod="table")
    return pd.DataFrame([
        {"тест": "Шапиро–Уилк", "статистика": sw.statistic, "p-value": sw.pvalue,
         "что проверяет": "форма в целом; мощный при n < 2000"},
        {"тест": "Д'Агостино–Пирсон K²", "статистика": dp.statistic, "p-value": dp.pvalue,
         "что проверяет": "асимметрия + эксцесс"},
        {"тест": "Андерсон–Дарлинг A²", "статистика": ad.statistic, "p-value": ad_p,
         "что проверяет": "хвосты; p по таблицам, обрезан до 0,01–0,15"},
        {"тест": "Лиллиефорс (КС с оценёнными μ, σ)", "статистика": lf_stat, "p-value": lf_p,
         "что проверяет": "максимальное расхождение ECDF и CDF"},
    ])


# ----------------------------------------------------------------------------- подбор распределений
def _dist(name: str):
    return getattr(stats, name)


def ks_parametric_bootstrap(x, name: str, params, n_boot: int, rng) -> float:
    """p-value критерия Колмогорова–Смирнова, когда параметры оценены по той же выборке.

    Обычный КС-тест в этом случае слишком «мягкий» (p завышено), поэтому распределение
    статистики D получаем параметрическим бутстрепом: генерируем выборки из подогнанного
    закона, снова оцениваем параметры и считаем D. p = доля D* ≥ D_наблюдаемого.
    """
    x = np.asarray(x, dtype=float)
    dist = _dist(name)
    d_obs = stats.kstest(x, dist.cdf, args=params).statistic
    hits = 0
    for _ in range(n_boot):
        xb = dist.rvs(*params, size=len(x), random_state=rng)
        pb = dist.fit(xb)
        hits += stats.kstest(xb, dist.cdf, args=pb).statistic >= d_obs
    return (hits + 1) / (n_boot + 1)


def fit_candidates(x, names=("norm", "laplace", "t"), n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """Оценка параметров методом максимального правдоподобия и сравнение кандидатов.

    AIC = 2k − 2·lnL (меньше — лучше; разница ΔAIC > 2 уже заметна, > 10 — решающая).
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for name in names:
        dist = _dist(name)
        params = dist.fit(x)
        loglik = dist.logpdf(x, *params).sum()
        k = len(params)
        rows.append({
            "код": name, "распределение": DIST_LABELS[name], "параметры": params,
            "lnL": loglik, "AIC": 2 * k - 2 * loglik, "BIC": k * np.log(len(x)) - 2 * loglik,
            "D (КС)": stats.kstest(x, dist.cdf, args=params).statistic,
            "p (КС, бутстреп)": ks_parametric_bootstrap(x, name, params, n_boot, rng),
        })
    table = pd.DataFrame(rows)
    table["ΔAIC"] = table["AIC"] - table["AIC"].min()
    return table


def describe_params(fits: pd.DataFrame) -> pd.Series:
    """Параметры в читаемом виде: μ, σ для нормального; μ, b для Лапласа; ν, μ, s для t."""
    out = {}
    for row in fits.itertuples():
        p = row.параметры
        if row.код == "norm":
            out[row.распределение] = f"μ = {p[0]:.4f}, σ = {p[1]:.4f}"
        elif row.код == "laplace":
            out[row.распределение] = f"μ = {p[0]:.4f}, b = {p[1]:.4f}"
        elif row.код == "t":
            out[row.распределение] = f"ν = {p[0]:.2f}, μ = {p[1]:.4f}, s = {p[2]:.4f}"
    return pd.Series(out, name="оценки параметров (ММП)")


def compare_moments(x, fits: pd.DataFrame) -> pd.DataFrame:
    """Эмпирические характеристики против теоретических у подогнанных законов."""
    x = np.asarray(x, dtype=float)
    probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    cols = {"выборка": [x.mean(), x.std(ddof=1), stats.skew(x, bias=False), stats.kurtosis(x, bias=False),
                        *np.quantile(x, probs)]}
    for row in fits.itertuples():
        dist = _dist(row.код)
        m, v, s, k = dist.stats(*row.параметры, moments="mvsk")
        cols[row.распределение] = [float(m), float(np.sqrt(v)), float(s), float(k), *dist.ppf(probs, *row.параметры)]
    index = ["среднее", "SD", "асимметрия", "эксцесс", "Q 5 %", "Q 25 %", "медиана", "Q 75 %", "Q 95 %"]
    return pd.DataFrame(cols, index=index)


# ----------------------------------------------------------------------------- независимость
def runs_test(x) -> tuple[float, float]:
    """Тест серий Вальда–Вольфовица: случайно ли чередуются значения выше и ниже медианы."""
    z, p = runstest_1samp(np.asarray(x, dtype=float), cutoff="median", correction=True)
    return float(z), float(p)


def lag1_autocorrelation(x) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def permutation_corr(x, y, n_perm: int = 9999, seed: int = 0, method: str = "spearman") -> tuple[float, float]:
    """Корреляция и её p-value перестановочным тестом (без предположения о нормальности)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if method == "spearman":
        x, y = stats.rankdata(x), stats.rankdata(y)
    r_obs = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    return float(r_obs), float((np.sum(np.abs(perm) >= abs(r_obs)) + 1) / (n_perm + 1))


def partial_corr_by_group(x, y, groups) -> tuple[float, float]:
    """Корреляция после вычитания средних по группам (убирает общий фактор, например C)."""
    df = pd.DataFrame({"x": x, "y": y, "g": groups})
    rx = df["x"] - df.groupby("g")["x"].transform("mean")
    ry = df["y"] - df.groupby("g")["y"].transform("mean")
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def random_effects_icc(values, groups) -> pd.Series:
    """Однофакторная модель со случайным эффектом группы: y_ij = μ + s_i + e_ij.

    Метод моментов (ANOVA-оценки): τ² = (MSB − MSW)/n₀, σ²_e = MSW,
    ICC = τ²/(τ² + σ²_e) — доля дисперсии, общей для наблюдений одной группы
    (внутриклассовая корреляция: так коррелируют два наблюдения из одной серии).
    """
    df = pd.DataFrame({"y": np.asarray(values, dtype=float), "g": np.asarray(groups)})
    sizes = df.groupby("g")["y"].size()
    k, big_n = len(sizes), len(df)
    grand = df["y"].mean()
    msb = (sizes * (df.groupby("g")["y"].mean() - grand) ** 2).sum() / (k - 1)
    msw = ((df["y"] - df.groupby("g")["y"].transform("mean")) ** 2).sum() / (big_n - k)
    n0 = (big_n - (sizes ** 2).sum() / big_n) / (k - 1)
    tau2 = max((msb - msw) / n0, 0.0)
    anova = stats.f_oneway(*[g["y"].to_numpy() for _, g in df.groupby("g")])
    return pd.Series({"групп (серий)": k, "наблюдений": big_n, "F (ANOVA)": anova.statistic, "p (ANOVA)": anova.pvalue,
                      "τ (SD эффекта серии)": np.sqrt(tau2), "σ_e (SD внутри серии)": np.sqrt(msw),
                      "ICC": tau2 / (tau2 + msw)})


# ----------------------------------------------------------------------------- состоятельность
def consistency_sim(draw: Callable, estimators: dict[str, Callable], ns, n_rep: int, seed: int) -> pd.DataFrame:
    """Для каждого объёма n генерирует n_rep выборок и считает оценки (векторно по строкам)."""
    rng = np.random.default_rng(seed)
    parts = []
    for n in ns:
        samples = draw(rng, (n_rep, n))
        for name, estimator in estimators.items():
            parts.append(pd.DataFrame({"n": n, "оценка": name, "значение": estimator(samples)}))
    return pd.concat(parts, ignore_index=True)


def consistency_summary(sim: pd.DataFrame, truth: dict[str, float], eps: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (name, n), values in sim.groupby(["оценка", "n"])["значение"]:
        v = values.to_numpy()
        rows.append({"оценка": name, "n": n, "истинное": truth[name], "среднее оценок": v.mean(),
                     "смещение": v.mean() - truth[name], "SD оценок": v.std(ddof=1),
                     "Q 2,5 %": np.quantile(v, 0.025), "Q 97,5 %": np.quantile(v, 0.975),
                     "ε": eps[name], "P(|θ̂ − θ| > ε)": np.mean(np.abs(v - truth[name]) > eps[name])})
    return pd.DataFrame(rows)


def loglog_slope(summary: pd.DataFrame, estimator: str) -> float:
    """Наклон lg(SD оценок) от lg(n); для √n-состоятельных оценок он близок к −1/2."""
    part = summary[summary["оценка"] == estimator]
    return float(np.polyfit(np.log10(part["n"]), np.log10(part["SD оценок"]), 1)[0])


# ----------------------------------------------------------------------------- выборки
def _sample_stats(x, population: bool = False) -> dict:
    x = np.asarray(x, dtype=float)
    ddof = 0 if population else 1
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": len(x), "среднее": x.mean(), "дисперсия": x.var(ddof=ddof), "SD": x.std(ddof=ddof),
            "медиана": med, "Q1": q1, "Q3": q3, "минимум": x.min(), "максимум": x.max()}


def population_vs_samples(population, n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Одна выборка с возвращением и одна без возвращения против всей совокупности.

    У совокупности дисперсия считается с делителем N (σ²), у выборок — с n − 1 (s²).
    """
    population = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    idx_with = rng.choice(len(population), size=n, replace=True)
    idx_without = rng.choice(len(population), size=n, replace=False)
    with_r, without_r = population[idx_with], population[idx_without]
    table = pd.DataFrame({
        "совокупность (N)": _sample_stats(population, population=True),
        "с возвращением": _sample_stats(with_r),
        "без возвращения": _sample_stats(without_r),
    })
    table.loc["разных элементов"] = [len(population), len(np.unique(idx_with)), len(np.unique(idx_without))]
    table.loc["КС: p против совокупности"] = [np.nan, stats.ks_2samp(with_r, population).pvalue,
                                              stats.ks_2samp(without_r, population).pvalue]
    return table, with_r, without_r


def sampling_distribution(population, ns, n_rep: int, seed: int) -> pd.DataFrame:
    """Дисперсия выборочного среднего: эксперимент против теории.

    С возвращением:   Var(x̄) = σ²/n.
    Без возвращения:  Var(x̄) = σ²/n · (N − n)/(N − 1)   — поправка на конечность совокупности (FPC).
    """
    population = np.asarray(population, dtype=float)
    big_n = len(population)
    sigma2 = population.var(ddof=0)
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        with_r = rng.choice(population, size=(n_rep, n), replace=True).mean(axis=1)
        keys = rng.random((n_rep, big_n))
        idx = np.argsort(keys, axis=1)[:, :n]  # случайная перестановка → первые n без повторов
        without_r = population[idx].mean(axis=1)
        rows.append({"n": n, "доля N": n / big_n,
                     "Var с возвр. (опыт)": with_r.var(ddof=1), "Var с возвр. (теория)": sigma2 / n,
                     "Var без возвр. (опыт)": without_r.var(ddof=1),
                     "Var без возвр. (теория)": sigma2 / n * (big_n - n) / (big_n - 1)})
    return pd.DataFrame(rows)


def stratified_vs_simple(population, strata, n: int, n_rep: int, seed: int) -> pd.DataFrame:
    """Простая случайная выборка без возвращения против стратифицированной (пропорциональное размещение)."""
    population = np.asarray(population, dtype=float)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(strata, return_counts=True)
    alloc = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
    while alloc.sum() > n:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n:
        alloc[np.argmin(alloc / counts)] += 1
    weights = counts / counts.sum()
    srs = np.array([rng.choice(population, n, replace=False).mean() for _ in range(n_rep)])
    st = np.empty(n_rep)
    for r in range(n_rep):
        st[r] = sum(w * rng.choice(population[strata == lab], k, replace=False).mean()
                    for lab, k, w in zip(labels, alloc, weights))
    return pd.DataFrame({
        "схема": ["простая случайная", "стратифицированная"],
        "среднее оценок": [srs.mean(), st.mean()],
        "SD оценок": [srs.std(ddof=1), st.std(ddof=1)],
        "истинное среднее": [population.mean()] * 2,
    }), dict(zip(labels, alloc))


# ----------------------------------------------------------------------------- гипотезы
def paired_test(d, confidence: float = 0.95) -> pd.Series:
    """Парное сравнение по разностям d: t-тест и знаково-ранговый тест Уилкоксона."""
    d = np.asarray(d, dtype=float)
    n, mean, sd = len(d), d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * se
    return pd.Series({
        "n пар": n, "средняя разность": mean, "SD разностей": sd,
        f"ДИ {int(confidence * 100)} % — низ": mean - half, f"ДИ {int(confidence * 100)} % — верх": mean + half,
        "медиана разностей": np.median(d), "d_z (размер эффекта)": mean / sd,
        "p (парный t-тест)": stats.ttest_1samp(d, 0.0).pvalue,
        "p (Уилкоксон)": stats.wilcoxon(d).pvalue,
        "p (Шапиро для разностей)": stats.shapiro(d).pvalue,
    })


def permutation_oneway(values, groups, n_perm: int = 9999, seed: int = 0) -> pd.Series:
    """Однофакторный дисперсионный анализ: классический F-тест + перестановочный p-value.

    Перестановочный тест не требует нормальности: перемешиваем метки групп и смотрим,
    как часто случайная разметка даёт F не меньше наблюдаемого.
    """
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    labels = np.unique(groups)

    def f_stat(g):
        return stats.f_oneway(*[values[g == lab] for lab in labels]).statistic

    f_obs = f_stat(groups)
    rng = np.random.default_rng(seed)
    perm = np.array([f_stat(rng.permutation(groups)) for _ in range(n_perm)])
    grand = values.mean()
    ss_between = sum(len(values[groups == lab]) * (values[groups == lab].mean() - grand) ** 2 for lab in labels)
    ss_total = ((values - grand) ** 2).sum()
    return pd.Series({"групп": len(labels), "наблюдений": len(values), "F": f_obs,
                      "p (F-тест)": stats.f_oneway(*[values[groups == lab] for lab in labels]).pvalue,
                      "p (перестановки)": (np.sum(perm >= f_obs) + 1) / (n_perm + 1),
                      "η² (доля дисперсии от фактора)": ss_between / ss_total})


def holm(pvalues, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Поправка Холма на множественные сравнения: возвращает (отвергнуть H0?, скорректированные p)."""
    reject, p_adj, _, _ = multipletests(np.asarray(pvalues, dtype=float), alpha=alpha, method="holm")
    return reject, p_adj
