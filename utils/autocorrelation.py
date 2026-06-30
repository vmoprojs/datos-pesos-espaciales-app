"""Estadísticos de autocorrelación espacial global.

La implementación es intencionalmente explícita para que pueda usarse en clase
sin depender de `esda`. Se apoya en la misma matriz W que construye la app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.weights import row_standardize


@dataclass(frozen=True)
class GlobalAutocorrelationResult:
    statistic: str
    observed: float
    expected: float
    z_sim: float
    p_sim: float
    simulations: np.ndarray
    interpretation: str


def _safe_weights(W: np.ndarray, standardize: bool = True) -> np.ndarray:
    matrix = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    return row_standardize(matrix) if standardize else matrix


def standardized_values(values: pd.Series | np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    centered = y - y.mean()
    std = centered.std(ddof=0)
    if np.isclose(std, 0):
        return np.zeros_like(centered, dtype=float)
    return centered / std


def moran_i(values: pd.Series | np.ndarray, W: np.ndarray, standardize: bool = True) -> float:
    matrix = _safe_weights(W, standardize)
    y = np.asarray(values, dtype=float)
    z = y - y.mean()
    denominator = float(np.sum(z**2))
    s0 = float(matrix.sum())
    if np.isclose(denominator, 0) or np.isclose(s0, 0):
        return float("nan")
    numerator = float(np.sum(matrix * np.outer(z, z)))
    return (len(y) / s0) * (numerator / denominator)


def geary_c(values: pd.Series | np.ndarray, W: np.ndarray, standardize: bool = True) -> float:
    matrix = _safe_weights(W, standardize)
    y = np.asarray(values, dtype=float)
    z = y - y.mean()
    denominator = float(np.sum(z**2))
    s0 = float(matrix.sum())
    if np.isclose(denominator, 0) or np.isclose(s0, 0):
        return float("nan")
    squared_differences = (y[:, None] - y[None, :]) ** 2
    numerator = float(np.sum(matrix * squared_differences))
    return ((len(y) - 1) / (2 * s0)) * (numerator / denominator)


def permutation_test(
    values: pd.Series | np.ndarray,
    W: np.ndarray,
    statistic: str,
    permutations: int = 499,
    standardize: bool = True,
    seed: int = 2026,
) -> GlobalAutocorrelationResult:
    y = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if statistic == "Moran's I":
        observed = moran_i(y, W, standardize)
        expected = -1 / (len(y) - 1) if len(y) > 1 else float("nan")
        stat_fn = moran_i
    elif statistic == "Geary's C":
        observed = geary_c(y, W, standardize)
        expected = 1.0
        stat_fn = geary_c
    else:
        raise ValueError(f"Estadístico no soportado: {statistic}")

    simulations = np.array(
        [stat_fn(rng.permutation(y), W, standardize) for _ in range(int(permutations))],
        dtype=float,
    )
    valid = simulations[np.isfinite(simulations)]
    if not np.isfinite(observed) or len(valid) == 0:
        z_sim = float("nan")
        p_sim = float("nan")
    else:
        sim_std = valid.std(ddof=1) if len(valid) > 1 else 0.0
        z_sim = (observed - valid.mean()) / sim_std if not np.isclose(sim_std, 0) else float("nan")
        distance_observed = abs(observed - expected)
        p_sim = (np.sum(np.abs(valid - expected) >= distance_observed) + 1) / (len(valid) + 1)

    return GlobalAutocorrelationResult(
        statistic=statistic,
        observed=float(observed),
        expected=float(expected),
        z_sim=float(z_sim),
        p_sim=float(p_sim),
        simulations=simulations,
        interpretation=interpret_global_statistic(statistic, float(observed), float(p_sim)),
    )


def moran_scatter_data(data: pd.DataFrame, variable: str, W: np.ndarray, standardize: bool = True) -> pd.DataFrame:
    matrix = _safe_weights(W, standardize)
    z = standardized_values(data[variable])
    lag_z = matrix @ z
    output = data[["id", "nombre", variable]].copy()
    output["z"] = z
    output["W_z"] = lag_z
    output["cuadrante_moran"] = [
        classify_moran_quadrant(own, lag) for own, lag in zip(output["z"], output["W_z"])
    ]
    return output


def local_moran_values(values: pd.Series | np.ndarray, W: np.ndarray, standardize: bool = True) -> np.ndarray:
    matrix = _safe_weights(W, standardize)
    y = np.asarray(values, dtype=float)
    z = y - y.mean()
    m2 = float(np.sum(z**2) / len(z))
    if np.isclose(m2, 0):
        return np.full_like(z, np.nan, dtype=float)
    return (z * (matrix @ z)) / m2


def local_geary_values(values: pd.Series | np.ndarray, W: np.ndarray, standardize: bool = True) -> np.ndarray:
    matrix = _safe_weights(W, standardize)
    z = standardized_values(values)
    return np.sum(matrix * (z[:, None] - z[None, :]) ** 2, axis=1)


def local_lisa_statistics(
    data: pd.DataFrame,
    variable: str,
    W: np.ndarray,
    permutations: int = 499,
    alpha: float = 0.05,
    standardize: bool = True,
    seed: int = 2028,
) -> pd.DataFrame:
    """Calcula LISA didáctico con pseudo p-valores por permutación.

    El procedimiento permuta la variable entre territorios y mantiene fija W. Es
    una aproximación pedagógica a la inferencia empírica usada en PySAL/esda.
    """

    y = data[variable].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    observed_i = local_moran_values(y, W, standardize)
    observed_c = local_geary_values(y, W, standardize)
    sim_i = np.vstack([local_moran_values(rng.permutation(y), W, standardize) for _ in range(int(permutations))])
    sim_c = np.vstack([local_geary_values(rng.permutation(y), W, standardize) for _ in range(int(permutations))])

    with np.errstate(invalid="ignore", divide="ignore"):
        p_moran = (np.sum(np.abs(sim_i) >= np.abs(observed_i), axis=0) + 1) / (sim_i.shape[0] + 1)
        p_geary_high = (np.sum(sim_c >= observed_c, axis=0) + 1) / (sim_c.shape[0] + 1)
        std_i = sim_i.std(axis=0, ddof=1)
        z_i = np.divide(observed_i - sim_i.mean(axis=0), std_i, out=np.zeros_like(observed_i), where=std_i != 0)

    scatter = moran_scatter_data(data, variable, W, standardize)
    output = data[["id", "nombre", variable]].copy()
    output["z"] = scatter["z"]
    output["W_z"] = scatter["W_z"]
    output["I_local"] = observed_i
    output["C_local_geary"] = observed_c
    output["p_moran_sim"] = p_moran
    output["z_moran_sim"] = z_i
    output["p_geary_alto_sim"] = p_geary_high
    output["cuadrante_moran"] = scatter["cuadrante_moran"]
    output["significativo"] = output["p_moran_sim"] < alpha
    output["cluster_lisa"] = [
        quadrant if significant else "No significativo"
        for quadrant, significant in zip(output["cuadrante_moran"], output["significativo"])
    ]
    return output


def classify_moran_quadrant(z: float, lag_z: float) -> str:
    if np.isclose(z, 0) or np.isclose(lag_z, 0):
        return "Indefinido"
    if z > 0 and lag_z > 0:
        return "Alto-Alto"
    if z < 0 and lag_z < 0:
        return "Bajo-Bajo"
    if z > 0 and lag_z < 0:
        return "Alto-Bajo"
    return "Bajo-Alto"


def interpret_global_statistic(statistic: str, observed: float, p_sim: float) -> str:
    if not np.isfinite(observed):
        return "No se puede calcular porque la variable no tiene variación suficiente o W no tiene enlaces."

    significance = "estadísticamente distinguible de aleatoriedad" if p_sim < 0.05 else "no concluyente frente a aleatoriedad"
    if statistic == "Moran's I":
        if observed > 0.05:
            pattern = "autocorrelación positiva: valores similares tienden a ubicarse cerca"
        elif observed < -0.05:
            pattern = "autocorrelación negativa: valores altos tienden a estar cerca de valores bajos"
        else:
            pattern = "un patrón cercano a aleatoriedad espacial"
        return f"Moran's I sugiere {pattern}; con permutaciones, el resultado es {significance}."

    if observed < 1:
        pattern = "similitud espacial entre vecinos"
    elif observed > 1:
        pattern = "disimilitud o alternancia espacial entre vecinos"
    else:
        pattern = "un patrón compatible con aleatoriedad espacial"
    return f"Geary's C indica {pattern}; con permutaciones, el resultado es {significance}."


def global_autocorrelation_table(
    moran: GlobalAutocorrelationResult,
    geary: GlobalAutocorrelationResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Estadístico": "Moran's I",
                "Valor observado": moran.observed,
                "Valor esperado": moran.expected,
                "z simulado": moran.z_sim,
                "p simulado": moran.p_sim,
                "Lectura rápida": "I > 0 agrupa similares; I < 0 alterna altos y bajos.",
            },
            {
                "Estadístico": "Geary's C",
                "Valor observado": geary.observed,
                "Valor esperado": geary.expected,
                "z simulado": geary.z_sim,
                "p simulado": geary.p_sim,
                "Lectura rápida": "C < 1 sugiere similitud; C > 1 sugiere disimilitud.",
            },
        ]
    )


def lisa_counts_table(local_stats: pd.DataFrame) -> pd.DataFrame:
    order = ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto", "No significativo"]
    counts = local_stats["cluster_lisa"].value_counts().reindex(order, fill_value=0)
    return counts.rename_axis("Clase LISA").reset_index(name="n_territorios")


def interpret_lisa_row(row: pd.Series, variable: str) -> str:
    territory = row.get("nombre", row.get("id", "el territorio"))
    variable_label = variable.replace("_", " ")
    cluster = row["cluster_lisa"]
    p_value = float(row["p_moran_sim"])

    if cluster == "No significativo":
        return (
            f"{territory} no muestra un patrón local estadísticamente claro para {variable_label} "
            f"con el umbral seleccionado (p_sim = {p_value:.3f})."
        )
    if cluster == "Alto-Alto":
        return (
            f"{territory} es un clúster Alto-Alto: valor alto rodeado de vecinos altos en {variable_label}. "
            f"Puede interpretarse como concentración territorial del fenómeno (p_sim = {p_value:.3f})."
        )
    if cluster == "Bajo-Bajo":
        return (
            f"{territory} es un clúster Bajo-Bajo: valor bajo rodeado de vecinos bajos en {variable_label}. "
            f"Indica continuidad espacial de valores bajos (p_sim = {p_value:.3f})."
        )
    if cluster == "Alto-Bajo":
        return (
            f"{territory} es un posible atípico Alto-Bajo: valor alto rodeado de vecinos bajos en {variable_label} "
            f"(p_sim = {p_value:.3f})."
        )
    if cluster == "Bajo-Alto":
        return (
            f"{territory} es un posible atípico Bajo-Alto: valor bajo rodeado de vecinos altos en {variable_label} "
            f"(p_sim = {p_value:.3f})."
        )
    return f"{territory} queda en la clase {cluster}."
