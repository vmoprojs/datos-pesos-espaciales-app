"""Construcción y diagnóstico de matrices de pesos espaciales."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightResult:
    """Resultado completo para una matriz de pesos."""

    W: np.ndarray
    labels: list[str]
    method: str
    description: str
    parameter_label: str = ""


def get_labels(data: pd.DataFrame) -> list[str]:
    if "id" in data.columns:
        return data["id"].astype(str).tolist()
    return [str(i) for i in range(len(data))]


def get_names(data: pd.DataFrame) -> list[str]:
    if "nombre" in data.columns:
        return data["nombre"].astype(str).tolist()
    return get_labels(data)


def get_centroids(data: pd.DataFrame) -> np.ndarray:
    """Extrae centroides desde columnas explícitas o desde x/y."""

    if {"centroide_x", "centroide_y"}.issubset(data.columns):
        return data[["centroide_x", "centroide_y"]].to_numpy(dtype=float)
    if {"x", "y"}.issubset(data.columns):
        return data[["x", "y"]].to_numpy(dtype=float)
    raise ValueError("La tabla necesita columnas centroide_x/centroide_y o x/y.")


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def build_rook_weights(polygons: pd.DataFrame) -> np.ndarray:
    """Vecindad por frontera compartida en una grilla regular."""

    n = len(polygons)
    W = np.zeros((n, n), dtype=float)
    rows = polygons["fila"].to_numpy()
    cols = polygons["columna"].to_numpy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(rows[i] - rows[j]) + abs(cols[i] - cols[j]) == 1:
                W[i, j] = 1.0
    return W


def build_queen_weights(polygons: pd.DataFrame) -> np.ndarray:
    """Vecindad por frontera o vértice compartido."""

    n = len(polygons)
    W = np.zeros((n, n), dtype=float)
    rows = polygons["fila"].to_numpy()
    cols = polygons["columna"].to_numpy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if max(abs(rows[i] - rows[j]), abs(cols[i] - cols[j])) == 1:
                W[i, j] = 1.0
    return W


def build_distance_threshold_weights(data: pd.DataFrame, threshold: float) -> np.ndarray:
    coords = get_centroids(data)
    distances = pairwise_distances(coords)
    W = ((distances <= threshold) & (distances > 0)).astype(float)
    return W


def build_knn_weights(data: pd.DataFrame, k: int, symmetric: bool = False) -> np.ndarray:
    coords = get_centroids(data)
    n = len(coords)
    k = int(max(1, min(k, n - 1)))
    distances = pairwise_distances(coords)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(distances[i])
        neighbors = [j for j in order if j != i][:k]
        W[i, neighbors] = 1.0
    if symmetric:
        W = np.maximum(W, W.T)
    return W


def build_inverse_distance_weights(
    data: pd.DataFrame,
    alpha: float = 1.0,
    max_distance: float | None = None,
) -> np.ndarray:
    coords = get_centroids(data)
    distances = pairwise_distances(coords)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = 1.0 / np.power(distances, alpha)
    W[~np.isfinite(W)] = 0.0
    np.fill_diagonal(W, 0.0)
    if max_distance is not None:
        W[distances > max_distance] = 0.0
    return W


def build_kernel_weights(data: pd.DataFrame, bandwidth: float) -> np.ndarray:
    """Kernel bi-cuadrado compacto: influencia cae a cero fuera del ancho."""

    coords = get_centroids(data)
    distances = pairwise_distances(coords)
    safe_bandwidth = max(float(bandwidth), 1e-9)
    scaled = distances / safe_bandwidth
    W = np.where((distances > 0) & (scaled <= 1), (1 - scaled**2) ** 2, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def row_standardize(W: np.ndarray) -> np.ndarray:
    row_sums = W.sum(axis=1, keepdims=True)
    return np.divide(
        W,
        row_sums,
        out=np.zeros_like(W, dtype=float),
        where=row_sums != 0,
    )


def compute_spatial_lag(W: np.ndarray, values: Iterable[float], standardized: bool = True) -> np.ndarray:
    matrix = row_standardize(W) if standardized else W
    y = np.asarray(list(values), dtype=float)
    return matrix @ y


def weights_to_dataframe(W: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.round(W, 3), index=labels, columns=labels)


def neighbors_from_weights(W: np.ndarray, labels: list[str], min_weight: float = 0.0) -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {}
    for i, label in enumerate(labels):
        idx = np.where(W[i] > min_weight)[0].tolist()
        neighbors[label] = [labels[j] for j in idx]
    return neighbors


def neighbors_table(W: np.ndarray, data: pd.DataFrame, min_weight: float = 0.0) -> pd.DataFrame:
    labels = get_labels(data)
    names = get_names(data)
    neighbors = neighbors_from_weights(W, labels, min_weight)
    rows = []
    for label, name in zip(labels, names):
        neigh = neighbors[label]
        rows.append(
            {
                "id": label,
                "territorio": name,
                "n_vecinos": len(neigh),
                "vecinos": ", ".join(neigh) if neigh else "Sin vecinos",
            }
        )
    return pd.DataFrame(rows)


def matrix_stats(W: np.ndarray, labels: list[str] | None = None) -> dict[str, object]:
    n = W.shape[0]
    labels = labels or [str(i) for i in range(n)]
    positive = W > 0
    np.fill_diagonal(positive, False)
    nonzero = int(positive.sum())
    possible = max(n * (n - 1), 1)
    row_counts = positive.sum(axis=1).astype(int)
    isolated_idx = np.where(row_counts == 0)[0].tolist()
    max_idx = int(np.argmax(row_counts)) if n else 0
    return {
        "n": n,
        "enlaces_dirigidos": nonzero,
        "densidad": nonzero / possible,
        "simétrica": bool(np.allclose(W, W.T)),
        "n_aisladas": len(isolated_idx),
        "unidades_aisladas": [labels[i] for i in isolated_idx],
        "vecinos_promedio": float(row_counts.mean()) if n else 0,
        "vecinos_max": int(row_counts.max()) if n else 0,
        "unidad_más_conectada": labels[max_idx] if n else "",
        "conteos": row_counts,
    }


def build_weights_by_method(
    method: str,
    data: pd.DataFrame,
    threshold: float = 2_600,
    k: int = 3,
    symmetric_knn: bool = False,
    alpha: float = 1.0,
    bandwidth: float = 3_200,
) -> WeightResult:
    """Despachador usado por la app para evitar duplicación en secciones."""

    if method == "Rook":
        W = build_rook_weights(data)
        description = "Contigüidad por frontera compartida."
        parameter_label = "frontera"
    elif method == "Queen":
        W = build_queen_weights(data)
        description = "Contigüidad por frontera o vértice compartido."
        parameter_label = "frontera o vértice"
    elif method == "Distancia umbral":
        W = build_distance_threshold_weights(data, threshold)
        description = "Vecinos si el centroide cae dentro del umbral de distancia."
        parameter_label = f"umbral = {threshold:,.0f} m"
    elif method == "k vecinos más cercanos":
        W = build_knn_weights(data, k, symmetric=symmetric_knn)
        suffix = "simétrico" if symmetric_knn else "dirigido"
        description = f"Cada unidad se conecta con sus {k} vecinos más cercanos ({suffix})."
        parameter_label = f"k = {k}"
    elif method == "Inversa de distancia":
        W = build_inverse_distance_weights(data, alpha)
        description = "Todos los pares tienen peso continuo decreciente con la distancia."
        parameter_label = f"alpha = {alpha:.2f}"
    elif method == "Kernel espacial":
        W = build_kernel_weights(data, bandwidth)
        description = "Influencia suave dentro de un ancho de banda; cero fuera de él."
        parameter_label = f"ancho = {bandwidth:,.0f} m"
    else:
        raise ValueError(f"Método de pesos no reconocido: {method}")
    return WeightResult(W=W, labels=get_labels(data), method=method, description=description, parameter_label=parameter_label)


def distance_summary(data: pd.DataFrame) -> pd.DataFrame:
    coords = get_centroids(data)
    labels = get_labels(data)
    distances = pairwise_distances(coords)
    rows = []
    for i, label in enumerate(labels):
        nonzero = distances[i][distances[i] > 0]
        rows.append(
            {
                "id": label,
                "distancia_vecino_más_cercano_m": round(float(nonzero.min()), 1),
                "distancia_media_a_otros_m": round(float(nonzero.mean()), 1),
                "distancia_máxima_m": round(float(nonzero.max()), 1),
            }
        )
    return pd.DataFrame(rows)

