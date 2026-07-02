"""Rutinas livianas de clustering y regionalización para la app docente."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterSolution:
    """Resultado común para algoritmos de agrupamiento."""

    name: str
    labels: np.ndarray
    description: str


def scale_features(data: pd.DataFrame, variables: list[str], method: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Escala variables numéricas y devuelve matriz más tabla de parámetros."""

    values = data[variables].to_numpy(dtype=float)
    method = method.lower()
    rows = []

    if method == "sin escalar":
        scaled = values.copy()
        center = np.zeros(values.shape[1])
        spread = np.ones(values.shape[1])
        label = "sin escalar"
    elif method == "min-max 0-1":
        center = values.min(axis=0)
        spread = values.max(axis=0) - values.min(axis=0)
        spread = np.where(np.isclose(spread, 0), 1.0, spread)
        scaled = (values - center) / spread
        label = "min-max"
    elif method == "robusta (mediana/iqr)":
        center = np.median(values, axis=0)
        q75 = np.percentile(values, 75, axis=0)
        q25 = np.percentile(values, 25, axis=0)
        spread = q75 - q25
        spread = np.where(np.isclose(spread, 0), 1.0, spread)
        scaled = (values - center) / spread
        label = "robusta"
    else:
        center = values.mean(axis=0)
        spread = values.std(axis=0, ddof=0)
        spread = np.where(np.isclose(spread, 0), 1.0, spread)
        scaled = (values - center) / spread
        label = "z-score"

    for variable, c, s in zip(variables, center, spread):
        rows.append({"variable": variable, "método": label, "centro": float(c), "escala": float(s)})
    return scaled, pd.DataFrame(rows)


def _clip_k(k: int, n: int) -> int:
    return int(max(1, min(k, n)))


def _relabel_by_min_index(labels: np.ndarray) -> np.ndarray:
    """Convierte etiquetas arbitrarias a 1..k según el primer territorio de cada grupo."""

    labels = np.asarray(labels, dtype=int)
    order = sorted(np.unique(labels), key=lambda lab: int(np.where(labels == lab)[0].min()))
    mapping = {lab: idx + 1 for idx, lab in enumerate(order)}
    return np.array([mapping[lab] for lab in labels], dtype=int)


def kmeans_labels(X: np.ndarray, k: int, seed: int = 2026, max_iter: int = 80) -> np.ndarray:
    """K-means pequeño y reproducible, suficiente para los datos sintéticos de clase."""

    n = X.shape[0]
    k = _clip_k(k, n)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    centroid_idx = [first]
    while len(centroid_idx) < k:
        centroids_so_far = X[centroid_idx]
        dists = ((X[:, None, :] - centroids_so_far[None, :, :]) ** 2).sum(axis=2)
        nearest = dists.min(axis=1)
        nearest[centroid_idx] = -1
        centroid_idx.append(int(np.argmax(nearest)))

    centroids = X[centroid_idx].copy()
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)

        counts = np.bincount(new_labels, minlength=k)
        if np.any(counts == 0):
            nearest = dists.min(axis=1)
            for empty in np.where(counts == 0)[0]:
                candidate = int(np.argmax(nearest))
                new_labels[candidate] = empty
                nearest[candidate] = -1

        new_centroids = centroids.copy()
        for cluster in range(k):
            members = X[new_labels == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)

        if np.array_equal(labels, new_labels) and np.allclose(centroids, new_centroids):
            break
        labels = new_labels
        centroids = new_centroids

    return _relabel_by_min_index(labels)


def ward_labels(X: np.ndarray, k: int, connectivity: np.ndarray | None = None) -> np.ndarray:
    """Agrupamiento aglomerativo tipo Ward, con restricción espacial opcional."""

    n = X.shape[0]
    k = _clip_k(k, n)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))
    next_id = n
    graph = None
    if connectivity is not None:
        graph = (np.asarray(connectivity) > 0) | (np.asarray(connectivity).T > 0)
        np.fill_diagonal(graph, False)

    def merge_allowed(left: list[int], right: list[int]) -> bool:
        if graph is None:
            return True
        return bool(graph[np.ix_(left, right)].any())

    def ward_cost(left: list[int], right: list[int]) -> float:
        a = X[left]
        b = X[right]
        mean_a = a.mean(axis=0)
        mean_b = b.mean(axis=0)
        return float((len(left) * len(right)) / (len(left) + len(right)) * np.sum((mean_a - mean_b) ** 2))

    while len(active) > k:
        best_pair: tuple[int, int] | None = None
        best_cost = np.inf
        for pos_i, cluster_i in enumerate(active[:-1]):
            for cluster_j in active[pos_i + 1 :]:
                left = clusters[cluster_i]
                right = clusters[cluster_j]
                if not merge_allowed(left, right):
                    continue
                cost = ward_cost(left, right)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (cluster_i, cluster_j)

        if best_pair is None:
            for pos_i, cluster_i in enumerate(active[:-1]):
                for cluster_j in active[pos_i + 1 :]:
                    cost = ward_cost(clusters[cluster_i], clusters[cluster_j])
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (cluster_i, cluster_j)

        left_id, right_id = best_pair
        merged = sorted(clusters[left_id] + clusters[right_id])
        active = [cluster_id for cluster_id in active if cluster_id not in {left_id, right_id}]
        del clusters[left_id]
        del clusters[right_id]
        clusters[next_id] = merged
        active.append(next_id)
        next_id += 1

    labels = np.zeros(n, dtype=int)
    for cluster_number, cluster_id in enumerate(sorted(active, key=lambda cid: min(clusters[cid])), start=1):
        labels[clusters[cluster_id]] = cluster_number
    return labels


def count_components_by_label(W: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Cuenta componentes conectados internos por etiqueta."""

    labels = np.asarray(labels, dtype=int)
    graph = (np.asarray(W) > 0) | (np.asarray(W).T > 0)
    np.fill_diagonal(graph, False)
    components: dict[int, int] = {}
    for label in sorted(np.unique(labels)):
        nodes = set(np.where(labels == label)[0].tolist())
        visited: set[int] = set()
        count = 0
        for node in sorted(nodes):
            if node in visited:
                continue
            count += 1
            stack = [node]
            visited.add(node)
            while stack:
                current = stack.pop()
                for neighbor in np.where(graph[current])[0]:
                    if neighbor in nodes and neighbor not in visited:
                        visited.add(int(neighbor))
                        stack.append(int(neighbor))
        components[int(label)] = count
    return components


def internal_neighbor_share(W: np.ndarray, labels: np.ndarray) -> float:
    positive = np.asarray(W) > 0
    np.fill_diagonal(positive, False)
    total = int(positive.sum())
    if total == 0:
        return 0.0
    labels = np.asarray(labels)
    internal = sum(1 for i, j in zip(*np.where(positive)) if labels[i] == labels[j])
    return float(internal / total)


def bbox_compactness(data: pd.DataFrame, labels: np.ndarray) -> float:
    """Compacidad simple en grilla: celdas del grupo / caja envolvente del grupo."""

    if not {"fila", "columna"}.issubset(data.columns):
        return float("nan")
    labels = np.asarray(labels, dtype=int)
    rows = data["fila"].to_numpy()
    cols = data["columna"].to_numpy()
    weighted = 0.0
    n = len(labels)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        height = int(rows[idx].max() - rows[idx].min() + 1)
        width = int(cols[idx].max() - cols[idx].min() + 1)
        compact = len(idx) / max(height * width, 1)
        weighted += compact * len(idx)
    return float(weighted / n) if n else 0.0


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = X.shape[0]
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1 or n <= k:
        return float("nan")
    grand = X.mean(axis=0)
    within = 0.0
    between = 0.0
    for label in unique:
        members = X[labels == label]
        centroid = members.mean(axis=0)
        within += float(((members - centroid) ** 2).sum())
        between += float(len(members) * ((centroid - grand) ** 2).sum())
    if np.isclose(within, 0):
        return float("nan")
    return float((between / (k - 1)) / (within / (n - k)))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) >= len(labels):
        return float("nan")
    distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for idx, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != idx]
        a = float(distances[idx, same].mean()) if len(same) else 0.0
        b = np.inf
        for other in unique:
            if other == label:
                continue
            other_idx = np.where(labels == other)[0]
            b = min(b, float(distances[idx, other_idx].mean()))
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 and np.isfinite(b) else 0.0)
    return float(np.mean(scores))


def cluster_profile_table(data: pd.DataFrame, labels: np.ndarray, variables: list[str]) -> pd.DataFrame:
    temp = data[["id", "nombre", *variables]].copy()
    temp["clúster"] = [f"C{label}" for label in labels]
    grouped = temp.groupby("clúster", sort=True)
    profile = grouped[variables].mean().round(2)
    profile.insert(0, "n_territorios", grouped.size())
    profile.insert(1, "territorios", grouped["id"].apply(lambda vals: ", ".join(vals)))
    return profile.reset_index()


def territory_cluster_table(data: pd.DataFrame, labels: np.ndarray, variables: list[str]) -> pd.DataFrame:
    table = data[["id", "nombre", *variables]].copy()
    table.insert(2, "clúster", [f"C{label}" for label in labels])
    return table


def solution_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    data: pd.DataFrame,
) -> dict[str, float | int]:
    components = count_components_by_label(W, labels)
    return {
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "silhouette": silhouette_score(X, labels),
        "componentes_internos": int(sum(components.values())),
        "fragmentación": int(sum(value - 1 for value in components.values())),
        "vecinos_internos_%": internal_neighbor_share(W, labels) * 100,
        "compacidad_bbox": bbox_compactness(data, labels),
    }
