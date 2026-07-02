"""Visualizaciones Plotly para la app Streamlit."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from utils.data_generation import geometry_coordinates, geometry_type, line_xy, point_xy, polygon_xy
from utils.weights import get_centroids, get_labels, matrix_stats, weights_to_dataframe


PALETTE = [
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#f4a261",
    "#6d597a",
    "#8ab17d",
    "#b56576",
    "#577590",
    "#e9c46a",
    "#43aa8b",
    "#9d4edd",
    "#f94144",
]

MORAN_QUADRANT_COLORS = {
    "Alto-Alto": "#d62828",
    "Bajo-Bajo": "#1d4ed8",
    "Alto-Bajo": "#f59e0b",
    "Bajo-Alto": "#60a5fa",
    "Indefinido": "#9ca3af",
    "No significativo": "#d1d5db",
}

CLUSTER_COLORS = {
    1: "#2a9d8f",
    2: "#e76f51",
    3: "#457b9d",
    4: "#f4a261",
    5: "#6d597a",
    6: "#8ab17d",
}


def _layout(fig: go.Figure, title: str = "", height: int = 480) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
    return fig


def _continuous_colors(values: pd.Series, colorscale: str = "Viridis") -> list[str]:
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        return [sample_colorscale(colorscale, 0.55)[0] for _ in values]
    normed = ((values - vmin) / (vmax - vmin)).clip(0, 1)
    return [sample_colorscale(colorscale, float(v))[0] for v in normed]


def fig_polygons(
    data: pd.DataFrame,
    color_column: str | None = None,
    selected_id: str | None = None,
    title: str = "",
    colorscale: str = "Viridis",
) -> go.Figure:
    fig = go.Figure()
    colors = _continuous_colors(data[color_column], colorscale) if color_column else PALETTE
    for idx, row in data.reset_index(drop=True).iterrows():
        x, y = polygon_xy(row["geometry"])
        label = f"{row['id']} · {row['nombre']}"
        if color_column:
            label += f"<br>{color_column}: {row[color_column]:,.2f}"
        line_width = 3 if row["id"] == selected_id else 1.2
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                mode="lines",
                line=dict(color="#263238", width=line_width),
                fillcolor=colors[idx % len(colors)] if not color_column else colors[idx],
                opacity=0.72 if row["id"] != selected_id else 0.95,
                hovertemplate=label + "<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data["centroide_x"],
            y=data["centroide_y"],
            mode="text",
            text=data["id"],
            textfont=dict(size=12, color="#1f2937"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if color_column:
        fig.add_trace(
            go.Scatter(
                x=[None, None],
                y=[None, None],
                mode="markers",
                marker=dict(
                    color=[float(data[color_column].min()), float(data[color_column].max())],
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title=color_column.replace("_", " "), thickness=14),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return _layout(fig, title=title)


def plot_spatial_weights_network(
    data: pd.DataFrame,
    W: np.ndarray,
    selected_id: str | None = None,
    title: str = "Red de vecindad espacial",
) -> go.Figure:
    fig = fig_polygons(data, selected_id=selected_id, title="")
    coords = get_centroids(data)
    labels = get_labels(data)
    positive = W > 0
    max_weight = float(W[positive].max()) if positive.any() else 1.0

    for i, j in zip(*np.where(positive)):
        if i == j:
            continue
        if np.allclose(W, W.T) and j < i:
            continue
        width = 1.0 + 4.0 * float(W[i, j]) / max_weight
        color = "#d62828" if selected_id in {labels[i], labels[j]} else "#264653"
        opacity = 0.85 if selected_id in {labels[i], labels[j]} else 0.45
        fig.add_trace(
            go.Scatter(
                x=[coords[i, 0], coords[j, 0]],
                y=[coords[i, 1], coords[j, 1]],
                mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                hovertemplate=f"{labels[i]} → {labels[j]}<br>w = {W[i, j]:.3f}<extra></extra>",
                showlegend=False,
            )
        )
    stats = matrix_stats(W, labels)
    degrees = stats["conteos"]
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=labels,
            textposition="middle center",
            marker=dict(
                size=28,
                color=degrees,
                colorscale="Bluered",
                cmin=0,
                showscale=True,
                colorbar=dict(title="n vecinos", thickness=12),
                line=dict(color="white", width=2),
            ),
            textfont=dict(color="white", size=11),
            hovertemplate="%{text}<br>vecinos: %{marker.color}<extra></extra>",
            showlegend=False,
        )
    )
    return _layout(fig, title=title)


def fig_points_network(
    data: pd.DataFrame,
    W: np.ndarray | None = None,
    title: str = "Puntos y conexiones",
) -> go.Figure:
    fig = go.Figure()
    if W is not None:
        coords = data[["x", "y"]].to_numpy()
        for i, j in zip(*np.where(W > 0)):
            if i == j:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[coords[i, 0], coords[j, 0]],
                    y=[coords[i, 1], coords[j, 1]],
                    mode="lines",
                    line=dict(color="#6c757d", width=1),
                    opacity=0.38,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    for tipo, group in data.groupby("tipo"):
        fig.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers",
                name=tipo,
                marker=dict(size=11, line=dict(color="white", width=1), color=PALETTE[len(fig.data) % len(PALETTE)]),
                hovertemplate="<b>%{customdata[0]}</b><br>tipo: %{customdata[1]}<br>usuarios: %{customdata[2]}<extra></extra>",
                customdata=group[["nombre", "tipo", "usuarios_estimados"]],
            )
        )
    return _layout(fig, title=title)


def fig_lines(lines: pd.DataFrame, title: str = "Líneas: rutas y corredores") -> go.Figure:
    fig = go.Figure()
    for idx, row in lines.iterrows():
        x, y = line_xy(row["geometry"])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(width=5, color=PALETTE[idx % len(PALETTE)]),
                marker=dict(size=7),
                name=row["nombre"],
                hovertemplate=f"{row['nombre']}<br>modo: {row['modo']}<br>frecuencia: {row['frecuencia_min']} min<extra></extra>",
            )
        )
    return _layout(fig, title=title)


def fig_raster(raster: dict[str, Any], title: str = "Raster sintético") -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=raster["values"],
            x=raster["x"],
            y=raster["y"],
            colorscale="Inferno",
            colorbar=dict(title="intensidad"),
            hovertemplate="x=%{x:.0f}<br>y=%{y:.0f}<br>valor=%{z:.1f}<extra></extra>",
        )
    )
    return _layout(fig, title=title)


def fig_network(graph: nx.Graph, title: str = "Red espacial") -> go.Figure:
    fig = go.Figure()
    for u, v, attrs in graph.edges(data=True):
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        color = "#e76f51" if attrs.get("tipo") == "conector" else "#457b9d"
        fig.add_trace(
            go.Scatter(
                x=[ux, vx],
                y=[uy, vy],
                mode="lines",
                line=dict(color=color, width=3),
                opacity=0.65,
                hovertemplate=f"{u} - {v}<br>{attrs.get('tipo')}<br>{attrs.get('longitud_m')} m<extra></extra>",
                showlegend=False,
            )
        )
    node_x = [attrs["x"] for _, attrs in graph.nodes(data=True)]
    node_y = [attrs["y"] for _, attrs in graph.nodes(data=True)]
    labels = [str(node) for node in graph.nodes]
    degrees = [graph.degree(node) for node in graph.nodes]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            marker=dict(size=24, color=degrees, colorscale="YlGnBu", line=dict(color="white", width=2), showscale=True),
            textfont=dict(color="#111827", size=10),
            hovertemplate="nodo %{text}<br>grado: %{marker.color}<extra></extra>",
            showlegend=False,
        )
    )
    return _layout(fig, title=title)


def fig_geometry(geometry: Any, title: str = "Geometría") -> go.Figure:
    fig = go.Figure()
    geom_type = geometry_type(geometry)
    coords = geometry_coordinates(geometry)

    def add_points(points: list[tuple[float, float]], mode: str, fill: str | None = None, name: str = "") -> None:
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=mode,
                fill=fill,
                name=name or geom_type,
                line=dict(color="#2a9d8f", width=4),
                marker=dict(size=12, color="#e76f51", line=dict(color="white", width=1)),
                hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
                showlegend=False,
            )
        )

    if geom_type == "Point":
        add_points([coords], "markers")
    elif geom_type == "LineString":
        add_points(coords, "lines+markers")
    elif geom_type == "Polygon":
        add_points(coords, "lines+markers", fill="toself")
    elif geom_type == "MultiPoint":
        add_points(coords, "markers")
    elif geom_type == "MultiLineString":
        for part in coords:
            add_points(part, "lines+markers")
    elif geom_type == "MultiPolygon":
        for part in coords:
            add_points(part, "lines+markers", fill="toself")

    fig.update_xaxes(visible=True, showgrid=True, zeroline=True)
    fig.update_yaxes(visible=True, showgrid=True, zeroline=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_weight_heatmap(W: np.ndarray, labels: list[str], title: str = "Heatmap de W") -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=np.round(W, 3),
            x=labels,
            y=labels,
            colorscale="YlGnBu",
            colorbar=dict(title="w_ij"),
            hovertemplate="i=%{y}<br>j=%{x}<br>w=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=450, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def plot_weight_matrix(W: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return weights_to_dataframe(W, labels)


def fig_neighbor_histogram(W: np.ndarray, labels: list[str], title: str = "Número de vecinos") -> go.Figure:
    stats = matrix_stats(W, labels)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=stats["conteos"],
            marker=dict(color="#2a9d8f"),
            hovertemplate="%{x}<br>vecinos: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=330,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Unidad territorial",
        yaxis_title="Vecinos",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def fig_lag_comparison(
    data: pd.DataFrame,
    value_col: str,
    lag_col: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[value_col],
            y=data[lag_col],
            mode="markers+text",
            text=data["id"],
            textposition="top center",
            marker=dict(size=14, color="#457b9d", line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{text}</b><br>valor propio=%{x:.2f}<br>rezago=%{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_shape(
        type="line",
        x0=float(min(data[value_col].min(), data[lag_col].min())),
        y0=float(min(data[value_col].min(), data[lag_col].min())),
        x1=float(max(data[value_col].max(), data[lag_col].max())),
        y1=float(max(data[value_col].max(), data[lag_col].max())),
        line=dict(color="#e76f51", dash="dash"),
    )
    fig.update_layout(
        title="Valor propio frente a rezago espacial",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title=value_col.replace("_", " "),
        yaxis_title=f"W·{value_col}",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def fig_moran_scatter(
    scatter_data: pd.DataFrame,
    moran_i: float,
    variable: str,
    title: str = "Diagrama de Moran",
) -> go.Figure:
    fig = go.Figure()
    for quadrant, group in scatter_data.groupby("cuadrante_moran"):
        fig.add_trace(
            go.Scatter(
                x=group["z"],
                y=group["W_z"],
                mode="markers+text",
                name=quadrant,
                text=group["id"],
                textposition="top center",
                marker=dict(
                    size=14,
                    color=MORAN_QUADRANT_COLORS.get(quadrant, "#9ca3af"),
                    line=dict(color="white", width=1.5),
                ),
                customdata=group[["nombre", variable, "cuadrante_moran"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{variable}: %{{customdata[1]:.2f}}<br>"
                    "z: %{x:.2f}<br>Wz: %{y:.2f}<br>"
                    "cuadrante: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    x_min = float(min(scatter_data["z"].min(), scatter_data["W_z"].min()) - 0.35)
    x_max = float(max(scatter_data["z"].max(), scatter_data["W_z"].max()) + 0.35)
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, line=dict(color="#6b7280", dash="dot"))
    fig.add_shape(type="line", x0=0, x1=0, y0=x_min, y1=x_max, line=dict(color="#6b7280", dash="dot"))
    if np.isfinite(moran_i):
        fig.add_shape(
            type="line",
            x0=x_min,
            x1=x_max,
            y0=moran_i * x_min,
            y1=moran_i * x_max,
            line=dict(color="#111827", width=2),
        )
    fig.update_layout(
        title=title,
        height=470,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title=f"{variable} estandarizada",
        yaxis_title=f"Rezago espacial de {variable} estandarizada",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
    )
    return fig


def fig_moran_quadrant_map(
    data: pd.DataFrame,
    title: str = "Mapa por cuadrantes de Moran",
) -> go.Figure:
    fig = go.Figure()
    for _, row in data.reset_index(drop=True).iterrows():
        x, y = polygon_xy(row["geometry"])
        quadrant = row["cuadrante_moran"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                mode="lines",
                line=dict(color="#263238", width=1.2),
                fillcolor=MORAN_QUADRANT_COLORS.get(quadrant, "#9ca3af"),
                opacity=0.78,
                name=quadrant,
                hovertemplate=f"{row['id']} · {row['nombre']}<br>{quadrant}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data["centroide_x"],
            y=data["centroide_y"],
            mode="text",
            text=data["id"],
            textfont=dict(size=12, color="white"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for quadrant, color in MORAN_QUADRANT_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                name=quadrant,
                hoverinfo="skip",
            )
        )
    return _layout(fig, title=title)


def fig_permutation_distribution(
    simulations: np.ndarray,
    observed: float,
    expected: float,
    statistic_name: str,
    title: str = "Distribución por permutaciones",
) -> go.Figure:
    valid = simulations[np.isfinite(simulations)]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid,
            nbinsx=28,
            marker=dict(color="#93c5fd", line=dict(color="white", width=1)),
            name="Permutaciones",
            hovertemplate=f"{statistic_name}: %{{x:.3f}}<br>frecuencia: %{{y}}<extra></extra>",
        )
    )
    fig.add_vline(x=observed, line_width=3, line_color="#d62828", annotation_text="observado")
    fig.add_vline(x=expected, line_width=2, line_dash="dash", line_color="#111827", annotation_text="esperado")
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title=statistic_name,
        yaxis_title="Frecuencia",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def fig_lisa_cluster_map(
    data: pd.DataFrame,
    cluster_column: str = "cluster_lisa",
    value_column: str | None = "I_local",
    p_column: str | None = "p_moran_sim",
    selected_id: str | None = None,
    title: str = "Mapa LISA",
) -> go.Figure:
    fig = go.Figure()
    for _, row in data.reset_index(drop=True).iterrows():
        x, y = polygon_xy(row["geometry"])
        cluster = row[cluster_column]
        line_width = 3 if row.get("id") == selected_id else 1.2
        value_text = ""
        if value_column and value_column in row:
            value_text += f"<br>{value_column}: {row[value_column]:.3f}"
        if p_column and p_column in row:
            value_text += f"<br>p_sim: {row[p_column]:.3f}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                mode="lines",
                line=dict(color="#263238", width=line_width),
                fillcolor=MORAN_QUADRANT_COLORS.get(cluster, "#9ca3af"),
                opacity=0.84 if cluster != "No significativo" else 0.58,
                hovertemplate=f"{row['id']} · {row['nombre']}<br>{cluster}{value_text}<extra></extra>",
                showlegend=False,
            )
        )
    labels = data["id"].astype(str)
    if value_column and value_column in data.columns:
        labels = data.apply(lambda row: f"{row['id']}<br>I={row[value_column]:.2f}", axis=1)
    fig.add_trace(
        go.Scatter(
            x=data["centroide_x"],
            y=data["centroide_y"],
            mode="text",
            text=labels,
            textfont=dict(size=11, color="#111827"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for cluster in ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto", "No significativo"]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=MORAN_QUADRANT_COLORS[cluster]),
                name=cluster,
                hoverinfo="skip",
            )
        )
    return _layout(fig, title=title)


def fig_local_stat_bar(
    local_stats: pd.DataFrame,
    value_column: str,
    color_column: str = "cluster_lisa",
    title: str = "Estadístico local por territorio",
) -> go.Figure:
    data = local_stats.sort_values(value_column, ascending=False)
    colors = [MORAN_QUADRANT_COLORS.get(label, "#9ca3af") for label in data[color_column]]
    fig = go.Figure(
        go.Bar(
            x=data["id"],
            y=data[value_column],
            marker=dict(color=colors),
            customdata=data[["nombre", color_column]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{x}<br>valor=%{y:.3f}<br>clase=%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Territorio",
        yaxis_title=value_column,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def fig_local_stat_histogram(
    local_stats: pd.DataFrame,
    value_column: str,
    title: str = "Distribución del estadístico local",
) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=local_stats[value_column],
            nbinsx=16,
            marker=dict(color="#2a9d8f", line=dict(color="white", width=1)),
            hovertemplate=f"{value_column}: %{{x:.3f}}<br>frecuencia: %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=330,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title=value_column,
        yaxis_title="Frecuencia",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def fig_cluster_map(
    data: pd.DataFrame,
    labels: np.ndarray,
    selected_id: str | None = None,
    variables: list[str] | None = None,
    title: str = "Mapa de clústeres",
) -> go.Figure:
    fig = go.Figure()
    labels = np.asarray(labels, dtype=int)
    variables = variables or []
    for idx, row in data.reset_index(drop=True).iterrows():
        x, y = polygon_xy(row["geometry"])
        cluster = int(labels[idx])
        cluster_name = f"C{cluster}"
        line_width = 3 if row["id"] == selected_id else 1.2
        values_text = "".join(f"<br>{variable}: {row[variable]:,.2f}" for variable in variables if variable in row.index)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                mode="lines",
                line=dict(color="#263238", width=line_width),
                fillcolor=CLUSTER_COLORS.get(cluster, PALETTE[(cluster - 1) % len(PALETTE)]),
                opacity=0.82 if row["id"] != selected_id else 0.96,
                hovertemplate=f"{row['id']} · {row['nombre']}<br>{cluster_name}{values_text}<extra></extra>",
                showlegend=False,
            )
        )
    text = [f"{row_id}<br>C{cluster}" for row_id, cluster in zip(data["id"], labels)]
    fig.add_trace(
        go.Scatter(
            x=data["centroide_x"],
            y=data["centroide_y"],
            mode="text",
            text=text,
            textfont=dict(size=11, color="#111827"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for cluster in sorted(np.unique(labels)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=CLUSTER_COLORS.get(int(cluster), PALETTE[(int(cluster) - 1) % len(PALETTE)])),
                name=f"C{int(cluster)}",
                hoverinfo="skip",
            )
        )
    return _layout(fig, title=title)


def fig_cluster_profile(
    profile: pd.DataFrame,
    variables: list[str],
    title: str = "Perfil medio por clúster",
) -> go.Figure:
    fig = go.Figure()
    for idx, variable in enumerate(variables):
        fig.add_trace(
            go.Bar(
                x=profile["clúster"],
                y=profile[variable],
                name=variable,
                marker=dict(color=PALETTE[idx % len(PALETTE)]),
                hovertemplate=f"%{{x}}<br>{variable}: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Clúster",
        yaxis_title="Media escalada",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="left", x=0),
    )
    return fig


def fig_cluster_scatter(
    data: pd.DataFrame,
    labels: np.ndarray,
    x_column: str,
    y_column: str,
    title: str = "Espacio de atributos",
) -> go.Figure:
    fig = go.Figure()
    labels = np.asarray(labels, dtype=int)
    temp = data.copy()
    temp["clúster"] = labels
    for cluster, group in temp.groupby("clúster", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group[x_column],
                y=group[y_column],
                mode="markers+text",
                name=f"C{int(cluster)}",
                text=group["id"],
                textposition="top center",
                marker=dict(
                    size=14,
                    color=CLUSTER_COLORS.get(int(cluster), PALETTE[(int(cluster) - 1) % len(PALETTE)]),
                    line=dict(color="white", width=1.5),
                ),
                customdata=group[["nombre"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{x_column}: %{{x:.2f}}<br>"
                    f"{y_column}: %{{y:.2f}}<br>"
                    f"C{int(cluster)}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title=x_column.replace("_", " "),
        yaxis_title=y_column.replace("_", " "),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, xanchor="left", x=0),
    )
    return fig
