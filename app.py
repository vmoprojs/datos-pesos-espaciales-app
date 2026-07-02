from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.autocorrelation import (
    global_autocorrelation_table,
    interpret_lisa_row,
    lisa_counts_table,
    local_lisa_statistics,
    moran_scatter_data,
    permutation_test,
)
from utils.clustering import (
    ClusterSolution,
    cluster_profile_table,
    count_components_by_label,
    kmeans_labels,
    scale_features,
    solution_metrics,
    territory_cluster_table,
    ward_labels,
)
from utils.spatial_regression import (
    MODEL_DESCRIPTIONS,
    breusch_pagan_test,
    coefficient_table,
    fit_spatial_regression,
    model_comparison_table,
    predict_spatial_shock,
    prepare_regression_data,
    spatial_diagnostic_signals,
)
from utils.data_generation import (
    GEOGRAPHIC_CRS,
    HAS_GEO,
    PROJECTED_CRS,
    create_geometry_example,
    create_intro_table,
    create_synthetic_lines,
    create_synthetic_network,
    create_synthetic_points,
    create_synthetic_polygons,
    create_synthetic_raster,
    distance_in_degrees,
    distance_in_meters,
    geometry_coordinates,
)
from utils.explanations import (
    DATA_TYPE_INFO,
    crs_warning,
    interpret_weights_matrix,
    lag_interpretation,
    method_guidance,
    structures_table,
)
from utils.plots import (
    fig_cluster_map,
    fig_cluster_profile,
    fig_cluster_scatter,
    fig_geometry,
    fig_lag_comparison,
    fig_lisa_cluster_map,
    fig_lines,
    fig_local_stat_bar,
    fig_local_stat_histogram,
    fig_moran_quadrant_map,
    fig_moran_scatter,
    fig_neighbor_histogram,
    fig_network,
    fig_points_network,
    fig_polygons,
    fig_permutation_distribution,
    fig_raster,
    plot_spatial_weights_network,
    plot_weight_heatmap,
    plot_weight_matrix,
)
from utils.weights import (
    build_distance_threshold_weights,
    build_knn_weights,
    build_weights_by_method,
    compute_spatial_lag,
    distance_summary,
    matrix_stats,
    neighbors_table,
    row_standardize,
)


st.set_page_config(
    page_title="Datos espaciales y pesos espaciales",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1380px;}
    h1, h2, h3 {letter-spacing: 0;}
    .concept-card {
        border: 1px solid #d8dee4;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
        min-height: 116px;
    }
    .small-note {
        color: #4b5563;
        font-size: .92rem;
        line-height: 1.35;
    }
    .discussion {
        border-left: 4px solid #2a9d8f;
        padding: .7rem .9rem;
        background: #f6fbfa;
        border-radius: 6px;
        margin-top: .6rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: .75rem;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


METHODS = [
    "Rook",
    "Queen",
    "Distancia umbral",
    "k vecinos más cercanos",
    "Inversa de distancia",
    "Kernel espacial",
]

SOCIAL_VARIABLES = [
    "ingreso_medio",
    "tasa_pobreza",
    "acceso_servicios",
    "índice_vulnerabilidad",
    "densidad_poblacional",
]


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, dict]:
    polygons = create_synthetic_polygons()
    points = create_synthetic_points()
    lines = create_synthetic_lines()
    graph = create_synthetic_network()
    raster = create_synthetic_raster()
    return polygons, points, lines, graph, raster


polygons, points, lines, graph, raster = load_data()


def section_title(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def discussion(question: str) -> None:
    st.markdown(f"<div class='discussion'><b>Pregunta para discusión:</b> {question}</div>", unsafe_allow_html=True)


def show_matrix_alerts(W: np.ndarray, labels: list[str]) -> None:
    stats = matrix_stats(W, labels)
    if stats["n_aisladas"]:
        st.warning(f"Hay unidades sin vecinos: {', '.join(stats['unidades_aisladas'])}.")
    if not stats["simétrica"]:
        st.info("La matriz no es simétrica: algunas relaciones van de i a j, pero no necesariamente de j a i.")
    if np.any(W.sum(axis=1) == 0):
        st.caption("Las filas sin vecinos quedan en cero al estandarizar por filas; así se evitan divisiones por cero.")


def weight_controls(prefix: str, data: pd.DataFrame, default_method: str = "Queen"):
    default_index = METHODS.index(default_method)
    method = st.selectbox("Criterio de pesos", METHODS, index=default_index, key=f"{prefix}_method")
    threshold = 2_600
    k = 3
    symmetric_knn = False
    alpha = 1.0
    bandwidth = 3_300

    if method == "Distancia umbral":
        threshold = st.slider(
            "Umbral de distancia (metros)",
            min_value=800,
            max_value=6_000,
            value=2_600,
            step=100,
            key=f"{prefix}_threshold",
        )
    elif method == "k vecinos más cercanos":
        k = st.slider(
            "Número de vecinos k",
            min_value=1,
            max_value=max(1, len(data) - 1),
            value=3,
            step=1,
            key=f"{prefix}_k",
        )
        symmetric_knn = st.checkbox("Hacer simétrica la matriz kNN", value=False, key=f"{prefix}_sym_knn")
    elif method == "Inversa de distancia":
        alpha = st.slider(
            "Decaimiento alpha",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key=f"{prefix}_alpha",
        )
    elif method == "Kernel espacial":
        bandwidth = st.slider(
            "Ancho de banda (metros)",
            min_value=1_200,
            max_value=7_000,
            value=3_300,
            step=100,
            key=f"{prefix}_bandwidth",
        )

    return build_weights_by_method(
        method,
        data,
        threshold=threshold,
        k=k,
        symmetric_knn=symmetric_knn,
        alpha=alpha,
        bandwidth=bandwidth,
    )


def stats_metrics(W: np.ndarray, labels: list[str]) -> None:
    stats = matrix_stats(W, labels)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Densidad de W", f"{stats['densidad']:.2f}")
    col2.metric("Vecinos promedio", f"{stats['vecinos_promedio']:.1f}")
    col3.metric("Unidades aisladas", stats["n_aisladas"])
    col4.metric("Simétrica", "Sí" if stats["simétrica"] else "No")


def method_stat_row(name: str, W: np.ndarray, labels: list[str]) -> dict:
    stats = matrix_stats(W, labels)
    return {
        "método": name,
        "densidad": round(float(stats["densidad"]), 3),
        "vecinos_promedio": round(float(stats["vecinos_promedio"]), 2),
        "vecinos_max": stats["vecinos_max"],
        "aisladas": stats["n_aisladas"],
        "simétrica": "sí" if stats["simétrica"] else "no",
    }


def compact_number(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "n/d"
    return f"{value:.{digits}f}"


def intro_section() -> None:
    section_title(
        "Datos espaciales, estructuras y pesos espaciales",
        "Clase interactiva para ciencias sociales, análisis territorial e IA aplicada.",
    )
    st.markdown(
        """
        Un dato se vuelve espacial cuando, además de sus atributos, tiene una localización o una geometría que permite
        preguntar por cercanía, contigüidad, distancia, pertenencia, accesibilidad o interacción territorial.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("<div class='concept-card'><b>Dato no espacial</b><br><span class='small-note'>Tabla de atributos sin ubicación explícita.</span></div>", unsafe_allow_html=True)
    c2.markdown("<div class='concept-card'><b>Dato espacial</b><br><span class='small-note'>Atributos vinculados a coordenadas o geometría.</span></div>", unsafe_allow_html=True)
    c3.markdown("<div class='concept-card'><b>Geometría</b><br><span class='small-note'>Forma: punto, línea, polígono, raster o red.</span></div>", unsafe_allow_html=True)
    c4.markdown("<div class='concept-card'><b>Relación espacial</b><br><span class='small-note'>Vecindad, distancia, conexión o superposición.</span></div>", unsafe_allow_html=True)

    selected = st.selectbox("Territorio destacado", polygons["id"] + " · " + polygons["nombre"], key="intro_selected")
    selected_id = selected.split(" · ")[0]

    left, right = st.columns([1.05, 1.35])
    with left:
        st.subheader("Como tabla")
        st.dataframe(create_intro_table(polygons), width="stretch", hide_index=True)
        row = polygons.loc[polygons["id"] == selected_id].iloc[0]
        st.markdown(
            f"""
            **Lectura tabular:** {row['nombre']} tiene {row['población']:,} habitantes, ingreso medio
            de {row['ingreso_medio']} y una geometría `Polygon`.
            """
        )
    with right:
        st.subheader("Como mapa")
        st.plotly_chart(fig_polygons(polygons, selected_id=selected_id, title="Barrios/parroquias ficticias"), use_container_width=True)
        st.markdown(
            "La misma fila ahora permite preguntar: ¿con quién limita?, ¿qué tan lejos está?, ¿qué territorios la rodean?"
        )
    discussion("¿Qué preguntas sociales aparecen solo cuando miramos el dato como territorio y no solo como tabla?")


def data_types_section() -> None:
    section_title("Tipos de datos espaciales", "Cada estructura representa una forma distinta de observar fenómenos sociales.")
    choice = st.selectbox("Tipo de dato espacial", list(DATA_TYPE_INFO.keys()), key="data_type")
    info = DATA_TYPE_INFO[choice]

    left, right = st.columns([0.9, 1.35])
    with left:
        st.subheader(choice)
        st.markdown(f"**Definición.** {info['definición']}")
        st.markdown(f"**Ejemplo social.** {info['ejemplo']}")
        st.markdown(f"**Estructura típica en Python.** `{info['python']}`")
        st.markdown(f"**Uso en analítica o IA social.** {info['ia']}")

        code_by_type = {
            "Puntos": "gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat), crs='EPSG:4326')",
            "Líneas": "ruta = shapely.geometry.LineString([(x1, y1), (x2, y2), (x3, y3)])",
            "Polígonos": "barrios = geopandas.GeoDataFrame(tabla, geometry='geometry', crs='EPSG:32717')",
            "Raster": "superficie = numpy.ndarray(shape=(filas, columnas))",
            "Redes": "G = networkx.Graph(); G.add_edge(origen, destino, longitud_m=850)",
            "Datos tabulares con coordenadas": "df[['lon', 'lat', 'ingreso', 'empleo']]",
            "Datos areales agregados": "gdf[['id', 'tasa_pobreza', 'geometry']]",
        }
        st.code(code_by_type[choice], language="python")
    with right:
        if choice == "Puntos":
            st.plotly_chart(fig_points_network(points, title="Equipamientos y eventos puntuales"), use_container_width=True)
            st.dataframe(points[["id", "tipo", "nombre", "usuarios_estimados"]].head(8), width="stretch", hide_index=True)
        elif choice == "Líneas":
            st.plotly_chart(fig_lines(lines), use_container_width=True)
            st.dataframe(lines[["id", "nombre", "modo", "frecuencia_min"]], width="stretch", hide_index=True)
        elif choice == "Polígonos":
            st.plotly_chart(fig_polygons(polygons, color_column="tasa_pobreza", title="Polígonos con indicador social"), use_container_width=True)
        elif choice == "Raster":
            st.plotly_chart(fig_raster(raster, title=raster["name"]), use_container_width=True)
        elif choice == "Redes":
            st.plotly_chart(fig_network(graph), use_container_width=True)
        elif choice == "Datos tabulares con coordenadas":
            sample = points[["id", "tipo", "x", "y", "usuarios_estimados"]].head(10).copy()
            st.dataframe(sample, width="stretch", hide_index=True)
            st.plotly_chart(fig_points_network(points, title="La tabla se espacializa con X/Y"), use_container_width=True)
        else:
            st.plotly_chart(fig_polygons(polygons, color_column="índice_vulnerabilidad", title="Datos agregados por territorio"), use_container_width=True)
            st.dataframe(
                polygons[["id", "nombre", "tasa_pobreza", "acceso_servicios", "índice_vulnerabilidad"]].head(10),
                width="stretch",
                hide_index=True,
            )
    discussion("¿Qué tipo de dato elegirías para estudiar acceso desigual a salud, y qué perderías si lo agregas por zonas?")


def geometries_section() -> None:
    section_title("Geometrías espaciales", "Las geometrías son los objetos que guardan forma, ubicación y relaciones espaciales.")
    if HAS_GEO:
        st.success("Este entorno tiene Shapely/GeoPandas disponible; las geometrías se crean como objetos reales.")
    else:
        st.info("Shapely/GeoPandas no están instalados en el entorno actual; la app usa una representación interna compatible para no detener la clase.")

    left, right = st.columns([0.85, 1.25])
    with left:
        kind = st.selectbox(
            "Geometría",
            ["Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"],
            key="geometry_kind",
        )
        scale = st.slider("Escala del ejemplo", 0.5, 3.0, 1.2, 0.1, key="geometry_scale")
        example = create_geometry_example(kind, scale)
        st.code(
            """from shapely.geometry import Point, LineString, Polygon

punto = Point(-78.49, -0.18)
linea = LineString([(0, 0), (1, 1), (2, 0)])
poligono = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])""",
            language="python",
        )
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Área", f"{example['área']:.2f}")
        metric_b.metric("Longitud / perímetro", f"{example['longitud']:.2f}")
        st.json({"tipo": example["tipo"], "coordenadas": example["coordenadas"]})
    with right:
        st.plotly_chart(fig_geometry(example["geometry"], title=f"Representación de {kind}"), use_container_width=True)
        st.markdown(
            "La geometría no reemplaza los atributos: los complementa. Una fila puede almacenar población, ingreso, tasa de pobreza y, al mismo tiempo, una forma espacial."
        )
    discussion("¿Por qué un polígono permite preguntas distintas a las de un punto, aunque ambos tengan coordenadas?")


def crs_section() -> None:
    section_title("Sistemas de referencia de coordenadas", "El CRS define cómo interpretar coordenadas, distancias y áreas.")
    st.markdown(
        f"""
        `{GEOGRAPHIC_CRS}` usa longitud/latitud en grados. Un CRS proyectado como `{PROJECTED_CRS}` usa metros,
        lo que permite calcular distancias y áreas con sentido métrico. En Ecuador continental, UTM 17S suele ser una referencia útil.
        """
    )
    left, right = st.columns([0.9, 1.25])
    with left:
        st.subheader("Dos puntos en Quito")
        lon_a = st.slider("Longitud A", -78.58, -78.38, -78.50, 0.005, key="lon_a")
        lat_a = st.slider("Latitud A", -0.35, -0.05, -0.20, 0.005, key="lat_a")
        lon_b = st.slider("Longitud B", -78.58, -78.38, -78.45, 0.005, key="lon_b")
        lat_b = st.slider("Latitud B", -0.35, -0.05, -0.16, 0.005, key="lat_b")
        a = (lon_a, lat_a)
        b = (lon_b, lat_b)
        deg_distance = distance_in_degrees(a, b)
        meter_distance = distance_in_meters(a, b)
        m1, m2 = st.columns(2)
        m1.metric("Distancia en grados", f"{deg_distance:.4f}")
        m2.metric("Distancia aproximada en metros", f"{meter_distance:,.0f}")
        st.warning("Los grados no son una unidad de distancia constante. Para distancias y áreas se debe reproyectar.")
        st.markdown(crs_warning(bool(polygons.attrs.get("crs"))))
    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[lon_a, lon_b],
                y=[lat_a, lat_b],
                mode="markers+lines+text",
                text=["A", "B"],
                textposition="top center",
                marker=dict(size=16, color=["#2a9d8f", "#e76f51"]),
                line=dict(width=3, color="#457b9d"),
                hovertemplate="lon=%{x:.4f}<br>lat=%{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Coordenadas geográficas",
            height=460,
            xaxis_title="longitud",
            yaxis_title="latitud",
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"CRS": "EPSG:4326", "Tipo": "geográfico", "Unidad": "grados", "Uso": "visualizar y compartir lon/lat"},
                    {"CRS": "EPSG:32717", "Tipo": "proyectado", "Unidad": "metros", "Uso": "distancias, áreas y buffers en Ecuador continental"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    discussion("¿Qué error de interpretación aparecería si calculas el área de barrios directamente en grados?")


def structures_section() -> None:
    section_title("Estructuras de datos espaciales en Python", "Del DataFrame al grafo: distintas estructuras responden distintas preguntas.")
    st.dataframe(structures_table(), width="stretch", hide_index=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["DataFrame", "GeoDataFrame", "Raster", "Matriz W", "Grafo"])
    with tab1:
        st.markdown("Un `DataFrame` almacena atributos. Puede tener coordenadas, pero todavía no sabe operar espacialmente.")
        st.dataframe(points[["id", "tipo", "x", "y", "usuarios_estimados"]].head(8), width="stretch", hide_index=True)
    with tab2:
        st.markdown("Un `GeoDataFrame` agrega una columna `geometry` y un `crs`; por eso puede mapear, reproyectar y cruzar capas.")
        sample = polygons[["id", "nombre", "geometry"]].head(5).copy()
        sample["geometry"] = sample["geometry"].map(lambda g: str(g)[:80] + "...")
        st.dataframe(sample, width="stretch", hide_index=True)
        st.code("gdf = geopandas.GeoDataFrame(df, geometry='geometry', crs='EPSG:32717')", language="python")
    with tab3:
        st.markdown("Un raster es una matriz: filas y columnas representan celdas de una superficie.")
        st.plotly_chart(fig_raster(raster, title="Matriz raster como superficie social/ambiental"), use_container_width=True)
    with tab4:
        st.markdown("Una matriz espacial `W` traduce la idea de vecindad en números.")
        W = build_weights_by_method("Queen", polygons).W
        st.dataframe(plot_weight_matrix(W, polygons["id"].tolist()), width="stretch")
        st.plotly_chart(plot_weight_heatmap(W, polygons["id"].tolist()), use_container_width=True)
    with tab5:
        st.markdown("Un grafo modela conectividad: nodos, aristas y atributos de conexión.")
        st.plotly_chart(fig_network(graph), use_container_width=True)

    discussion("¿Qué estructura usarías para modelar accesibilidad a transporte y por qué no bastaría una tabla plana?")


def weights_concept_section() -> None:
    section_title("Pesos espaciales: concepto central", "Una matriz W convierte vecindad e influencia territorial en una estructura computable.")
    st.latex(r"y_i^{lag} = \sum_j w_{ij} y_j")
    st.markdown(
        """
        `w_ij` representa cuánto pesa la unidad `j` en el contexto espacial de la unidad `i`.
        Normalmente `w_ii = 0` porque una unidad no se considera vecina de sí misma. Al estandarizar por filas,
        cada fila suma 1 y el rezago espacial se interpreta como promedio ponderado de los vecinos.
        """
    )
    left, right = st.columns([0.85, 1.35])
    with left:
        method = st.radio("Matriz de ejemplo", ["Rook", "Queen"], horizontal=True, key="concept_method")
        variable = st.selectbox("Variable social", SOCIAL_VARIABLES, key="concept_variable")
        selected = st.selectbox("Unidad i", polygons["id"] + " · " + polygons["nombre"], key="concept_selected")
        selected_id = selected.split(" · ")[0]
        result = build_weights_by_method(method, polygons)
        W_binary = result.W
        W_std = row_standardize(W_binary)
        idx = result.labels.index(selected_id)
        lag = compute_spatial_lag(W_binary, polygons[variable], standardized=True)
        st.metric("Valor propio", f"{polygons.loc[idx, variable]:,.2f}")
        st.metric("Rezago espacial Wy", f"{lag[idx]:,.2f}")
        st.markdown(method_guidance(method))
    with right:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Matriz binaria**")
            st.dataframe(plot_weight_matrix(W_binary, result.labels), width="stretch")
        with col_b:
            st.markdown("**Matriz estandarizada por filas**")
            st.dataframe(plot_weight_matrix(W_std, result.labels), width="stretch")
        st.plotly_chart(plot_spatial_weights_network(polygons, W_binary, selected_id=selected_id, title="Vecindad usada en W"), use_container_width=True)
    discussion("¿Cuándo conviene interpretar el rezago como suma y cuándo como promedio de los vecinos?")


def weights_types_section() -> None:
    section_title("Tipos de pesos espaciales", "La definición de vecino cambia el diagnóstico territorial.")
    left, right = st.columns([0.85, 1.35])
    with left:
        result = weight_controls("types", polygons, default_method="Rook")
        st.markdown(f"**Concepto.** {result.description}")
        st.markdown(f"**Parámetro.** {result.parameter_label}")
        st.markdown(method_guidance(result.method))
        show_matrix_alerts(result.W, result.labels)
        stats_metrics(result.W, result.labels)
        if result.method == "Inversa de distancia":
            st.latex(r"w_{ij} = \frac{1}{d_{ij}^{\alpha}}")
        elif result.method == "Kernel espacial":
            st.latex(r"w_{ij} = (1 - (d_{ij}/b)^2)^2 \quad \text{si } d_{ij} \leq b")
    with right:
        st.plotly_chart(plot_spatial_weights_network(polygons, result.W, title=f"Red de vecindad: {result.method}"), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Matriz", "Heatmap", "Vecinos"])
    with tab1:
        col_a, col_b = st.columns(2)
        col_a.markdown("**W original**")
        col_a.dataframe(plot_weight_matrix(result.W, result.labels), width="stretch")
        col_b.markdown("**W estandarizada por filas**")
        col_b.dataframe(plot_weight_matrix(row_standardize(result.W), result.labels), width="stretch")
    with tab2:
        st.plotly_chart(plot_weight_heatmap(result.W, result.labels, title=f"Intensidad de pesos: {result.method}"), use_container_width=True)
    with tab3:
        st.dataframe(neighbors_table(result.W, polygons), width="stretch", hide_index=True)
        st.plotly_chart(fig_neighbor_histogram(result.W, result.labels), use_container_width=True)
    discussion("¿Qué cambia en la interpretación de un barrio cuando pasas de Rook a Queen o a kNN?")


def comparator_section() -> None:
    section_title("Comparador de matrices W", "Comparar criterios ayuda a justificar una decisión metodológica.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matriz A")
        result_a = weight_controls("comp_a", polygons, default_method="Queen")
    with col_b:
        st.subheader("Matriz B")
        result_b = weight_controls("comp_b", polygons, default_method="k vecinos más cercanos")

    map_a, map_b = st.columns(2)
    with map_a:
        st.plotly_chart(plot_spatial_weights_network(polygons, result_a.W, title=f"A: {result_a.method}"), use_container_width=True)
    with map_b:
        st.plotly_chart(plot_spatial_weights_network(polygons, result_b.W, title=f"B: {result_b.method}"), use_container_width=True)

    heat_a, heat_b = st.columns(2)
    with heat_a:
        st.plotly_chart(plot_weight_heatmap(result_a.W, result_a.labels, title=f"Heatmap A: {result_a.method}"), use_container_width=True)
    with heat_b:
        st.plotly_chart(plot_weight_heatmap(result_b.W, result_b.labels, title=f"Heatmap B: {result_b.method}"), use_container_width=True)

    st.subheader("Diagnóstico comparativo")
    stats_df = pd.DataFrame(
        [
            method_stat_row(result_a.method, result_a.W, result_a.labels),
            method_stat_row(result_b.method, result_b.W, result_b.labels),
        ]
    )
    st.dataframe(stats_df, width="stretch", hide_index=True)

    focus = st.radio("Detalle", ["Matriz A", "Matriz B"], horizontal=True, key="comp_focus")
    focus_result = result_a if focus == "Matriz A" else result_b
    detail_left, detail_right = st.columns([1.05, 1])
    with detail_left:
        st.markdown(f"**Matriz seleccionada: {focus_result.method}**")
        st.dataframe(plot_weight_matrix(focus_result.W, focus_result.labels), width="stretch")
    with detail_right:
        st.plotly_chart(fig_neighbor_histogram(focus_result.W, focus_result.labels), use_container_width=True)
    st.dataframe(neighbors_table(focus_result.W, polygons), width="stretch", hide_index=True)
    st.info(interpret_weights_matrix(focus_result.W, focus_result.labels, focus_result.method))
    discussion("¿Qué matriz defenderías para analizar difusión de vulnerabilidad social entre parroquias y con qué argumento?")


def lag_section() -> None:
    section_title("Rezago espacial interactivo", "Wy resume el contexto social de cada territorio según la matriz de pesos elegida.")
    left, right = st.columns([0.85, 1.35])
    with left:
        variable = st.selectbox("Variable y", SOCIAL_VARIABLES, index=3, key="lag_variable")
        result = weight_controls("lag", polygons, default_method="Queen")
        standardized = st.checkbox("Usar W estandarizada por filas", value=True, key="lag_standardized")
        W_used = row_standardize(result.W) if standardized else result.W
        lag_values = compute_spatial_lag(result.W, polygons[variable], standardized=standardized)
        lag_col = f"W_{variable}"
        lag_data = polygons.copy()
        lag_data[lag_col] = lag_values
        lag_data["diferencia_y_menos_Wy"] = lag_data[variable] - lag_data[lag_col]
        selected = st.selectbox("Territorio para interpretar", lag_data["id"] + " · " + lag_data["nombre"], key="lag_selected")
        selected_id = selected.split(" · ")[0]
        selected_row = lag_data.loc[lag_data["id"] == selected_id].iloc[0]
        st.metric("Valor propio y", f"{selected_row[variable]:,.2f}")
        st.metric("Rezago Wy", f"{selected_row[lag_col]:,.2f}")
        st.info(lag_interpretation(selected_row, variable, lag_col))
        show_matrix_alerts(W_used, result.labels)
    with right:
        map_y, map_wy = st.columns(2)
        with map_y:
            st.plotly_chart(fig_polygons(lag_data, color_column=variable, selected_id=selected_id, title=f"Mapa de {variable}"), use_container_width=True)
        with map_wy:
            st.plotly_chart(fig_polygons(lag_data, color_column=lag_col, selected_id=selected_id, title=f"Mapa de rezago espacial W·y"), use_container_width=True)
    st.plotly_chart(fig_lag_comparison(lag_data, variable, lag_col), use_container_width=True)
    st.dataframe(
        lag_data[["id", "nombre", variable, lag_col, "diferencia_y_menos_Wy"]].round(2),
        width="stretch",
        hide_index=True,
    )
    discussion("¿Qué territorio parece concentrar un problema junto a sus vecinos y cuál parece un caso atípico local?")


def global_autocorrelation_section() -> None:
    section_title(
        "Autocorrelación espacial global",
        "Moran y Geary permiten evaluar si una variable se agrupa espacialmente más de lo esperable por azar.",
    )
    st.markdown(
        """
        La autocorrelación espacial global pregunta si, en todo el sistema territorial, los valores de una misma
        variable son más parecidos entre vecinos que entre unidades alejadas. Es el puente natural entre la matriz
        de pesos `W`, el rezago espacial `Wy` y pruebas como Moran's I o Geary's C.
        """
    )
    st.caption("Tema basado en: https://vmoprojs.github.io/SpatialEconPython/autocorrelacionglobal/")

    left, right = st.columns([0.85, 1.35])
    with left:
        variable = st.selectbox("Variable social", SOCIAL_VARIABLES, index=3, key="global_auto_variable")
        result = weight_controls("global_auto", polygons, default_method="Queen")
        standardize = st.checkbox(
            "Estandarizar W por filas",
            value=True,
            key="global_auto_standardize",
            help="Hace que cada fila sume 1; el rezago se lee como promedio ponderado de vecinos.",
        )
        permutations = st.slider(
            "Permutaciones Monte Carlo",
            min_value=99,
            max_value=1999,
            value=499,
            step=100,
            key="global_auto_permutations",
        )
        if not np.allclose(result.W, result.W.T):
            st.warning("La matriz seleccionada no es simétrica. Geary's C debe interpretarse con cautela.")
        elif standardize:
            st.info("Aunque la matriz binaria sea simétrica, la estandarización por filas puede cambiar la simetría efectiva.")

        st.latex(r"I = \frac{n}{S_0}\frac{\sum_i\sum_j w_{ij}z_i z_j}{\sum_i z_i^2}")
        st.latex(r"C = \frac{(n-1)}{2S_0}\frac{\sum_i\sum_j w_{ij}(x_i-x_j)^2}{\sum_i z_i^2}")
        st.markdown(
            """
            **Lectura rápida**

            - `Moran's I > 0`: valores similares se agrupan.
            - `Moran's I < 0`: valores altos y bajos se alternan.
            - `Geary's C < 1`: vecinos similares.
            - `Geary's C > 1`: vecinos disímiles.
            """
        )

    moran_result = permutation_test(
        polygons[variable],
        result.W,
        "Moran's I",
        permutations=permutations,
        standardize=standardize,
        seed=2026,
    )
    geary_result = permutation_test(
        polygons[variable],
        result.W,
        "Geary's C",
        permutations=permutations,
        standardize=standardize,
        seed=2027,
    )
    scatter_data = moran_scatter_data(polygons, variable, result.W, standardize=standardize)
    quadrant_data = polygons.copy()
    quadrant_data["z"] = scatter_data["z"]
    quadrant_data["W_z"] = scatter_data["W_z"]
    quadrant_data["cuadrante_moran"] = scatter_data["cuadrante_moran"]

    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Moran's I", f"{moran_result.observed:.3f}")
        m2.metric("p simulado", f"{moran_result.p_sim:.3f}")
        m3.metric("Geary's C", f"{geary_result.observed:.3f}")
        m4.metric("p simulado", f"{geary_result.p_sim:.3f}")
        st.info(moran_result.interpretation)
        st.info(geary_result.interpretation)
        st.dataframe(
            global_autocorrelation_table(moran_result, geary_result).round(4),
            width="stretch",
            hide_index=True,
        )

    tab1, tab2, tab3, tab4 = st.tabs(["Diagrama de Moran", "Mapa de cuadrantes", "Permutaciones", "Tabla territorial"])
    with tab1:
        st.plotly_chart(
            fig_moran_scatter(
                scatter_data,
                moran_result.observed,
                variable,
                title=f"Diagrama de Moran: {variable}",
            ),
            use_container_width=True,
        )
        st.markdown(
            "La pendiente de la línea del diagrama se interpreta como Moran's I cuando se usa una matriz `W` estandarizada por filas."
        )
    with tab2:
        st.plotly_chart(
            fig_moran_quadrant_map(quadrant_data, title=f"Cuadrantes del diagrama de Moran: {variable}"),
            use_container_width=True,
        )
        st.markdown(
            "`Alto-Alto` y `Bajo-Bajo` sugieren continuidad espacial; `Alto-Bajo` y `Bajo-Alto` señalan posibles atípicos espaciales."
        )
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                fig_permutation_distribution(
                    moran_result.simulations,
                    moran_result.observed,
                    moran_result.expected,
                    "Moran's I",
                    title="Moran's I bajo aleatoriedad espacial",
                ),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                fig_permutation_distribution(
                    geary_result.simulations,
                    geary_result.observed,
                    geary_result.expected,
                    "Geary's C",
                    title="Geary's C bajo aleatoriedad espacial",
                ),
                use_container_width=True,
            )
        st.markdown(
            "La permutación reordena los valores entre territorios y mantiene fija la matriz `W`. Así se construye una referencia de aleatoriedad espacial."
        )
    with tab4:
        st.dataframe(
            scatter_data[["id", "nombre", variable, "z", "W_z", "cuadrante_moran"]].round(3),
            width="stretch",
            hide_index=True,
        )

    discussion("¿La variable elegida parece espacialmente aleatoria o hay evidencia de agrupamiento territorial?")


def local_autocorrelation_section() -> None:
    section_title(
        "Autocorrelación espacial local",
        "LISA permite ubicar clústeres y atípicos espaciales que un estadístico global puede ocultar.",
    )
    st.markdown(
        """
        La autocorrelación global resume todo el mapa en un solo número. La autocorrelación local pregunta, en cambio:
        **¿dónde están los clústeres territoriales y dónde aparecen casos atípicos?** El estadístico local de Moran
        combina el valor estandarizado de cada territorio con el rezago espacial de sus vecinos.
        """
    )
    st.caption("Tema basado en: https://vmoprojs.github.io/SpatialEconPython/autocorrelacionlocal/")

    left, right = st.columns([0.85, 1.35])
    with left:
        variable = st.selectbox("Variable social", SOCIAL_VARIABLES, index=3, key="local_auto_variable")
        result = weight_controls("local_auto", polygons, default_method="Queen")
        standardize = st.checkbox(
            "Estandarizar W por filas",
            value=True,
            key="local_auto_standardize",
            help="Hace comparable el contexto de cada territorio como promedio ponderado de vecinos.",
        )
        alpha = st.select_slider(
            "Umbral de significancia",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            key="local_auto_lisa_alpha",
        )
        map_mode = st.radio(
            "Modo del mapa LISA",
            ["Cuadrantes LISA (todos)", "Solo clústeres significativos"],
            index=0,
            horizontal=True,
            key="local_auto_map_mode",
            help="El modo pedagógico muestra el cuadrante de cada territorio aunque su p_sim no sea menor al umbral.",
        )
        permutations = st.slider(
            "Permutaciones Monte Carlo",
            min_value=99,
            max_value=1999,
            value=499,
            step=100,
            key="local_auto_permutations",
        )
        st.latex(r"I_i = z_i \sum_j w_{ij} z_j")
        st.markdown(
            """
            **Clases LISA**

            - `Alto-Alto`: valor alto rodeado de valores altos.
            - `Bajo-Bajo`: valor bajo rodeado de valores bajos.
            - `Alto-Bajo`: posible atípico alto.
            - `Bajo-Alto`: posible atípico bajo.
            - `No significativo`: la evidencia local no supera el umbral elegido.
            """
        )

    local_stats = local_lisa_statistics(
        polygons,
        variable,
        result.W,
        permutations=permutations,
        alpha=alpha,
        standardize=standardize,
        seed=2028,
    )
    local_data = polygons.copy()
    for column in ["z", "W_z", "I_local", "C_local_geary", "p_moran_sim", "p_geary_alto_sim", "cuadrante_moran", "significativo", "cluster_lisa"]:
        local_data[column] = local_stats[column].values

    cluster_display_col = "cuadrante_moran" if map_mode == "Cuadrantes LISA (todos)" else "cluster_lisa"
    local_stats["cluster_mapa"] = local_stats[cluster_display_col]
    local_data["cluster_mapa"] = local_stats["cluster_mapa"].values
    counts = lisa_counts_table(local_stats)
    visible_counts = (
        local_stats["cluster_mapa"]
        .value_counts()
        .reindex(["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto", "No significativo"], fill_value=0)
        .rename_axis("Clase visible en mapa")
        .reset_index(name="n_territorios")
    )
    significant_count = int(local_stats["significativo"].sum())
    selected = st.selectbox("Territorio para interpretar", local_stats["id"] + " · " + local_stats["nombre"], key="local_auto_selected")
    selected_id = selected.split(" · ")[0]
    selected_row = local_stats.loc[local_stats["id"] == selected_id].iloc[0]

    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Territorios significativos", significant_count)
        m2.metric("Alto-Alto visibles", int((local_stats["cluster_mapa"] == "Alto-Alto").sum()))
        m3.metric("Bajo-Bajo visibles", int((local_stats["cluster_mapa"] == "Bajo-Bajo").sum()))
        m4.metric("Atípicos visibles", int(local_stats["cluster_mapa"].isin(["Alto-Bajo", "Bajo-Alto"]).sum()))
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("I local seleccionado", f"{selected_row['I_local']:.3f}")
        s2.metric("p_sim seleccionado", f"{selected_row['p_moran_sim']:.3f}")
        s3.metric("z", f"{selected_row['z']:.2f}")
        s4.metric("Wz", f"{selected_row['W_z']:.2f}")
        st.info(interpret_lisa_row(selected_row, variable))
        st.markdown("**Conteos del mapa visible**")
        st.dataframe(visible_counts, width="stretch", hide_index=True)
        with st.expander("Conteos usando solo significancia"):
            st.dataframe(counts, width="stretch", hide_index=True)
        if significant_count == 0:
            st.warning(
                "Con el umbral actual no hay clústeres LISA significativos. "
                "El mapa pedagógico sigue mostrando cuadrantes para discutir la dirección del patrón; "
                "usa 'Solo clústeres significativos' para ver el filtro estadístico estricto."
            )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Mapa LISA", "I local", "Geary local", "Significancia", "Tabla"])
    with tab1:
        map_title = (
            f"Cuadrantes LISA de todos los territorios: {variable}"
            if map_mode == "Cuadrantes LISA (todos)"
            else f"Clústeres LISA significativos: {variable}"
        )
        st.plotly_chart(
            fig_lisa_cluster_map(
                local_data,
                cluster_column="cluster_mapa",
                value_column="I_local",
                p_column="p_moran_sim",
                selected_id=selected_id,
                title=map_title,
            ),
            use_container_width=True,
        )
        st.markdown(
            "Cada recuadro muestra el id del territorio y su `I local`. El color indica el cuadrante LISA visible según el modo seleccionado."
        )
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                fig_polygons(local_data, color_column="I_local", selected_id=selected_id, title="Mapa de I local de Moran"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                fig_local_stat_bar(local_stats, "I_local", color_column="cluster_mapa", title="I local de Moran por territorio"),
                use_container_width=True,
            )
        st.plotly_chart(
            fig_local_stat_histogram(local_stats, "I_local", title="Distribución de I local"),
            use_container_width=True,
        )
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                fig_polygons(local_data, color_column="C_local_geary", selected_id=selected_id, title="Mapa de Geary local"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                fig_local_stat_bar(local_stats, "C_local_geary", color_column="cluster_mapa", title="Geary local por territorio"),
                use_container_width=True,
            )
        st.markdown(
            "Valores altos de Geary local indican mayor diferencia respecto a los vecinos; valores bajos sugieren similitud local."
        )
    with tab4:
        sig_data = local_data.copy()
        sig_data["-log10_p"] = -np.log10(sig_data["p_moran_sim"].clip(lower=1e-6))
        st.plotly_chart(
            fig_polygons(sig_data, color_column="-log10_p", selected_id=selected_id, title="Evidencia local: -log10(p_sim)"),
            use_container_width=True,
        )
        st.dataframe(
            local_stats[["id", "nombre", "p_moran_sim", "z_moran_sim", "cluster_lisa"]].sort_values("p_moran_sim").round(4),
            width="stretch",
            hide_index=True,
        )
    with tab5:
        st.dataframe(
            local_stats[
                [
                    "id",
                    "nombre",
                    variable,
                    "z",
                    "W_z",
                    "I_local",
                    "C_local_geary",
                    "p_moran_sim",
                    "p_geary_alto_sim",
                    "cuadrante_moran",
                    "cluster_mapa",
                    "cluster_lisa",
                ]
            ].round(4),
            width="stretch",
            hide_index=True,
        )

    discussion("¿Qué cambia en la interpretación territorial cuando pasamos de un Moran global a un mapa LISA local?")


def clustering_regionalization_section() -> None:
    section_title(
        "Clustering y regionalización",
        "Del parecido multivariado a regiones territorialmente coherentes.",
    )
    st.markdown(
        """
        El clustering agrupa territorios con perfiles sociales similares. La regionalización agrega una condición
        territorial: los miembros de una misma región deberían estar conectados por una relación espacial, como
        contigüidad `Queen`, `Rook` o una red `kNN`.
        """
    )
    st.caption("Tema basado en: https://vmoprojs.github.io/SpatialEconPython/clusteringregionalization/")

    left, right = st.columns([0.82, 1.38])
    with left:
        variables = st.multiselect(
            "Variables del perfil multivariado",
            SOCIAL_VARIABLES,
            default=["ingreso_medio", "tasa_pobreza", "acceso_servicios", "índice_vulnerabilidad"],
            key="cluster_variables",
        )
        if len(variables) < 2:
            st.warning("Selecciona al menos dos variables. Se usará un par base para mantener activos los gráficos.")
            variables = SOCIAL_VARIABLES[:2]

        scale_method = st.selectbox(
            "Estandarización",
            ["robusta (mediana/IQR)", "z-score", "min-max 0-1", "sin escalar"],
            key="cluster_scale_method",
            help="Las distancias multivariadas son sensibles a la escala de cada variable.",
        )
        k = st.slider(
            "Número de clústeres o regiones",
            min_value=2,
            max_value=min(6, len(polygons)),
            value=4,
            step=1,
            key="cluster_k",
        )
        algorithm = st.selectbox(
            "Solución principal",
            ["K-means", "Ward jerárquico", "Regionalización Ward + conectividad"],
            key="cluster_algorithm",
        )
        seed = st.slider("Semilla K-means", 1, 9999, 2026, 1, key="cluster_seed")
        connectivity_method = st.selectbox(
            "Conectividad para regionalización",
            ["Queen", "Rook", "k vecinos más cercanos"],
            key="cluster_connectivity_method",
        )
        regional_k = 3
        if connectivity_method == "k vecinos más cercanos":
            regional_k = st.slider(
                "k de conectividad regional",
                min_value=1,
                max_value=max(1, len(polygons) - 1),
                value=3,
                step=1,
                key="cluster_regional_k",
            )
        selected = st.selectbox("Territorio para inspeccionar", polygons["id"] + " · " + polygons["nombre"], key="cluster_selected")
        selected_id = selected.split(" · ")[0]

    X, scaling_table = scale_features(polygons, variables, scale_method)
    regional_weights = build_weights_by_method(
        connectivity_method,
        polygons,
        k=regional_k,
        symmetric_knn=True,
    )
    solutions = {
        "K-means": ClusterSolution(
            "K-means",
            kmeans_labels(X, k, seed=seed),
            "Agrupa por cercanía al centroide del clúster, sin exigir continuidad territorial.",
        ),
        "Ward jerárquico": ClusterSolution(
            "Ward jerárquico",
            ward_labels(X, k),
            "Fusiona grupos minimizando la pérdida de homogeneidad interna, sin restricción espacial.",
        ),
        "Regionalización Ward + conectividad": ClusterSolution(
            "Regionalización Ward + conectividad",
            ward_labels(X, k, connectivity=regional_weights.W),
            "Fusiona solo grupos conectados por la matriz espacial elegida.",
        ),
    }
    selected_solution = solutions[algorithm]
    labels = selected_solution.labels
    selected_idx = int(polygons.index[polygons["id"] == selected_id][0])
    selected_cluster = int(labels[selected_idx])

    scaled_profile_data = polygons[["id", "nombre"]].copy()
    scaled_scatter_data = polygons[["id", "nombre"]].copy()
    for idx, variable in enumerate(variables):
        scaled_profile_data[variable] = X[:, idx]
        scaled_scatter_data[variable] = X[:, idx]

    profile_original = cluster_profile_table(polygons, labels, variables)
    profile_scaled = cluster_profile_table(scaled_profile_data, labels, variables)
    territory_table = territory_cluster_table(polygons, labels, variables)
    selected_profile = profile_original.loc[profile_original["clúster"] == f"C{selected_cluster}"].iloc[0]

    metrics_rows = []
    for solution in solutions.values():
        row = {"solución": solution.name}
        row.update(solution_metrics(X, solution.labels, regional_weights.W, polygons))
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    selected_metrics = metrics_df.loc[metrics_df["solución"] == selected_solution.name].iloc[0]

    with right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ajuste CH", compact_number(float(selected_metrics["Calinski-Harabasz"])))
        c2.metric("Silhouette", compact_number(float(selected_metrics["silhouette"])))
        c3.metric("Fragmentación", int(selected_metrics["fragmentación"]))
        c4.metric("Vecinos internos", f"{float(selected_metrics['vecinos_internos_%']):.0f}%")
        st.info(
            f"{selected_solution.description} El territorio {selected_id} queda en el clúster C{selected_cluster}."
        )
        cards = st.columns(3)
        cards[0].markdown(
            "<div class='concept-card'><b>Clustering</b><br><span class='small-note'>Busca similitud estadística entre perfiles multivariados.</span></div>",
            unsafe_allow_html=True,
        )
        cards[1].markdown(
            "<div class='concept-card'><b>Regionalización</b><br><span class='small-note'>Mantiene similitud, pero exige conectividad territorial.</span></div>",
            unsafe_allow_html=True,
        )
        cards[2].markdown(
            "<div class='concept-card'><b>Trade-off</b><br><span class='small-note'>Más coherencia geográfica puede reducir ajuste estadístico puro.</span></div>",
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Idea central", "Clustering", "Regionalización", "Evaluación", "Actividad guiada"]
    )

    with tab1:
        st.markdown(
            """
            **Ruta conceptual**

            1. Elegimos variables que describen un perfil social multivariado.
            2. Estandarizamos para que una variable grande no domine las distancias.
            3. Agrupamos territorios con perfiles parecidos.
            4. Revisamos si esos grupos forman manchas conectadas o parches dispersos.
            5. Si necesitamos regiones operables, imponemos conectividad espacial.
            """
        )
        col_a, col_b = st.columns([0.92, 1.08])
        with col_a:
            st.markdown("**Parámetros de escalado**")
            st.dataframe(scaling_table.round(3), width="stretch", hide_index=True)
            st.markdown(
                "Las etiquetas `C1`, `C2`, etc. no son rangos ni prioridades: solo indican pertenencia a grupos."
            )
        with col_b:
            x_axis = st.selectbox("Eje X del espacio multivariado", variables, key="cluster_concept_x")
            y_options = [variable for variable in variables if variable != x_axis] or variables
            y_axis = st.selectbox("Eje Y del espacio multivariado", y_options, key="cluster_concept_y")
            st.plotly_chart(
                fig_cluster_scatter(
                    scaled_scatter_data,
                    labels,
                    x_axis,
                    y_axis,
                    title=f"Territorios en espacio escalado: {x_axis} vs {y_axis}",
                ),
                use_container_width=True,
            )

    with tab2:
        col_a, col_b = st.columns([1.05, 0.95])
        with col_a:
            st.plotly_chart(
                fig_cluster_map(
                    polygons,
                    labels,
                    selected_id=selected_id,
                    variables=variables,
                    title=f"{selected_solution.name}: {k} grupos",
                ),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                fig_cluster_profile(
                    profile_scaled,
                    variables,
                    title=f"Perfil medio escalado: {selected_solution.name}",
                ),
                use_container_width=True,
            )
        st.markdown("**Perfil en unidades originales**")
        st.dataframe(profile_original, width="stretch", hide_index=True)
        st.markdown("**Asignación territorial**")
        st.dataframe(territory_table.round(2), width="stretch", hide_index=True)

    with tab3:
        st.markdown(
            f"La restricción espacial usa `{regional_weights.method}` ({regional_weights.parameter_label}). "
            "Un grupo regionalizado solo puede crecer fusionando territorios conectados por esa matriz."
        )
        map_a, map_b, map_c = st.columns(3)
        with map_a:
            st.plotly_chart(
                fig_cluster_map(polygons, solutions["K-means"].labels, title="K-means sin restricción"),
                use_container_width=True,
            )
        with map_b:
            st.plotly_chart(
                fig_cluster_map(polygons, solutions["Ward jerárquico"].labels, title="Ward sin restricción"),
                use_container_width=True,
            )
        with map_c:
            st.plotly_chart(
                fig_cluster_map(
                    polygons,
                    solutions["Regionalización Ward + conectividad"].labels,
                    title="Ward con conectividad",
                ),
                use_container_width=True,
            )
        col_a, col_b = st.columns([0.95, 1.05])
        with col_a:
            component_rows = []
            for solution in solutions.values():
                components = count_components_by_label(regional_weights.W, solution.labels)
                for cluster, component_count in components.items():
                    component_rows.append(
                        {
                            "solución": solution.name,
                            "clúster": f"C{cluster}",
                            "n_territorios": int((solution.labels == cluster).sum()),
                            "componentes": component_count,
                            "lectura": "conectado" if component_count == 1 else "fragmentado",
                        }
                    )
            st.dataframe(pd.DataFrame(component_rows), width="stretch", hide_index=True)
        with col_b:
            st.plotly_chart(
                plot_spatial_weights_network(
                    polygons,
                    regional_weights.W,
                    selected_id=selected_id,
                    title=f"Conectividad usada: {regional_weights.method}",
                ),
                use_container_width=True,
            )

    with tab4:
        st.markdown(
            """
            Una solución no se evalúa con un solo número. El ajuste estadístico mira homogeneidad interna y separación
            entre grupos; la coherencia espacial mira si esos grupos forman regiones compactas y conectadas.
            """
        )
        display_metrics = metrics_df.copy()
        numeric_cols = [
            "Calinski-Harabasz",
            "silhouette",
            "vecinos_internos_%",
            "compacidad_bbox",
        ]
        display_metrics[numeric_cols] = display_metrics[numeric_cols].round(3)
        st.dataframe(display_metrics, width="stretch", hide_index=True)
        fig_fit = go.Figure()
        fig_fit.add_trace(
            go.Bar(
                x=metrics_df["solución"],
                y=metrics_df["Calinski-Harabasz"],
                name="Calinski-Harabasz",
                marker=dict(color="#457b9d"),
            )
        )
        fig_fit.update_layout(
            title="Ajuste estadístico: mayor suele ser mejor",
            height=330,
            margin=dict(l=10, r=10, t=55, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        fig_space = go.Figure()
        fig_space.add_trace(
            go.Bar(
                x=metrics_df["solución"],
                y=metrics_df["vecinos_internos_%"],
                name="Vecinos internos",
                marker=dict(color="#2a9d8f"),
            )
        )
        fig_space.add_trace(
            go.Bar(
                x=metrics_df["solución"],
                y=metrics_df["compacidad_bbox"] * 100,
                name="Compacidad bbox",
                marker=dict(color="#f4a261"),
            )
        )
        fig_space.update_layout(
            title="Coherencia espacial: mayor indica regiones más compactas o conectadas",
            height=330,
            margin=dict(l=10, r=10, t=55, b=10),
            yaxis_title="Porcentaje / índice x100",
            barmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
        )
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(fig_fit, use_container_width=True)
        col_b.plotly_chart(fig_space, use_container_width=True)
        st.info(
            "Para planificación territorial suele ser preferible una solución algo menos óptima en distancia, "
            "pero más clara como región conectada y comunicable."
        )

    with tab5:
        st.markdown(
            """
            **Actividad: diseña y defiende una regionalización**

            1. Formula una pregunta social donde tenga sentido agrupar territorios.
            2. Elige variables y explica por qué necesitan estandarización.
            3. Compara K-means, Ward y regionalización: ¿qué solución comunica mejor el patrón?
            4. Revisa el clúster del territorio foco y su perfil promedio.
            5. Decide qué pesa más en tu conclusión: ajuste estadístico o coherencia territorial.
            """
        )
        a1, a2, a3 = st.columns(3)
        a1.metric("Territorio foco", selected_id)
        a2.metric("Clúster foco", f"C{selected_cluster}")
        a3.metric("Tamaño del clúster", int(selected_profile["n_territorios"]))
        st.markdown("**Perfil del clúster foco**")
        st.dataframe(pd.DataFrame([selected_profile]), width="stretch", hide_index=True)
        question = st.text_input(
            "Pregunta social del grupo",
            placeholder="Ejemplo: ¿qué territorios comparten un perfil de vulnerabilidad y baja accesibilidad?",
            key="cluster_activity_question",
        )
        interpretation = st.text_area(
            "Interpretación y decisión metodológica",
            height=220,
            placeholder=(
                "Explica variables, escalado, número de grupos, diferencias entre clustering y regionalización, "
                "y cómo usarías la solución en una decisión territorial."
            ),
            key="cluster_activity_interpretation",
        )
        summary = (
            f"Pregunta social: {question}\n"
            f"Variables: {', '.join(variables)}\n"
            f"Estandarización: {scale_method}\n"
            f"Número de grupos/regiones: {k}\n"
            f"Solución principal: {selected_solution.name}\n"
            f"Conectividad regional: {regional_weights.method} ({regional_weights.parameter_label})\n"
            f"Territorio foco: {selected}\n"
            f"Clúster foco: C{selected_cluster}\n"
            f"Calinski-Harabasz: {compact_number(float(selected_metrics['Calinski-Harabasz']), 4)}\n"
            f"Silhouette: {compact_number(float(selected_metrics['silhouette']), 4)}\n"
            f"Fragmentación: {int(selected_metrics['fragmentación'])}\n"
            f"Vecinos internos: {float(selected_metrics['vecinos_internos_%']):.2f}%\n\n"
            f"Interpretación:\n{interpretation}"
        )
        st.download_button(
            "Descargar actividad",
            data=summary.encode("utf-8"),
            file_name="actividad_clustering_regionalizacion.txt",
            mime="text/plain",
        )

    discussion("¿Cuándo aceptarías perder un poco de ajuste estadístico para ganar una región conectada y comunicable?")


def spatial_regression_section() -> None:
    section_title(
        "Regresión espacial",
        "Cuando la relación entre variables también viaja por la vecindad.",
    )
    st.markdown(
        """
        En datos territoriales, una regresión lineal puede fallar si ignora dependencia espacial, spillovers o
        heterogeneidad territorial. Esta sección usa modelos didácticos para mostrar cuándo MCO es insuficiente
        y cómo cambian la interpretación, los residuos y las predicciones al incorporar `W`.
        """
    )
    st.caption("Tema basado en: https://vmoprojs.github.io/SpatialEconPython/spatialregression/")

    left, right = st.columns([0.82, 1.38])
    with left:
        y_name = st.selectbox(
            "Variable dependiente y",
            SOCIAL_VARIABLES,
            index=3,
            key="spatial_reg_y",
        )
        x_options = [variable for variable in SOCIAL_VARIABLES if variable != y_name]
        default_x = [variable for variable in ["ingreso_medio", "acceso_servicios", "tasa_pobreza"] if variable in x_options][:2]
        if len(default_x) < 2:
            default_x = x_options[:2]
        x_names = st.multiselect(
            "Variables explicativas X",
            x_options,
            default=default_x,
            key="spatial_reg_x",
        )
        if not x_names:
            st.warning("Selecciona al menos una covariable. Se usará la primera disponible para mantener activo el modelo.")
            x_names = x_options[:1]
        if len(x_names) > 3:
            st.warning("Para mantener grados de libertad en estos datos sintéticos, se usarán las primeras tres covariables.")
            x_names = x_names[:3]

        model_name = st.selectbox(
            "Modelo a interpretar",
            list(MODEL_DESCRIPTIONS.keys()),
            key="spatial_reg_model",
        )
        standardize_model = st.checkbox(
            "Estandarizar y y X",
            value=True,
            key="spatial_reg_standardize",
            help="Hace comparables los coeficientes. Las predicciones quedan en desviaciones estándar de y.",
        )
        weights_result = weight_controls("spatial_reg", polygons, default_method="Queen")
        permutations = st.slider(
            "Permutaciones para Moran de residuos",
            min_value=99,
            max_value=999,
            value=499,
            step=100,
            key="spatial_reg_permutations",
        )
        selected = st.selectbox(
            "Territorio para impactos",
            polygons["id"] + " · " + polygons["nombre"],
            key="spatial_reg_selected",
        )
        selected_id = selected.split(" · ")[0]

    model_names = list(MODEL_DESCRIPTIONS.keys())
    fitted_models = [
        fit_spatial_regression(polygons, y_name, x_names, weights_result.W, name, standardize=standardize_model)
        for name in model_names
    ]
    selected_model = next(result for result in fitted_models if result.model_name == model_name)
    model_data, scaling_table = prepare_regression_data(polygons, y_name, x_names, standardize_model)
    selected_idx = int(polygons.index[polygons["id"] == selected_id][0])

    residual_moran = permutation_test(
        selected_model.residuals,
        weights_result.W,
        "Moran's I",
        permutations=permutations,
        standardize=True,
        seed=2040,
    )
    bp_result = breusch_pagan_test(selected_model)
    comparison = model_comparison_table(fitted_models)
    coeffs = coefficient_table(selected_model)

    residual_data = polygons.copy()
    residual_data["observado_modelo"] = selected_model.y_values
    residual_data["ajustado"] = selected_model.fitted
    residual_data["residuo"] = selected_model.residuals
    residual_data["abs_residuo"] = np.abs(selected_model.residuals)

    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", compact_number(selected_model.r2))
        m2.metric("AIC", compact_number(selected_model.aic))
        m3.metric("Moran residuos", compact_number(residual_moran.observed, 3))
        m4.metric("p Moran", compact_number(residual_moran.p_sim, 3))
        st.info(MODEL_DESCRIPTIONS[model_name])
        cards = st.columns(3)
        cards[0].markdown(
            "<div class='concept-card'><b>Dependencia espacial</b><br><span class='small-note'>Los residuos o la propia y se parecen entre vecinos.</span></div>",
            unsafe_allow_html=True,
        )
        cards[1].markdown(
            "<div class='concept-card'><b>Spillovers</b><br><span class='small-note'>Cambios en un territorio pueden afectar a sus vecinos.</span></div>",
            unsafe_allow_html=True,
        )
        cards[2].markdown(
            "<div class='concept-card'><b>Heterogeneidad</b><br><span class='small-note'>La varianza o los parámetros no son estables en el espacio.</span></div>",
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Modelo", "Diagnóstico", "Tipología", "Impactos", "Actividad guiada"]
    )

    with tab1:
        equation_by_model = {
            "MCO / OLS": r"y = X\beta + \varepsilon",
            "SLX (X y WX)": r"y = X\beta + WX\theta + \varepsilon",
            "SLM / SAR didáctico (Wy)": r"y = \rho Wy + X\beta + \varepsilon",
            "SDM didáctico (Wy + WX)": r"y = \rho Wy + X\beta + WX\theta + \varepsilon",
        }
        st.latex(equation_by_model[model_name])
        col_a, col_b = st.columns([0.95, 1.05])
        with col_a:
            fit_fig = go.Figure()
            fit_fig.add_trace(
                go.Scatter(
                    x=residual_data["observado_modelo"],
                    y=residual_data["ajustado"],
                    mode="markers+text",
                    text=residual_data["id"],
                    textposition="top center",
                    marker=dict(size=14, color="#457b9d", line=dict(color="white", width=1.5)),
                    hovertemplate="<b>%{text}</b><br>observado=%{x:.2f}<br>ajustado=%{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )
            axis_min = float(min(residual_data["observado_modelo"].min(), residual_data["ajustado"].min()))
            axis_max = float(max(residual_data["observado_modelo"].max(), residual_data["ajustado"].max()))
            fit_fig.add_shape(
                type="line",
                x0=axis_min,
                y0=axis_min,
                x1=axis_max,
                y1=axis_max,
                line=dict(color="#e76f51", dash="dash"),
            )
            fit_fig.update_layout(
                title="Observado frente a ajustado",
                height=420,
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis_title=f"{y_name} observado",
                yaxis_title=f"{y_name} ajustado",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fit_fig, use_container_width=True)
        with col_b:
            coef_plot = coeffs.loc[coeffs["término"] != "constante"].copy()
            coef_plot["significativo"] = np.where(coef_plot["p_valor"] < 0.05, "p < 0.05", "p >= 0.05")
            coef_fig = go.Figure()
            colors = ["#2a9d8f" if value < 0.05 else "#9ca3af" for value in coef_plot["p_valor"]]
            coef_fig.add_trace(
                go.Bar(
                    x=coef_plot["término"],
                    y=coef_plot["coeficiente"],
                    marker=dict(color=colors),
                    customdata=coef_plot[["p_valor", "error_estándar"]],
                    hovertemplate="%{x}<br>coef=%{y:.3f}<br>p=%{customdata[0]:.3f}<br>se=%{customdata[1]:.3f}<extra></extra>",
                )
            )
            coef_fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
            coef_fig.update_layout(
                title="Coeficientes del modelo",
                height=420,
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis_title="Término",
                yaxis_title="Coeficiente",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(coef_fig, use_container_width=True)
        st.markdown("**Tabla de coeficientes**")
        st.dataframe(coeffs.round(4), width="stretch", hide_index=True)
        st.markdown("**Comparación de especificaciones**")
        st.dataframe(comparison.round(4), width="stretch", hide_index=True)

    with tab2:
        st.markdown(
            """
            El diagnóstico parte de una pregunta simple: después de explicar `y`, ¿los residuos todavía tienen patrón
            espacial? Si Moran de residuos es alto y significativo, el modelo lineal dejó estructura territorial sin explicar.
            """
        )
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Moran I residuos", compact_number(residual_moran.observed, 3))
        d2.metric("p simulado", compact_number(residual_moran.p_sim, 3))
        d3.metric("Breusch-Pagan LM", compact_number(bp_result["LM"], 2))
        d4.metric("p BP", compact_number(bp_result["p_valor"], 3))
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                fig_polygons(residual_data, color_column="residuo", selected_id=selected_id, title="Mapa de residuos"),
                use_container_width=True,
            )
        with col_b:
            residual_scatter = moran_scatter_data(residual_data, "residuo", weights_result.W, standardize=True)
            st.plotly_chart(
                fig_moran_scatter(
                    residual_scatter,
                    residual_moran.observed,
                    "residuo",
                    title="Diagrama de Moran para residuos",
                ),
                use_container_width=True,
            )
        signals = spatial_diagnostic_signals(selected_model, weights_result.W)
        st.dataframe(signals.round(3), width="stretch", hide_index=True)
        if residual_moran.p_sim < 0.05:
            st.warning(
                "Los residuos muestran autocorrelación espacial. Revisa SLM/SDM si la dependencia parece sustantiva "
                "o SEM/variables omitidas si el patrón queda en el error."
            )
        else:
            st.success("Con esta especificación, los residuos no muestran evidencia fuerte de autocorrelación espacial.")
        st.info(
            "En una aplicación formal se usarían pruebas LM robustas, razón de verosimilitud o Wald. "
            "Aquí se muestran señales didácticas para entender la lógica de selección del modelo."
        )

    with tab3:
        taxonomy = pd.DataFrame(
            [
                {
                    "modelo": "MCO / OLS",
                    "ecuación": "y = Xβ + ε",
                    "qué espacializa": "Nada",
                    "cuándo discutirlo": "Base de comparación; útil si residuos no tienen patrón espacial.",
                },
                {
                    "modelo": "SLX",
                    "ecuación": "y = Xβ + WXθ + ε",
                    "qué espacializa": "Covariables de vecinos",
                    "cuándo discutirlo": "Spillovers de variables explicativas sin feedback de y.",
                },
                {
                    "modelo": "SLM / SAR",
                    "ecuación": "y = ρWy + Xβ + ε",
                    "qué espacializa": "Variable dependiente",
                    "cuándo discutirlo": "Interdependencia sustantiva entre territorios.",
                },
                {
                    "modelo": "SDM",
                    "ecuación": "y = ρWy + Xβ + WXθ + ε",
                    "qué espacializa": "y y X",
                    "cuándo discutirlo": "Dependencia de resultados y spillovers de covariables.",
                },
                {
                    "modelo": "SEM",
                    "ecuación": "y = Xβ + u; u = λWu + ε",
                    "qué espacializa": "Error",
                    "cuándo discutirlo": "Variables omitidas o shocks no observados con patrón espacial.",
                },
            ]
        )
        st.dataframe(taxonomy, width="stretch", hide_index=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                **Lectura de parámetros**

                - `β`: relación propia entre `X` e `y`.
                - `θ`: efecto de covariables vecinas `WX`.
                - `ρ`: dependencia de `y` respecto a `Wy`.
                - `λ`: dependencia espacial en el error.
                """
            )
            st.latex(r"(I - \rho W)y = X\beta + WX\theta + \varepsilon")
        with col_b:
            st.code(
                """# En una aplicación formal con PySAL/spreg:
from spreg import ML_Lag, ML_Error

modelo_slm = ML_Lag(y, X, w=w_queen)
modelo_sem = ML_Error(y, X, w=w_queen)

# Luego: revisar rho/lambda, pseudo R², AIC,
# heterocedasticidad y Moran de residuos.""",
                language="python",
            )

    with tab4:
        shock_variable = st.selectbox(
            "Variable que recibe el cambio",
            x_names,
            key="spatial_reg_shock_variable",
        )
        if standardize_model:
            shock = st.slider(
                "Cambio aplicado al territorio foco (desviaciones estándar)",
                min_value=-2.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="spatial_reg_shock_std",
            )
            unit_label = "desv. est. de y"
        else:
            raw_std = float(polygons[shock_variable].std(ddof=0))
            shock = st.slider(
                f"Cambio aplicado a {shock_variable}",
                min_value=-2.0 * raw_std,
                max_value=2.0 * raw_std,
                value=raw_std,
                step=max(raw_std / 20, 0.1),
                key="spatial_reg_shock_raw",
            )
            unit_label = f"unidades de {y_name}"
        impacts = predict_spatial_shock(model_data, selected_model, shock_variable, selected_idx, shock)
        impact_map = polygons.copy()
        impact_map["cambio_predicho"] = impacts["cambio_predicho"].values
        direct_effect = float(impacts.loc[selected_idx, "cambio_predicho"])
        indirect_effect = float(impacts.loc[impacts.index != selected_idx, "cambio_predicho"].sum())
        total_effect = float(impacts["cambio_predicho"].sum())
        i1, i2, i3 = st.columns(3)
        i1.metric("Impacto directo", f"{direct_effect:.3f}")
        i2.metric("Impacto indirecto total", f"{indirect_effect:.3f}")
        i3.metric("Impacto total", f"{total_effect:.3f}")
        col_a, col_b = st.columns([1.05, 0.95])
        with col_a:
            st.plotly_chart(
                fig_polygons(
                    impact_map,
                    color_column="cambio_predicho",
                    selected_id=selected_id,
                    title=f"Cambio predicho en {y_name}",
                    colorscale="RdBu",
                ),
                use_container_width=True,
            )
        with col_b:
            impact_fig = go.Figure(
                go.Bar(
                    x=impacts["id"],
                    y=impacts["cambio_predicho"],
                    marker=dict(color=np.where(impacts["tipo_impacto"] == "directo", "#e76f51", "#457b9d")),
                    customdata=impacts[["nombre", "tipo_impacto"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>%{x}<br>cambio=%{y:.3f}<br>%{customdata[1]}<extra></extra>",
                )
            )
            impact_fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
            impact_fig.update_layout(
                title=f"Respuesta territorial ({unit_label})",
                height=420,
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis_title="Territorio",
                yaxis_title="Cambio predicho",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(impact_fig, use_container_width=True)
        st.dataframe(impacts.round(4), width="stretch", hide_index=True)
        st.markdown(
            "En modelos con `Wy`, un cambio puede circular por el multiplicador espacial. "
            "En modelos con `WX`, parte del efecto aparece en vecinos por las covariables rezagadas."
        )

    with tab5:
        st.markdown(
            """
            **Actividad: selecciona y defiende un modelo espacial**

            1. Plantea una hipótesis sustantiva para `y`.
            2. Justifica las covariables y la matriz `W`.
            3. Estima MCO y revisa residuos: ¿queda patrón espacial?
            4. Compara MCO, SLX, SLM y SDM con AIC/BIC, `R²` y diagnóstico.
            5. Decide si el problema parece de spillovers, rezago de `y`, error espacial o heterogeneidad.
            6. Interpreta un impacto directo e indirecto para el territorio foco.
            """
        )
        response = st.text_area(
            "Respuesta del grupo",
            height=240,
            placeholder="Ejemplo: La vulnerabilidad parece depender de ingreso, servicios y de la situación de territorios vecinos...",
            key="spatial_reg_activity_response",
        )
        summary = (
            f"Variable dependiente: {y_name}\n"
            f"Covariables: {', '.join(x_names)}\n"
            f"Matriz W: {weights_result.method} ({weights_result.parameter_label})\n"
            f"Modelo interpretado: {selected_model.model_name}\n"
            f"Estandarizado: {'sí' if standardize_model else 'no'}\n"
            f"R2: {selected_model.r2:.4f}\n"
            f"AIC: {selected_model.aic:.4f}\n"
            f"Moran residuos: {residual_moran.observed:.4f} | p_sim: {residual_moran.p_sim:.4f}\n"
            f"Breusch-Pagan LM: {bp_result['LM']:.4f} | p: {bp_result['p_valor']:.4f}\n"
            f"Territorio foco: {selected}\n\n"
            f"Respuesta:\n{response}"
        )
        st.download_button(
            "Descargar actividad",
            data=summary.encode("utf-8"),
            file_name="actividad_regresion_espacial.txt",
            mime="text/plain",
        )

    discussion("¿El patrón espacial está en la variable dependiente, en las covariables vecinas o en lo que el modelo dejó sin explicar?")


def activity_section() -> None:
    section_title(
        "Actividad guiada para estudiantes",
        "Cierre aplicado: del problema social a W, rezago espacial, Moran global y LISA local.",
    )
    left, right = st.columns([0.9, 1.1])
    with left:
        data_type = st.selectbox("Tipo de dato para tu problema", list(DATA_TYPE_INFO.keys()), key="activity_type")
        variable = st.selectbox("Variable social a interpretar", SOCIAL_VARIABLES, index=3, key="activity_var")
        result = weight_controls("activity", polygons, default_method="Queen")
        standardize = st.checkbox(
            "Usar W estandarizada por filas",
            value=True,
            key="activity_standardize",
            help="Hace que el rezago y los estadísticos usen un promedio ponderado de vecinos.",
        )
        alpha = st.select_slider("Alpha para LISA", options=[0.01, 0.05, 0.10], value=0.05, key="activity_lisa_alpha")
        permutations = st.slider(
            "Permutaciones",
            min_value=99,
            max_value=999,
            value=499,
            step=100,
            key="activity_permutations",
        )
        selected = st.selectbox("Territorio foco", polygons["id"] + " · " + polygons["nombre"], key="activity_selected")
        selected_id = selected.split(" · ")[0]
        st.markdown(f"**Pista conceptual.** {method_guidance(result.method)}")
        st.markdown(f"**Tipo de dato elegido.** {DATA_TYPE_INFO[data_type]['definición']}")
        if result.method in {"Rook", "Queen"} and data_type not in {"Polígonos", "Datos areales agregados"}:
            st.warning("Rook/Queen suelen requerir unidades areales. Para puntos, distancia, kNN o kernel suele ser más natural.")

    moran_result = permutation_test(
        polygons[variable],
        result.W,
        "Moran's I",
        permutations=permutations,
        standardize=standardize,
        seed=2030,
    )
    geary_result = permutation_test(
        polygons[variable],
        result.W,
        "Geary's C",
        permutations=permutations,
        standardize=standardize,
        seed=2031,
    )
    lag_values = compute_spatial_lag(result.W, polygons[variable], standardized=standardize)
    activity_data = polygons.copy()
    activity_data[f"W_{variable}"] = lag_values
    activity_data["diferencia_y_menos_Wy"] = activity_data[variable] - activity_data[f"W_{variable}"]
    local_stats = local_lisa_statistics(
        polygons,
        variable,
        result.W,
        permutations=permutations,
        alpha=alpha,
        standardize=standardize,
        seed=2032,
    )
    local_data = polygons.copy()
    for column in ["z", "W_z", "I_local", "C_local_geary", "p_moran_sim", "cuadrante_moran", "cluster_lisa"]:
        local_data[column] = local_stats[column].values
    local_data["cluster_mapa"] = local_stats["cuadrante_moran"].values
    selected_row = activity_data.loc[activity_data["id"] == selected_id].iloc[0]
    selected_lisa = local_stats.loc[local_stats["id"] == selected_id].iloc[0]

    with right:
        st.markdown(
            """
            1. Define un problema social territorial y la unidad de análisis.
            2. Justifica el tipo de dato espacial y el criterio de pesos `W`.
            3. Interpreta el rezago espacial del territorio foco: ¿se parece a sus vecinos?
            4. Lee Moran's I y Geary's C: ¿hay agrupamiento global o patrón aleatorio?
            5. Usa LISA: ¿hay clústeres locales o atípicos territoriales?
            6. Explica qué cambiaría si se usa otro criterio de vecindad.
            7. Señala riesgos de una mala elección de `W` para política pública o IA territorial.
            """
        )
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Moran's I", f"{moran_result.observed:.3f}")
        g2.metric("p Moran", f"{moran_result.p_sim:.3f}")
        g3.metric("Geary's C", f"{geary_result.observed:.3f}")
        g4.metric("p Geary", f"{geary_result.p_sim:.3f}")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("y territorio", f"{selected_row[variable]:.2f}")
        l2.metric("Wy vecinos", f"{selected_row[f'W_{variable}']:.2f}")
        l3.metric("I local", f"{selected_lisa['I_local']:.3f}")
        l4.metric("p LISA", f"{selected_lisa['p_moran_sim']:.3f}")
        st.info(moran_result.interpretation)
        st.info(interpret_lisa_row(selected_lisa, variable))

    tab1, tab2, tab3, tab4 = st.tabs(["Mapa y W", "Global", "Local LISA", "Respuesta"])
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_spatial_weights_network(polygons, result.W, selected_id=selected_id, title=f"W seleccionada: {result.method}"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                fig_lag_comparison(activity_data, variable, f"W_{variable}"),
                use_container_width=True,
            )
        st.dataframe(
            activity_data[["id", "nombre", variable, f"W_{variable}", "diferencia_y_menos_Wy"]].round(3),
            width="stretch",
            hide_index=True,
        )
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(
                global_autocorrelation_table(moran_result, geary_result).round(4),
                width="stretch",
                hide_index=True,
            )
        with col_b:
            scatter_data = moran_scatter_data(polygons, variable, result.W, standardize=standardize)
            st.plotly_chart(
                fig_moran_scatter(scatter_data, moran_result.observed, variable, title=f"Diagrama de Moran: {variable}"),
                use_container_width=True,
            )
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                fig_lisa_cluster_map(
                    local_data,
                    cluster_column="cluster_mapa",
                    value_column="I_local",
                    p_column="p_moran_sim",
                    selected_id=selected_id,
                    title=f"Cuadrantes LISA: {variable}",
                ),
                use_container_width=True,
            )
        with col_b:
            st.dataframe(lisa_counts_table(local_stats), width="stretch", hide_index=True)
            st.dataframe(
                local_stats[["id", "nombre", "I_local", "p_moran_sim", "cuadrante_moran", "cluster_lisa"]].round(4),
                width="stretch",
                hide_index=True,
            )
    with tab4:
        response = st.text_area(
            "Respuesta del grupo",
            height=300,
            placeholder="Ejemplo: Queremos estudiar vulnerabilidad territorial en parroquias urbanas...",
            key="activity_response",
        )
        summary = (
            f"Tipo de dato: {data_type}\n"
            f"Criterio de pesos: {result.method} ({result.parameter_label})\n"
            f"Variable social: {variable}\n"
            f"Territorio foco: {selected}\n"
            f"Moran's I: {moran_result.observed:.4f} | p_sim: {moran_result.p_sim:.4f}\n"
            f"Geary's C: {geary_result.observed:.4f} | p_sim: {geary_result.p_sim:.4f}\n"
            f"I local territorio foco: {selected_lisa['I_local']:.4f} | p_sim: {selected_lisa['p_moran_sim']:.4f}\n"
            f"Clase LISA foco: {selected_lisa['cluster_lisa']}\n\n"
            f"Respuesta:\n{response}"
        )
        st.download_button(
            "Descargar respuesta",
            data=summary.encode("utf-8"),
            file_name="actividad_autocorrelacion_espacial.txt",
            mime="text/plain",
        )

    discussion("¿La conclusión cambia cuando pasas del diagnóstico global al mapa local LISA?")


def diagnostics_section() -> None:
    """Pequeño panel usado dentro de la app para mostrar distancias base."""

    st.subheader("Distancias entre centroides")
    st.dataframe(distance_summary(polygons), width="stretch", hide_index=True)


SECTIONS = {
    "1. Introducción": intro_section,
    "2. Tipos de datos": data_types_section,
    "3. Geometrías": geometries_section,
    "4. CRS": crs_section,
    "5. Estructuras Python": structures_section,
    "6. Pesos espaciales": weights_concept_section,
    "7. Tipos de pesos": weights_types_section,
    "8. Comparador W": comparator_section,
    "9. Rezago espacial": lag_section,
    "10. Autocorrelación global": global_autocorrelation_section,
    "11. Autocorrelación local": local_autocorrelation_section,
    "12. Actividad guiada": activity_section,
    "13. Clustering y regionalización": clustering_regionalization_section,
    "14. Regresión espacial": spatial_regression_section,
}

SECTION_GROUPS = {
    "Datos espaciales": [
        "1. Introducción",
        "2. Tipos de datos",
        "3. Geometrías",
        "4. CRS",
        "5. Estructuras Python",
    ],
    "Pesos espaciales y vecindades": [
        "6. Pesos espaciales",
        "7. Tipos de pesos",
        "8. Comparador W",
        "9. Rezago espacial",
    ],
    "Autocorrelación espacial": [
        "10. Autocorrelación global",
        "11. Autocorrelación local",
        "12. Actividad guiada",
    ],
    "Clustering y regionalización": [
        "13. Clustering y regionalización",
    ],
    "Modelos espaciales": [
        "14. Regresión espacial",
    ],
}


with st.sidebar:
    st.header("Contenidos")
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = next(iter(SECTIONS))

    for group_name, group_sections in SECTION_GROUPS.items():
        st.markdown(f"**{group_name}**")
        for index, section_name in enumerate(group_sections, start=1):
            is_active = st.session_state["active_section"] == section_name
            if st.button(
                section_name,
                key=f"nav_{group_name}_{index}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["active_section"] = section_name

    section = st.session_state["active_section"]
    st.divider()
    st.caption("Materia: GeoAnalítica de Datos")
    st.divider()
    with st.expander("Datos sintéticos"):
        st.write(f"Territorios: {len(polygons)}")
        st.write(f"Puntos: {len(points)}")
        st.write(f"Rutas: {len(lines)}")
        st.write(f"CRS de trabajo: {polygons.attrs.get('crs', PROJECTED_CRS)}")
        diagnostics_section()


SECTIONS[section]()
