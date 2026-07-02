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
