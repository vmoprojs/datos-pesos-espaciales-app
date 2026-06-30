"""Textos pedagógicos e interpretaciones automáticas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.weights import matrix_stats


DATA_TYPE_INFO = {
    "Puntos": {
        "definición": "Objetos representados por una ubicación puntual: una escuela, un delito, una encuesta o un centro de salud.",
        "ejemplo": "Ubicación de escuelas y centros de salud para estudiar desigualdad de acceso entre barrios.",
        "python": "GeoDataFrame con geometry = Point; también puede iniciar como DataFrame con lon/lat.",
        "ia": "Modelos de riesgo, clustering de eventos, detección de puntos calientes y asignación óptima de servicios.",
    },
    "Líneas": {
        "definición": "Objetos lineales formados por una secuencia de coordenadas.",
        "ejemplo": "Rutas de transporte público, calles, tuberías o trayectorias de movilidad.",
        "python": "GeoDataFrame con geometry = LineString; para redes, grafo con nodos y aristas.",
        "ia": "Predicción de congestión, accesibilidad, optimización de rutas y análisis de conectividad.",
    },
    "Polígonos": {
        "definición": "Áreas cerradas que delimitan territorios o zonas de análisis.",
        "ejemplo": "Parroquias urbanas, cantones, barrios o zonas censales con indicadores sociales.",
        "python": "GeoDataFrame con geometry = Polygon o MultiPolygon.",
        "ia": "Modelos territoriales de vulnerabilidad, segmentación espacial y priorización de política pública.",
    },
    "Raster": {
        "definición": "Superficie dividida en celdas; cada celda almacena un valor.",
        "ejemplo": "Luminosidad nocturna, temperatura, elevación, cobertura de suelo o exposición ambiental.",
        "python": "Arreglo NumPy 2D/3D con metadatos de resolución, extensión y CRS.",
        "ia": "Clasificación de imágenes, extracción de rasgos territoriales y aprendizaje profundo geoespacial.",
    },
    "Redes": {
        "definición": "Sistema de nodos y conexiones donde importa la relación topológica.",
        "ejemplo": "Red vial, transporte público, flujos entre territorios o cadenas de abastecimiento.",
        "python": "networkx.Graph con atributos en nodos y aristas; puede conectarse con GeoDataFrames.",
        "ia": "Medidas de centralidad, propagación, detección de comunidades y modelos de grafos.",
    },
    "Datos tabulares con coordenadas": {
        "definición": "Tabla convencional que contiene columnas de longitud/latitud o X/Y.",
        "ejemplo": "Encuestas georreferenciadas con variables de ingreso, empleo o acceso a servicios.",
        "python": "pandas.DataFrame que se convierte a GeoDataFrame usando points_from_xy.",
        "ia": "Modelos predictivos con variables contextuales, geocodificación y enriquecimiento espacial.",
    },
    "Datos areales agregados": {
        "definición": "Indicadores resumidos por unidades territoriales.",
        "ejemplo": "Tasa de pobreza por cantón, cobertura educativa por parroquia o vulnerabilidad por zona censal.",
        "python": "GeoDataFrame de polígonos con columnas de atributos agregados.",
        "ia": "Modelos espaciales, clasificación de territorios y sistemas de alerta para intervención pública.",
    },
}


def structures_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Estructura": "pandas.DataFrame",
                "Qué almacena": "Filas y columnas sin geometría activa",
                "Ejemplo": "encuestas con lon/lat",
                "Uso típico": "limpieza, joins, estadísticas",
                "Librería": "pandas",
            },
            {
                "Estructura": "geopandas.GeoDataFrame",
                "Qué almacena": "Atributos + columna geometry + CRS",
                "Ejemplo": "parroquias con polígonos",
                "Uso típico": "mapas, joins espaciales, reproyección",
                "Librería": "geopandas",
            },
            {
                "Estructura": "shapely.geometry",
                "Qué almacena": "Objetos geométricos individuales",
                "Ejemplo": "Point, LineString, Polygon",
                "Uso típico": "área, longitud, distancia, intersección",
                "Librería": "shapely",
            },
            {
                "Estructura": "Raster simplificado",
                "Qué almacena": "Matriz de celdas con valor",
                "Ejemplo": "luminosidad nocturna",
                "Uso típico": "superficies, imágenes, variables ambientales",
                "Librería": "numpy / rasterio",
            },
            {
                "Estructura": "Matriz espacial W",
                "Qué almacena": "Relaciones o intensidades entre unidades",
                "Ejemplo": "vecindad Rook o kNN",
                "Uso típico": "rezagos, Moran, LISA, modelos espaciales",
                "Librería": "numpy / libpysal",
            },
            {
                "Estructura": "Grafo",
                "Qué almacena": "Nodos, aristas y atributos",
                "Ejemplo": "red de transporte",
                "Uso típico": "rutas, accesibilidad, centralidad",
                "Librería": "networkx",
            },
        ]
    )


def method_guidance(method: str) -> str:
    messages = {
        "Rook": "Adecuado para datos areales cuando se quiere una noción estricta de vecindad territorial: compartir frontera.",
        "Queen": "Útil cuando los contactos por esquina también pueden representar interacción social o administrativa.",
        "Distancia umbral": "Conveniente para puntos o centroides cuando existe una distancia máxima con sentido sustantivo.",
        "k vecinos más cercanos": "Evita unidades aisladas, pero puede imponer relaciones artificiales entre territorios lejanos.",
        "Inversa de distancia": "Representa influencia continua: todos influyen, pero los cercanos pesan mucho más.",
        "Kernel espacial": "Útil cuando la influencia se difumina dentro de una banda y desaparece más allá del alcance definido.",
    }
    return messages.get(method, "")


def interpret_weights_matrix(W: np.ndarray, labels: list[str], method: str) -> str:
    stats = matrix_stats(W, labels)
    density = stats["densidad"]
    pieces = []

    if stats["n_aisladas"]:
        isolated = ", ".join(stats["unidades_aisladas"])
        pieces.append(f"Hay {stats['n_aisladas']} unidad(es) sin vecinos: {isolated}.")
    else:
        pieces.append("No hay unidades aisladas; todas tienen al menos una relación espacial.")

    if density < 0.18:
        pieces.append("La matriz es dispersa: la vecindad es selectiva y local.")
    elif density < 0.45:
        pieces.append("La matriz tiene densidad media: captura varias relaciones sin conectar todo con todo.")
    else:
        pieces.append("La matriz es densa: muchas unidades influyen sobre muchas otras.")

    if stats["simétrica"]:
        pieces.append("La relación es simétrica: si i es vecino de j, j también lo es de i.")
    else:
        pieces.append("La relación no es necesariamente simétrica; esto suele aparecer en kNN dirigido.")

    pieces.append(
        f"La unidad más conectada es {stats['unidad_más_conectada']} con {stats['vecinos_max']} vecino(s)."
    )
    pieces.append(method_guidance(method))
    return " ".join(pieces)


def lag_interpretation(row: pd.Series, variable: str, lag_col: str) -> str:
    own = float(row[variable])
    lag = float(row[lag_col])
    diff = own - lag
    territory = row.get("nombre", row.get("id", "el territorio"))
    variable_label = variable.replace("_", " ")
    if abs(diff) < 2:
        return (
            f"{territory} tiene un valor de {variable_label} muy parecido al promedio ponderado de sus vecinos. "
            "Esto sugiere continuidad territorial del fenómeno."
        )
    if diff > 0:
        return (
            f"{territory} está por encima de sus vecinos en {variable_label}. "
            "Puede interpretarse como un posible caso atípico local alto."
        )
    return (
        f"{territory} está por debajo de sus vecinos en {variable_label}. "
        "Puede ser un posible caso atípico local bajo o un territorio rodeado por mayores presiones sociales."
    )


def crs_warning(has_crs: bool) -> str:
    if has_crs:
        return "La capa declara un CRS, por lo que es posible interpretar correctamente sus unidades."
    return "La capa no declara CRS: cualquier distancia, área o superposición queda en duda."

