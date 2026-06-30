"""Generación de datos espaciales sintéticos para la app.

Los datos son ficticios, pero están construidos para parecer plausibles en una
clase de análisis territorial aplicada a Ecuador. Las coordenadas principales
usan metros en una grilla local compatible con la idea de un CRS proyectado
como UTM 17S (EPSG:32717).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sqrt
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

try:  # Geometría real cuando el entorno la tenga instalada.
    import geopandas as gpd
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

    HAS_GEO = True
except Exception:  # pragma: no cover - se usa como fallback de aula.
    gpd = None
    Point = LineString = Polygon = MultiPoint = MultiLineString = MultiPolygon = None
    HAS_GEO = False


PROJECTED_CRS = "EPSG:32717"
GEOGRAPHIC_CRS = "EPSG:4326"
BASE_X = 772_000.0
BASE_Y = 9_974_000.0


@dataclass(frozen=True)
class SimpleGeometry:
    """Representación mínima si Shapely/GeoPandas no están disponibles."""

    geom_type: str
    coordinates: Any

    @property
    def is_fallback(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover - solo presentación.
        return f"{self.geom_type}({self.coordinates})"


def _point(x: float, y: float) -> Any:
    return Point(x, y) if HAS_GEO else SimpleGeometry("Point", (x, y))


def _line(coords: list[tuple[float, float]]) -> Any:
    return LineString(coords) if HAS_GEO else SimpleGeometry("LineString", coords)


def _polygon(coords: list[tuple[float, float]]) -> Any:
    return Polygon(coords) if HAS_GEO else SimpleGeometry("Polygon", coords)


def _multi_point(coords: list[tuple[float, float]]) -> Any:
    return MultiPoint(coords) if HAS_GEO else SimpleGeometry("MultiPoint", coords)


def _multi_line(parts: list[list[tuple[float, float]]]) -> Any:
    return MultiLineString(parts) if HAS_GEO else SimpleGeometry("MultiLineString", parts)


def _multi_polygon(parts: list[list[tuple[float, float]]]) -> Any:
    if HAS_GEO:
        return MultiPolygon([Polygon(coords) for coords in parts])
    return SimpleGeometry("MultiPolygon", parts)


def as_geodataframe(df: pd.DataFrame, crs: str = PROJECTED_CRS) -> pd.DataFrame:
    """Devuelve GeoDataFrame si es posible; si no, conserva DataFrame."""

    result = df.copy()
    result.attrs["crs"] = crs
    if HAS_GEO and "geometry" in result.columns:
        return gpd.GeoDataFrame(result, geometry="geometry", crs=crs)
    return result


def geometry_type(geometry: Any) -> str:
    if hasattr(geometry, "geom_type"):
        return geometry.geom_type
    if isinstance(geometry, SimpleGeometry):
        return geometry.geom_type
    return type(geometry).__name__


def geometry_coordinates(geometry: Any) -> Any:
    """Extrae coordenadas en una forma amigable para tablas y gráficos."""

    if isinstance(geometry, SimpleGeometry):
        return geometry.coordinates
    if HAS_GEO and hasattr(geometry, "geom_type"):
        if geometry.geom_type == "Point":
            return (round(geometry.x, 2), round(geometry.y, 2))
        if geometry.geom_type == "LineString":
            return [(round(x, 2), round(y, 2)) for x, y in geometry.coords]
        if geometry.geom_type == "Polygon":
            return [(round(x, 2), round(y, 2)) for x, y in geometry.exterior.coords]
        if geometry.geom_type.startswith("Multi"):
            return [geometry_coordinates(part) for part in geometry.geoms]
    return geometry


def polygon_xy(geometry: Any) -> tuple[list[float], list[float]]:
    coords = geometry_coordinates(geometry)
    if coords and isinstance(coords[0], tuple):
        return [p[0] for p in coords], [p[1] for p in coords]
    return [], []


def line_xy(geometry: Any) -> tuple[list[float], list[float]]:
    coords = geometry_coordinates(geometry)
    if coords and isinstance(coords[0], tuple):
        return [p[0] for p in coords], [p[1] for p in coords]
    return [], []


def point_xy(geometry: Any) -> tuple[float, float]:
    coords = geometry_coordinates(geometry)
    if isinstance(coords, tuple):
        return float(coords[0]), float(coords[1])
    return np.nan, np.nan


def create_synthetic_polygons(rows: int = 3, cols: int = 4) -> pd.DataFrame:
    """Crea parroquias/barrrios ficticios de Quito como polígonos en grilla."""

    names = [
        "La Floresta",
        "Iñaquito",
        "Carcelén",
        "Calderón",
        "Solanda",
        "Chillogallo",
        "La Magdalena",
        "Cumbayá",
        "San Juan",
        "Cotocollao",
        "Tumbaco",
        "Guamaní",
    ]
    width = 1_800.0
    height = 1_400.0
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            x0 = BASE_X + col * width
            y0 = BASE_Y + row * height
            coords = [
                (x0, y0),
                (x0 + width, y0),
                (x0 + width, y0 + height),
                (x0, y0 + height),
                (x0, y0),
            ]
            cx = x0 + width / 2
            cy = y0 + height / 2
            # Valores sociales sintéticos: no son aleatorios puros, tienen patrón espacial.
            centrality = 1.0 - (abs(col - (cols - 1) / 2) + abs(row - (rows - 1) / 2)) / 4
            population = int(9_000 + row * 1_100 + col * 700 + rng.integers(-650, 650))
            ingreso = int(520 + centrality * 420 + col * 35 + rng.integers(-45, 45))
            pobreza = max(6, min(48, int(38 - centrality * 18 - col * 2 + row * 3 + rng.integers(-3, 4))))
            servicios = max(40, min(96, int(58 + centrality * 24 + col * 2 - row + rng.integers(-4, 5))))
            vulnerabilidad = max(5, min(85, int(0.75 * pobreza + 0.45 * (100 - servicios) + rng.integers(-3, 4))))
            densidad = round(population / ((width * height) / 1_000_000), 1)
            lon = -78.55 + (cx - BASE_X) / 105_000
            lat = -0.31 + (cy - BASE_Y) / 111_000
            records.append(
                {
                    "id": f"P{idx + 1:02d}",
                    "nombre": names[idx],
                    "fila": row,
                    "columna": col,
                    "población": population,
                    "ingreso_medio": ingreso,
                    "tasa_pobreza": pobreza,
                    "acceso_servicios": servicios,
                    "índice_vulnerabilidad": vulnerabilidad,
                    "densidad_poblacional": densidad,
                    "lon": round(lon, 5),
                    "lat": round(lat, 5),
                    "coordenadas": f"({round(lon, 5)}, {round(lat, 5)})",
                    "centroide_x": cx,
                    "centroide_y": cy,
                    "geometry": _polygon(coords),
                }
            )

    return as_geodataframe(pd.DataFrame(records), PROJECTED_CRS)


def create_intro_table(polygons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla pequeña para contrastar dato tabular y dato espacial."""

    if polygons is None:
        polygons = create_synthetic_polygons()
    columns = ["id", "nombre", "población", "ingreso_medio", "coordenadas", "geometry"]
    table = polygons.loc[:5, columns].copy()
    table["geometry"] = table["geometry"].map(lambda geom: f"{geometry_type(geom)}(...)")
    return table


def create_synthetic_points(n: int = 24) -> pd.DataFrame:
    """Crea puntos de equipamientos y eventos sociales ficticios."""

    rng = np.random.default_rng(7)
    tipos = np.array(["escuela", "centro de salud", "mercado", "parada BRT", "incidente"])
    records: list[dict[str, Any]] = []
    for i in range(n):
        x = BASE_X + rng.uniform(300, 6_900)
        y = BASE_Y + rng.uniform(250, 3_950)
        tipo = str(rng.choice(tipos, p=[0.28, 0.18, 0.18, 0.22, 0.14]))
        records.append(
            {
                "id": f"E{i + 1:02d}",
                "tipo": tipo,
                "nombre": f"{tipo.title()} {i + 1}",
                "usuarios_estimados": int(rng.integers(40, 900)),
                "x": x,
                "y": y,
                "geometry": _point(x, y),
            }
        )
    return as_geodataframe(pd.DataFrame(records), PROJECTED_CRS)


def create_synthetic_lines() -> pd.DataFrame:
    """Rutas de transporte sintéticas que cruzan los territorios."""

    routes = [
        {
            "id": "R1",
            "nombre": "Corredor norte-sur",
            "modo": "BRT",
            "frecuencia_min": 8,
            "coords": [
                (BASE_X + 900, BASE_Y + 100),
                (BASE_X + 1_400, BASE_Y + 1_500),
                (BASE_X + 1_750, BASE_Y + 2_700),
                (BASE_X + 2_200, BASE_Y + 4_150),
            ],
        },
        {
            "id": "R2",
            "nombre": "Ruta transversal",
            "modo": "bus urbano",
            "frecuencia_min": 12,
            "coords": [
                (BASE_X + 120, BASE_Y + 1_800),
                (BASE_X + 2_100, BASE_Y + 1_900),
                (BASE_X + 4_200, BASE_Y + 2_050),
                (BASE_X + 7_000, BASE_Y + 2_300),
            ],
        },
        {
            "id": "R3",
            "nombre": "Conexión periurbana",
            "modo": "alimentador",
            "frecuencia_min": 18,
            "coords": [
                (BASE_X + 5_200, BASE_Y + 100),
                (BASE_X + 5_600, BASE_Y + 1_100),
                (BASE_X + 6_300, BASE_Y + 2_400),
                (BASE_X + 6_900, BASE_Y + 4_050),
            ],
        },
    ]
    records = []
    for route in routes:
        coords = route.pop("coords")
        records.append({**route, "geometry": _line(coords)})
    return as_geodataframe(pd.DataFrame(records), PROJECTED_CRS)


def create_synthetic_network() -> nx.Graph:
    """Red vial simple: nodos como intersecciones y aristas como calles."""

    graph = nx.Graph()
    node_id = 0
    for row in range(4):
        for col in range(5):
            x = BASE_X + 500 + col * 1_450
            y = BASE_Y + 350 + row * 1_050
            graph.add_node(node_id, x=x, y=y, barrio=f"Nodo {node_id}")
            node_id += 1

    def node(row: int, col: int) -> int:
        return row * 5 + col

    for row in range(4):
        for col in range(5):
            if col < 4:
                graph.add_edge(node(row, col), node(row, col + 1), tipo="calle", longitud_m=1_450)
            if row < 3:
                graph.add_edge(node(row, col), node(row + 1, col), tipo="calle", longitud_m=1_050)
    graph.add_edge(node(0, 0), node(1, 1), tipo="conector", longitud_m=1_790)
    graph.add_edge(node(2, 3), node(3, 4), tipo="conector", longitud_m=1_790)
    return graph


def create_synthetic_raster(size: tuple[int, int] = (45, 60)) -> dict[str, Any]:
    """Superficie raster sintética: luminosidad nocturna y presión urbana."""

    rows, cols = size
    y, x = np.mgrid[0:rows, 0:cols]
    center_1 = np.exp(-(((x - 23) ** 2) / 190 + ((y - 22) ** 2) / 120))
    center_2 = 0.75 * np.exp(-(((x - 46) ** 2) / 80 + ((y - 12) ** 2) / 95))
    corridor = 0.28 * np.exp(-((y - (0.35 * x + 7)) ** 2) / 18)
    rng = np.random.default_rng(12)
    values = 20 + 68 * (center_1 + center_2 + corridor) + rng.normal(0, 2.2, size)
    values = np.clip(values, 0, 100)
    return {
        "values": values,
        "x": np.linspace(BASE_X, BASE_X + 7_200, cols),
        "y": np.linspace(BASE_Y, BASE_Y + 4_200, rows),
        "name": "Luminosidad nocturna sintética",
        "crs": PROJECTED_CRS,
    }


def local_project_lonlat_to_meters(
    lon: float,
    lat: float,
    origin_lon: float = -78.5,
    origin_lat: float = -0.2,
) -> tuple[float, float]:
    """Aproximación didáctica para convertir grados a metros cerca de Quito."""

    meters_per_degree_lat = 110_574.0
    meters_per_degree_lon = 111_320.0 * cos(radians(origin_lat))
    x = (lon - origin_lon) * meters_per_degree_lon
    y = (lat - origin_lat) * meters_per_degree_lat
    return x, y


def distance_in_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distance_in_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = local_project_lonlat_to_meters(*a)
    bx, by = local_project_lonlat_to_meters(*b)
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def create_geometry_example(kind: str, scale: float = 1.0) -> dict[str, Any]:
    """Construye geometrías básicas para la sección de Shapely."""

    s = scale
    examples = {
        "Point": _point(0, 0),
        "LineString": _line([(0, 0), (1.5 * s, 0.8 * s), (3.0 * s, 0.2 * s)]),
        "Polygon": _polygon([(0, 0), (3 * s, 0), (2.6 * s, 1.8 * s), (0.4 * s, 2.2 * s), (0, 0)]),
        "MultiPoint": _multi_point([(0, 0), (1.2 * s, 1.4 * s), (2.6 * s, 0.6 * s), (3.1 * s, 1.9 * s)]),
        "MultiLineString": _multi_line(
            [
                [(0, 0), (1.2 * s, 1.0 * s), (2.2 * s, 0.4 * s)],
                [(0.3 * s, 1.8 * s), (1.4 * s, 2.1 * s), (2.6 * s, 1.5 * s)],
            ]
        ),
        "MultiPolygon": _multi_polygon(
            [
                [(0, 0), (1.2 * s, 0), (1.2 * s, 1.1 * s), (0, 1.1 * s), (0, 0)],
                [(1.8 * s, 0.4 * s), (3.0 * s, 0.4 * s), (3.0 * s, 1.6 * s), (1.8 * s, 1.6 * s), (1.8 * s, 0.4 * s)],
            ]
        ),
    }
    geometry = examples[kind]
    area = float(getattr(geometry, "area", np.nan)) if HAS_GEO else _fallback_area(geometry)
    length = float(getattr(geometry, "length", np.nan)) if HAS_GEO else _fallback_length(geometry)
    return {
        "geometry": geometry,
        "tipo": geometry_type(geometry),
        "coordenadas": geometry_coordinates(geometry),
        "área": area,
        "longitud": length,
        "usa_shapely": HAS_GEO,
    }


def _fallback_area(geometry: Any) -> float:
    if not isinstance(geometry, SimpleGeometry):
        return float("nan")
    if geometry.geom_type == "Polygon":
        return _shoelace(geometry.coordinates)
    if geometry.geom_type == "MultiPolygon":
        return sum(_shoelace(coords) for coords in geometry.coordinates)
    return 0.0


def _fallback_length(geometry: Any) -> float:
    if not isinstance(geometry, SimpleGeometry):
        return float("nan")
    if geometry.geom_type == "LineString":
        return _path_length(geometry.coordinates)
    if geometry.geom_type == "Polygon":
        return _path_length(geometry.coordinates)
    if geometry.geom_type == "MultiLineString":
        return sum(_path_length(coords) for coords in geometry.coordinates)
    if geometry.geom_type == "MultiPolygon":
        return sum(_path_length(coords) for coords in geometry.coordinates)
    return 0.0


def _path_length(coords: Iterable[tuple[float, float]]) -> float:
    points = list(coords)
    return float(
        sum(
            sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2)
            for i in range(len(points) - 1)
        )
    )


def _shoelace(coords: Iterable[tuple[float, float]]) -> float:
    points = list(coords)
    return float(
        abs(
            sum(points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1] for i in range(len(points) - 1))
        )
        / 2
    )

