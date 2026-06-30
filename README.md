# App didáctica: Datos espaciales y pesos espaciales

Aplicación interactiva en Streamlit para apoyar una clase de 2 horas sobre datos espaciales, estructuras de datos geográficos y matrices de pesos espaciales en ciencias sociales.

La app está completamente en español, usa datos sintéticos inspirados en Quito/Ecuador y no requiere archivos externos ni conexión a internet para funcionar.

## Objetivo

Permitir que estudiantes de ciencias sociales exploren, visualicen y manipulen:

- datos tabulares, vectoriales, raster, redes y datos puntuales;
- geometrías `Point`, `LineString`, `Polygon` y variantes múltiples;
- sistemas de referencia de coordenadas;
- estructuras de datos en Python para análisis espacial;
- matrices de pesos espaciales `W`;
- criterios Rook, Queen, distancia umbral, k vecinos, inversa de distancia y kernel;
- rezago espacial `Wy` e interpretación territorial;
- autocorrelación espacial global con Moran's I, Geary's C y pruebas por permutación;
- autocorrelación espacial local con LISA, clústeres Alto-Alto/Bajo-Bajo y atípicos espaciales.

## Instalación

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si ya tienes un entorno con Streamlit y las librerías científicas instaladas, puedes usarlo directamente.

## Ejecución

```bash
streamlit run app.py
```

La app abre un navegador local con navegación lateral por secciones.

## Estructura de archivos

```text
app.py
requirements.txt
README.md
utils/
  __init__.py
  autocorrelation.py
  data_generation.py
  explanations.py
  plots.py
  weights.py
```

## Contenidos cubiertos

1. Introducción: qué hace espacial a un dato.
2. Tipos de datos espaciales.
3. Geometrías espaciales.
4. Sistemas de referencia de coordenadas.
5. Estructuras de datos espaciales en Python.
6. Concepto central de pesos espaciales.
7. Tipos de pesos espaciales.
8. Comparador de matrices `W`.
9. Rezago espacial interactivo.
10. Autocorrelación espacial global.
11. Autocorrelación espacial local.
12. Actividad guiada para estudiantes.

## Guía sugerida para una clase de 2 horas

| Tiempo | Actividad |
|---|---|
| 0-10 min | Introducción: dato espacial vs dato no espacial |
| 10-25 min | Tipos de datos espaciales |
| 25-40 min | Geometrías y CRS |
| 40-55 min | Estructuras de datos espaciales en Python |
| 55-75 min | Concepto de pesos espaciales |
| 75-95 min | Comparación Rook, Queen, distancia, kNN, inversa de distancia y kernel |
| 95-105 min | Rezago espacial e interpretación social |
| 105-113 min | Moran's I, Geary's C y permutaciones |
| 113-118 min | LISA local: clústeres y atípicos territoriales |
| 118-120 min | Actividad final: comparar `W`, rezago, autocorrelación global y LISA local |

## Actividades sugeridas

- Elegir un problema social territorial y justificar el tipo de dato espacial más adecuado.
- Comparar Rook y Queen para discutir qué significa "vecindad" en datos areales.
- Mover el umbral de distancia y observar cuándo aparecen unidades aisladas.
- Cambiar `k` en kNN y discutir el riesgo de forzar vecinos lejanos.
- Calcular `Wy` para vulnerabilidad o pobreza e interpretar casos de continuidad territorial o atipicidad local.
- Comparar Moran's I y Geary's C para decidir si una variable muestra agrupamiento espacial global.
- Usar el diagrama de Moran para distinguir cuadrantes Alto-Alto, Bajo-Bajo, Alto-Bajo y Bajo-Alto.
- Usar el mapa LISA para identificar dónde están los clústeres significativos y posibles atípicos espaciales.
- Alternar el mapa LISA entre modo pedagógico, que colorea todos los cuadrantes, y modo estricto, que muestra solo clústeres significativos.
- Resolver la actividad guiada integrando una misma matriz `W` con rezago espacial, Moran's I, Geary's C y LISA local.
- Defender una matriz de pesos para una pregunta sustantiva concreta.

## Notas sobre dependencias

`geopandas` y `shapely` pueden requerir ruedas binarias compatibles con tu versión de Python. Si la instalación falla:

- actualiza `pip` con `pip install --upgrade pip`;
- usa Python 3.10 o 3.11 si tu sistema tiene conflictos geoespaciales;
- instala primero `shapely`, `pyproj` y `pyogrio`, y luego `geopandas`;
- como alternativa docente, la app incluye una representación interna mínima para geometrías cuando `shapely/geopandas` no están disponibles.

La lógica de matrices de pesos está implementada con `numpy`, sin depender de `libpysal`, para que el ejemplo sea transparente para estudiantes.

## Recomendaciones docentes

- Empieza con una pregunta social concreta antes de mostrar la matriz.
- Pide al grupo que nombre qué relación territorial representa cada criterio.
- Usa el comparador para mostrar que no existe una matriz "neutral".
- Enfatiza que `W` es una decisión teórica y metodológica, no solo técnica.
- Cierra con el rezago espacial, la autocorrelación global y LISA local para conectar la clase con Moran, Geary, modelos espaciales e IA territorial.
