"""Modelos de regresión espacial didácticos para la app Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from utils.weights import row_standardize


MODEL_DESCRIPTIONS = {
    "MCO / OLS": "Regresión lineal clásica: la variable dependiente se explica solo por covariables propias.",
    "SLX (X y WX)": "Incluye rezagos espaciales de las covariables para capturar spillovers exógenos.",
    "SLM / SAR didáctico (Wy)": "Incluye el rezago espacial de la variable dependiente como señal de dependencia sustantiva.",
    "SDM didáctico (Wy + WX)": "Combina rezago de y y rezagos de X; permite discutir efectos directos e indirectos.",
}


@dataclass(frozen=True)
class SpatialRegressionResult:
    """Resultado compacto de un modelo lineal espacial didáctico."""

    model_name: str
    y_name: str
    x_names: list[str]
    feature_names: list[str]
    coefficients: pd.Series
    std_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    fitted: np.ndarray
    residuals: np.ndarray
    r2: float
    adj_r2: float
    aic: float
    bic: float
    sigma2: float
    design: np.ndarray
    y_values: np.ndarray
    W_used: np.ndarray
    standardized: bool

    @property
    def rho(self) -> float:
        return float(self.coefficients.get("W_y", 0.0))


def _safe_std(values: np.ndarray) -> float:
    std = float(values.std(ddof=0))
    return 1.0 if np.isclose(std, 0) else std


def prepare_regression_data(
    data: pd.DataFrame,
    y_name: str,
    x_names: list[str],
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve datos para modelar y tabla de escalas usadas."""

    columns = [y_name, *x_names]
    output = data[["id", "nombre", *columns]].copy()
    rows = []
    if standardize:
        for column in columns:
            values = output[column].to_numpy(dtype=float)
            mean = float(values.mean())
            std = _safe_std(values)
            output[column] = (values - mean) / std
            rows.append({"variable": column, "centro": mean, "escala": std, "método": "z-score"})
    else:
        for column in columns:
            rows.append({"variable": column, "centro": 0.0, "escala": 1.0, "método": "original"})
    return output, pd.DataFrame(rows)


def build_design_matrix(
    data: pd.DataFrame,
    y_name: str,
    x_names: list[str],
    W: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, list[str]]:
    """Construye matriz de diseño con intercepto y términos espaciales opcionales."""

    W_std = row_standardize(W)
    parts = [np.ones(len(data), dtype=float)]
    names = ["constante"]

    if model_name in {"SLM / SAR didáctico (Wy)", "SDM didáctico (Wy + WX)"}:
        parts.append(W_std @ data[y_name].to_numpy(dtype=float))
        names.append("W_y")

    for variable in x_names:
        parts.append(data[variable].to_numpy(dtype=float))
        names.append(variable)

    if model_name in {"SLX (X y WX)", "SDM didáctico (Wy + WX)"}:
        x_values = data[x_names].to_numpy(dtype=float)
        wx_values = W_std @ x_values
        for idx, variable in enumerate(x_names):
            parts.append(wx_values[:, idx])
            names.append(f"W_{variable}")

    return np.column_stack(parts), names


def fit_spatial_regression(
    data: pd.DataFrame,
    y_name: str,
    x_names: list[str],
    W: np.ndarray,
    model_name: str,
    standardize: bool = True,
) -> SpatialRegressionResult:
    model_data, _ = prepare_regression_data(data, y_name, x_names, standardize)
    y = model_data[y_name].to_numpy(dtype=float)
    X, feature_names = build_design_matrix(model_data, y_name, x_names, W, model_name)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    residuals = y - fitted
    n, p = X.shape
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if not np.isclose(tss, 0) else float("nan")
    df_resid = max(n - p, 1)
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid if np.isfinite(r2) and n > 1 else float("nan")
    sigma2 = rss / df_resid
    xtx_inv = np.linalg.pinv(X.T @ X)
    std_errors = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(beta, std_errors, out=np.full_like(beta, np.nan), where=std_errors != 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=df_resid)
    aic = n * np.log(max(rss / n, 1e-12)) + 2 * p
    bic = n * np.log(max(rss / n, 1e-12)) + p * np.log(n)

    return SpatialRegressionResult(
        model_name=model_name,
        y_name=y_name,
        x_names=x_names,
        feature_names=feature_names,
        coefficients=pd.Series(beta, index=feature_names, dtype=float),
        std_errors=pd.Series(std_errors, index=feature_names, dtype=float),
        t_values=pd.Series(t_values, index=feature_names, dtype=float),
        p_values=pd.Series(p_values, index=feature_names, dtype=float),
        fitted=fitted,
        residuals=residuals,
        r2=float(r2),
        adj_r2=float(adj_r2),
        aic=float(aic),
        bic=float(bic),
        sigma2=float(sigma2),
        design=X,
        y_values=y,
        W_used=row_standardize(W),
        standardized=standardize,
    )


def coefficient_table(result: SpatialRegressionResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "término": result.feature_names,
            "coeficiente": result.coefficients.values,
            "error_estándar": result.std_errors.values,
            "t": result.t_values.values,
            "p_valor": result.p_values.values,
        }
    )


def model_comparison_table(results: list[SpatialRegressionResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "modelo": result.model_name,
                "R2": result.r2,
                "R2_ajustado": result.adj_r2,
                "AIC": result.aic,
                "BIC": result.bic,
                "sigma2": result.sigma2,
                "rho_Wy": result.rho,
                "n_términos": len(result.feature_names),
            }
        )
    return pd.DataFrame(rows)


def breusch_pagan_test(result: SpatialRegressionResult) -> dict[str, float]:
    """Prueba Breusch-Pagan clásica sobre residuos del modelo."""

    e2 = result.residuals**2
    Z = result.design
    beta_aux, *_ = np.linalg.lstsq(Z, e2, rcond=None)
    fitted_aux = Z @ beta_aux
    rss_aux = float(np.sum((e2 - fitted_aux) ** 2))
    tss_aux = float(np.sum((e2 - e2.mean()) ** 2))
    r2_aux = 1 - rss_aux / tss_aux if not np.isclose(tss_aux, 0) else 0.0
    lm = len(e2) * max(r2_aux, 0.0)
    df = max(Z.shape[1] - 1, 1)
    p_value = float(stats.chi2.sf(lm, df))
    return {"LM": float(lm), "gl": float(df), "p_valor": p_value, "R2_aux": float(r2_aux)}


def spatial_diagnostic_signals(result: SpatialRegressionResult, W: np.ndarray) -> pd.DataFrame:
    """Señales didácticas para elegir entre rezago y error espacial."""

    W_std = row_standardize(W)
    residuals = result.residuals
    y_lag = W_std @ result.y_values
    residual_lag = W_std @ residuals

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.isclose(a.std(ddof=0), 0) or np.isclose(b.std(ddof=0), 0):
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    lag_signal = corr(residuals, y_lag)
    error_signal = corr(residuals, residual_lag)
    return pd.DataFrame(
        [
            {
                "señal": "rezago espacial omitido",
                "indicador": "corr(residuo, W_y)",
                "valor": lag_signal,
                "lectura": "alto en valor absoluto sugiere revisar SLM/SDM",
            },
            {
                "señal": "error espacial",
                "indicador": "corr(residuo, W_residuo)",
                "valor": error_signal,
                "lectura": "alto en valor absoluto sugiere revisar SEM o variables omitidas",
            },
        ]
    )


def predict_spatial_shock(
    data: pd.DataFrame,
    result: SpatialRegressionResult,
    x_name: str,
    selected_index: int,
    shock: float,
) -> pd.DataFrame:
    """Simula un cambio en una covariable y propaga el efecto según el modelo."""

    x_values = data[result.x_names].to_numpy(dtype=float)
    shocked = x_values.copy()
    x_pos = result.x_names.index(x_name)
    shocked[selected_index, x_pos] += shock

    def linear_part(values: np.ndarray) -> np.ndarray:
        xb = np.full(values.shape[0], result.coefficients.get("constante", 0.0), dtype=float)
        for idx, variable in enumerate(result.x_names):
            xb += result.coefficients.get(variable, 0.0) * values[:, idx]
        if any(name.startswith("W_") for name in result.feature_names):
            wx = result.W_used @ values
            for idx, variable in enumerate(result.x_names):
                xb += result.coefficients.get(f"W_{variable}", 0.0) * wx[:, idx]
        return xb

    rho = result.rho if "W_y" in result.feature_names else 0.0
    rho = float(np.clip(rho, -0.95, 0.95))
    eye = np.eye(len(data))
    base_linear = linear_part(x_values)
    shocked_linear = linear_part(shocked)
    if not np.isclose(rho, 0):
        base_prediction = np.linalg.solve(eye - rho * result.W_used, base_linear)
        shocked_prediction = np.linalg.solve(eye - rho * result.W_used, shocked_linear)
    else:
        base_prediction = base_linear
        shocked_prediction = shocked_linear

    output = data[["id", "nombre"]].copy()
    output["predicción_base"] = base_prediction
    output["predicción_con_choque"] = shocked_prediction
    output["cambio_predicho"] = shocked_prediction - base_prediction
    output["tipo_impacto"] = np.where(output.index == selected_index, "directo", "indirecto")
    return output
