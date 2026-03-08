from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Literal

# ── Cargar modelo ─────────────────────────────────────────────────────────────
model = joblib.load("xgboost.pkl")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Default Scoring API",
    description="""
API para predicción de riesgo de incumplimiento crediticio.

**Modelo:** XGBoost optimizado con Optuna  
**AUC-ROC:** 0.7788 | **Gini:** 0.5576 | **KS:** 0.4212  
**Dataset:** Default of Credit Card Clients (UCI)
    """,
    version="1.0.0"
)

# ── Schema de entrada ─────────────────────────────────────────────────────────
class ClienteInput(BaseModel):
    conteo_retrasos: int = Field(..., ge=0, description="Número total de meses con retraso en el historial")
    comportamiento_septiembre: int = Field(..., description="Estado de pago en septiembre (-2 a 8)")
    conteo_meses_inactivos: int = Field(..., ge=0, description="Meses sin movimiento en la cuenta")
    conteo_impagos: int = Field(..., ge=0, description="Número de veces que no realizó pago")
    abono_promedio: float = Field(..., ge=0, description="Promedio mensual de pagos realizados (TWD)")
    utilizacion_septiembre: float = Field(..., ge=0, le=1, description="Porcentaje de crédito utilizado en septiembre (0-1)")
    estado_cuenta_septiembre: int = Field(..., description="Estado de la cuenta en septiembre")
    threshold: float = Field(0.70, ge=0.0, le=1.0, description="Threshold de clasificación (default: 0.70)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "conteo_retrasos": 3,
                "comportamiento_septiembre": 2,
                "conteo_meses_inactivos": 1,
                "conteo_impagos": 2,
                "abono_promedio": 1500.0,
                "utilizacion_septiembre": 0.85,
                "estado_cuenta_septiembre": 1,
                "threshold": 0.70
            }
        }
    }

# ── Schema de salida ──────────────────────────────────────────────────────────
class PrediccionOutput(BaseModel):
    probabilidad_default: float
    prediccion: int
    segmento_riesgo: str
    threshold_usado: float
    recomendacion: str

# ── Lógica de segmentación ────────────────────────────────────────────────────
def clasificar_riesgo(probabilidad: float, prediccion: int) -> tuple[str, str]:
    if probabilidad < 0.30:
        return "Bajo", "Cliente con bajo riesgo de incumplimiento. Candidato a mejores productos crediticios."
    elif probabilidad < 0.50:
        return "Medio", "Cliente con riesgo moderado. Se recomienda monitoreo periódico de comportamiento."
    elif probabilidad < 0.70:
        return "Alto", "Cliente con riesgo elevado. Considerar restricción de límite de crédito y seguimiento activo."
    else:
        return "Crítico", "Cliente con riesgo crítico. Candidato a programa de reestructuración de deuda."

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {"status": "ok", "mensaje": "Credit Default Scoring API activa"}

@app.get("/health", tags=["Status"])
def health():
    return {"status": "healthy", "modelo": "XGBoost", "version": "1.0.0"}

@app.post("/predict", response_model=PrediccionOutput, tags=["Scoring"])
def predecir(cliente: ClienteInput):
    try:
        features = pd.DataFrame([{
            "conteo_retrasos":            cliente.conteo_retrasos,
            "comportamiento_septiembre":  cliente.comportamiento_septiembre,
            "conteo_meses_inactivos":     cliente.conteo_meses_inactivos,
            "conteo_impagos":             cliente.conteo_impagos,
            "abono_promedio":             cliente.abono_promedio,
            "utilizacion_septiembre":     cliente.utilizacion_septiembre,
            "estado_cuenta_septiembre":   cliente.estado_cuenta_septiembre,
        }])

        probabilidad = float(model.predict_proba(features)[0][1])
        prediccion   = int(probabilidad >= cliente.threshold)
        segmento, recomendacion = clasificar_riesgo(probabilidad, prediccion)

        return PrediccionOutput(
            probabilidad_default = round(probabilidad, 4),
            prediccion           = prediccion,
            segmento_riesgo      = segmento,
            threshold_usado      = cliente.threshold,
            recomendacion        = recomendacion
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))