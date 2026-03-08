\# 💳 Credit Default Scoring — Predicción de Riesgo Crediticio



> Modelo de machine learning para predecir el incumplimiento de pago en tarjetas de crédito, con pipeline completo de ciencia de datos, API REST y aplicación interactiva de scoring.



---



\## 📌 Contexto del Problema



Una institución financiera necesita anticipar qué clientes incumplirán su pago el próximo mes para tomar acciones preventivas: ajustar límites de crédito, activar cobranza proactiva, ofrecer reestructuración de deuda o diseñar productos adecuados a cada perfil de riesgo.



Con base en el historial de pagos, estados de cuenta y comportamiento de abril a septiembre, el objetivo es predecir qué clientes incumplirán su pago en octubre — permitiendo a la institución tomar acciones preventivas con anticipación.



El desafío central es que \*\*el costo de los errores no es simétrico\*\*:

\- Un \*\*falso negativo\*\* (no detectar a un moroso) implica pérdida directa de capital, costos de cobranza y castigo contable de la deuda.

\- Un \*\*falso positivo\*\* (marcar como riesgoso a un buen cliente) representa únicamente un costo de oportunidad — los intereses que habría generado.



Por esta razón, se priorizaron métricas sensibles a la clase minoritaria: \*\*AUC-ROC, Gini y KS\*\*, y se realizó un análisis explícito de thresholds para dejar la decisión del punto de corte al criterio de negocio.



---



\## 📂 Dataset



\*\*Default of Credit Card Clients\*\* — UCI Machine Learning Repository  

\- 30,000 clientes de una institución financiera de Taiwán

\- 23 variables originales: historial de pagos, estados de cuenta, abonos, datos demográficos

\- Variable objetivo: `DEFAULT` (1 = incumplimiento el mes siguiente)

\- Desbalance de clases: 77.9% pago / 22.1% impago



---



\## 🗂️ Estructura del Proyecto



```

credit\_risk/

├── data/

│   ├── raw.csv

│   ├── datos\_procesados.csv

│   ├── datos\_entrenamiento.csv

│   ├── datos\_prueba.csv

│   └── datos\_para\_modelado.csv

├── notebooks/

│   ├── 01\_preparacion\_de\_datos.ipynb

│   ├── 02\_objetivo\_del\_negocio.ipynb

│   ├── 03\_limpieza\_datos.ipynb

│   ├── 04\_modelado.ipynb

│   └── 05\_evaluacion\_final.ipynb

├── scripts/

│   ├── graficos.py

│   └── modelos.py

├── API/

│   ├── main.py        ← FastAPI

│   ├── app.py         ← Streamlit

│   └── xgboost.pkl

└── README.md

```



---



\## 🔬 Metodología



\### 1. Preparación de Datos

\- Renombrado de columnas para mayor legibilidad del código

\- Agrupación de retrasos ≥ 4 meses en una sola categoría (criterio: todos equivalen a morosidad severa >90 días)

\- Recodificación de variables demográficas con categorías de baja frecuencia

\- \*\*Split estratificado train/test (70/30) como primer paso\*\*, antes de cualquier transformación, para garantizar ausencia de data leakage



\### 2. Análisis Exploratorio y Definición del Negocio

\- Análisis de la distribución de clases y sus implicaciones para el modelado

\- Identificación de tendencia de deterioro progresivo: los clientes con retraso prácticamente se duplicaron de abril (10.3%) a septiembre (22.7%)

\- Detección de anomalía en `comportamiento\_septiembre`: salto de 28 a 3,688 clientes en la categoría de 1 mes de retraso entre agosto y septiembre — documentada con hipótesis alternativas (problema de recolección vs. contracción económica puntual)

\- Justificación de métricas desde el negocio: AUC-ROC y Recall penalizan más los falsos negativos, que representan el error más costoso para la institución



\### 3. Feature Engineering y Selección de Variables



Se realizó AUC univariado como análisis exploratorio inicial, confirmando que el \*\*comportamiento de pago reciente es el predictor más fuerte\*\* (AUC=0.69), mientras que variables demográficas y montos absolutos tienen poder discriminante cercano al azar.



A partir de las 23 variables originales se construyeron features con justificación financiera:



| Variable | Descripción | Justificación |

|---|---|---|

| `conteo\_retrasos` | Meses con retraso >1 en 6 meses | Distingue mora crónica de evento aislado |

| `comportamiento\_septiembre` | Estado de pago más reciente | Indicador de alerta temprana más confiable (AUC=0.69) |

| `conteo\_meses\_inactivos` | Meses sin saldo pendiente | Inactividad no siempre es señal positiva |

| `conteo\_impagos` | Meses con saldo sin abono | Clientes que no pagan a pesar de tener deuda |

| `abono\_promedio` | Promedio de abonos en 6 meses | Estabiliza señal de capacidad de pago |

| `utilizacion\_septiembre` | % de crédito usado (deuda/límite) | Medida relativa de presión financiera real |

| `estado\_cuenta\_septiembre` | Saldo en el último mes | Situación financiera más reciente |



\*\*Selección final:\*\* XGBoost de importancia → top 10 variables → análisis VIF → eliminación por multicolinealidad (VIF > 10) con criterio de recencia → \*\*7 variables con VIF < 3.0\*\*



\### 4. Modelado



Se implementaron dos modelos con pipeline completo:



\*\*Estrategias aplicadas:\*\*

\- `StratifiedKFold` para preservar proporción de clases en cada fold

\- `scale\_pos\_weight` para penalizar errores sobre defaults sin resampling

\- `Optuna con TPESampler` para optimización bayesiana de hiperparámetros

\- \*\*Nested Cross-Validation\*\* para separar optimización de evaluación y evitar optimism bias

\- `cross\_val\_predict` con `predict\_proba` para métricas out-of-fold honestas

\- Análisis de thresholds (0.2 a 0.8) para exponer el trade-off precisión/recall

\- Reentrenamiento final con 100% de datos de entrenamiento usando los mejores hiperparámetros



---



\## 📊 Resultados



\### Comparativa de Modelos (Validación — Nested CV)



| Métrica | XGBoost | Regresión Logística |

|---|---|---|

| AUC-ROC | \*\*0.7871\*\* | 0.7561 |

| Gini | \*\*0.5742\*\* | 0.5122 |

| KS | \*\*0.4332\*\* | 0.4136 |



\### Evaluación Final en Test Set (datos no vistos)



| Métrica | XGBoost | Regresión Logística |

|---|---|---|

| AUC-ROC | \*\*0.7788\*\* | 0.7435 |

| Gini | \*\*0.5576\*\* | 0.4869 |

| KS | \*\*0.4212\*\* | 0.3953 |

| Threshold | 0.70 | 0.75 |

| Precision | 0.6203 | 0.6335 |

| Recall | \*\*0.4184\*\* | 0.3169 |

| F1 | \*\*0.4997\*\* | 0.4225 |



La caída mínima entre validación y test (~1% en AUC) confirma que el modelo \*\*generaliza correctamente sin overfitting\*\*.



\### Selección de Threshold



Se eligió \*\*threshold = 0.70\*\* con criterio de negocio: permite identificar una población de alto riesgo representativa para aplicar acciones diferenciadas:

\- Clientes clasificados como default → candidatos a programas de reestructuración de deuda

\- Clientes en zona gris → monitoreo activo y restricción de límite

\- Clientes de bajo riesgo → candidatos a mejores productos crediticios



El threshold puede ajustarse dinámicamente según la respuesta del portafolio a las acciones implementadas.



---



\## 🚀 Deployment



\### FastAPI — Backend de Scoring



```bash

cd API

uvicorn main:app --reload

```



Documentación interactiva disponible en `http://localhost:8000/docs`



\*\*Endpoint principal:\*\*

```

POST /predict

```



```json

{

&nbsp; "conteo\_retrasos": 3,

&nbsp; "comportamiento\_septiembre": 2,

&nbsp; "conteo\_meses\_inactivos": 1,

&nbsp; "conteo\_impagos": 2,

&nbsp; "abono\_promedio": 1500.0,

&nbsp; "utilizacion\_septiembre": 0.85,

&nbsp; "estado\_cuenta\_septiembre": 1,

&nbsp; "threshold": 0.70

}

```



\*\*Respuesta:\*\*

```json

{

&nbsp; "probabilidad\_default": 0.8248,

&nbsp; "prediccion": 1,

&nbsp; "segmento\_riesgo": "Crítico",

&nbsp; "threshold\_usado": 0.70,

&nbsp; "recomendacion": "Cliente con riesgo crítico. Candidato a programa de reestructuración de deuda."

}

```



\### Streamlit — Interfaz de Scoring



```bash

streamlit run API/app.py

```



Disponible en `http://localhost:8501`



---



\## 🛠️ Tecnologías



!\[Python](https://img.shields.io/badge/Python-3.11-blue)

!\[XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-red)

!\[FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)

!\[Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

!\[Optuna](https://img.shields.io/badge/Optuna-3.x-purple)



---



\## ⚙️ Instalación



```bash

git clone https://github.com/tu-usuario/credit-risk-scoring

cd credit-risk-scoring

python -m venv .venv

.venv\\Scripts\\activate  # Windows

pip install -r API/requirements.txt

```



---



\## 👤 Autor



\*\*Roberto García\*\*  

Maestría en Ciencias Computacionales  

\[LinkedIn](#) · \[GitHub](#)

