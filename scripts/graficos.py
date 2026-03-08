import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')


# ===== GRAFICO DEFAULT =====

def grafico_default(df):
    conteo = df['DEFAULT'].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(['Pago (0)', 'Impago (1)'], conteo.values, 
                  color=['green', 'red'], edgecolor='white', linewidth=1.5, width=1.0)

    total = len(df)
    for bar, val in zip(bars, conteo.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')

    ax.set_ylabel('Clientes', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, conteo.max() * 1.15) 

    plt.tight_layout()
    plt.show()


# ===== GRAFICO COMPORTAMIENTO =====

def grafico_comportamiento(df):
    meses = ['abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre']
    labels = ['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep']

    grupo_0 = []
    grupo_1 = []

    for mes in meses:
        col = f'comportamiento_{mes}'
        grupo_0.append((df[col] <= 0).sum())
        grupo_1.append((df[col] >= 1).sum())

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(labels, grupo_0, marker='o', color='green', linewidth=3.5, label='Sin retraso (≤ 0)')
    ax.plot(labels, grupo_1, marker='o', color='red', linewidth=3.5, label='Con retraso (≥ 1)')

    for i, (v0, v1) in enumerate(zip(grupo_0, grupo_1)):
        ax.text(i, v0 + 600, f'{v0:,}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2ecc71')
        ax.text(i, v1 - 800, f'{v1:,}', ha='center', va='top', fontsize=12, fontweight='bold', color='#e74c3c')

    ax.set_ylim(0, max(grupo_0) + 2500)
    ax.set_title('Comportamiento en los últimos 6 meses', fontsize=15, fontweight='bold')
    ax.set_xlabel('Mes', fontsize=13)
    ax.set_ylabel('Número de clientes', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()


# ===== GRAFICO MOROSIDAD =====

def grafico_morosidad(df):
    meses = ['abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre']
    comportamiento_cols = [f'comportamiento_{m}' for m in meses]

    resumen = df[comportamiento_cols].melt().groupby(['variable', 'value']).size().unstack(level=1)
    resumen_filtrado = resumen[[col for col in resumen.columns if col > 0]].copy()
    resumen_filtrado = resumen_filtrado.fillna(0)
    resumen_filtrado.index = [c.replace('comportamiento_', '').capitalize() for c in comportamiento_cols]

    ax = resumen_filtrado.plot(kind='line', marker='o', figsize=(10, 5), linewidth=2.5)

    ax.set_title('Distribución del comportamiento de morosidad', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de clientes')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

# ===== GRAFICO UTILIZACION CREDITO =====

def grafico_utilizacion_credito(df):
    mask = df['comportamiento_septiembre'] >= 1
    suma_bill = df.loc[mask, 'estado_cuenta_septiembre'].sum()
    suma_limit = df.loc[mask, 'linea_credito'].sum()
    resta = suma_limit - suma_bill

    fig, ax = plt.subplots(figsize=(7, 5))

    categorias = ['Crédito', 'Capital Disponible', 'Deuda']
    valores = [suma_limit, resta, suma_bill]
    colores = ['green', 'yellow', 'red']

    bars = ax.bar(categorias, valores, color=colores, edgecolor='white', linewidth=1.5, width=0.6)

    for bar, val in zip(bars, valores):
        pct = val / suma_limit * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + suma_limit * 0.01,
                f'${val:,.0f}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_ylabel('Monto', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, suma_limit * 1.2)
    ax.set_title('Utilización del crédito por clientes morosos en el último mes', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ===== GRAFICO AUC UNIVARIADO =====

def grafico_auc_univariado(df):
    ax = (
        df.drop(columns='DEFAULT')
          .apply(lambda x: roc_auc_score(df['DEFAULT'], x))
          .sort_values()
          .plot.barh(figsize=(8, 6), color='darkblue', width=0.9)
    )

    for bar in ax.patches:
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.2f}', va='center', ha='center', fontsize=9, color='white', fontweight='bold')

    ax.axvline(0.5, color='red', linewidth=5, linestyle='--', alpha=0.7, label='azar')
    ax.legend()
    ax.set_title("Capacidad de Discriminación de Variables frente al Default (AUC)", fontsize=14)
    plt.show()

# ===== SELECCIÓN DE CARACTERÍSTICAS =====

def importancia_de_caracteristicas(df, target='DEFAULT', top_n=10):
    """
    Entrena un modelo XGBoost para clasificación de riesgo crediticio
    y grafica las variables predictoras más importantes.
    Las importancias se calculan sobre un split 80/20 interno
    para evitar importancias infladas por sobreajuste.

    Parámetros:
    -----------
    df     : DataFrame de entrenamiento (sin el test set reservado)
    target : Nombre de la variable objetivo (default: 'DEFAULT')
    top_n  : Número de variables a mostrar en el gráfico (default: 10)
    """

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    counts = y_train.value_counts()
    spw = counts[0] / counts[1]

    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='logloss',
         verbose=0
    )

    model_xgb.fit(X_train, y_train)

    y_probs = model_xgb.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_probs)

    importancias = pd.DataFrame({
        'feature': X.columns,
        'importance': model_xgb.feature_importances_
    }).sort_values(by='importance', ascending=False)

    top = importancias.head(top_n)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top['feature'][::-1], top['importance'][::-1], color='steelblue')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1%}', va='center', fontsize=9)

    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Variables Predictoras — AUC-ROC Val: {auc_score:.4f}')
    plt.xlim(0, top['importance'].max() + 0.06)
    plt.tight_layout()
    plt.show()

    return model_xgb, auc_score, importancias


# ===== Factor de inflación de varianza =====

def factor_inflacion_varianza(X):
    """
    Calcula el Factor de Inflación de Varianza (VIF) para detectar multicolinealidad
    y grafica los resultados con una línea de referencia en VIF=10.

    Parámetros:
    -----------
    X : DataFrame con las variables predictoras (sin el target)
    """

    vif = pd.DataFrame({
        'Variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values('VIF', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(vif['Variable'][::-1], vif['VIF'][::-1], color='steelblue')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}', va='center', fontsize=9)

    plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10')
    plt.xlabel('VIF')
    plt.title('Factor de Inflación de Varianza (VIF)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return vif