def modelo_optimizado(X, y, modelo='logistica', n_trials=30):
    """
    Optimiza hiperparámetros con Optuna, evalúa métricas honestas con
    nested cross-validation y guarda el modelo final entrenado con todos los datos.

    Parámetros:
    -----------
    X        : DataFrame con las variables predictoras
    y        : Serie con la variable objetivo
    modelo   : Tipo de modelo a entrenar ('logistica' o 'xgboost')
    n_trials : Número de combinaciones que prueba Optuna (default: 30)
    """
    from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from scipy.stats import ks_2samp
    import xgboost as xgb
    import optuna
    import joblib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Espacio de búsqueda según modelo ──────────────────────────────────────
    def objective(trial):
        if modelo == 'logistica':
            params = {
                'C':        trial.suggest_float('C', 1e-5, 100, log=True),
                'solver':   'lbfgs',
                'max_iter': 1000
            }
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(**params, class_weight='balanced', random_state=42))
            ])

        elif modelo == 'xgboost':
            counts = y.value_counts()
            spw    = counts[0] / counts[1]
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 50, 300),
                'max_depth':        trial.suggest_int('max_depth', 3, 8),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            pipeline = Pipeline([
                ('xgb', xgb.XGBClassifier(**params, scale_pos_weight=spw,
                                           eval_metric='logloss', random_state=42))
            ])

        scores = cross_val_score(pipeline, X, y, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    # ── Optimización ──────────────────────────────────────────────────────────
    print(f"\nOptimizando {modelo}...")
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Mejores parámetros  : {study.best_params}")
    print(f"Mejor AUC (inner CV): {study.best_value:.4f}")

    # ── Modelo final con mejores parámetros ───────────────────────────────────
    if modelo == 'logistica':
        mejor_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=study.best_params['C'],
                                      solver='lbfgs',
                                      max_iter=1000,
                                      class_weight='balanced',
                                      random_state=42))
        ])
    elif modelo == 'xgboost':
        counts = y.value_counts()
        spw    = counts[0] / counts[1]
        mejor_pipeline = Pipeline([
            ('xgb', xgb.XGBClassifier(**study.best_params, scale_pos_weight=spw,
                                       eval_metric='logloss', random_state=42))
        ])

    # ── Métricas honestas con outer CV ────────────────────────────────────────
    y_proba = cross_val_predict(mejor_pipeline, X, y, cv=outer_cv,
                                method='predict_proba', n_jobs=-1)[:, 1]

    auc    = roc_auc_score(y, y_proba)
    gini   = 2 * auc - 1
    ks, _  = ks_2samp(y_proba[y == 0], y_proba[y == 1])

    print(f"\nAUC-ROC : {auc:.4f}")
    print(f"Gini    : {gini:.4f}")
    print(f"KS      : {ks:.4f}")

    # ── Análisis de thresholds ────────────────────────────────────────────────
    thresholds = np.arange(0.2, 0.8, 0.05)
    resultados = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred_t).ravel()
        resultados.append({
            'Threshold' : round(t, 2),
            'TP'        : tp,
            'FP'        : fp,
            'TN'        : tn,
            'FN'        : fn,
            'Precision' : round(precision_score(y, y_pred_t, zero_division=0), 4),
            'Recall'    : round(recall_score(y, y_pred_t), 4),
            'F1'        : round(f1_score(y, y_pred_t), 4),
        })
    print(f"\nAnálisis de Thresholds:")
    print(pd.DataFrame(resultados).to_string(index=False))

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC — {modelo}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── Importancia de variables ──────────────────────────────────────────────
    mejor_pipeline.fit(X, y)

    if modelo == 'logistica':
        coefs  = mejor_pipeline.named_steps['lr'].coef_[0]
        df_imp = pd.DataFrame({
            'Variable':    X.columns,
            'Importancia': coefs
        }).sort_values('Importancia', key=abs, ascending=False)

    elif modelo == 'xgboost':
        coefs  = mejor_pipeline.named_steps['xgb'].feature_importances_
        df_imp = pd.DataFrame({
            'Variable':    X.columns,
            'Importancia': coefs
        }).sort_values('Importancia', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_imp['Variable'][::-1], df_imp['Importancia'][::-1], color='steelblue')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', va='center', fontsize=9)
    plt.xlabel('Importancia')
    plt.title(f'Importancia de Variables — {modelo}')
    plt.tight_layout()
    plt.show()

    # ── Guardar modelo ────────────────────────────────────────────────────────
    nombre = f'{modelo}.pkl'
    joblib.dump(mejor_pipeline, nombre)
    print(f"\nModelo guardado como {nombre} ✅")

    return mejor_pipeline, study.best_params, auc


#=======================
# EVALUACIÓN DEL MODELO   
#=======================

def evaluar_test(X_test, y_test, modelo, threshold=0.5):
    
    """
    Evalúa el modelo final sobre el test set con el threshold elegido.

    Parámetros:
    -----------
    X_test    : DataFrame con las variables predictoras del test set
    y_test    : Serie con la variable objetivo del test set
    modelo    : Modelo entrenado cargado con joblib
    threshold : Threshold elegido para clasificación (default: 0.5)
    
    """
    
    from sklearn.metrics import roc_auc_score, roc_curve, precision_score
    from sklearn.metrics import recall_score, f1_score, confusion_matrix
    from scipy.stats import ks_2samp
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    y_proba  = modelo.predict_proba(X_test)[:, 1]
    y_pred   = (y_proba >= threshold).astype(int)

    # ── Métricas generales ────────────────────────────────────────────────────
    auc    = roc_auc_score(y_test, y_proba)
    gini   = 2 * auc - 1
    ks, _  = ks_2samp(y_proba[y_test == 0], y_proba[y_test == 1])

    print(f"AUC-ROC : {auc:.4f}")
    print(f"Gini    : {gini:.4f}")
    print(f"KS      : {ks:.4f}")

    # ── Métricas con threshold elegido ────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\nThreshold : {threshold}")
    print(f"TP        : {tp}")
    print(f"FP        : {fp}")
    print(f"TN        : {tn}")
    print(f"FN        : {fn}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1        : {f1_score(y_test, y_pred):.4f}")

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC — Test Set')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── Matriz de confusión ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title(f'Matriz de Confusión — Threshold {threshold}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks([0, 1], ['No Default', 'Default'])
    plt.yticks([0, 1], ['No Default', 'Default'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=14)
    plt.tight_layout()
    plt.show()

    return auc, gini, ks