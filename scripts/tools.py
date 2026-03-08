def preprocesar(df):
    """
    Aplica todas las transformaciones de feature engineering al dataset.
    Funciona tanto para train como para test.
    """

    meses = ['abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre']

    # ── Conteo de retrasos ────────────────────────────────────────────────────
    cols_comportamiento = [f'comportamiento_{mes}' for mes in meses]
    df['conteo_retrasos'] = (df[cols_comportamiento] > 1).sum(axis=1)

    # ── Abonos ────────────────────────────────────────────────────────────────
    cols_abono = [f'abono_{mes}' for mes in meses]
    df['abono_promedio'] = df[cols_abono].mean(axis=1)
    df['abono_maximo']   = df[cols_abono].max(axis=1)
    df['abono_minimo']   = df[cols_abono].min(axis=1)

    # ── Utilización ───────────────────────────────────────────────────────────
    cols_estado      = [f'estado_cuenta_{mes}' for mes in meses]
    cols_utilizacion = [f'utilizacion_{mes}' for mes in meses]
    df[cols_utilizacion] = df[cols_estado].div(df['linea_credito'], axis=0)
    df['delta_endeudamiento_6meses']      = df['utilizacion_septiembre'] - df['utilizacion_abril']
    df['conteo_meses_estres_financiero']  = (df[cols_utilizacion] > 0.90).sum(axis=1)
    df['utilizacion_promedio']            = df[cols_utilizacion].mean(axis=1)
    df['utilizacion_maxima']              = df[cols_utilizacion].max(axis=1)
    df['utilizacion_minima']              = df[cols_utilizacion].min(axis=1)

    # ── Clipping ──────────────────────────────────────────────────────────────
    df[cols_utilizacion]                   = df[cols_utilizacion].clip(0, 1)
    df['conteo_meses_estres_financiero']   = df['conteo_meses_estres_financiero'].clip(0, 6)
    df['delta_endeudamiento_6meses']       = df['delta_endeudamiento_6meses'].clip(-1, 1)

    # ── Impagos ───────────────────────────────────────────────────────────────
    meses_analisis = [
        ('abril', 'mayo'), ('mayo', 'junio'), ('junio', 'julio'),
        ('julio', 'agosto'), ('agosto', 'septiembre')
    ]
    df['conteo_impagos'] = sum(
        ((df[f'abono_{sig}'] == 0) & (df[f'estado_cuenta_{act}'] > 0)).astype(int)
        for act, sig in meses_analisis
    )
    df['impago_agosto'] = ((df['abono_septiembre'] == 0) & (df['estado_cuenta_agosto'] > 0)).astype(int)

    # ── Pago mínimo ───────────────────────────────────────────────────────────
    meses_pago = {'abril': 'mayo', 'mayo': 'junio', 'junio': 'julio',
                  'julio': 'agosto', 'agosto': 'septiembre'}
    df['conteo_pago_minimo'] = sum(
        ((df[f'estado_cuenta_{est}'] > 0) &
         (df[f'abono_{abo}'] > 0) &
         (df[f'abono_{abo}'] <= df[f'estado_cuenta_{est}'] * 0.1)).astype(int)
        for est, abo in meses_pago.items()
    )
    df['pago_minimo_sept'] = (
        (df['estado_cuenta_agosto'] > 0) &
        (df['abono_septiembre'] > 0) &
        (df['abono_septiembre'] <= df['estado_cuenta_septiembre'] * 0.1)
    ).astype(int)

    # ── Meses inactivos ───────────────────────────────────────────────────────
    df['conteo_meses_inactivos'] = (df[cols_estado] == 0).sum(axis=1)

    # ── Selección de columnas finales ─────────────────────────────────────────
    df = df[[
        'conteo_retrasos',
        'comportamiento_septiembre',
        'conteo_meses_inactivos',
        'conteo_impagos',
        'abono_promedio',
        'utilizacion_septiembre',
        'estado_cuenta_septiembre',
        'DEFAULT'
    ]]

    return df