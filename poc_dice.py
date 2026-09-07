"""
PoC-02 | Objetivo unico: comprobar que dice_ml genera un contrafactual
valido sobre el MISMO modelo de la PoC-01, modificando unicamente las
variables que declaramos accionables.

NO valida el modelo. NO vuelve a validar SHAP. NO usa datos reales.
Los datos son SINTETICOS: sirven solo para alimentar la herramienta.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import dice_ml
from dice_ml import Dice
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------
# PASO 1 - Mismo dataset sintetico de la PoC-01 (misma semilla)
# ---------------------------------------------------------------
VARIABLES = [
    "ingreso_mensual_promedio", "ingreso_mes_peor", "coef_variacion_ingreso",
    "gasto_mensual_promedio", "ratio_cuota_ingreso_minimo", "num_acreedores",
    "deuda_total", "cuota_mensual_total", "antiguedad_negocio_meses",
    "num_empleados", "ahorro_disponible", "dias_atraso_max_12m",
    "num_creditos_ultimos_12m", "tiene_cuenta_bancaria", "usa_credito_informal",
    "carga_familiar", "nivel_educativo", "formalidad_negocio",
]

N = 800
ingreso = np.random.gamma(4, 400, N) + 500
peor = ingreso * np.random.uniform(0.25, 0.95, N)
cv = (ingreso - peor) / ingreso
cuota = ingreso * np.random.uniform(0.05, 0.75, N)

X = pd.DataFrame({
    "ingreso_mensual_promedio": ingreso,
    "ingreso_mes_peor": peor,
    "coef_variacion_ingreso": cv,
    "gasto_mensual_promedio": ingreso * np.random.uniform(0.3, 0.9, N),
    "ratio_cuota_ingreso_minimo": cuota / peor,
    "num_acreedores": np.random.poisson(2, N).astype(float),
    "deuda_total": cuota * np.random.uniform(6, 30, N),
    "cuota_mensual_total": cuota,
    "antiguedad_negocio_meses": np.random.randint(1, 180, N).astype(float),
    "num_empleados": np.random.poisson(1.5, N).astype(float),
    "ahorro_disponible": np.random.gamma(2, 300, N),
    "dias_atraso_max_12m": np.random.choice([0, 0, 0, 5, 15, 30, 60], N).astype(float),
    "num_creditos_ultimos_12m": np.random.poisson(1.8, N).astype(float),
    "tiene_cuenta_bancaria": np.random.binomial(1, 0.6, N).astype(float),
    "usa_credito_informal": np.random.binomial(1, 0.45, N).astype(float),
    "carga_familiar": np.random.poisson(2, N).astype(float),
    "nivel_educativo": np.random.randint(1, 5, N).astype(float),
    "formalidad_negocio": np.random.binomial(1, 0.35, N).astype(float),
})[VARIABLES]

riesgo = (
    2.5 * X["ratio_cuota_ingreso_minimo"] + 1.8 * X["coef_variacion_ingreso"]
    + 0.35 * X["num_acreedores"] + 0.02 * X["dias_atraso_max_12m"]
    + 0.7 * X["usa_credito_informal"] - 0.006 * X["antiguedad_negocio_meses"]
    - 0.0009 * X["ahorro_disponible"]
)
y = (riesgo + np.random.normal(0, 0.4, N) > riesgo.median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# ---------------------------------------------------------------
# PASO 2 - HALLAZGO 1: DiCE no acepta el Booster nativo
# ---------------------------------------------------------------
print("=" * 72)
print("PASO 2 | COMPATIBILIDAD DEL MODELO")

booster_nativo = lgb.train(
    {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
     "min_data_in_leaf": 20, "verbose": -1, "seed": RANDOM_SEED},
    lgb.Dataset(X_train, y_train), num_boost_round=150,
)
print(f"  lgb.Booster tiene predict_proba? "
      f"{hasattr(booster_nativo, 'predict_proba')}")
print("  => DiCE exige predict_proba. El Booster de la PoC-01 NO sirve aqui.")

# Solucion: entrenar con la API sklearn de LightGBM.
clf = lgb.LGBMClassifier(
    n_estimators=150, learning_rate=0.05, num_leaves=15,
    min_child_samples=20, random_state=RANDOM_SEED, verbose=-1,
)
clf.fit(X_train, y_train)
print(f"  lgb.LGBMClassifier tiene predict_proba? {hasattr(clf, 'predict_proba')}")

# Verificacion critica: el .booster_ interno es el MISMO modelo.
caso = X_test.iloc[[0]]
p_clf = float(clf.predict_proba(caso)[0][1])
p_booster_interno = float(clf.booster_.predict(caso)[0])
print(f"  clf.predict_proba          : {p_clf:.6f}")
print(f"  clf.booster_.predict       : {p_booster_interno:.6f}")
print(f"  Mismo modelo para SHAP y DiCE: "
      f"{'SI' if abs(p_clf - p_booster_interno) < 1e-9 else 'NO'}")

# ---------------------------------------------------------------
# PASO 3 - INPUT: declarar que variables son accionables
# ---------------------------------------------------------------
ACCIONABLES = [
    "gasto_mensual_promedio", "ahorro_disponible", "num_acreedores",
    "deuda_total", "cuota_mensual_total", "num_creditos_ultimos_12m",
    "usa_credito_informal", "tiene_cuenta_bancaria", "formalidad_negocio",
]
NO_ACCIONABLES = [v for v in VARIABLES if v not in ACCIONABLES]

print("\nPASO 3 | VARIABLES ACCIONABLES")
print(f"  Accionables    ({len(ACCIONABLES)}): {', '.join(ACCIONABLES)}")
print(f"  No accionables ({len(NO_ACCIONABLES)}): {', '.join(NO_ACCIONABLES)}")
print("  Criterio: solo puede recomendarse lo que la emprendedora decide.")

df_train = X_train.copy()
df_train["riesgo_sobreendeudamiento"] = y_train.values

data_dice = dice_ml.Data(
    dataframe=df_train,
    continuous_features=VARIABLES,
    outcome_name="riesgo_sobreendeudamiento",
)
model_dice = dice_ml.Model(model=clf, backend="sklearn")
explicador = Dice(data_dice, model_dice, method="random")

# ---------------------------------------------------------------
# PASO 4 - VUELTA (C): generar los contrafactuales
# ---------------------------------------------------------------
print("\nPASO 4 | GENERACION DE CONTRAFACTUALES")
print(f"  Probabilidad original: {p_clf:.2%}  (clase {clf.predict(caso)[0]})")

resultado = explicador.generate_counterfactuals(
    caso,
    total_CFs=8,
    desired_class="opposite",
    features_to_vary=ACCIONABLES,
    random_seed=RANDOM_SEED,
)
cfs = resultado.cf_examples_list[0].final_cfs_df
if cfs is None or len(cfs) == 0:
    print("  NO se genero ningun contrafactual. Revisar restricciones.")
    raise SystemExit(1)

cfs_X = cfs[VARIABLES].astype(float)
probs_cf = clf.predict_proba(cfs_X)[:, 1]
print(f"  Contrafactuales generados: {len(cfs_X)}")

# ---------------------------------------------------------------
# PASO 5 - OUTPUT: que cambio en cada uno
# ---------------------------------------------------------------
print("\nPASO 5 | LECTURA DE CADA CONTRAFACTUAL")
orig = caso.iloc[0]
detalle = []
for i in range(len(cfs_X)):
    fila = cfs_X.iloc[i]
    cambios = []
    for v in VARIABLES:
        if not np.isclose(fila[v], orig[v], rtol=1e-6):
            cambios.append({
                "variable": v,
                "actual": round(float(orig[v]), 2),
                "propuesto": round(float(fila[v]), 2),
                "delta": round(float(fila[v] - orig[v]), 2),
            })
    print(f"\n  --- Contrafactual {i+1} ---")
    print(f"  Probabilidad: {p_clf:.2%} -> {probs_cf[i]:.2%} "
          f"({probs_cf[i]-p_clf:+.2%})")
    for c in cambios:
        print(f"    {c['variable']:<28} {c['actual']:>10.2f} -> "
              f"{c['propuesto']:>10.2f}  ({c['delta']:+.2f})")
    # Validacion de inmutabilidad
    tocadas_prohibidas = [c["variable"] for c in cambios
                          if c["variable"] in NO_ACCIONABLES]
    print(f"    Variables no accionables tocadas: "
          f"{tocadas_prohibidas if tocadas_prohibidas else 'ninguna'}")
    detalle.append({
        "id": i + 1,
        "probabilidad_original": round(p_clf, 4),
        "probabilidad_contrafactual": round(float(probs_cf[i]), 4),
        "cambios": cambios,
        "respeta_inmutabilidad": len(tocadas_prohibidas) == 0,
    })

# ---------------------------------------------------------------
# PASO 6 - HALLAZGO 2: coherencia entre variables dependientes
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("PASO 6 | CHEQUEO DE COHERENCIA (variables derivadas)")
print("  ratio_cuota_ingreso_minimo deberia ser cuota / ingreso_mes_peor.")
incoherentes = 0
for i in range(len(cfs_X)):
    fila = cfs_X.iloc[i]
    esperado = fila["cuota_mensual_total"] / fila["ingreso_mes_peor"]
    declarado = fila["ratio_cuota_ingreso_minimo"]
    coherente = np.isclose(esperado, declarado, rtol=0.01)
    if not coherente:
        incoherentes += 1
    print(f"  CF{i+1}: declarado {declarado:.4f} | recalculado {esperado:.4f} "
          f"| coherente: {'SI' if coherente else 'NO'}")
print(f"\n  Contrafactuales incoherentes: {incoherentes}/{len(cfs_X)}")
print("  => DiCE trata cada variable como independiente. Si el perfil tiene")
print("     variables derivadas, hay que recalcularlas o excluirlas.")

# ---------------------------------------------------------------
# PASO 7 - CONTRATO JSON de la capa 4
# ---------------------------------------------------------------
salida = {
    "poc_id": "PoC-02-DiCE",
    "capa_4_contrafactuales": {
        "componente": "dice_ml.Dice",
        "method": "random",
        "backend": "sklearn",
        "modelo": "lightgbm.LGBMClassifier",
        "features_to_vary": ACCIONABLES,
        "features_inmutables": NO_ACCIONABLES,
        "total_CFs_solicitados": 8,
        "total_CFs_generados": len(cfs_X),
        "coherencia_variables_derivadas_ok": incoherentes == 0,
        "contrafactuales": detalle,
    }
}
with open("resultado_poc2.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, indent=2, ensure_ascii=False)
print("\nPASO 7 | Archivo generado: resultado_poc2.json")
print("=" * 72)
