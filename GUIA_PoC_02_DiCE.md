# PoC-02 — DiCE, generación de contrafactuales

**Objetivo único:** comprobar que `dice_ml` genera un contrafactual válido sobre el mismo modelo de la PoC-01, modificando únicamente las variables declaradas accionables.

**Lo que esta PoC NO hace:** no vuelve a validar SHAP, no valida el modelo, no usa datos reales, no arma la recomendación en lenguaje natural.

---

## 1. Preparación

Ya tienes el entorno de la PoC-01. Solo falta una librería:

```bash
# con el .venv activado, en la misma carpeta
pip install dice-ml
```

Versión verificada: `dice_ml 0.12` sobre `lightgbm 4.7.0`, Python 3.12.

Guarda `poc_dice.py` junto a `poc_shap.py` y córrelo:

```bash
python poc_dice.py
```

Aparecerán barras de progreso (`0%|  | 0/8`) mientras genera. Es normal.

---

## 2. Resultado: la PoC pasa, pero con reservas serias

DiCE **sí genera contrafactuales**. Pidió 8, entregó 8, todos bajan la probabilidad de 92.85% a un rango de 5% a 37%, y ninguno tocó una variable inmutable. En términos estrictos, el objetivo se cumplió.

El problema es lo que dicen esos contrafactuales.

---

## 3. Los cuatro hallazgos

### Hallazgo 1 — El `Booster` de la PoC-01 no sirve para DiCE

```
lgb.Booster tiene predict_proba?        False
lgb.LGBMClassifier tiene predict_proba? True
```

DiCE interroga al modelo por `predict_proba`, un método de la API de scikit-learn. El `Booster` nativo no lo tiene. **Esto habría roto la integración si lo descubrían en desarrollo y no ahora.**

La solución no es entrenar dos modelos. Es entrenar un `LGBMClassifier` y usar su atributo interno para SHAP:

```
clf.predict_proba          : 0.928498
clf.booster_.predict       : 0.928498
Mismo modelo para SHAP y DiCE: SI
```

Un solo modelo entrenado; C3 consume `clf.booster_`, C4 consume `clf`. Es una decisión de arquitectura, no un detalle de implementación, y hay que escribirla.

### Hallazgo 2 — Los contrafactuales son válidos pero absurdos

Este es el hallazgo importante. Lee lo que el sistema propone:

| CF | Recomendación implícita | Prob. |
|---|---|---|
| 1 | Aumentar el gasto mensual de S/ 912 a S/ 2,267 | 15.58% |
| 4 | **Empezar a usar crédito informal** (0 → 1) | 36.58% |
| 5 | Subir la cuota mensual de S/ 1,690 a S/ 2,472 | 14.31% |
| 7 | Gastar S/ 3,955 en lugar de S/ 912 | 5.62% |
| 8 | **Tomar 2.3 créditos más** en los últimos 12 meses | 24.98% |

Matemáticamente correctos: son puntos del espacio de variables donde el modelo predice bajo riesgo. Como consejo a una emprendedora endeudada, son indefendibles. "Para no sobreendeudarte, gasta cuatro veces más y toma un préstamo informal."

Esta es la comprobación empírica de la objeción que te hizo Herrera con la analogía del tráfico. **Un contrafactual no es una recomendación.** Es un punto vecino en el espacio de features, y la dirección del cambio no la controla nadie salvo que se la impongas.

Implicación de diseño: la capa 4 necesita una **restricción de dirección** por variable — `permitted_range` que impida que el gasto suba, que la deuda crezca o que el crédito informal se active. Sin eso, el output no puede llegar al usuario.

### Hallazgo 3 — DiCE rompe la coherencia entre variables dependientes

```
CF5: declarado 0.9459 | recalculado 1.3834 | coherente: NO
Contrafactuales incoherentes: 1/8
```

En el CF5, DiCE subió `cuota_mensual_total` pero dejó `ratio_cuota_ingreso_minimo` congelado en su valor original. El ratio se calcula a partir de la cuota, así que ese perfil **es aritméticamente imposible**: describe a alguien cuya cuota subió pero cuyo ratio cuota/ingreso no se movió.

DiCE trata cada columna como independiente. No sabe que unas se derivan de otras.

Salida: o excluyes las derivadas de `features_to_vary` y las recalculas después, o las sacas del modelo. Y esa decisión hay que tomarla sobre las 18 variables reales, no sobre las sintéticas — cuando tengan la lista definitiva, marquen cuáles son derivadas.

### Hallazgo 4 — Las variables discretas salen con decimales

```
tiene_cuenta_bancaria      0.00 -> 0.60
num_creditos_ultimos_12m   2.00 -> 4.30
```

Una cuenta bancaria no se tiene al 60%, y nadie toma 4.3 créditos. Están declaradas como `continuous_features` y DiCE las mueve como continuas. Hay que declarar el tipo correcto o redondear antes de mostrar.

---

## 4. Criterio de éxito, revisado

| Chequeo | Resultado |
|---|---|
| Genera contrafactuales sobre el modelo | Sí — 8/8 |
| Baja la probabilidad al otro lado de la frontera | Sí — de 92.85% a 5–37% |
| Respeta las variables inmutables | Sí — 0 violaciones |
| El contenido es accionable | **No** — ver hallazgos 2, 3 y 4 |

Los tres primeros son de la herramienta. El cuarto es de ustedes, y todavía no está resuelto.

---

## 5. Lo que esto le dice al asesor

La frase con la que abrir la próxima sesión:

> "DiCE funciona y genera contrafactuales válidos, pero comprobamos que un contrafactual no equivale a una recomendación: en tres de ocho casos nos propuso aumentar el gasto o tomar crédito informal para bajar el riesgo. La capa 4 necesita restricciones de dirección antes de que su salida pueda mostrarse."

Eso responde exactamente lo que él te planteó y lo responde con datos, no con papers. Es la diferencia entre "leímos que DiCE genera contrafactuales" y "lo corrimos y encontramos su límite".

---

## 6. Lo que sigue

- **PoC-03** (después, no ahora): las mismas restricciones de dirección aplicadas con `permitted_range`, y comprobar si aún así se generan contrafactuales. Es posible que con restricciones realistas **no encuentre ninguno** — ese sería un resultado igual de valioso.
- La traducción de contrafactual a texto en español no es una PoC: es implementación, y va después.
- Sigue pendiente el conteo de variables en ENAHO y Findex. Eso no lo destraba ninguna PoC.

---

## Anexo — Los bloques que importan del script

Configuración del explicador:

```python
data_dice = dice_ml.Data(
    dataframe=df_train,                      # X_train + columna de etiqueta
    continuous_features=VARIABLES,
    outcome_name="riesgo_sobreendeudamiento",
)
model_dice = dice_ml.Model(model=clf, backend="sklearn")   # clf, NO booster
explicador = Dice(data_dice, model_dice, method="random")
```

Generación con inmutabilidad:

```python
resultado = explicador.generate_counterfactuals(
    caso,                                    # 1 fila x 18 columnas
    total_CFs=8,
    desired_class="opposite",                # de clase 1 a clase 0
    features_to_vary=ACCIONABLES,            # solo las 9 accionables
    random_seed=42,
)
cfs = resultado.cf_examples_list[0].final_cfs_df
```

Chequeo de coherencia (el que reveló el hallazgo 3):

```python
esperado  = fila["cuota_mensual_total"] / fila["ingreso_mes_peor"]
declarado = fila["ratio_cuota_ingreso_minimo"]
coherente = np.isclose(esperado, declarado, rtol=0.01)
```

El script completo está en `poc_dice.py` y la salida estructurada en `resultado_poc2.json`.
