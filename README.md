# Telecom X – Predicción de Churn (Parte 2)

Proyecto completo de analítica y machine learning para estimar el riesgo de cancelación (churn) en una telco LATAM. Incluye pipeline reproducible de **ingestión, limpieza, ingeniería ligera, selección de variables, modelado comparativo, evaluación avanzada (ROC / PR / threshold tuning), interpretabilidad (importancias, SHAP / fallback), persistencia de artefactos y recomendaciones estratégicas** contenido en el notebook `TelecomX2_LATAM.ipynb`.

---

## 🧾 Resumen Ejecutivo

> (Sustituir los marcadores tras ejecutar el notebook)
>
> El mejor modelo final fue **`<BEST_MODEL>`** con **F1 = `<F1>`**, **Recall = `<RECALL>`** y **ROC-AUC = `<ROC>`** sobre el conjunto de test. Se adoptó un umbral operativo = **`<THRESHOLD>`** optimizado para maximizar F1 (alternativa con Recall≥0.80 evaluada). Las variables con mayor impacto en la probabilidad de churn incluyen: `<TOP_FEATURES>`. Con estas señales se prioriza la segmentación proactiva de clientes de alto riesgo, optimizando campañas de retención y reduciendo costos asociados a pérdida de ingresos recurrentes.
>
> Se recomienda monitorear mensualmente el desempeño del modelo (F1, Recall, drift en distribución de probabilidades) y recalibrar el umbral ante cambios significativos.

Tabla rápida (ejemplo a completar):

| Métrica            | Valor            |
| ------------------- | ---------------- |
| F1 Test             | `<F1>`         |
| Recall Test         | `<RECALL>`     |
| Precision Test      | `<PRECISION>`  |
| ROC-AUC Test        | `<ROC>`        |
| Threshold operativo | `<THRESHOLD>`  |
| Modelo              | `<BEST_MODEL>` |

---

---

## 📂 Estructura del Repositorio

```
├── TelecomX2_LATAM.ipynb        # Notebook principal (pipeline end‑to‑end)
├── data/
│   ├── TelecomX_model_ready.parquet  # Dataset base (procesado parte 1)
│   └── telecomX_model_ready.csv      # Export generado al cargar
├── best_churn_model.joblib       # (Se genera tras ejecutar sección 4.4)
└── README.md
```

> El archivo `best_churn_model.joblib` se crea sólo después de ejecutar la evaluación extendida (4.4) y guardar artefactos.

---

## 🧪 Objetivo

Construir y comparar modelos supervisados para predecir probabilidad de churn, priorizando **F1** y **Recall** (sensibles para retención), generando además un umbral operativo y explicaciones de variables clave.

---

## 🔍 Flujo Metodológico (Resumen de Secciones)

| Sección       | Contenido                                                              | Notas                                         |
| -------------- | ---------------------------------------------------------------------- | --------------------------------------------- |
| 2.1            | Carga y exploración inicial                                           | Detección heurística de la columna objetivo |
| 2.2–2.3       | Limpieza, eliminación de columnas irrelevantes, codificación One-Hot | Manejo robusto de target heterogéneo         |
| 2.4–2.6       | Análisis de desbalance, (opcional SMOTE), escalado                    | Escalado sólo para modelos sensibles         |
| 3.1–3.2       | Correlaciones y selección simple de features                          | Heurística top-N por                         |
| 4.1–4.3       | Split, entrenamiento y métricas base                                  | Logistic, KNN, RandomForest, Linear SVC       |
| 4.4            | ROC, Precision-Recall, comparación train/test, tuning de umbral       | Guarda artefactos + threshold                 |
| 4.5 (opcional) | Validación cruzada + RandomizedSearch ligero                          | Sólo actualiza si mejora F1                  |
| 5.1            | Importancia/coeficientes de modelos                                    | Tabla unificada + gráficos                   |
| 5.3 (opcional) | Interpretabilidad SHAP / fallback permutation importance               | Muestra reducida para velocidad               |
| 6              | Borrador automatizado de conclusiones estratégicas                    | Plantilla editable                            |

---

## 🧾 Datos

- Archivo base: `data/TelecomX_model_ready.parquet` (resultado de la parte 1).
- El notebook exporta a CSV (`telecomX_model_ready.csv`) para referencia externa.
- La columna objetivo se mapea de valores textuales/booleanos/numéricos a {0,1} con validaciones (umbral ≤5% de valores no mapeables).

---

## 🛠️ Dependencias Principales

| Grupo             | Librerías                                      |
| ----------------- | ----------------------------------------------- |
| Core              | pandas, numpy, pyarrow                          |
| ML                | scikit-learn, imbalanced-learn (SMOTE opcional) |
| Visualización    | matplotlib, seaborn                             |
| Interpretabilidad | shap (opcional)                                 |
| Persistencia      | joblib                                          |

### Instalación Rápida (Windows PowerShell)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install pandas numpy pyarrow scikit-learn imbalanced-learn seaborn matplotlib joblib shap
```

> Si no deseas SHAP inicialmente, puedes omitir `shap` (la celda 5.3 mostrará un mensaje y usará fallback si corresponde).

---

## ▶️ Ejecución del Notebook

1. Activa el entorno virtual y abre VS Code / Jupyter.
2. Ejecuta secuencialmente las celdas (secciones 2 → 6).
3. Verifica que `metrics_df` y el modelo con mejor F1 se impriman en 4.3.
4. Ejecuta 4.4 para curvas, threshold y guardado `best_churn_model.joblib`.
5. (Opcional) Ejecuta 4.5 para tuning ligero.
6. Re-ejecuta 5.1 y (opcional) 5.3 si hubo tuning que cambie importancias.
7. Ejecuta 6.1 para generar el borrador de conclusiones y edita según criterio de negocio.

### Carga del Modelo Guardado (Ejemplo)

```python
import joblib
art = joblib.load('best_churn_model.joblib')
model = art['model']
scaler = art['scaler']
features = art['features']
thr = art['threshold']
# Supón nuevo DataFrame df_new con mismas columnas originales antes de dummies
# -> Debes replicar pipeline de preparación para crear X_encoded[features]
```

---

## 📊 Métricas

Se construye una tabla comparativa con: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC_AUC` (si hay probabilidades).
La **optimización de umbral** escanea valores 0.05→0.95 buscando máximo F1 y alternativa con `Recall ≥ 0.80`.

### Ejemplo de interpretación

- Gap `F1_train - F1_test` > 0.05 → alerta de posible overfitting.
- Si RandomForest domina, se reportan importancias; si Logistic, coeficientes (signo = dirección del efecto).

---

## 🔍 Interpretabilidad

- **Importancias**: RandomForest (`feature_importances_`), Logistic / LinearSVC (`coef_`).
- **SHAP (5.3)**: Usa `TreeExplainer` sobre muestra (≤400 filas) para impacto global; fallback a permutation importance si `shap` no está.

---

## 🧪 Tuning (4.5)

RandomizedSearch con mallas discretas:

- RandomForest: profundidad, n_estimators, min_samples_split/leaf, max_features.
- LogisticRegression: C y solver.
  Sólo reemplaza el modelo en `models` y actualiza `metrics_df` si mejora F1 en test ≥ 0.001.

---

## 💾 Persistencia y Deployment Inicial

`best_churn_model.joblib` contiene:

```python
{
  'model_name': <str>,
  'model': <estimador sklearn>,
  'features': <lista str>,
  'scaler': <StandardScaler or None>,
  'threshold': <float>,
  'metrics_table': metrics_df
}
```

Para un API / batch scoring futuro:

1. Replicar pipeline de limpieza + dummies.
2. Seleccionar las mismas `features` (rellenar faltantes con 0 si aparecen nuevas dummy columns).
3. Aplicar `scaler` si no es None.
4. Obtener `proba = model.predict_proba(X)[:,1]` y clasificar con `threshold` guardado.

---

## ✅ Buenas Prácticas Implementadas

- Validaciones tempranas de variables clave.
- Manejo de target heterogéneo con tokens positivos/negativos.
- Control de desbalance (SMOTE condicional).
- Separación lógica de pasos y reproducibilidad de features.
- Threshold data-driven (F1 y alternativa recall alto).
- Guardado de artefactos y metadata para reproducibilidad.

---

## 🚀 Próximos Pasos Recomendados

- Grid/Random/Bayesian Search más profundo (recursos permitting).
- Añadir **permutation importance** estable para todos los modelos.
- Implementar **validación temporal** si hay componente temporal en el churn.
- Monitoreo de drift (PSI, KS) y reentrenos programados.
- Integrar métricas de negocio: LTV, costo de retención, priorización económica.
- Despliegue como servicio REST / batch (FastAPI + contenedor).

---

## 🔐 Consideraciones

- Verificar compliance y privacidad si se añaden datos sensibles.
- Documentar versiones exactas (se recomienda `pip freeze > requirements.txt`).

Ejemplo rápido:

```powershell
pip freeze > requirements.txt
```

---

## 📄 Licencia

(Agregar tipo de licencia aquí, ej.: MIT, Apache 2.0).

---

## 👤 Autor

Proyecto desarrollado como parte del desafío "Challenge ONE Data Science – Telecom X (Parte 2)".

---

## 🆘 Soporte

Si encuentras un issue:

1. Crear un *Issue* con: descripción, paso para reproducir, traceback y sección del notebook.
2. Adjuntar versión de Python y scikit-learn (`python -c "import sklearn, sys; print(sys.version); print(sklearn.__version__)"`).

---

## ✨ Resumen Ejecutivo (plantilla)

> El mejor modelo fue `<MODEL>` con F1=`<F1>`. Principales impulsores: `<FEATURES>`. Umbral operativo = `<THR>`. Próximos pasos: mejorar hiperparámetros, monitorear drift, evaluar impacto económico.

---

## 🔁 Reproducibilidad Rápida

1. Clonar repositorio.
2. Crear y activar entorno virtual.
3. `pip install -r requirements.txt`.
4. Abrir y ejecutar `TelecomX2_LATAM.ipynb` en orden.
5. Verificar creación de `best_churn_model.joblib`.

### Script de extracción de métricas (opcional)

```python
import joblib, pandas as pd
art = joblib.load('best_churn_model.joblib')
print('Modelo:', art['model_name'])
print('Threshold:', art['threshold'])
print(art['metrics_table'])
```

---

## 🧪 Verificación Rápida del Entorno

```powershell
python -c "import sklearn, pandas, numpy; print('sklearn', sklearn.__version__)"
```

---

## 📌 Checklist de Entrega Final

- [ ] Notebook ejecutado sin errores (celdas en orden)
- [ ] Métricas finales reemplazadas en Resumen Ejecutivo
- [ ] Archivo `best_churn_model.joblib` generado
- [ ] README actualizado (sin placeholders)
- [ ] `requirements.txt` congelado si se desea reproducibilidad exacta (`pip freeze > requirements.txt`)

---

## 🗂 Versionado y Congelación (Opcional)

Para entornos productivos, fijar versiones exactas:

```powershell
pip freeze | Out-File -Encoding UTF8 requirements_exact.txt
```

---

---

¡Listo! Tras reemplazar marcadores, el documento queda listo para presentación / repositorio público.
