Ejecuta este snippet en una consola Python dentro del entorno después de haber generado best_churn_model.joblib para volcar métricas exactas:

import joblib, json
art = joblib.load('best_churn_model.joblib')
print('Modelo:', art['model_name'])
print('Threshold:', art['threshold'])
print(art['metrics_table'])

# (Opcional) Guardar métricas a JSON para automatizar actualización README
art['metrics_table'].to_json('metrics_table.json', orient='records', lines=False)
