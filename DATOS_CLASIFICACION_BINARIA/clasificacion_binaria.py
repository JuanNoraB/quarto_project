import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('default')
sns.set_palette("husl")

print("=== SISTEMA DE CLASIFICACIÓN BINARIA - SINIESTROS LETALES ===")
print("Cargando datos...")

# Cargar datos
df = pd.read_csv('inec_anuario-de-estadisticas-de-transporte_siniestros-de-transito_2019.csv', 
                 sep=';', encoding='utf-8')

print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
print("\nPrimeras 5 filas:")
print(df.head())

# Exploración inicial
print("\n=== EXPLORACIÓN DE DATOS ===")
print(f"Columnas: {list(df.columns)}")
print(f"\nValores faltantes por columna:")
print(df.isnull().sum())

# Crear variable objetivo binaria
print("\n=== CREANDO VARIABLE OBJETIVO ===")
df['SINIESTRO_LETAL'] = (df['NUM_FALLECIDO'] > 0).astype(int)

print(f"Distribución de la variable objetivo:")
print(df['SINIESTRO_LETAL'].value_counts())
print(f"Porcentaje de siniestros letales: {df['SINIESTRO_LETAL'].mean()*100:.2f}%")

# Preparar features
print("\n=== PREPARANDO FEATURES ===")
# Seleccionar columnas para el modelo (excluyendo las de conteo de víctimas para evitar data leakage)
feature_cols = ['MES', 'DIA', 'HORA', 'PROVINCIA', 'CANTON', 'ZONA', 'CLASE', 'CAUSA']
X = df[feature_cols].copy()
y = df['SINIESTRO_LETAL']

print(f"Features seleccionadas: {feature_cols}")
print(f"Forma de X: {X.shape}")

# Identificar columnas categóricas y numéricas
categorical_features = ['MES', 'DIA', 'PROVINCIA', 'CANTON', 'ZONA', 'CLASE', 'CAUSA']
numerical_features = []  # Inicialmente vacío

# Procesar la columna HORA para extraer la hora numérica
def extract_hour(hora_str):
    try:
        # Extraer el primer número de la cadena "HH:00 A HH:59"
        return int(hora_str.split(':')[0])
    except:
        return 12  # valor por defecto

X['HORA_NUM'] = X['HORA'].apply(extract_hour)
X = X.drop('HORA', axis=1)

# Actualizar listas de features
numerical_features = ['HORA_NUM']

print(f"Features categóricas: {categorical_features}")
print(f"Features numéricas: {numerical_features}")

# Dividir datos
print("\n=== DIVIDIENDO DATOS ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# Crear pipeline de preprocesamiento
print("\n=== CREANDO PIPELINE ===")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Pipeline completo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Entrenar modelo
print("Entrenando modelo LogisticRegression...")
pipeline.fit(X_train, y_train)

# Predicciones
print("\n=== REALIZANDO PREDICCIONES ===")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluación del modelo
print("\n=== EVALUACIÓN DEL MODELO ===")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Letal', 'Letal']))

# Visualizaciones
print("\n=== CREANDO VISUALIZACIONES ===")

# 1. Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Letal', 'Letal'], 
            yticklabels=['No Letal', 'Letal'])
plt.title('Matriz de Confusión - Siniestros Letales', fontsize=14, fontweight='bold')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Curva ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC - Clasificación de Siniestros Letales', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('curva_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Distribución de probabilidades predichas
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba[y_test==0], bins=40, alpha=0.7, label='No Letal', color='skyblue', density=True)
plt.hist(y_pred_proba[y_test==1], bins=40, alpha=0.7, label='Letal', color='salmon', density=True)
plt.xlabel('Probabilidad Predicha de Siniestro Letal')
plt.ylabel('Densidad')
plt.title('Distribución de Probabilidades Predichas por Clase', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('distribucion_probabilidades.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Métricas de evaluación
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
plt.title('Métricas de Evaluación del Modelo', fontsize=14, fontweight='bold')
plt.ylabel('Valor de la Métrica')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('metricas_evaluacion.png', dpi=300, bbox_inches='tight')
plt.show()

# Análisis adicional
print("\n=== ANÁLISIS ADICIONAL ===")

# Importancia de features (coeficientes del modelo logístico)
feature_names = (numerical_features + 
                list(pipeline.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(categorical_features)))

coefficients = pipeline.named_steps['classifier'].coef_[0]

# Crear DataFrame con importancias
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

print("Top 10 features más importantes (por valor absoluto del coeficiente):")
print(feature_importance.head(10))

# Visualizar top features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = ['red' if coef < 0 else 'green' for coef in top_features['coefficient']]
plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coeficiente')
plt.title('Top 15 Features más Importantes (Coeficientes del Modelo Logístico)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== DISCUSIÓN DE RESULTADOS ===")
print(f"""
RESUMEN DEL MODELO:
- Dataset: {df.shape[0]} siniestros de tránsito del 2019
- Variable objetivo: Siniestro letal (NUM_FALLECIDO > 0)
- Prevalencia de siniestros letales: {df['SINIESTRO_LETAL'].mean()*100:.2f}%
- Modelo: Regresión Logística con pipeline de preprocesamiento

MÉTRICAS DE EVALUACIÓN:
- Accuracy: {accuracy:.4f} - El modelo clasifica correctamente el {accuracy*100:.1f}% de los casos
- Precision: {precision:.4f} - De los casos predichos como letales, {precision*100:.1f}% son realmente letales
- Recall: {recall:.4f} - El modelo detecta el {recall*100:.1f}% de los siniestros letales reales
- F1-Score: {f1:.4f} - Balance entre precision y recall
- ROC-AUC: {roc_auc:.4f} - Capacidad de discriminación del modelo

INTERPRETACIÓN:
""")

if accuracy > 0.8:
    print("✓ ACCURACY: Excelente capacidad de clasificación general")
elif accuracy > 0.7:
    print("✓ ACCURACY: Buena capacidad de clasificación general")
else:
    print("⚠ ACCURACY: La capacidad de clasificación podría mejorar")

if precision > 0.7:
    print("✓ PRECISION: Baja tasa de falsos positivos")
else:
    print("⚠ PRECISION: Alta tasa de falsos positivos - muchos casos no letales predichos como letales")

if recall > 0.7:
    print("✓ RECALL: Buena detección de siniestros letales")
else:
    print("⚠ RECALL: Baja detección de siniestros letales - muchos casos letales no detectados")

if roc_auc > 0.8:
    print("✓ ROC-AUC: Excelente capacidad de discriminación")
elif roc_auc > 0.7:
    print("✓ ROC-AUC: Buena capacidad de discriminación")
else:
    print("⚠ ROC-AUC: Capacidad de discriminación limitada")

print(f"""
RECOMENDACIONES:
1. El modelo muestra un desempeño {'bueno' if accuracy > 0.7 else 'que requiere mejora'}
2. {'Considerar técnicas de balanceo de clases si la precisión es baja' if precision < 0.7 else 'La precisión es aceptable'}
3. {'Evaluar features adicionales o ingeniería de features para mejorar recall' if recall < 0.7 else 'El recall es satisfactorio'}
4. Las features más importantes incluyen información sobre la causa, clase del siniestro y ubicación
5. Considerar validación cruzada para una evaluación más robusta

¡Modelo entrenado y evaluado exitosamente!
""")
