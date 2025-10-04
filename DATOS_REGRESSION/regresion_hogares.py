import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=== MODELO DE REGRESIÓN - DATOS DE HOGARES ECUADOR ===")

# Cargar datos
print("Cargando datos...")
df = pd.read_csv('hogares_rs18.csv')
print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# Mapeo de variables basado en el diccionario
variable_mapping = {
    's1_id03': 'Area_urbano_rural',
    's3_vi01': 'Tipo_vivienda', 
    's3_vi02': 'Via_acceso_principal_vivienda',
    's3_vi03': 'Material_techo',
    's3_vi04': 'Material_piso', 
    's3_vi05': 'Estado_paredes',
    's4_ho01': 'Vivienda_con_cocina',
    's4_ho06': 'Agua_beber_hogar',
    's4_ho08': 'Servicio_higienico_hogar',
    's4_ho12': 'Eliminacion_basura_hogar',
    's4_ho16': 'Hogar_cocina_principalmente_con',
    's4_ho17': 'Alumbrado_principalmente_con',
    's4_ho19': 'Hogar_acceso_internet',
    's4_ho21': 'Hogar_acceso_telefono_convencional',
    's4_ho22': 'Hogar_acceso_television_cable_satelital',
    'decil': 'Decil_socioeconomico',
    'tipo_pob_rs18': 'Tipo_pobreza_RS18',
    'tot_hogares': 'Total_hogares',
    'tot_nucleos': 'Total_nucleos_familiares', 
    'tot_personas': 'Total_personas'
}

print(f"\nVariables disponibles y su significado:")
for code, meaning in variable_mapping.items():
    if code in df.columns:
        print(f"  {code} -> {meaning}")

# Explorar variables objetivo potenciales
print(f"\n=== ANÁLISIS DE VARIABLES OBJETIVO POTENCIALES ===")

# Opción 1: Total de personas
print("1. TOTAL DE PERSONAS:")
print(f"   Rango: {df['tot_personas'].min()} - {df['tot_personas'].max()}")
print(f"   Media: {df['tot_personas'].mean():.2f}")
print(f"   Desv. Std: {df['tot_personas'].std():.2f}")

# Opción 2: Total de hogares  
print("2. TOTAL DE HOGARES:")
print(f"   Rango: {df['tot_hogares'].min()} - {df['tot_hogares'].max()}")
print(f"   Media: {df['tot_hogares'].mean():.2f}")
print(f"   Desv. Std: {df['tot_hogares'].std():.2f}")

# Opción 3: Decil socioeconómico
print("3. DECIL SOCIOECONÓMICO:")
print(f"   Rango: {df['decil'].min()} - {df['decil'].max()}")
print(f"   Media: {df['decil'].mean():.2f}")
print(f"   Distribución:")
print(df['decil'].value_counts().sort_index())

# DECISIÓN: Predecir DECIL SOCIOECONÓMICO basado en características de vivienda y servicios
print(f"\n=== PROBLEMA DE REGRESIÓN DEFINIDO ===")
print("OBJETIVO: Predecir DECIL SOCIOECONÓMICO basado en características de vivienda y servicios")
print("JUSTIFICACIÓN: Es una variable continua ordenada (1-10) que refleja nivel socioeconómico")
print("UTILIDAD: Permite identificar factores que determinan el nivel socioeconómico")

# Seleccionar features predictoras
feature_cols = [
    's1_id03',  # Área urbano/rural
    's3_vi01',  # Tipo de vivienda
    's3_vi02',  # Vía de acceso principal
    's3_vi03',  # Material del techo
    's3_vi04',  # Material del piso
    's3_vi05',  # Estado de paredes
    's4_ho01',  # Vivienda con cocina
    's4_ho06',  # Agua para beber
    's4_ho08',  # Servicio higiénico
    's4_ho12',  # Eliminación de basura
    's4_ho16',  # Cocina principalmente con
    's4_ho17',  # Alumbrado principalmente con
    's4_ho19',  # Acceso a internet
    's4_ho21',  # Teléfono convencional
    's4_ho22',  # TV cable/satelital
    'tot_hogares',  # Total hogares (como control)
    'tot_personas'  # Total personas (como control)
]

target_col = 'decil'

print(f"\nFeatures seleccionadas: {len(feature_cols)} variables")
print(f"Variable objetivo: {target_col} (Decil socioeconómico)")

# Preparar datos
X = df[feature_cols].copy()
y = df[target_col]

print(f"\nForma inicial - X: {X.shape}, y: {y.shape}")

# Verificar y limpiar datos
print(f"\nVerificando datos:")
print(f"Valores faltantes en X: {X.isnull().sum().sum()}")
print(f"Valores faltantes en y: {y.isnull().sum()}")

# Eliminar filas con valores faltantes
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"Después de limpiar - X: {X.shape}, y: {y.shape}")

# Identificar variables categóricas vs numéricas
categorical_features = [col for col in feature_cols if col.startswith('s')]
numerical_features = ['tot_hogares', 'tot_personas']

print(f"\nFeatures categóricas: {len(categorical_features)}")
print(f"Features numéricas: {len(numerical_features)}")

# Codificar variables categóricas
le_dict = {}
for col in categorical_features:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# Dividir datos
print(f"\n=== DIVIDIENDO DATOS ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# Crear pipeline
print(f"\n=== ENTRENANDO MODELO ===")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)

# Predicciones
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Evaluación
print(f"\n=== EVALUACIÓN DEL MODELO ===")
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"MSE Entrenamiento: {mse_train:.4f}")
print(f"MSE Prueba: {mse_test:.4f}")
print(f"RMSE Prueba: {np.sqrt(mse_test):.4f}")
print(f"R² Entrenamiento: {r2_train:.4f}")
print(f"R² Prueba: {r2_test:.4f}")

# Visualizaciones
print(f"\n=== CREANDO VISUALIZACIONES ===")

# 1. Valores reales vs predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea perfecta')
plt.xlabel('Decil Real')
plt.ylabel('Decil Predicho')
plt.title('Predicción de Decil Socioeconómico: Valores Reales vs Predichos', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('decil_reales_vs_predichos.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Curva de aprendizaje
plt.figure(figsize=(10, 6))
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validación')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('R² Score')
plt.title('Curva de Aprendizaje - Predicción de Decil Socioeconómico', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('curva_aprendizaje_decil.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Distribución de errores
plt.figure(figsize=(10, 6))
errores = y_test - y_pred_test
plt.hist(errores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Error de Predicción (Decil Real - Decil Predicho)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores de Predicción', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', label='Error = 0')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('distribucion_errores_decil.png', dpi=300, bbox_inches='tight')
plt.show()

# Importancia de features
print(f"\n=== ANÁLISIS DE IMPORTANCIA ===")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': pipeline.named_steps['regressor'].coef_,
    'abs_coefficient': np.abs(pipeline.named_steps['regressor'].coef_),
    'description': [variable_mapping.get(col, col) for col in feature_cols]
}).sort_values('abs_coefficient', ascending=False)

print("Top 10 factores más importantes para determinar el decil socioeconómico:")
for idx, row in feature_importance.head(10).iterrows():
    direction = "↑" if row['coefficient'] > 0 else "↓"
    print(f"{direction} {row['description']} (coef: {row['coefficient']:.4f})")

# Visualizar importancia
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(12)
colors = ['green' if coef > 0 else 'red' for coef in top_features['coefficient']]
plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), top_features['description'])
plt.xlabel('Coeficiente de Regresión')
plt.title('Factores más Importantes para Determinar el Decil Socioeconómico', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('importancia_factores_decil.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== DISCUSIÓN DE RESULTADOS ===")
print(f"""
MODELO DE REGRESIÓN - DECIL SOCIOECONÓMICO:
- Dataset: {df.shape[0]:,} registros de hogares ecuatorianos
- Variable objetivo: Decil socioeconómico (1-10)
- Features: Características de vivienda, servicios y ubicación
- Modelo: Regresión Lineal con normalización

MÉTRICAS DE EVALUACIÓN:
- MSE: {mse_test:.4f} - Error cuadrático medio
- RMSE: {np.sqrt(mse_test):.4f} - Error promedio en deciles
- R²: {r2_test:.4f} - El modelo explica el {r2_test*100:.1f}% de la varianza

INTERPRETACIÓN:
""")

if r2_test > 0.7:
    print("✓ R²: Excelente capacidad predictiva del nivel socioeconómico")
elif r2_test > 0.5:
    print("✓ R²: Buena capacidad predictiva del nivel socioeconómico")
elif r2_test > 0.3:
    print("⚠ R²: Capacidad predictiva moderada")
else:
    print("⚠ R²: Capacidad predictiva limitada")

rmse = np.sqrt(mse_test)
if rmse < 1:
    print("✓ RMSE: Error bajo - predicciones muy precisas")
elif rmse < 2:
    print("✓ RMSE: Error moderado - predicciones aceptables")
else:
    print("⚠ RMSE: Error alto - revisar modelo")

print(f"""
FACTORES CLAVE IDENTIFICADOS:
Los factores más importantes para determinar el nivel socioeconómico incluyen:
- Características de la vivienda (materiales, servicios)
- Acceso a servicios básicos (agua, electricidad, internet)
- Ubicación geográfica (urbano/rural)

APLICACIONES PRÁCTICAS:
- Identificación de hogares vulnerables para programas sociales
- Planificación de inversión en infraestructura
- Evaluación de políticas de desarrollo social

¡Modelo de regresión completado exitosamente!
""")
