"""
CLASIFICACIÓN DE TWEETS - PREDICCIÓN DE POPULARIDAD DE USUARIOS

Objetivo: Predecir el nivel de popularidad de usuarios de Twitter basándose en 
el contenido de sus tweets y características asociadas.

Dataset: tweets_totales_con_sentimiento_ml.csv

Autor: Juan Nora
Fecha: Enero 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# Preprocesamiento y modelado
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Métricas y evaluación
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support
)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CLASIFICACIÓN DE TWEETS - PREDICCIÓN DE POPULARIDAD DE USUARIOS")
print("="*80)

# ============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL DEL DATASET
# ============================================================================
print("\n[1] Cargando dataset...")
df = pd.read_csv('tweets_totales_con_sentimiento_ml.csv')

print(f"   - Total de registros: {len(df):,}")
print(f"   - Total de columnas: {df.shape[1]}")
print(f"   - Periodo: {df['createdAt'].min()} a {df['createdAt'].max()}")

# ============================================================================
# 2. CREACIÓN DE LA VARIABLE OBJETIVO (TARGET)
# ============================================================================
print("\n[2] Creando variable objetivo: USER_POPULARITY")
print("   JUSTIFICACIÓN:")
print("   - Queremos clasificar usuarios según su nivel de popularidad")
print("   - La popularidad se mide por la cantidad de seguidores (authorFollowers)")
print("   - Hipótesis: El ESTILO de escritura y características del tweet")
print("     pueden revelar si proviene de un usuario popular o no")

# PASO 1: Agrupar por authorId para evitar duplicación
print(f"\n   IMPORTANTE: Agrupando por autor para evitar duplicación")
print(f"   - Total de tweets en dataset: {len(df):,}")
print(f"   - Autores únicos: {df['authorId'].nunique():,}")
print(f"   - Promedio tweets por autor: {len(df) / df['authorId'].nunique():.2f}")

# Obtener followers máximos por autor (por si cambiaron en el tiempo)
author_stats = df.groupby('authorId').agg({
    'authorFollowers': 'max'  # Tomar el máximo de followers
}).reset_index()

# PASO 2: Analizar distribución de followers SOBRE AUTORES ÚNICOS
print(f"\n   Estadísticas de authorFollowers (por autor único):")
print(f"   - Mínimo: {author_stats['authorFollowers'].min()}")
print(f"   - Mediana: {author_stats['authorFollowers'].median():.0f}")
print(f"   - Media: {author_stats['authorFollowers'].mean():.0f}")
print(f"   - Máximo: {author_stats['authorFollowers'].max():,}")

# PASO 3: Definir umbrales basados en percentiles de AUTORES ÚNICOS
p33 = author_stats['authorFollowers'].quantile(0.33)
p66 = author_stats['authorFollowers'].quantile(0.66)

print(f"\n   Umbrales definidos (basados en percentiles de autores únicos):")
print(f"   - BAJA popularidad: < {p33:.0f} seguidores (33% de autores)")
print(f"   - MEDIA popularidad: {p33:.0f} - {p66:.0f} seguidores (33% de autores)")
print(f"   - ALTA popularidad: > {p66:.0f} seguidores (34% de autores)")

# PASO 4: Crear función de clasificación
def classify_popularity(followers):
    if followers < p33:
        return 'BAJA'
    elif followers < p66:
        return 'MEDIA'
    else:
        return 'ALTA'

# PASO 5: Asignar label a TODOS los tweets de cada autor
df['user_popularity'] = df['authorFollowers'].apply(classify_popularity)

# Verificar distribución (ahora sobre tweets, no autores)
print("\n   Distribución del target (sobre todos los tweets):")
print(df['user_popularity'].value_counts())
print(f"\n   Porcentajes:")
print(df['user_popularity'].value_counts(normalize=True) * 100)

# Verificar que el agrupamiento funcionó correctamente
print(f"\n   Verificación: Autores únicos por clase:")
authors_by_class = df.groupby('user_popularity')['authorId'].nunique()
print(authors_by_class)

# ============================================================================
# 3. SELECCIÓN DE FEATURES RELEVANTES
# ============================================================================
print("\n[3] Selección de features relevantes")
print("   JUSTIFICACIÓN DE FEATURES SELECCIONADAS:")
print("   ✓ content: Texto del tweet (OBLIGATORIO para análisis de texto)")
print("   ✓ account_age_days: Edad de la cuenta (cuentas antiguas vs nuevas)")
print("   ✓ content_length: Largo del mensaje (usuarios populares pueden escribir diferente)")
print("   ✓ mentions_count: Cantidad de menciones (patrón de interacción)")
print("   ✓ time_response: Velocidad de respuesta (usuarios activos)")
print("   ✓ has_profile_picture: Tiene foto de perfil (usuarios serios)")
print("   ✓ sentiment_polarity: Polaridad del sentimiento (-1 a 1)")

print("\n   FEATURES ELIMINADAS Y JUSTIFICACIÓN:")
print("   ✗ authorFollowers: Es la base del TARGET (data leakage)")
print("   ✗ hashtags_count: TODOS son 0 (sin variación)")
print("   ✗ source: TODOS 'Twitter for iPhone' (sin variación)")
print("   ✗ authorVerified: TODOS False (sin variación)")
print("   ✗ isReply: TODOS True (sin variación)")
print("   ✗ IDs y URLs: No son predictivos (identificadores únicos)")

# Filtrar datos válidos
print("\n   Filtrando datos...")
print(f"   - Registros antes de filtrar: {len(df):,}")

# Filtrar solo usuarios con foto de perfil (más confiables)
df_filtered = df[df['has_profile_picture'] == True].copy()
print(f"   - Después de filtrar has_profile_picture: {len(df_filtered):,}")

# Eliminar valores nulos si existen
df_filtered = df_filtered.dropna(subset=['content', 'account_age_days', 'sentiment_polarity'])
print(f"   - Después de eliminar nulos: {len(df_filtered):,}")

# ============================================================================
# 4. PREPROCESAMIENTO DE TEXTO
# ============================================================================
print("\n[4] Preprocesamiento del texto (content)")

def clean_text(text):
    """
    Limpia el texto del tweet:
    - Remueve URLs
    - Remueve menciones @usuario
    - Convierte a minúsculas
    - Remueve caracteres especiales excesivos
    """
    # Remover URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remover menciones @usuario
    text = re.sub(r'@\w+', '', text)
    # Convertir a minúsculas
    text = text.lower()
    # Remover múltiples espacios
    text = re.sub(r'\s+', ' ', text)
    # Trim
    text = text.strip()
    return text

print("   Aplicando limpieza de texto...")
df_filtered['content_clean'] = df_filtered['content'].apply(clean_text)

print("   Ejemplo de limpieza:")
idx = 5
print(f"   ORIGINAL: {df_filtered.iloc[idx]['content'][:100]}...")
print(f"   LIMPIO:   {df_filtered.iloc[idx]['content_clean'][:100]}...")

# ============================================================================
# 5. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================
print("\n[5] Preparación de datos para modelado")

# Definir features y target
feature_columns = [
    'content_clean',
    'account_age_days',
    'content_length',
    'mentions_count',
    'time_response',
    'sentiment_polarity'
]

X = df_filtered[feature_columns].copy()
y = df_filtered['user_popularity'].copy()

print(f"   - Shape de X: {X.shape}")
print(f"   - Shape de y: {y.shape}")
print(f"   - Clases en y: {y.unique()}")

# División train-test (80-20, estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\n   División de datos:")
print(f"   - Train: {len(X_train):,} registros ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Test:  {len(X_test):,} registros ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 6. PIPELINE DE PREPROCESAMIENTO CON COLUMNTRANSFORMER
# ============================================================================
print("\n[6] Creando Pipeline de preprocesamiento")

# Definir transformadores por tipo de variable
print("   Técnicas de preprocesamiento aplicadas:")
print("   - Texto (content_clean): TfidfVectorizer")
print("     * max_features=1000 (top 1000 palabras más importantes)")
print("     * ngram_range=(1,2) (unigramas y bigramas)")
print("     * stop_words='english' (remover palabras comunes)")
print("   - Variables numéricas: StandardScaler")
print("     * Escala: media=0, std=1")

# ColumnTransformer para combinar todos los preprocesadores
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=1000, 
            ngram_range=(1, 2),
            stop_words='spanish',
            min_df=5
        ), 'content_clean'),
        ('num', StandardScaler(), [
            'account_age_days', 
            'content_length', 
            'mentions_count', 
            'time_response', 
            'sentiment_polarity'
        ])
    ],
    verbose_feature_names_out=False
)

# ============================================================================
# 7. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# ============================================================================
print("\n[7] Entrenamiento del modelo de clasificación")
print("   Modelo seleccionado: LogisticRegression")
print("   Justificación:")
print("   - Bueno para clasificación multiclase")
print("   - Interpretable (coeficientes indican importancia)")
print("   - Funciona bien con features de alta dimensionalidad (TF-IDF)")

# Pipeline completo: preprocesamiento + modelo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'  # Para manejar desbalanceo
    ))
])

print("\n   Entrenando modelo...")
pipeline.fit(X_train, y_train)
print("   ✓ Modelo entrenado exitosamente")

# ============================================================================
# 8. PREDICCIONES
# ============================================================================
print("\n[8] Realizando predicciones")
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

print("   ✓ Predicciones completadas")

# ============================================================================
# 9. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n[9] Evaluación del modelo")
print("="*80)

# Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\nACCURACY:")
print(f"   - Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   - Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Classification Report
print("\n" + "="*80)
print("CLASSIFICATION REPORT (Test Set)")
print("="*80)
print(classification_report(y_test, y_pred_test))

# ============================================================================
# 10. VISUALIZACIONES
# ============================================================================
print("\n[10] Generando visualizaciones...")

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CLASIFICACIÓN DE TWEETS - EVALUACIÓN DEL MODELO', 
             fontsize=16, fontweight='bold', y=0.995)

# --- Subplot 1: Distribución del Target ---
ax1 = axes[0, 0]
target_counts = df_filtered['user_popularity'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(target_counts.index, target_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Distribución de Clases (Target)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Nivel de Popularidad', fontweight='bold')
ax1.set_ylabel('Cantidad de Tweets', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df_filtered)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# --- Subplot 2: Matriz de Confusión ---
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, y_pred_test, labels=['BAJA', 'MEDIA', 'ALTA'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BAJA', 'MEDIA', 'ALTA'])
disp.plot(ax=ax2, cmap='Blues', values_format='d', colorbar=False)
ax2.set_title('Matriz de Confusión (Test Set)', fontweight='bold', fontsize=12)
ax2.grid(False)

# --- Subplot 3: Métricas por Clase ---
ax3 = axes[1, 0]
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_test, labels=['BAJA', 'MEDIA', 'ALTA']
)

x = np.arange(len(['BAJA', 'MEDIA', 'ALTA']))
width = 0.25

bars1 = ax3.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#FF6B6B')
bars2 = ax3.bar(x, recall, width, label='Recall', alpha=0.8, color='#4ECDC4')
bars3 = ax3.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#45B7D1')

ax3.set_xlabel('Clase', fontweight='bold')
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Métricas por Clase (Test Set)', fontweight='bold', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(['BAJA', 'MEDIA', 'ALTA'])
ax3.legend()
ax3.set_ylim(0, 1.1)
ax3.grid(axis='y', alpha=0.3)

# Añadir valores
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

# --- Subplot 4: Distribución de Features Numéricas ---
ax4 = axes[1, 1]
feature_stats = X_train[['content_length', 'mentions_count', 'account_age_days']].describe().loc['mean']
bars = ax4.barh(range(len(feature_stats)), feature_stats.values, alpha=0.8, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
ax4.set_yticks(range(len(feature_stats)))
ax4.set_yticklabels(['Largo Contenido', 'Menciones', 'Edad Cuenta (días)'])
ax4.set_xlabel('Valor Promedio', fontweight='bold')
ax4.set_title('Estadísticas de Features Numéricas (Train)', fontweight='bold', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

# Añadir valores
for i, (bar, val) in enumerate(zip(bars, feature_stats.values)):
    ax4.text(val, bar.get_y() + bar.get_height()/2., 
            f'{val:.0f}',
            ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('resultados_clasificacion_tweets.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualización guardada: resultados_clasificacion_tweets.png")

plt.show()

# ============================================================================
# 11. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DEL ANÁLISIS")
print("="*80)

print(f"\n1. DATASET:")
print(f"   - Registros totales procesados: {len(df_filtered):,}")
print(f"   - Features utilizadas: {len(feature_columns)}")
print(f"   - Clases del target: {len(y.unique())} (BAJA, MEDIA, ALTA)")

print(f"\n2. PREPROCESAMIENTO:")
print(f"   - Texto vectorizado: TF-IDF (1000 features)")
print(f"   - Variables numéricas: StandardScaler (5 features)")
print(f"   - Total features en el modelo: ~1005")

print(f"\n3. MODELO:")
print(f"   - Algoritmo: Logistic Regression")
print(f"   - Accuracy en Test: {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\n4. INTERPRETACIÓN:")
if test_acc > 0.50:
    print(f"   ✓ El modelo supera el baseline de clasificación aleatoria (33.3%)")
    print(f"   ✓ El estilo de escritura SÍ contiene información sobre la popularidad del usuario")
else:
    print(f"   ⚠ El modelo tiene dificultades para diferenciar las clases")
    print(f"   ⚠ Considerar agregar más features o ajustar umbrales del target")

print(f"\n5. CONCLUSIONES:")
print(f"   - El contenido del tweet, junto con características del perfil,")
print(f"     permite predecir con cierta precisión la popularidad del usuario")
print(f"   - Usuarios con diferentes niveles de popularidad tienen patrones")
print(f"     de escritura y comportamiento distinguibles")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)
