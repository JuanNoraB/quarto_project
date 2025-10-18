# %% [markdown]
# # ANÁLISIS EXPLORATORIO DE DATOS (EDA) - TWEETS CON TOXICIDAD

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import skew, kurtosis
from scipy import stats
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS - TWEETS CON TOXICIDAD")
print("="*80)

df = pd.read_csv('/home/juanchx/Documents/Maestria_IA/machine learning/final_examen_data/1500_tweets_con_toxicity.csv')
print(f"\n✓ Dataset cargado: {df.shape[0]:,} registros x {df.shape[1]} columnas")

# %%
# BLOQUE 1: TIPOS DE DATOS Y CLASIFICACIÓN
print("\n" + "="*80)
print("TIPOS DE DATOS Y CLASIFICACIÓN DE VARIABLES")
print("="*80)

print('Exploracion inicial tipos de datos y valores nulos')
print(df.info())

print('Exploracion inicial valores nulos')
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

print(f"\nVARIABLES NUMÉRICAS ({len(numeric_cols)}): {numeric_cols}")
print(f"VARIABLES CATEGÓRICAS ({len(categorical_cols)}): {categorical_cols[:5]}... (truncado)")
print(f"VARIABLES BOOLEANAS ({len(boolean_cols)}): {boolean_cols}")

# %%
# BLOQUE 2: IDENTIFICACIÓN DE COLUMNAS CON POCA INFORMACIÓN
print("\n" + "="*80)
print("ANÁLISIS DE COLUMNAS CON POCA INFORMACIÓN")
print("="*80)

for col in df.columns:
    unique_ratio = df[col].nunique() / len(df)
    if unique_ratio > 0.98:
        print(f"⚠️  {col}: {df[col].nunique():,} únicos ({unique_ratio*100:.1f}%) - ALTA CARDINALIDAD")

for col in df.columns:
    null_ratio = df[col].isnull().sum() / len(df)
    if null_ratio > 0.8:
        print(f"⚠️  {col}: {null_ratio*100:.1f}% nulos - ALTA PROPORCIÓN DE NULOS")

#analisis por nombre de variables
print("\n" + "="*80)
print("ANÁLISIS POR NOMBRE DE VARIABLES")
print("="*80)
print(df.columns)
columnas_identificadores_unicos = [
    'tweetId', 'tweetUrl',
    'authorId', 'authorName', 'authorUsername'
]
#analisis de columnas identificadores unicos
print("\n" + "="*80)
print("ANÁLISIS DE COLUMNAS IDENTIFICADORES UNICOS")
print("="*80)
print(df[columnas_identificadores_unicos].nunique())

# %%
# BLOQUE 3: VALORES NULOS Y DUPLICADOS
print("\n" + "="*80)
print("ANÁLISIS DE VALORES NULOS Y DUPLICADOS")
print("="*80)

null_counts = df.isnull().sum()
null_percentages = (null_counts / len(df)) * 100
null_df = pd.DataFrame({'Columna': null_counts.index, 'Nulos': null_counts.values, 
                        'Porcentaje': null_percentages.values}).sort_values('Nulos', ascending=False)
print("\n", null_df[null_df['Nulos'] > 0].to_string(index=False))

duplicates = df.duplicated().sum()
print(f"\nDUPLICADOS: {duplicates}")

toxicity_nulls = df['toxicity_score'].isnull().sum()
print(f"\n⚠️  toxicity_score NULOS: {toxicity_nulls} ({(toxicity_nulls/len(df))*100:.1f}%)")

fig, ax = plt.subplots(figsize=(12, 6))
null_data = null_df[null_df['Nulos'] > 0]
ax.barh(null_data['Columna'], null_data['Porcentaje'], color='coral')
ax.set_xlabel('Porcentaje de Valores Nulos (%)')
ax.set_title('Valores Nulos por Columna', fontweight='bold')
plt.tight_layout()
plt.savefig('01_valores_nulos.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: 01_valores_nulos.png")
plt.close()

# %%
# BLOQUE 4: ANÁLISIS DE toxicity_score
print("\n" + "="*80)
print("ANÁLISIS DE toxicity_score")
print("="*80)

toxicity_data = df['toxicity_score'].dropna()
print("\nESTADÍSTICAS:\n", toxicity_data.describe())

print(f"\nAsimetría: {skew(toxicity_data):.4f}")
print(f"Curtosis: {kurtosis(toxicity_data):.4f}")

Q1, Q3 = toxicity_data.quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = toxicity_data[(toxicity_data < Q1 - 1.5*IQR) | (toxicity_data > Q3 + 1.5*IQR)]
print(f"\nOUTLIERS (IQR): {len(outliers)} ({(len(outliers)/len(toxicity_data))*100:.2f}%)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(toxicity_data, bins=50, edgecolor='black', alpha=0.7, density=True)
toxicity_data.plot(kind='kde', ax=axes[0, 0], color='red', linewidth=2)
axes[0, 0].axvline(toxicity_data.mean(), color='green', linestyle='--', label=f'Media: {toxicity_data.mean():.3f}')
axes[0, 0].axvline(toxicity_data.median(), color='orange', linestyle='--', label=f'Mediana: {toxicity_data.median():.3f}')
axes[0, 0].set_title('Distribución Toxicity Score', fontweight='bold')
axes[0, 0].legend()

axes[0, 1].boxplot(toxicity_data)
axes[0, 1].set_title('Boxplot - Outliers', fontweight='bold')

stats.probplot(toxicity_data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot', fontweight='bold')

axes[1, 1].violinplot(toxicity_data, showmeans=True, showmedians=True)
axes[1, 1].set_title('Violin Plot', fontweight='bold')

plt.tight_layout()
plt.savefig('02_toxicity_distribucion.png', dpi=300)
print("✓ Guardado: 02_toxicity_distribucion.png")
plt.close()

# %%
# BLOQUE 5: VARIABLES NUMÉRICAS
print("\n" + "="*80)
print("ANÁLISIS DE VARIABLES NUMÉRICAS")
print("="*80)

numeric_features = ['authorFollowers', 'time_response', 'account_age_days', 
                   'mentions_count', 'hashtags_count', 'content_length', 'sentiment_polarity']

print("\nESTADÍSTICAS:\n", df[numeric_features].describe().T)

correlations = df[numeric_features + ['toxicity_score']].corr()['toxicity_score'].drop('toxicity_score').sort_values(ascending=False)
print("\nCORRELACIÓN CON toxicity_score:\n", correlations)

fig, ax = plt.subplots(figsize=(10, 6))
correlations.plot(kind='barh', ax=ax, color=['green' if x > 0 else 'red' for x in correlations])
ax.set_xlabel('Correlación con toxicity_score')
ax.set_title('Correlaciones', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('03_correlaciones.png', dpi=300)
print("✓ Guardado: 03_correlaciones.png")
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for idx, col in enumerate(numeric_features):
    axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].axvline(df[col].mean(), color='red', linestyle='--', label='Media')
    axes[idx].legend(fontsize=8)
for idx in range(len(numeric_features), len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.savefig('04_histogramas_numericas.png', dpi=300)
print("✓ Guardado: 04_histogramas_numericas.png")
plt.close()

df_clean = df.dropna(subset=['toxicity_score'])
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, col in enumerate(numeric_features):
    axes[idx].scatter(df_clean[col], df_clean['toxicity_score'], alpha=0.5, s=20)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('toxicity_score')
    axes[idx].set_title(f'{col} vs Toxicity', fontweight='bold')
    # Solo hacer línea de tendencia si hay varianza
    if df_clean[col].std() > 0:
        try:
            z = np.polyfit(df_clean[col], df_clean['toxicity_score'], 1)
            p = np.poly1d(z)
            axes[idx].plot(df_clean[col], p(df_clean[col]), "r--", alpha=0.8)
        except:
            pass
for idx in range(len(numeric_features), len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.savefig('05_scatterplots_toxicity.png', dpi=300)
print("✓ Guardado: 05_scatterplots_toxicity.png")
plt.close()

# %%
# BLOQUE 6: VARIABLES CATEGÓRICAS
print("\n" + "="*80)
print("ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("="*80)

categorical_features = ['isReply', 'authorVerified', 'has_profile_picture']

for col in categorical_features:
    print(f"\n{col}:\n", df[col].value_counts())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_features):
    df[col].value_counts().plot(kind='bar', ax=axes[idx], color='lightgreen', edgecolor='black')
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('06_categoricas_distribucion.png', dpi=300)
print("✓ Guardado: 06_categoricas_distribucion.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(categorical_features):
    df_clean.boxplot(column='toxicity_score', by=col, ax=axes[idx])
    axes[idx].set_title(f'Toxicity por {col}', fontweight='bold')
plt.tight_layout()
plt.savefig('07_toxicity_por_categoria.png', dpi=300)
print("✓ Guardado: 07_toxicity_por_categoria.png")
plt.close()

print("\nTOXICITY PROMEDIO POR CATEGORÍA:")
for col in categorical_features:
    print(f"\n{col}:\n", df_clean.groupby(col)['toxicity_score'].agg(['mean', 'median', 'std', 'count']))

# %%
# BLOQUE 7: ANÁLISIS DE TEXTO
print("\n" + "="*80)
print("ANÁLISIS DE TEXTO")
print("="*80)

print("\nLONGITUD:\n", df['content_length'].describe())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

stopwords_es = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'por', 'con', 
                'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 
                'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese', 'si', 'me', 'ya', 'ver', 'porque', 
                'dar', 'cuando', 'muy', 'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'yo', 
                'también', 'hasta', 'dos', 'querer', 'así', 'desde', 'eso', 'ni', 'nos', 'llegar', 
                'tiempo', 'ella', 'sí', 'día', 'uno', 'bien', 'poco', 'entonces', 'tanto', 'donde', 
                'ahora', 'después', 'siempre', 'nada', 'cada', 'algo', 'solo', 'estos', 'momento', 
                'tal', 'cual', 'dentro', 'al', 'del', 'los', 'las', 'una', 'unos', 'unas', 'es', 'son'}

all_text = ' '.join(df['content'].apply(clean_text))
words = [w for w in all_text.split() if w and w not in stopwords_es and len(w) > 2]

print(f"\nTotal palabras (sin stopwords): {len(words)}")
print(f"Palabras únicas: {len(set(words))}")

word_freq = Counter(words)
top_words = word_freq.most_common(30)
print(f"\nTOP 30 PALABRAS:")
for word, count in top_words:
    print(f"  {word:20s}: {count:5d}")

fig, ax = plt.subplots(figsize=(12, 8))
words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
ax.barh(words_df['Palabra'], words_df['Frecuencia'], color='steelblue', edgecolor='black')
ax.set_xlabel('Frecuencia')
ax.set_title('Top 30 Palabras Más Frecuentes', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('08_top_palabras.png', dpi=300)
print("✓ Guardado: 08_top_palabras.png")
plt.close()

wordcloud = WordCloud(width=1600, height=800, background_color='white',
                     stopwords=stopwords_es, max_words=100, colormap='viridis').generate(' '.join(words))
fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Nube de Palabras', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('09_wordcloud.png', dpi=300)
print("✓ Guardado: 09_wordcloud.png")
plt.close()

# %%
# BLOQUE 8: HALLAZGOS CLAVE
print("\n" + "="*80)
print("HALLAZGOS CLAVE DEL EDA")
print("="*80)

print("\n🔍 HALLAZGO 1: Calidad de Datos")
print(f"   - toxicity_score tiene {toxicity_nulls} nulos (10.2%)")
print(f"   - hashtags tiene 92% de nulos (mayoría de tweets sin hashtags)")
print(f"   - Múltiples columnas ID con alta cardinalidad (>95% únicos)")
print(f"   → Decisión: Eliminar columnas ID, manejar nulos de toxicity en preprocesamiento")

print("\n🔍 HALLAZGO 2: Distribución de Toxicidad")
print(f"   - Media: {toxicity_data.mean():.3f}, Mediana: {toxicity_data.median():.3f}")
print(f"   - Asimetría positiva ({skew(toxicity_data):.3f}) → Más tweets con baja toxicidad")
print(f"   - Rango: {toxicity_data.min():.3f} a {toxicity_data.max():.3f}")
print(f"   → La mayoría de tweets tienen toxicidad baja-moderada")

print("\n🔍 HALLAZGO 3: Correlaciones con Toxicidad")
print(f"   - sentiment_polarity: {correlations['sentiment_polarity']:.3f} (correlación más fuerte)")
print(f"   - content_length: {correlations['content_length']:.3f}")
print(f"   - Variables demográficas (followers, account_age) tienen poca correlación")
print(f"   → El sentimiento y longitud del texto son predictores importantes")

print("\n" + "="*80)
print("EDA COMPLETADO - Todos los gráficos guardados")
print("="*80)
