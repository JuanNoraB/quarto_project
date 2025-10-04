# Resumen de Modelos de Machine Learning - Aprendizaje Supervisado

## Descripción General

Este documento presenta los resultados de dos problemas independientes de aprendizaje supervisado utilizando datos abiertos de Ecuador: un modelo de **regresión** para predecir nivel socioeconómico y un modelo de **clasificación binaria** para predecir siniestros de tránsito letales.

---

## 🏠 MODELO DE REGRESIÓN - Predicción de Decil Socioeconómico

### Problema Planteado
Predecir el decil socioeconómico (1-10) de hogares ecuatorianos basado en características de vivienda y acceso a servicios básicos.

### Dataset
- **Fuente:** Datos Abiertos Ecuador - Encuesta de Hogares 2018
- **Tamaño:** 2,565,433 registros de hogares
- **Variables:** 17 features (materiales de vivienda, servicios básicos, ubicación geográfica)
- **Variable objetivo:** Decil socioeconómico (1 = más pobre, 10 = más rico)

### Metodología
- **Modelo:** LinearRegression con Pipeline
- **Preprocesamiento:** StandardScaler + LabelEncoder
- **División:** 80% entrenamiento, 20% prueba (estratificada)
- **Features principales:** Tipo de vivienda, materiales (techo/piso/paredes), servicios (agua, electricidad, internet, TV cable)

### Resultados Clave
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R²** | 0.38 | Explica 38% de la varianza (capacidad moderada) |
| **MSE** | 3.46 | Error cuadrático medio aceptable |
| **RMSE** | 1.86 | Error promedio de ~2 deciles |

### Hallazgos Principales
1. **Total de personas por hogar** (coef: -1.96) → Hogares más grandes tienden a menor decil
2. **Total de hogares por área** (coef: +1.86) → Más hogares concentrados = mayor decil
3. **Material del piso** (coef: -0.50) → Mejores materiales = mayor decil
4. **Acceso a TV cable/satelital** (coef: -0.39) → Acceso mejora el decil
5. **Ubicación urbana/rural** (coef: -0.29) → Área urbana = mayor decil

### Interpretación y Aplicaciones
- El modelo identifica **patrones claros** entre infraestructura de vivienda y nivel socioeconómico
- **Servicios básicos** (internet, TV, teléfono) son predictores importantes del bienestar
- **Área geográfica** influye significativamente en el nivel socioeconómico
- **Aplicaciones:** Identificación de hogares vulnerables, planificación de programas sociales, inversión en infraestructura

---

## 🚗 MODELO DE CLASIFICACIÓN BINARIA - Predicción de Siniestros Letales

### Problema Planteado
Predecir si un siniestro de tránsito será letal (con fallecidos) o no letal basado en características del accidente.

### Dataset
- **Fuente:** Datos Abiertos Ecuador - INEC Siniestros de Tránsito 2019
- **Tamaño:** 24,595 siniestros de tránsito
- **Variable objetivo:** Binaria (Letal vs No Letal)
- **Distribución:** 8.01% letales, 91.99% no letales (clases desbalanceadas)

### Metodología
- **Modelo:** LogisticRegression con Pipeline
- **Preprocesamiento:** OneHotEncoder + StandardScaler
- **División:** 80% entrenamiento, 20% prueba (estratificada)
- **Features principales:** Ubicación (provincia, cantón), tipo de siniestro, causa, hora, zona urbana/rural

### Resultados Clave
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | 92.4% | Clasifica correctamente 9 de cada 10 casos |
| **Precision** | 66.1% | De los predichos como letales, 66% son realmente letales |
| **Recall** | 9.9% | Solo detecta 1 de cada 10 siniestros letales reales |
| **F1-Score** | 0.17 | Balance bajo entre precision y recall |
| **ROC-AUC** | 0.81 | Excelente capacidad de discriminación |

### Hallazgos Principales
1. **Cantón El Carmen** → Mayor riesgo de siniestros letales
2. **Provincia Cotopaxi** → Zona de alto riesgo
3. **Clase "Caída de Pasajeros"** → Menor probabilidad de ser letal (coef negativo)
4. **Cantón Puerto Quito** → Alto riesgo de letalidad
5. **Zona urbana** → Menor riesgo que zonas rurales

### Interpretación y Aplicaciones
- El modelo es **conservador**: prefiere no clasificar como letal (bajo recall)
- **Excelente accuracy general** pero **baja detección de casos letales**
- **Factores geográficos** son los más predictivos de letalidad
- **Problema de clases desbalanceadas** afecta el rendimiento
- **Aplicaciones:** Prevención de accidentes, asignación de recursos de emergencia, políticas de seguridad vial

---

## 📊 Comparación de Modelos

| Aspecto | Regresión | Clasificación |
|---------|-----------|---------------|
| **Tipo de problema** | Predicción continua | Predicción binaria |
| **Rendimiento** | Moderado (R²=0.38) | Alto (Acc=92.4%) |
| **Principal desafío** | Varianza no explicada | Clases desbalanceadas |
| **Factor más importante** | Total personas/hogares | Ubicación geográfica |
| **Aplicación práctica** | Políticas sociales | Seguridad vial |

## 🛠️ Metodología Técnica

### Pipeline Utilizado
```python
# Regresión
Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Clasificación
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])),
    ('classifier', LogisticRegression())
])
```

### Visualizaciones Generadas
- **Regresión:** Valores reales vs predichos, curva de aprendizaje, distribución de errores
- **Clasificación:** Matriz de confusión, curva ROC, distribución de probabilidades, métricas de evaluación

## 🎯 Conclusiones Generales

1. **Ambos modelos** demuestran la aplicabilidad del aprendizaje supervisado a problemas reales ecuatorianos
2. **Factores geográficos** emergen como predictores importantes en ambos casos
3. **Pipeline de Scikit-learn** facilita el preprocesamiento y entrenamiento sistemático
4. **Datos gubernamentales abiertos** proporcionan oportunidades valiosas para análisis predictivo
5. **Consideraciones éticas** son importantes, especialmente en aplicaciones de política pública

---

## 📁 Archivos del Proyecto

### Regresión
- `regresion_hogares.py` - Código principal del modelo
- `decil_reales_vs_predichos.png` - Comparación de predicciones
- `curva_aprendizaje_decil.png` - Curva de aprendizaje
- `importancia_factores_decil.png` - Importancia de features

### Clasificación
- `clasificacion_binaria.py` - Código principal del modelo
- `matriz_confusion.png` - Matriz de confusión
- `curva_roc.png` - Curva ROC
- `distribucion_probabilidades.png` - Distribución de probabilidades
- `metricas_evaluacion.png` - Métricas de evaluación

---

*Proyecto desarrollado como parte del curso de Machine Learning - Aplicación de técnicas de aprendizaje supervisado con datos abiertos de Ecuador.*
