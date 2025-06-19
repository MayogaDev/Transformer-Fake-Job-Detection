# Detección de Ofertas de Trabajo Fraudulentas

Este repositorio contiene un proyecto para la detección de ofertas de trabajo fraudulentas utilizando modelos de aprendizaje profundo. Se han implementado dos enfoques principales: un modelo basado en BERT (Bidirectional Encoder Representations from Transformers) y un modelo basado en una Red Convolucional (CNN). Además, se ha mejorado el sistema de visualización de datos para facilitar el análisis y comparación de los resultados de ambos modelos.

## Mejoras Realizadas

### Sistema de Visualización de Datos
Se ha mejorado significativamente el sistema de visualización de datos con el siguiente script: `visualize_results.py`. Las principales mejoras incluyen:
- Generación de gráficos de alta calidad con una resolución de 300 DPI y ajustes automáticos de espaciado (`bbox_inches='tight'`).
- Creación de un gráfico de líneas (`accuracy_comparison.png`) que muestra la evolución de la precisión de entrenamiento y validación por época para ambos modelos, promediando los resultados sobre los pliegues de validación cruzada (K-fold).
- Generación de un gráfico de barras (`test_metrics_comparison.png`) que compara las métricas de prueba (precisión, recall, F1-score y exactitud) entre BERT y CNN.
- Uso de una paleta de colores consistente (#1f77b4 para BERT y #ff7f0e para CNN) y cuadrículas para mejorar la legibilidad.

Estas mejoras permiten una comparación visual más clara y profesional de los resultados, facilitando la toma de decisiones sobre el modelo más adecuado para el problema.

## Arquitecturas de los Modelos

### Modelo BERT
- **Descripción**: El modelo utiliza una arquitectura Transformer preentrenada (`bert-base-uncased`) de la biblioteca `transformers` de Hugging Face. Esta arquitectura es bidireccional, lo que le permite capturar el contexto semántico de los textos en ambas direcciones.
- **Estructura**:
  - Capa base: `BertModel` preentrenado con 12 capas de Transformer, 768 dimensiones ocultas y 12 cabezas de atención.
  - Capa de dropout (p=0.3) para regularización.
  - Capa lineal de salida con 2 clases (fraudulento/no fraudulento).
- **Entrenamiento**:
  - Optimizador: Adam con un learning rate de 2e-5.
  - Épocas: 3.
  - Validación cruzada: K-fold con 5 pliegues.
  - Batch size: 16.
- **Ventajas**: Alta capacidad para capturar dependencias contextuales de largo alcance, ideal para textos complejos.

### Modelo CNN
- **Descripción**: El modelo utiliza una red convolucional personalizada diseñada para procesar secuencias de texto tokenizadas. Se basa en embeddings inicializados aleatoriamente (sin embeddings preentrenados como GloVe en esta implementación).
- **Estructura**:
  - Capa de embedding con un tamaño de vocabulario basado en el tokenizador BERT y una dimensión de 100.
  - Capas convolucionales con 100 filtros y tamaños de kernel [3, 4, 5] para capturar patrones locales.
  - Max pooling para reducir la dimensionalidad.
  - Capa fully connected de salida con 2 clases.
  - Dropout (p=0.5) para prevenir sobreajuste.
- **Entrenamiento**:
  - Optimizador: Adam con un learning rate de 1e-3.
  - Épocas: 3.
  - Validación cruzada: K-fold con 5 pliegues.
  - Batch size: 16.
- **Ventajas**: Menor consumo computacional y mayor rapidez en el entrenamiento, adecuada para entornos con recursos limitados.

## Comparación y Análisis de Visualizaciones

### Gráfico de Precisión de Entrenamiento y Validación (`accuracy_comparison.png`)
![accuracy_comparison](https://github.com/user-attachments/assets/d7474e39-5587-4c2e-bd24-7674d51533b2)

- **Descripción**: Este gráfico muestra la evolución de la precisión de entrenamiento y validación a lo largo de las 3 épocas para ambos modelos. Las curvas de BERT están representadas en azul (líneas continuas para entrenamiento, discontinuas para validación), mientras que las de CNN están en naranja.
- **Análisis**:
  - **BERT**: La precisión de entrenamiento comienza en aproximadamente 0.965 y alcanza casi 0.995 en la tercera época, mientras que la precisión de validación se mantiene entre 0.98 y 0.99. Esto indica una convergencia rápida y una generalización excelente, con poca diferencia entre entrenamiento y validación, sugiriendo un modelo bien regularizado.
  - **CNN**: La precisión de entrenamiento comienza en alrededor de 0.96 y llega a aproximadamente 0.98 en la tercera época, con la precisión de validación oscilando entre 0.975 y 0.985. Las curvas muestran una convergencia más lenta y una brecha ligeramente mayor entre entrenamiento y validación, lo que podría indicar un leve sobreajuste o menor capacidad para capturar patrones complejos.
  - **Comparación**: BERT supera consistentemente a la CNN en precisión, con una diferencia notable que se amplía con el tiempo (alrededor de 1-2%). Esto resalta la superioridad de BERT para capturar el contexto semántico en textos largos.

### Gráfico de Métricas de Prueba (`test_metrics_comparison.png`)
![test_metrics_comparison](https://github.com/user-attachments/assets/3b382e67-281d-4dd6-93ba-8da9a51b2800)

- **Descripción**: Este gráfico de barras compara las métricas de prueba (exactitud, precisión, recall y F1-score) entre BERT (azul) y CNN (naranja) en el conjunto de prueba.
- **Análisis**:
  - **BERT**: Muestra valores de aproximadamente 0.986 para exactitud, 0.986 para precisión, 0.987 para recall y 0.986 para F1-score. Estas métricas son extremadamente altas y equilibradas, indicando un rendimiento excepcional en la detección de ofertas fraudulentas y no fraudulentas.
  - **CNN**: Presenta valores de alrededor de 0.983 para exactitud, 0.983 para precisión, 0.983 para recall y 0.981 para F1-score. Aunque sólidas, estas métricas son ligeramente inferiores a las de BERT, con una diferencia de aproximadamente 0.3-0.5%.
  - **Comparación**: BERT tiene un rendimiento superior en todas las métricas, con una ventaja más pronunciada en F1-score (0.986 vs. 0.981), lo que sugiere que maneja mejor el equilibrio entre precisión y recall. La CNN, aunque competitiva, no alcanza el nivel de BERT, probablemente debido a su menor capacidad para modelar dependencias de largo alcance.

## Conclusión
- **Modelo Recomendado**: Basado en las visualizaciones, **BERT** es el modelo ideal para este proyecto debido a su mayor precisión y capacidad de generalización, especialmente en un problema donde la detección precisa de fraudes es crítica. Sin embargo, la **CNN** ofrece un rendimiento sólido y es una alternativa viable en entornos con recursos computacionales limitados o para prototipos rápidos.
- **Siguientes Pasos**: Se pueden explorar optimizaciones como el uso de embeddings preentrenados (e.g., GloVe) para la CNN, ajuste de hiperparámetros, o aumento del número de épocas para mejorar su rendimiento. Además, se recomienda analizar el balance de clases en el dataset para ajustar la función de pérdida si es necesario.

## Instrucciones de Uso
1. **Requisitos**: Asegúrate de tener instaladas las bibliotecas `torch`, `transformers`, `pandas`, `sklearn`, `tqdm`, `matplotlib` y `seaborn`.
   ```bash
   pip install torch transformers pandas scikit-learn tqdm matplotlib seaborn
