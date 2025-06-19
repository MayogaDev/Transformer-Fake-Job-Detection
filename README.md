# Detección de Ofertas de Trabajo Fraudulentas

Este repositorio alberga un proyecto desarrollado para la detección de ofertas de trabajo fraudulentas mediante modelos de aprendizaje profundo. Se han implementado dos enfoques principales: un modelo basado en BERT (Bidirectional Encoder Representations from Transformers) y un modelo basado en una Red Convolucional (CNN). El proyecto incluye mejoras significativas en su estructura y capacidades analíticas, permitiendo una evaluación comparativa detallada de ambos modelos.

## Mejoras Realizadas

El proyecto ha sido actualizado con dos mejoras clave. En primer lugar, se ha incorporado un sistema de visualizaciones para facilitar el análisis de los resultados, mediante la adición del script `visualize_results.py`, que genera representaciones gráficas de las métricas de rendimiento. En segundo lugar, se ha desarrollado una CNN diseñada como una alternativa competitiva frente al modelo BERT, optimizando su arquitectura para ofrecer un rendimiento comparable en la tarea de detección de fraudes, con un enfoque en la eficiencia computacional.

## Arquitecturas de los Modelos

### Modelo BERT
- **Descripción**: El modelo emplea una arquitectura Transformer preentrenada (`bert-base-uncased`) de la biblioteca `transformers` de Hugging Face. Su diseño bidireccional permite capturar el contexto semántico de los textos en ambas direcciones.
- **Estructura**:
  - Capa base: `BertModel` preentrenado con 12 capas de Transformer, 768 dimensiones ocultas y 12 cabezas de atención.
  - Capa de dropout (p=0.3) para regularización.
  - Capa lineal de salida con 2 clases (fraudulento/no fraudulento).
- **Entrenamiento**:
  - Optimizador: Adam con un learning rate de 2e-5.
  - Épocas: 3.
  - Validación cruzada: K-fold con 5 pliegues.
  - Batch size: 16.
- **Ventajas**: Excelente capacidad para modelar dependencias contextuales de largo alcance, lo que lo hace idóneo para textos complejos.

### Modelo CNN
- **Descripción**: El modelo utiliza una red convolucional personalizada, diseñada para procesar secuencias de texto tokenizadas. Se basa en embeddings inicializados aleatoriamente, con el objetivo de competir con BERT en términos de rendimiento.
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
- **Ventajas**: Menor demanda computacional y mayor rapidez en el entrenamiento, constituyéndose como una alternativa viable en entornos con recursos limitados.

## Comparación y Análisis de Resultados

### Gráfico de Precisión de Entrenamiento y Validación (`accuracy_comparison.png`)
![accuracy_comparison](https://github.com/user-attachments/assets/22fdcdf2-c13c-44b6-85ca-adda7d004d0d)

- **Descripción**: Este gráfico de líneas presenta la evolución de la precisión de entrenamiento y validación a lo largo de las 3 épocas para ambos modelos. Las curvas de BERT se representan en azul (líneas continuas para entrenamiento, discontinuas para validación), mientras que las de CNN se muestran en naranja.
- **Análisis**: 
  - El modelo BERT exhibe una precisión de entrenamiento que inicia en aproximadamente 0.965 y alcanza cerca de 0.995 en la tercera época, con una precisión de validación que oscila entre 0.98 y 0.99. Esta tendencia refleja una convergencia rápida y una generalización efectiva.
  - La CNN muestra una precisión de entrenamiento que parte de alrededor de 0.96 y llega a aproximadamente 0.98 en la tercera época, con una precisión de validación entre 0.975 y 0.985. La convergencia es más gradual, con una ligera brecha entre entrenamiento y validación, sugiriendo una capacidad algo menor para capturar patrones complejos.
  - **Comparación**: BERT supera consistentemente a la CNN, con una diferencia de 1-2% en precisión, destacando su superioridad en el modelado del contexto semántico.

### Gráfico de Métricas de Prueba (`test_metrics_comparison.png`)
![test_metrics_comparison](https://github.com/user-attachments/assets/bc058a6a-1da4-4a21-9f96-0838bd953aee)

- **Descripción**: Este gráfico de barras compara las métricas de prueba (exactitud, precisión, recall y F1-score) entre BERT (azul) y CNN (naranja) en el conjunto de prueba.
- **Análisis**: 
  - BERT presenta valores de aproximadamente 0.986 para exactitud, 0.986 para precisión, 0.987 para recall y 0.986 para F1-score, reflejando un rendimiento equilibrado y elevado.
  - La CNN obtiene valores de alrededor de 0.983 para exactitud, 0.983 para precisión, 0.983 para recall y 0.981 para F1-score, mostrando un desempeño sólido pero inferior.
  - **Comparación**: BERT exhibe un rendimiento superior en todas las métricas, con una ventaja más pronunciada en F1-score (0.986 vs. 0.981), lo que indica una mejor gestión del equilibrio entre precisión y recall.

## Conclusión
- **Modelo Recomendado**: Con base en los resultados, BERT se identifica como la opción preferida para este proyecto, gracias a su mayor precisión y capacidad de generalización, particularmente en un contexto donde la detección precisa de fraudes es esencial. No obstante, la CNN representa una alternativa competitiva en escenarios con restricciones computacionales o para prototipos iniciales.
- **Siguientes Pasos**: Se sugieren exploraciones adicionales, como la integración de embeddings preentrenados (e.g., GloVe) para la CNN, el ajuste de hiperparámetros o el análisis del balance de clases en el dataset para optimizar el rendimiento.

## Instrucciones de Uso
1. **Requisitos**: Se recomienda contar con las bibliotecas `torch`, `transformers`, `pandas`, `sklearn`, `tqdm`, `matplotlib` y `seaborn` instaladas.
   ```bash
   pip install torch transformers pandas scikit-learn tqdm matplotlib seaborn
