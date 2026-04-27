# Hito 2: Multicategorización de Comandos de Voz

Este hito implementa la clasificación multiclase de espectrogramas de audio en 7 categorías principales, comparando dos enfoques arquitectónicos distintos.

## Estructura del Proyecto

- `src/data/`: Gestión de datasets y transformaciones, con división por locutor (speaker hash).
- `src/models/`: Implementación de arquitecturas.
  - **CLIP + MLP**: Extracción de embeddings usando CLIP (congelado) y clasificación mediante un Perceptrón Multicapa.
  - **CNN**: Red Neuronal Convolucional personalizada entrenada desde cero.
- `src/training/`: Lógica de entrenamiento y validación modular.
- `src/utils/`: Utilidades para la gestión de etiquetas y categorías.

## Categorías (Labels)

Las 30 palabras originales se agrupan en:
1. `numbers`: zero, one, ..., nine
2. `animals`: bird, cat, dog
3. `directions`: up, down, left, right
4. `commands`: go, stop, on, off, yes, no
5. `objects`: bed, house, tree
6. `names`: marvin, sheila
7. `emotions`: happy, wow

## Modelos

### 1. CLIP + MLP
Utiliza el modelo preentrenado `openai/clip-vit-base-patch32` para obtener un vector de características (embedding) de 768 dimensiones. Este vector se pasa por un MLP de 3 capas para la clasificación final.
- **Ventaja**: Aprovecha el conocimiento visual previo de un modelo masivo.

### 2. CNN Custom
Una red convolucional de 3 bloques (Conv + ReLU + MaxPool) seguida de una cabeza clasificadora.
- **Ventaja**: Aprende patrones específicos de los espectrogramas desde cero.

## Guía de Ejecución

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para la gestión de dependencias y el entorno virtual.

### 1. Requisitos Previos
Tener instalado `uv`. Si no lo tienes:
```bash
curl -LsSf https://astral-sh/uv/install.sh | sh
```

### 2. Instalación
Clona el repositorio y sincroniza el entorno:
```bash
git clone git@github.com:ThaielSC/Hito-2-AI.git
cd Hito-2-AI
uv sync
```

### 3. Entrenamiento de Modelos
Para entrenar cualquiera de los modelos, ejecuta el script principal desde la raíz:

**Entrenar CNN (Recomendado):**
```bash
uv run python main.py --model cnn --epochs 10
```

**Entrenar CLIP + MLP:**
```bash
uv run python main.py --model clip --epochs 10
```

### 4. Predicción (Inferencia sin entrenar)
Puedes probar los modelos ya entrenados incluidos en el repositorio con cualquier espectrograma:
```bash
uv run python predict.py --image "ruta/a/tu/espectrograma.png" --model cnn
```

### 5. Evaluación Comparativa
Para obtener un informe detallado de métricas (F1-score, Precision, Recall) comparando ambos modelos sobre el conjunto de prueba (test set):
```bash
uv run python evaluate.py
```

### 6. Ejecución de Pruebas
Para verificar que el dataset y los modelos están correctamente configurados:
```bash
uv run python tests/test_sanity.py
```
*(Nota: Asegúrate de estar en la raíz del proyecto para que las importaciones funcionen correctamente).*

## Comparación de Resultados

Tras entrenar ambos modelos durante 10 épocas con las mismas condiciones (batch_size: 64, lr: 1e-4), se obtuvieron los siguientes resultados:

| Métrica | CNN Custom | CLIP + MLP |
|---------|------------|------------|
| **Precisión (Val)** | **77.86%** | 53.28% |
| **Pérdida (Val)** | **0.6394** | 1.2030 |
| **Tiempo/Época** | **~90 segundos** | ~210 segundos |
| **Enfoque** | Aprendizaje desde cero (Feature Learning) | Transfer Learning (Frozen Backbone) |

### Análisis y Conclusiones

1. **Especialización vs. Generalización**: La CNN superó a CLIP por un amplio margen ( ~24%). Esto se debe a que los espectrogramas son representaciones visuales de señales de audio con patrones de textura y frecuencia específicos que difieren radicalmente de las imágenes naturales en las que CLIP fue entrenado.
2. **Eficiencia Computacional**: La CNN es significativamente más liviana y rápida de entrenar. CLIP, al ser un Transformer (ViT-B/32), requiere mucha más memoria y cómputo para procesar cada imagen, incluso con los pesos congelados.
3. **Desafío de CLIP**: El "dominio" de los datos es la clave. Los embeddings de CLIP son excelentes para objetos cotidianos, pero "ven" los espectrogramas como ruido o texturas abstractas, dificultando la tarea del MLP clasificador.
4. **Hito 2 vs Hito 1**: Hemos pasado exitosamente de una clasificación binaria a una multiclase (7 categorías), logrando una precisión sólida con una arquitectura convolucional simple pero bien adaptada al dominio del problema.
