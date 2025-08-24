# Clasificador de Literatura Biomédica

Este proyecto implementa un sistema de clasificación de literatura médica utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático.

## Características

- Preprocesamiento de texto biomédico
- Normalización semántica con MeSH
- Embeddings biomédicos (PubMedBERT)
- Clasificación híbrida
- Interfaz web para demostraciones

## Estructura del Proyecto

```
Classification-App/
├── data/
│   ├── raw/          # Datos originales
│   ├── processed/    # Datos procesados
│   └── external/     # Recursos externos (MeSH, FDA, etc.)
├── src/
│   ├── exploration/  # EDA y baseline
│   ├── preprocessing/# Limpieza y normalización
│   ├── features/     # Extracción de características
│   ├── models/       # Modelos de clasificación
│   └── evaluation/   # Métricas y análisis
├── notebooks/        # Jupyter notebooks
├── app/             # Interfaz Streamlit
└── requirements.txt  # Dependencias
```

## Instalación

1. Crear entorno virtual:

```bash
python -m venv venv
```

2. Activar entorno:

```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instalar dependencias:

```bash
uv pip install -r requirements.txt
```

4. Descargar modelos necesarios:

```bash
python -m spacy download en_core_sci_lg
```

## Uso

1. Ejecutar EDA:

```bash
python src/exploration/eda.py
```

2. Entrenar modelo:

```bash
python src/models/train.py
```

3. Iniciar interfaz web:

```bash
streamlit run app/streamlit_app.py
```

## Métricas de Evaluación

- Weighted F1 Score
- Accuracy
- Matriz de Confusión
- ROC-AUC
- Top-k Accuracy
