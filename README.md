# Sistema Inteligente de Recomendación de Películas

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movies-recsys.streamlit.app)

- **Universidad de La Laguna**
- **Grado de Ingeniería en Informática**
- **Asignatura:** Sistemas Inteligentes
- **Curso:** 2025-26
- **Autores:** Carolina Acosta Acosta, Samuel Frías Hernández, Salvador González Cueto
- **Tutora del proyecto:** Elena Sánchez Nielsen

---

## Descripción del Proyecto

Este proyecto implementa un sistema de recomendación de películas accesible vía web. Combina técnicas de filtrado colaborativo y basado en contenido para ofrecer recomendaciones personalizadas y precisas, gestionando eficientemente grandes volúmenes de datos.

## Ejecución y Gestión

La aplicación está contenerizada para facilitar su ejecución sin dependencias locales complejas.

### Requisitos

- Docker y Docker Compose.

### Puesta en marcha

1.  Inicia la aplicación desde la raíz del proyecto:
    ```bash
    docker compose up
    ```
2.  Accede a la interfaz web en: [http://localhost:8501](http://localhost:8501)

Para detener el sistema, usa `Ctrl+C` en la terminal o ejecuta `docker compose down` en otra ventana.

> **Nota:** El contenedor instalará automáticamente las dependencias necesarias al iniciarse.

## Funcionamiento Técnico del Sistema

El núcleo del recomendador es un **sistema híbrido** que integra dos enfoques complementarios para maximizar la relevancia de las sugerencias:

### 1. Filtrado Colaborativo (SVD)

Utilizamos el algoritmo **Singular Value Decomposition (SVD)** para capturar patrones latentes en las valoraciones de los usuarios.

- **Matriz de Factores**: Descomponemos la matriz de interacción Usuario-Ítem en vectores de características latentes para usuarios ($p_u$) e ítems ($q_i$).
- **Predicción**: La valoración estimada se calcula como $\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u$.
- **Optimización**: Para manejar el dataset de 32M de interacciones de manera eficiente, no cargamos el modelo completo en memoria. En su lugar, extraemos las matrices ($P, Q, B_u, B_i$) y utilizamos **Memory Mapping** (`numpy.load(mmap_mode='r')`) junto con operaciones vectorizadas. Esto permite generar predicciones para todo el catálogo en milisegundos con un consumo mínimo de RAM.

### 2. Filtrado Basado en Contenido (Géneros)

Para refinar las recomendaciones y respetar las preferencias explícitas del usuario, el sistema incorpora un **score de afinidad** de género. Este se calcula midiendo la intersección entre los géneros de cada película y los seleccionados por el usuario en su perfil, normalizando por el tamaño de la selección (similar a un índice de Jaccard adaptado).

### 3. Enfoque Híbrido y "Fold-In"

- **Score Final**: La lista final de recomendaciones se ordena mediante una combinación lineal ponderada del score normalizado del SVD (calidad global/personalizada) y el score de género (preferencia actual).
- **Cold Start (Fold-In)**: Para nuevos usuarios o sesiones anónimas, el sistema ejecuta un proceso de **optimización SGD en tiempo real** ("Fold-in"). Esto calcula un vector latente temporal ($p_u$) basándose exclusivamente en las valoraciones que el usuario realiza durante la sesión, permitiendo generar recomendaciones personalizadas instantáneamente sin necesidad de reentrenar el modelo global.

## Estructura del Proyecto

A continuación se describen los ficheros clave para facilitar la evaluación del código:

- **`src/`**: Directorio principal del código fuente.
  - **`app.py`**: Punto de entrada de la aplicación Streamlit. Orquesta la interfaz y el flujo de navegación.
  - **`model.py`**: **[CRÍTICO]** Contiene la lógica del sistema recomendador. Aquí se encuentran:
    - `load_optimized_components()`: Carga eficiente de matrices.
    - `fold_in_user()`: Algoritmo para nuevos usuarios.
    - `get_recommendations()`: Lógica híbrida de puntuación y ranking.
  - **`database.py`**: Manejo de la base de datos SQLite (usuarios y ratings).
  - **`data_loader.py`**: Carga de datasets estáticos (títulos de películas).
  - **`ui/`**: Módulos para la interfaz de usuario (componentes de recomendaciones, perfil, etc.).
- **`models/`**: Contiene los archivos binarios (`.npy`) del modelo SVD entrenado y optimizado.
- **`data/`**: Base de datos SQLite (`movie_recsys.db`).
- **`docker-compose.yml`**: Definición de la infraestructura para el despliegue.
