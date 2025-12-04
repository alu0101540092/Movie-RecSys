# Movie-RecSys

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movies-recsys.streamlit.app)

## Ejecución con Docker Compose

Este proyecto está configurado para ejecutarse completamente utilizando Docker Compose, eliminando la necesidad de instalar Python o dependencias en tu máquina local.

### Requisitos previos

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Iniciar la aplicación

Para iniciar la aplicación Streamlit, ejecuta el siguiente comando en la raíz del proyecto:

```bash
docker compose up
```

La aplicación estará disponible en [http://localhost:8501](http://localhost:8501).

> **Nota:** El contenedor instalará automáticamente las dependencias listadas en `requirements.txt` cada vez que se inicie.

### Detener la aplicación

Para detener la aplicación, presiona `Ctrl+C` en la terminal donde se está ejecutando, o ejecuta en otra terminal:

```bash
docker compose down
```

### Ejecutar comandos personalizados

Puedes ejecutar cualquier comando dentro del entorno del contenedor utilizando `docker compose run`. Esto es útil para ejecutar scripts de Python, pruebas o tareas de mantenimiento sin ensuciar tu entorno local.

**Sintaxis general:**

```bash
docker compose run --rm app <comando>
```

#### Ejemplos útiles:

1.  **Ejecutar un script de Python:**

    ```bash
    docker compose run --rm app python src/evaluate.py
    ```

2.  **Abrir una terminal interactiva (shell):**

    ```bash
    docker compose run --rm app /bin/bash
    ```

3.  **Verificar la versión de Python:**

    ```bash
    docker compose run --rm app python --version
    ```

4.  **Instalar una dependencia temporalmente:**

    ```bash
    docker compose run --rm app pip install pandas
    ```

### Gestión de Dependencias

Para agregar una nueva librería al proyecto de forma permanente:

1.  Agrega el nombre de la librería al archivo `requirements.txt`.
2.  Reinicia el contenedor con `docker compose up`.

## Funcionamiento del Sistema Recomendador

El sistema utiliza un algoritmo de **Filtrado Colaborativo** basado en **Descomposición en Valores Singulares (SVD)**. A continuación se detallan los aspectos técnicos de su implementación y optimización.

### 1. Modelo Matemático (SVD)

El modelo SVD descompone la matriz de valoraciones usuario-ítem $R$ en el producto de factores latentes. La predicción de la valoración $\hat{r}_{ui}$ que un usuario $u$ daría a un ítem $i$ se calcula como:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T p_u
$$

Donde:

- $\mu$: Media global de todas las valoraciones.
- $b_u$: Sesgo (bias) del usuario $u$ (tendencia del usuario a valorar alto o bajo).
- $b_i$: Sesgo (bias) del ítem $i$ (tendencia de la película a recibir valoraciones altas o bajas).
- $q_i$: Vector de factores latentes del ítem $i$.
- $p_u$: Vector de factores latentes del usuario $u$.

### 2. Optimización de Memoria y Rendimiento

Originalmente, cargar el modelo completo (1.1GB) en memoria causaba problemas de rendimiento. Se ha implementado una solución optimizada:

#### Extracción de Matrices

En lugar de cargar el objeto Python completo (pickled), extraemos las matrices numéricas esenciales ($P$, $Q$, $B_u$, $B_i$) y las guardamos en formato binario de Numpy (`.npy`).

#### Memory Mapping

Utilizamos `numpy.load(..., mmap_mode='r')` para cargar estas matrices. Esto permite al sistema operativo mapear el archivo en disco directamente a la memoria virtual, cargando solo los fragmentos necesarios en la RAM física bajo demanda. Esto reduce drásticamente el consumo de memoria inicial.

#### Vectorización

La generación de recomendaciones se realiza mediante operaciones vectoriales de Numpy en lugar de bucles de Python. Calculamos las predicciones para **todas** las películas simultáneamente:

```python
# Cálculo vectorizado para todos los ítems
scores = np.dot(qi, user_factors) + bi + user_bias + global_mean
```

Esto es significativamente más rápido que iterar sobre cada película y llamar al método `predict()` de la librería Surprise.

### 3. Flujo de Recomendación

1.  **Carga Lazy**: Los componentes del modelo solo se mapean en memoria cuando se solicitan recomendaciones.
2.  **Cálculo de Scores**: Se calcula el score predicho para todas las películas del catálogo utilizando la fórmula vectorial.
3.  **Filtrado**: Se eliminan las películas que el usuario ya ha valorado.
4.  **Ranking**: Se ordenan las películas restantes por score descendente.
5.  **Enriquecimiento**: Se añaden metadatos (título, género) a los top-N resultados para mostrarlos en la UI.
