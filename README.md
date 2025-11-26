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
