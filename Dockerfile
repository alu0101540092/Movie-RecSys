# 1. Imagen base
FROM python:3.11

# 2. Directorio de trabajo
WORKDIR /app

# 3. Copiar SÓLO el archivo de requisitos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el código
COPY . .

# 5. Exponer el puerto que usará Streamlit
EXPOSE 8501

# 6. Comando para ejecutar la app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]