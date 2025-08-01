# Usa una imagen oficial con Python
FROM python:3.8-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instalar librerías del sistema necesarias (si usas opencv, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos y código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expone el puerto en el contenedor
EXPOSE 8000

# Comando para correr la app
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "60", "--log-level", "info"]