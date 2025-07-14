#!/bin/bash

# Configuración de servidor
WORKERS=8            # Ajusta según CPU y pruebas de carga
HOST="0.0.0.0"       # Escucha en todas las interfaces (útil en servidores)
PORT=8000

echo "🚀 Iniciando servidor en producción con Gunicorn + UvicornWorker..."
gunicorn app:app \
  --workers $WORKERS \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind ${HOST}:${PORT} \
  --log-level info \
  --timeout 60
