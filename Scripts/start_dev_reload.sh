# scripts/start_dev_reload.sh
# Solo para desarrollo con recarga automática
uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload