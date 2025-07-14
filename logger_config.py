import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

# Carpeta por día (logs/AAAA-MM-DD/)
today_str = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join("logs", today_str)
os.makedirs(LOG_DIR, exist_ok=True)

# Ruta del archivo de log
log_file_path = os.path.join(LOG_DIR, "app.log")

# Logger personalizado
logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)

# Formato del log
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Handler de archivo por día
file_handler = RotatingFileHandler(log_file_path, maxBytes=5_000_000, backupCount=5)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Handler para consola (opcional)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Evitar agregar múltiples handlers si ya están
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.propagate = False  # Evita conflictos con uvicorn
