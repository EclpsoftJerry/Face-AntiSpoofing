import os
from logger_config import logger
from fastapi import UploadFile, HTTPException
from typing import Set

def validate_file_extension(file: UploadFile, allowed_extensions: Set[str]):
    """
    Valida que la extensión del archivo esté permitida.

    Args:
        file (UploadFile): Archivo subido por el usuario.
        allowed_extensions (Set[str]): Conjunto de extensiones permitidas, sin el punto (ej: {"jpg", "png"}).

    Raises:
        HTTPException: Si la extensión del archivo no está permitida.
    """
    ext = os.path.splitext(file.filename)[1].lower().strip(".")
    if ext not in allowed_extensions:
        message = (
            f"Archivo '{file.filename}' tiene una extensión no permitida. "
            f"Solo se permiten: {', '.join(sorted(allowed_extensions))}"
        )
        logger.warning(" %s", message)
        raise HTTPException(
            status_code=400,
            detail=message
        )
