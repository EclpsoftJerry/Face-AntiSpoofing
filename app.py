from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
from ultralytics import YOLO
from logger_config import logger
from config import MODEL_PATH, YOLOV8_PATH, TEMP_DIR, YOLO5FACE_PATH, ALLOWED_EXTENSIONS, IOU_THRESHOLD, SCORE_THRESHOLD
from utils.validators import validate_file_extension
from security import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_db
)
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from models.user import User
from models.audit_log import AuditLog
from predict_module import process_image

import shutil
import uuid
import os
import cv2
import json

# Cargar los modelos una sola vez
logger.info("Cargando modelos una sola vez...")
yolo8_model = YOLO(YOLOV8_PATH)
face_det_model = YOLOv5(YOLO5FACE_PATH)
antispoof_model = AntiSpoof(MODEL_PATH)
logger.info("Modelos cargados.")


app = FastAPI()

@app.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    logger.info("Intento de login para usuario: %s", form_data.username)
    user_authenticated = authenticate_user(db, form_data.username, form_data.password)
    if not user_authenticated:
        logger.warning("Login fallido para usuario: %s", form_data.username)
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    access_token = create_access_token(data={"sub": user_authenticated.username})
    logger.info("Login exitoso para usuario: %s", form_data.username)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    #image1: UploadFile = File(None),
    #image2: UploadFile = File(None) 
    images: List[UploadFile] = File(...)   
):
    logger.info("Inicio del endpoint /predict por usuario: %s", user.username)   
    # Filtrar solo imágenes válidas (que no sean None y tengan filename)
    valid_images = [img for img in images if img and img.filename] 
    if not valid_images:
        logger.warning("Solicitud sin imágenes válidas por parte del usuario: %s", user.username)
        raise HTTPException(
            status_code=400,
            detail="Debe subir al menos una imagen válida en el campo 'images'."
        )
    if len(images) > 3:
        logger.warning("Demasiadas imágenes enviadas por usuario: %s", user.username)
        raise HTTPException(
            status_code=400,
            detail="Máximo 2 imágenes permitidas."
        )
    
    # Validar extensiones
    for image in images:
        if image and image.filename:
            validate_file_extension(image, ALLOWED_EXTENSIONS)

    os.makedirs(TEMP_DIR, exist_ok=True)
    paths = []
    uploads = []
    results = {}

    try:
        for idx, uploaded in enumerate(valid_images, start=1):
            ext = uploaded.filename.split('.')[-1]
            temp_path = f"{TEMP_DIR}/{uuid.uuid4()}.{ext}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(uploaded.file, buffer)
            paths.append(temp_path)
            uploads.append(uploaded.filename)

        for idx, path in enumerate(paths):
            key = uploads[idx]
            logger.info(f"Analizando imagen: {key}")
            img = cv2.imread(path)
            if img is None:
                results[key] = {
                    "result": "UNKNOWN",
                    "reason": "Imagen no válida"
                }
                continue

            #result = process_image(img, MODEL_PATH, YOLOV8_PATH, YOLO5FACE_PATH)            
            result = process_image(
                                    img,
                                    yolo8=yolo8_model,
                                    face_det=face_det_model,
                                    anti_spoof=antispoof_model,
                                    iou_thresh=IOU_THRESHOLD,
                                    score_thresh=SCORE_THRESHOLD
                                )
            
            results[key] = result

        if len(results) == 2:
            final = "REAL" if all(r["result"] == "REAL" for r in results.values()) else "FAKE"
        else:
            final = list(results.values())[0]["result"]

        response_json = {
           **results,
            "final_inference": final
        }

        # Registrar log en base de datos
        log = AuditLog(
            username=user.username,
            endpoint=str(request.url.path),
            method=request.method,            
            request_data=", ".join(uploads),
            response_data=json.dumps(response_json),
            status_code=200
        )
        db.add(log)
        db.commit()
        logger.info("Procesamiento exitoso en /predict. Resultado: %s", final)
        return JSONResponse(response_json)        
    except Exception as e:
        # Registrar error
        log = AuditLog(
            username=user.username,
            endpoint=str(request.url.path),
            method=request.method,
            request_data="Error al subir imágenes",
            response_data=str(e),
            status_code=500
        )
        db.add(log)
        db.commit()
        logger.error("Error interno en /predict: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    finally:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
