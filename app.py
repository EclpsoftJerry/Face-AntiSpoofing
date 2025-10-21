from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Form
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
from typing import Optional
from skimage.metrics import structural_similarity as ssim

import shutil
import uuid
import os
import cv2
import json
import fitz
import numpy as np

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
    images: List[UploadFile] = File(...),
    id: Optional[str] = Form(None)
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
            document_id=id,
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
            document_id=id,
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

@app.post("/compare-pdf")
async def compare_pdf(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    id: Optional[str] = Form(None)
):
    """
    Endpoint que recibe un PDF, extrae las dos primeras imágenes
    y evalúa si son idénticas usando SSIM.
    """
    logger.info(f"Inicio del endpoint /compare-pdf por usuario: {user.username}")

    try:
        # Validar que se haya enviado el archivo
        if not file:
            raise HTTPException(status_code=400, detail="No se envió ningún archivo.")
        # Validar que el archivo sea un PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="El archivo debe ser de tipo PDF.")
        
        pdf_bytes = await file.read()
        # Validar que el PDF no esté vacío
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="El PDF está vacío o no se pudo leer.")
        
        # Abrir el PDF con PyMuPDF
        try:
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"El archivo no es un PDF válido: {e}")

        images = []

        # Extraer imágenes del PDF
        for page_index in range(len(pdf)):
            for img in pdf.get_page_images(page_index):
                xref = img[0]
                base = pdf.extract_image(xref)
                np_img = np.frombuffer(base["image"], np.uint8)
                img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if img_cv is not None:
                    images.append(img_cv)
        pdf.close()

        logger.info(f"Total de imágenes detectadas: {len(images)}")

        if len(images) < 2:
            raise HTTPException(status_code=400, detail="No se encontraron al menos dos imágenes en el PDF.")

        # Tomar las dos primeras imágenes
        img1, img2 = images[0], images[1]
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(gray1, gray2, full=True)
        same_image = score >= 0.99

        result_json = {            
            "images_identical": bool(same_image), 
            "similarity_score": float(round(float(score), 4)),                       
            "interpretation": (
                "The images appear identical (possible duplicate or retry with the same photo)."
                if same_image
                else "The images are different (valid liveness test)."
            ),
            "total_images_detected": len(images)
        }        
        logger.info(
            f"SSIM result: {score:.4f} - "
            f"{'Duplicate detected' if same_image else 'Different images'} | "
            f"Total images: {len(images)}"
        )

        # Registrar log en base de datos
        log = AuditLog(
            username=user.username,
            document_id=id,
            endpoint=str(request.url.path),
            method=request.method,
            request_data=file.filename,
            response_data=json.dumps(result_json),
            status_code=200
        )
        db.add(log)
        db.commit()

        return JSONResponse(result_json)
        
    except HTTPException as e:
        logger.warning(f"Error validado en /compare-pdf: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error general en /compare-pdf: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {e}")