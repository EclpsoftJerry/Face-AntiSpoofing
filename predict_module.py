import cv2
import numpy as np
from ultralytics import YOLO
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from logger_config import logger
from config import IOU_THRESHOLD, SCORE_THRESHOLD
import time

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2-ix1)*(iy2-iy1)
    areaA  = (ax2-ax1)*(ay2-ay1)
    areaB  = (bx2-bx1)*(by2-by1)
    return inter / (areaA + areaB - inter)

def increased_crop(img, bbox, bbox_inc=1.5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)[:4]
    l   = max(x2-x1, y2-y1)
    xc, yc = x1+(x2-x1)/2, y1+(y2-y1)/2
    x , y  = int(xc-l*bbox_inc/2), int(yc-l*bbox_inc/2)
    x1c, y1c = max(0,x), max(0,y)
    x2c, y2c = min(w, x+int(l*bbox_inc)), min(h, y+int(l*bbox_inc))
    patch = img[y1c:y2c, x1c:x2c]
    padT, padB = y1c-y, int(l*bbox_inc)-(y2c-y)
    padL, padR = x1c-x, int(l*bbox_inc)-(x2c-x)
    return cv2.copyMakeBorder(patch, padT, padB, padL, padR,
                              cv2.BORDER_CONSTANT, value=[0,0,0])

def process_image(img, yolo8, face_det, anti_spoof, iou_thresh, score_thresh):

    start_total = time.time()
    logger.info(f"IOU_THRESHOLD: {iou_thresh}, SCORE_THRESHOLD: {score_thresh}")

    if img is None:
        logger.warning("Imagen recibida es None.")
        return {"result": "UNKNOWN", "reason": "Imagen no válida"}    
    try:
        logger.info("Iniciando análisis de imagen...")        

        classNames = yolo8.names
        PHONE_ID  = [k for k, v in classNames.items() if v == "cell phone"][0]    
        PERSON_ID = [k for k, v in classNames.items() if v == "person"][0]
        TV_ID     = [k for k, v in classNames.items() if v == "tv"][0]        

        start = time.time()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.info("Tiempo conversión a RGB: %.2f s", time.time() - start)

        start = time.time()
        results = yolo8(img, verbose=False)
        logger.info("Tiempo predicción YOLOv8: %.2f s", time.time() - start)

        faces_boxes, phones_boxes, tv_boxes = [], [], []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                if cls==PERSON_ID:
                    faces_boxes.append(np.array([x1,y1,x2,y2]))
                elif cls==PHONE_ID:
                    phones_boxes.append(np.array([x1,y1,x2,y2]))
                elif cls==TV_ID:
                    tv_boxes.append(np.array([x1, y1, x2, y2]))

        logger.info("Detecciones: Rostros=%d, Celulares=%d, TVs=%d", len(faces_boxes), len(phones_boxes), len(tv_boxes))

        for f in faces_boxes:
            for p in phones_boxes + tv_boxes:
                iou_val = iou(f, p)
                if iou_val > iou_thresh:
                    logger.warning("IoU alto entre rostro y objeto: %.2f", iou_val)
                    return {
                        "result": "FAKE",
                        "reason": "IoU alto entre rostro y celular/TV"
                    }
        start = time.time()
        bboxes = face_det([imgRGB])[0]
        logger.info("Tiempo detección rostros YOLOv5-face: %.2f s", time.time() - start)

        if bboxes.shape[0]==0:
            logger.warning("No se detectó rostro.")
            return {
                "result": "UNKNOWN",
                "reason": "No se detectó rostro"
            }

        for bbox in bboxes:
            crop = increased_crop(imgRGB, bbox)
            start = time.time()
            pred = anti_spoof([crop])[0]
            logger.info("Tiempo predicción AntiSpoof: %.2f s", time.time() - start)            
            score = pred[0][0]
            label = np.argmax(pred)
            result = {
                "result": "REAL" if label==0 and score > score_thresh else "UNKNOWN" if label==0 else "FAKE",
                "real_score": float(pred[0][0]),
                "fake_score": float(pred[0][1])
            }
            logger.info("Resultado del modelo: %s", result)
            logger.info("Tiempo total de análisis: %.2f s", time.time() - start_total)
            return result

        logger.warning("Final inesperado: ninguna predicción devuelta.")
        logger.info("Tiempo total de análisis: %.2f s", time.time() - start_total)
        return {"result": "UNKNOWN", "reason": "Error inesperado"}

    except Exception as e:
        logger.exception("Error crítico en process_image: %s", str(e))
        return {"result": "UNKNOWN", "reason": "Error interno"}
