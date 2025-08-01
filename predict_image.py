#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

COLOR_REAL     = (0, 255, 0)
COLOR_FAKE     = (0,   0,255)
COLOR_UNKNOWN  = (127,127,127)
COLOR_PHONE    = (255,  0, 0)
COLOR_PERSON   = (  0,255,255)
COLOR_TV       = (255, 165, 0)

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YOLOv8 + Face Anti-Spoof")
    ap.add_argument("-i","--input" ,required=True ,help="Imagen de entrada")
    ap.add_argument("-o","--output",required=True ,help="Imagen de salida anotada")
    ap.add_argument("-m","--model_path",required=True, help="Ruta .onnx del modelo Anti-Spoof")
    ap.add_argument("-t","--threshold",type=float,default=0.5, help="Umbral REAL vs UNKNOWN")
    ap.add_argument("--iou",type=float,default=0.30, help="IoU mínimo rostro-celular para marcar spoof")
    ap.add_argument("--yolo8","--y8",default="yolov8m.pt", help="Pesos YOLOv8 (pt u onnx)")
    args = ap.parse_args()

    # Clases consideradas como SPOOF si se superponen con rostro
    SPOOF_CLASSES = ["cell phone", "tv", "laptop", "tablet", "monitor", "book"]  # ← puedes añadir más etc.

    yolo8  = YOLO(args.yolo8)
    classNames  = yolo8.names

    # Identificadores de clases  
    CLASS_IDS = {v: k for k, v in classNames.items() if v in SPOOF_CLASSES + ["person"]}      
    PHONE_ID  = CLASS_IDS.get("cell phone")
    TV_ID     = CLASS_IDS.get("tv")
    PERSON_ID = CLASS_IDS.get("person")
    print(f"IDs => PHONE: {PHONE_ID}, TV: {TV_ID}, PERSON: {PERSON_ID}")

    # Modelos
    face_det = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(args.model_path)

    # Carga y detección
    img    = cv2.imread(args.input)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("🔍 Paso 1: Detección de objetos con YOLOv8...")
    results = yolo8(img, verbose=False)
    faces_boxes = []
    spoof_boxes = []
    all_det = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0]); cls = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            label = classNames[cls]
            all_det.append((label, conf, bbox))
            # Dibujar cajas
            color = (200,0,200)            
            #color = COLOR_PHONE if cls==PHONE_ID else COLOR_PERSON if cls==PERSON_ID else COLOR_TV if cls==TV_ID else (200,0,200)
            if label in SPOOF_CLASSES:
                color = COLOR_PHONE if label == "cell phone" else COLOR_TV
            elif label == "person":
                color = COLOR_PERSON

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)            
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # Clasificación
            if label == "person":
                faces_boxes.append(np.array([x1, y1, x2, y2]))
            elif label in SPOOF_CLASSES:
                #spoof_boxes.append(np.array([x1, y1, x2, y2]))
                spoof_boxes.append((label, np.array([x1, y1, x2, y2])))

    print(f"🧠 Objetos detectados: {len(all_det)}")
    for n,c,b in all_det:
        print(f"🔹 {n:<12} conf={c:.2f} bbox={b}")

    print("🧪 Paso 2: Verificación de SPOOF (superposición)...")
    spoof = False
    for f in faces_boxes:
        for label,p in spoof_boxes:
            iou_val = iou(f, p)            
            print(f"IOU entre rostro y objeto sospechoso [{label}]: {iou_val:.2f}")
            #print(f"IOU: {iou_val}")
            if iou_val > args.iou:
                spoof = True
                mensaje = f"❌ SPOOF DETECTED (IoU={iou_val:.2f})"
                cv2.putText(img,mensaje,(f[0],f[1]-35),
                            cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)                
                print(f"⚠️ SPOOF DETECTADO por superposición con objeto")
                break
        if spoof: 
            break
    
    if not spoof:
        print("🧠 Paso 3: Evaluación Anti-Spoof con modelo CNN...")
        print("🔍 Paso 3.1: Detección de rostro con YOLOv5-face...")
        bboxes = face_det([imgRGB])[0]
        if bboxes.shape[0]==0:
            print("❌ No se detectaron rostros en la imagen para evaluación Anti-Spoof.")
        else:
            print("🔍 Paso 3.2: Análisis del rostro detectado con el modelo Anti-Spoofing (CNN entrenado)...")
        for bbox in bboxes:
            crop   = increased_crop(imgRGB, bbox)
            pred   = anti_spoof([crop])[0]
            score  = pred[0][0]
            label  = np.argmax(pred)
            x1,y1,x2,y2 = bbox.astype(int)[:4]
            if label==0 and score>args.threshold:
                text, color = f"REAL {score:.2f}", COLOR_REAL
            elif label==0:
                text, color = f"UNKNOWN {score:.2f}", COLOR_UNKNOWN
            else:
                text, color = f"FAKE {score:.2f}", COLOR_FAKE
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.putText(img,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            cv2.imwrite("debug_crop.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            print(f"📊 Scores predichos: Real={pred[0][0]:.4f} | Fake={pred[0][1]:.4f}")
            print("✅ Resultado Anti-Spoof:", text)
    else:
        print("🔒 SPOOF detectado — se omite evaluación CNN.")

    print("💾 Paso 4: Guardando resultado final con anotaciones...")
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), img)
    print(f"📝 Guardado en: {out_path}")
