#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video anti-spoof checker
  – Paso-1: YOLOv8  (persona / celular)
  – Paso-2: IOU      (spoof ↔ rostro-celular)
  – Paso-3: CNN anti-spoof (solo si no es spoof)
Genera un resumen único al final.
"""
import cv2, argparse, time, numpy as np, datetime
from pathlib import Path
from ultralytics import YOLO
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

# ---------- colores ----------
C_REAL, C_FAKE, C_UNK = (0,255,0), (0,0,255), (127,127,127)
C_PHONE, C_PERSON     = (255,0,0), (0,255,255)
PHONE_ID, PERSON_ID   = 67, 0     # etiquetas COCO

# ---------- helpers ----------
def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    return inter / ((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

def inc_crop(img,b,inc=1.5):
    h,w = img.shape[:2]; x1,y1,x2,y2 = b.astype(int)[:4]
    l   = max(x2-x1,y2-y1); xc,yc = x1+(x2-x1)/2, y1+(y2-y1)/2
    x,y = int(xc-l*inc/2), int(yc-l*inc/2)
    x1c,y1c = max(0,x), max(0,y)
    x2c,y2c = min(w,x+int(l*inc)), min(h,y+int(l*inc))
    patch = img[y1c:y2c, x1c:x2c]
    padT,padB = y1c-y, int(l*inc)-(y2c-y)
    padL,padR = x1c-x, int(l*inc)-(x2c-x)
    return cv2.copyMakeBorder(patch,padT,padB,padL,padR,
                              cv2.BORDER_CONSTANT,value=[0,0,0])

# ---------- argumentos ----------
ap = argparse.ArgumentParser("Video anti-spoof")
ap.add_argument("-i","--input",  help="video (vacío=webcam)")
ap.add_argument("-o","--output", help="video salida", default=None)
ap.add_argument("-m","--model_path", required=True, help="CNN .onnx")
ap.add_argument("--yolo8",   default="yolov8l.pt")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--iou",       type=float, default=0.30)
ap.add_argument("--display",   choices=["y","n"], default="n")
ap.add_argument("--verbose",   choices=["y","n"], default="n")
args = ap.parse_args()

# ---------- modelos ----------
y8   = YOLO(args.yolo8); names = y8.names
face = YOLOv5('saved_models/yolov5s-face.onnx')
cnn  = AntiSpoof(args.model_path)

# ---------- captura ----------
cap = cv2.VideoCapture(args.input if args.input else 0)
if not cap.isOpened(): raise RuntimeError("❌ No se pudo abrir fuente.")
fps = cap.get(cv2.CAP_PROP_FPS) or 24
w,h = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(args.output,
         cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h)) if args.output else None

print("🔴 Procesando video… (Q para salir si display=y)")

# ---------- contadores ----------
total_frames = det_frames = spoof_frames = cnn_frames = 0
scores_real  = []          # para promedio

# ---------- bucle ----------
while True:
    ret,frame = cap.read()
    if not ret: break
    total_frames += 1

    # Paso-1 YOLOv8
    res = y8(frame, verbose=False)
    faces = []; phones = []
    for r in res:
        for b in r.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            col = C_PHONE if cls==PHONE_ID else C_PERSON if cls==PERSON_ID else (200,0,200)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            if cls in (PHONE_ID,PERSON_ID):
                cv2.putText(frame,f"{names[cls]} {conf:.2f}",(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
            if cls==PERSON_ID: faces.append(np.array([x1,y1,x2,y2]))
            if cls==PHONE_ID:  phones.append(np.array([x1,y1,x2,y2]))
    if faces or phones: det_frames += 1

    # Paso-2 SPOOF
    spoof = any(iou(f,p)>args.iou for f in faces for p in phones)
    if spoof:
        spoof_frames += 1
        for f in faces:
            cv2.putText(frame,"❌ SPOOF",(f[0],f[1]-25),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    # Paso-3 CNN (solo si no es spoof)
    if not spoof and faces:
        bboxes = face([cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)])[0]
        if bboxes.shape[0]:
            cnn_frames += 1
            bx = bboxes[0]
            crop = inc_crop(frame,bx)
            pred = cnn([crop])[0]; score = pred[0][0]; label = np.argmax(pred)
            tag,color = ("REAL",C_REAL) if label==0 and score>args.threshold else \
                        ("UNKNOWN",C_UNK) if label==0 else ("FAKE",C_FAKE)
            scores_real.append(score if tag=="REAL" else 0.0)
            x1,y1,x2,y2 = bx.astype(int)[:4]
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{tag} {score:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    # salida
    writer and writer.write(frame)
    # --- comentar en caso de quitar parametro display -----
    if args.display=="y":
        cv2.imshow("Spoof-Checker",frame)
        if cv2.waitKey(1)&0xFF in (27,ord('q')): break

# ---------- cierre ----------
cap.release(); writer and writer.release()
# --- comentar en caso de quitar parametro display -----
args.display=="y" and cv2.destroyAllWindows()

# ---------- resumen ----------
avg_real = np.mean(scores_real) if scores_real else 0.0
final_tag = "REAL" if avg_real>=args.threshold and spoof_frames==0 else "FAKE"

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"""
📊 RESUMEN DE PROCESAMIENTO DEL VIDEO — {stamp}
----------------------------------------------------
🎞️ Total de frames procesados   : {total_frames}
📌 Frames con detecciones       : {det_frames}
⚠️  Frames con SPOOF detectado  : {spoof_frames}
🧪 Frames evaluados por CNN     : {cnn_frames}
🧠 Puntaje promedio de realidad : {avg_real:.2f}
✅ Resultado final del video    : {final_tag}
""")
