#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_image_folder.py
-----------------------
✓ Detecta objetos (YOLOv8) → celular/persona
✓ Marca spoof si el rostro está dentro de un celular (IoU > umbral)
✓ Si NO spoof, evalúa el rostro con el modelo Anti-Spoof (CNN Silent-Face)
✓ Guarda:
   • Imagenes anotadas en <out_dir>/<nombre>_out.jpg
   • CSV con resumen en la carpeta de salida

Ejemplo:
python predict_image_folder.py \
   --input_dir ./test_imgs \
   --output_dir ./out_imgs \
   --model_path saved_models/AntiSpoofing_bin_1.5_128.onnx \
   --yolo8 yolov8l.pt --threshold 0.76 --iou 0.30
"""
import cv2, csv, datetime, argparse, os, sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

# ---------- Colores ----------
COLOR_REAL    = (  0,255,  0)
COLOR_FAKE    = (  0,  0,255)
COLOR_UNKNOWN = (127,127,127)
COLOR_PHONE   = (255,  0,  0)
COLOR_PERSON  = (  0,255,255)

PHONE_ID, PERSON_ID = 67, 0  # COCO IDs

# ---------- Helpers ----------
def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    areaA=(ax2-ax1)*(ay2-ay1); areaB=(bx2-bx1)*(by2-by1)
    return inter/(areaA+areaB-inter)

def increased_crop(img,bbox,bbox_inc=1.58):
    h,w=img.shape[:2]
    x1,y1,x2,y2=bbox.astype(int)[:4]; l=max(x2-x1,y2-y1)
    xc,yc=x1+(x2-x1)/2, y1+(y2-y1)/2
    x,y=int(xc-l*bbox_inc/2), int(yc-l*bbox_inc/2)
    x1c,y1c=max(0,x),max(0,y)
    x2c,y2c=min(w,x+int(l*bbox_inc)),min(h,y+int(l*bbox_inc))
    patch=img[y1c:y2c,x1c:x2c]
    padT,padB=y1c-y,int(l*bbox_inc)-(y2c-y)
    padL,padR=x1c-x,int(l*bbox_inc)-(x2c-x)
    return cv2.copyMakeBorder(patch,padT,padB,padL,padR,cv2.BORDER_CONSTANT,value=[0,0,0])

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser(description="Folder Anti-Spoof batch")
    ap.add_argument("--input_dir","-i",required=True,help="Directorio con imágenes")
    ap.add_argument("--output_dir","-o",required=True,help="Directorio de salida")
    ap.add_argument("--model_path","-m",required=True,help="Modelo Anti-Spoof (.onnx)")
    ap.add_argument("--yolo8","-y8",default="yolov8l.pt",help="Pesos YOLOv8 (pt/onnx)")
    ap.add_argument("--threshold","-t",type=float,default=0.5,help="Umbral REAL vs UNKNOWN")
    ap.add_argument("--iou",type=float,default=0.30,help="IoU min rostro-celular")
    args=ap.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)

    # Modelos
    y8 = YOLO(args.yolo8); classNames=y8.names
    y5_face = YOLOv5('saved_models/yolov5s-face.onnx')
    anti = AntiSpoof(args.model_path)

    # CSV setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = out_dir/f"resultados_spoof_{timestamp}.csv"
    csv_file  = open(csv_path,"w",newline="",encoding="utf-8")
    writer    = csv.writer(csv_file)
    writer.writerow(["imagen","spoof","cnn_label","cnn_score"])

    # Recorrer imágenes
    for img_path in sorted(in_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}: continue
        img = cv2.imread(str(img_path)); imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # ---------- Paso 1: YOLOv8 objetos ----------
        results=y8(img,verbose=False)
        faces,phones=[],[]
        for r in results:
            for b in r.boxes:
                cls=int(b.cls[0]); conf=float(b.conf[0])
                x1,y1,x2,y2=map(int,b.xyxy[0])
                if cls==PERSON_ID: faces.append(np.array([x1,y1,x2,y2]))
                elif cls==PHONE_ID: phones.append(np.array([x1,y1,x2,y2]))
                # dibuja (opcional)
                color = COLOR_PHONE if cls==PHONE_ID else COLOR_PERSON if cls==PERSON_ID else (200,0,200)
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                cv2.putText(img,f"{classNames[cls]} {conf:.2f}",(x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        # ---------- Paso 2: SPOOF ----------
        spoof=False
        for f in faces:
            if spoof: break
            for p in phones:
                if iou(f,p)>args.iou:
                    spoof=True
                    cv2.putText(img,"❌ SPOOF",(f[0],f[1]-25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)
                    break

        # ---------- Paso 3: CNN si no spoof ----------
        cnn_label,cnn_score="N/A","N/A"
        if not spoof:
            bboxes=y5_face([imgRGB])[0]
            if bboxes.shape[0]:
                crop=increased_crop(imgRGB,bboxes[0])
                pred=anti([crop])[0]; cnn_score=float(pred[0][0]); lbl=np.argmax(pred)
                if lbl==0 and cnn_score>args.threshold:
                    text,color=f"REAL {cnn_score:.2f}",COLOR_REAL
                    cnn_label="REAL"
                elif lbl==0:
                    text,color=f"UNKNOWN {cnn_score:.2f}",COLOR_UNKNOWN
                    cnn_label="UNKNOWN"
                else:
                    text,color=f"FAKE {cnn_score:.2f}",COLOR_FAKE
                    cnn_label="FAKE"
                x1,y1,x2,y2=bboxes[0].astype(int)[:4]
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                cv2.putText(img,text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            else:
                cnn_label="NO_FACE"

        # ---------- Guardar imagen anotada ----------
        out_img = out_dir/f"{img_path.stem}_out{img_path.suffix}"
        cv2.imwrite(str(out_img),img)

        # ---------- Escribir CSV ----------
        writer.writerow([img_path.name,"YES" if spoof else "NO",cnn_label,cnn_score])

        print(f"[✓] {img_path.name:30} → spoof={spoof}, cnn={cnn_label}")

    csv_file.close()
    print(f"\nCSV guardado en: {csv_path}")

if __name__=="__main__":
    main()
