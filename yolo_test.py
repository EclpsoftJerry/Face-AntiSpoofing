
import cv2
import numpy as np
from ultralytics import YOLO

# Configura el modelo
#model = YOLO("saved_models/yolov8m.pt")  
#model = YOLO("yolov8x.pt")
#model = YOLO("saved_models/yolov8l.pt")

model = YOLO("yolov8m.pt")
#model = YOLO("yolov9c.pt")
#model = YOLO("yolov9e.pt")

# Carga imagen original
image_path = "prueba4.jpeg"  # Cambia al nombre de tu imagen
img = cv2.imread(image_path)

# Puedes mostrarla si deseas:
#cv2.imshow("Mejorada", enhanced_img)
#cv2.waitKey(0)

# Ejecuta predicción
results = model(img)

# Muestra resultados
print(f"🧠 Objetos detectados: {len(results[0].boxes)}")
for box in results[0].boxes:
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    label = model.names[cls]
    print(f"🔹 {label:<12} conf={conf:.2f} bbox=({x1}, {y1}, {x2}, {y2})")

# Guarda resultado visual
img_result = results[0].plot()
# Guardar resultado
cv2.imwrite("resultado_yolo8_simple.jpg", img_result)
print("📝 Imagen guardada como: resultado_yolo8_simple.jpg")
