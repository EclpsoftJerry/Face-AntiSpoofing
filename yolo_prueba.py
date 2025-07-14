import cv2
import numpy as np
import onnxruntime as ort
import time

# Ruta al modelo ONNX
model_path = "saved_models/yolov8l.onnx"  # Cambia si usas otro modelo .onnx
image_path = "prueba15.jpeg"       # Cambia al nombre de tu imagen

# Iniciar sesión ONNX
ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Leer imagen
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (640, 640))  # Asegúrate de usar el tamaño correcto
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0
img_input = np.transpose(img_normalized, (2, 0, 1))  # Canales primero
img_input = np.expand_dims(img_input, axis=0)        # Añadir batch dim

# Obtener nombre de entrada del modelo
input_name = ort_session.get_inputs()[0].name

# Iniciar temporizador
start_time = time.time()

# Ejecutar inferencia
outputs = ort_session.run(None, {input_name: img_input})

# Medir tiempo
end_time = time.time()
inference_time = (end_time - start_time) * 1000
print(f"⏱ Tiempo de inferencia ONNX: {inference_time:.2f} ms")

# Procesar salidas (formato YOLOv8)
output = outputs[0][0]  # Batch 1
boxes = []
for det in output:
    x1, y1, x2, y2, conf, cls = det[:6]
    if conf > 0.3:
        boxes.append((int(cls), float(conf), int(x1), int(y1), int(x2), int(y2)))

print(f"🧠 Objetos detectados: {len(boxes)}")
for cls, conf, x1, y1, x2, y2 in boxes:
    print(f"🔹 Clase {int(cls)} conf={conf:.2f} bbox=({x1}, {y1}, {x2}, {y2})")
    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_resized, f"{int(cls)}: {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Guardar resultado
cv2.imwrite("resultado_yolo8_onnx.jpg", img_resized)
print("📝 Imagen guardada como: resultado_yolo8_onnx.jpg")
