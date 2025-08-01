from ultralytics import YOLO

# Cargar modelo YOLOv8 en formato .pt
model = YOLO("saved_models/yolov8l.pt")

# Exportar a formato ONNX
model.export(format="onnx", opset=12, dynamic=True)

print("✅ Conversión exitosa: yolov8l.onnx generado")
