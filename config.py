from decouple import config

# Rutas de los modelos
YOLOV8_PATH = config("YOLOV8_PATH", default="saved_models/yolov8l.pt")
MODEL_PATH = config("MODEL_PATH", default="saved_models/AntiSpoofing_bin_1.5_128.onnx")
YOLO5FACE_PATH = config("YOLO5FACE_PATH", default="saved_models/yolov5s-face.onnx")

# Umbrales para predicción
IOU_THRESHOLD = config("IOU_THRESHOLD", cast=float, default=0.3)
SCORE_THRESHOLD = config("SCORE_THRESHOLD", cast=float, default=0.7)

# Otros valores reutilizables (opcional)
TEMP_DIR = config("TEMP_DIR", default="temp")

#Extensiones permitidas
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
