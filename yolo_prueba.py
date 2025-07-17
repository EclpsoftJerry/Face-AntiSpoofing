import mediapipe as mp
import cv2

mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializar el detector de objetos con el modelo EfficientDet-Lite0
with mp_object_detection.ObjectDetector(model_selection=0, min_detection_confidence=0.5) as detector:
  # Cargar una imagen
  image = cv2.imread("prueba1.jpg")

  # Convertir la imagen a formato RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Realizar la detección de objetos
  results = detector.process(image_rgb)

  # Si se detectaron objetos
  if results.detections:
    for detection in results.detections:
      # Dibujar los resultados en la imagen
      mp_drawing.draw_detection(image, detection)

  # Mostrar la imagen con los resultados
  cv2.imshow("Detección de Objetos", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()