import os
import cv2
import numpy as np
import argparse
import csv
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox: np.ndarray, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)
    w, h = x2 - x1, y2 - y1
    l = max(w, h)
    xc, yc = x1 + w / 2, y1 + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1_crop = max(0, x)
    y1_crop = max(0, y)
    x2_crop = min(real_w, x + int(l * bbox_inc))
    y2_crop = min(real_h, y + int(l * bbox_inc))
    img_cropped = img[y1_crop:y2_crop, x1_crop:x2_crop, :]
    border_top = y1_crop - y
    border_bottom = int(l * bbox_inc) - (y2_crop - y)
    border_left = x1_crop - x
    border_right = int(l * bbox_inc) - (x2_crop - x)
    img_cropped = cv2.copyMakeBorder(img_cropped,
                                     border_top, border_bottom,
                                     border_left, border_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img_cropped

def make_prediction(img_rgb, face_detector, anti_spoof, threshold):
    bboxes = face_detector([img_rgb])[0]
    results = []
    if bboxes.shape[0] == 0:
        return results

    for bbox in bboxes:
        crop = increased_crop(img_rgb, bbox)
        pred = anti_spoof([crop])[0]
        score = pred[0][0]
        label = np.argmax(pred)

        x1, y1, x2, y2 = bbox[:4].astype(int)
        if label == 0:
            if score > threshold:
                result_text = f"REAL {score:.2f}"
                label_str = "REAL"
                color = COLOR_REAL
            else:
                result_text = f"UNKNOWN {score:.2f}"
                label_str = "UNKNOWN"
                color = COLOR_UNKNOWN
        else:
            result_text = f"FAKE {score:.2f}"
            label_str = "FAKE"
            color = COLOR_FAKE

        results.append({
            "bbox": (x1, y1, x2, y2),
            "label": label_str,
            "score": score,
            "color": color,
            "text": result_text
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spoof detection for folder of images")
    parser.add_argument("-i", "--input_folder", required=True, help="Folder with images")
    parser.add_argument("-o", "--output_folder", required=True, help="Folder to save annotated images")
    parser.add_argument("-m", "--model_path", required=True, help="Path to ONNX model")
    parser.add_argument("-t", "--threshold", type=float, default=0.75, help="Threshold for classification")
    parser.add_argument("--csv", type=str, default="resultados.csv", help="CSV file to store results")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    face_detector = YOLOv5("saved_models/yolov5s-face.onnx")
    anti_spoof = AntiSpoof(args.model_path)

    with open(args.csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Label", "Score"])

        for filename in os.listdir(args.input_folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(args.input_folder, filename)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            preds = make_prediction(img_rgb, face_detector, anti_spoof, args.threshold)

            if not preds:
                print(f"❌ No face in: {filename}")
                writer.writerow([filename, "NO_FACE", "0.00"])
                cv2.imwrite(os.path.join(args.output_folder, filename), img)
                continue

            for pred in preds:
                x1, y1, x2, y2 = pred["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), pred["color"], 2)
                cv2.putText(img, pred["text"], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred["color"], 2)
                writer.writerow([filename, pred["label"], f"{pred['score']:.2f}"])

            cv2.imwrite(os.path.join(args.output_folder, filename), img)
            print(f"✅ {filename} → {preds[0]['text']}")

    print("📁 Proceso terminado. Revisa el archivo CSV:", args.csv)
