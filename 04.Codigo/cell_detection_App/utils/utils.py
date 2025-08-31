import streamlit as st
import cv2
import io
import os
import numpy as np
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix


st.set_page_config(
    page_title="🔬 Detector de Células", layout="wide", initial_sidebar_state="expanded"
)


# Cache del modelo para mejorar rendimiento
@st.cache_resource
def load_model(model_name):
    """Carga el modelo YOLO seleccionado"""
    model_paths = {
        "YOLOv12s": "models/final_model_yolov12s/weights/best.pt",
        "YOLOv12l": "models/final_model_yolov12l/weights/best.pt",
        "YOLOv11s": "models/final_model_yolov11s/weights/best.pt",
        "YOLOv10s": "models/final_model_yolov10s/weights/best.pt",
        "YOLOv9s": "models/final_model_yolov9s/weights/best.pt",
        "YOLOv8s": "models/final_model_yolov8s/weights/best.pt",
        "Custom": "models/final_model_custom/weights/best.pt",
    }
    model_path = model_paths[model_name]
    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró el modelo en: {model_path}")
        st.stop()
    return YOLO(model_path)


def process_image_YOLO(image, model, confidence_threshold=0.5):
    """Procesa una imagen con YOLO y devuelve los resultados"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = model(img_array, conf=confidence_threshold)

        processed_results = []
        for r in results:
            # Crear una copia de la imagen original en lugar de usar r.plot()
            im_array = img_array.copy()
            
            num_detections = len(r.boxes) if r.boxes is not None else 0
            detection_info = []
            
            # Color azul turquesa en formato BGR
            turquoise_color = (209, 156, 0)  # Esto es turquesa en BGR
            
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Dibujar el rectángulo con color turquesa
                    cv2.rectangle(im_array, (x1, y1), (x2, y2), turquoise_color, 2)
                    
                    # Añadir el texto con la probabilidad
                    label = f"{conf:.2f}"
                    text_y = max(y1 - 0, 0)
                    
                    # Fondo para el texto con el mismo color turquesa
                    (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(im_array, (x1, text_y - text_height), 
                                 (x1 + text_width, text_y), turquoise_color, -1)
                    
                    # Texto en negro sobre fondo turquesa
                    cv2.putText(im_array, label, (x1, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    detection_info.append(
                        {
                            "id": i + 1,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "area": int((x2 - x1) * (y2 - y1)),
                        }
                    )

            # Convertir de BGR a RGB para mostrar en Streamlit
            im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            processed_results.append(
                {
                    "image": im_array,
                    "num_detections": num_detections,
                    "detection_info": detection_info,
                }
            )

        return processed_results

    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {str(e)}")
        return None


def read_voc_xml(xml_file):
    """Lee un archivo xml en formato Pascal VOC y devuelve las bounding box Ground Truth"""
    gt_boxes = []
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            x1 = int(float(bbox.find("xmin").text))
            y1 = int(float(bbox.find("ymin").text))
            x2 = int(float(bbox.find("xmax").text))
            y2 = int(float(bbox.find("ymax").text))
            gt_boxes.append([x1, y1, x2, y2])
    except Exception as e:
        st.warning(f"No se puede leer el xml: {e}")
    return gt_boxes


def draw_gt_boxes(image_array, gt_boxes, color=(255, 0, 0)):
    """Se dibujan las bounding boxes sobre la imagen"""
    img = image_array.copy()
    num_boxes = 0
    for box in gt_boxes:
        num_boxes += 1
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img, num_boxes


def sidebar_config(device):
    st.sidebar.header("⚙️ Configuración")

    model_options = ["YOLOv12s", "YOLOv12l", "YOLOv11s", "YOLOv10s", "YOLOv9s", "YOLOv8s", "Custom"]
    selected_model = st.sidebar.selectbox(
        "Selecciona el modelo", model_options, index=0
    )

    model = load_model(selected_model)
    model = model.to(device)

    confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Ajusta la sensibilidad del detector",
    )

    enable_zoom = st.sidebar.checkbox("🔍 Zoom", value=False)
    enable_gt = st.sidebar.checkbox("Ground Truth", value=False)

    st.sidebar.info(f"**Dispositivo**: {device}")

    return model, confidence_threshold, enable_zoom, enable_gt


def save_pascal_voc_xml_to_buffer(image_filename, image_size, detections, folder="original_images", path="", database="Unknown"):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = image_filename
    ET.SubElement(annotation, "path").text = path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = database

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_size[0])
    ET.SubElement(size, "height").text = str(image_size[1])
    ET.SubElement(size, "depth").text = str(image_size[2])

    ET.SubElement(annotation, "segmented").text = "0"

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "cell"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(det.get("truncated", 0))
        ET.SubElement(obj, "difficult").text = str(det.get("difficult", 0))
        bbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = det["bbox"]
        ET.SubElement(bbox, "xmin").text = str(x1)
        ET.SubElement(bbox, "ymin").text = str(y1)
        ET.SubElement(bbox, "xmax").text = str(x2)
        ET.SubElement(bbox, "ymax").text = str(y2)

    xml_buffer = io.BytesIO()
    tree = ET.ElementTree(annotation)
    tree.write(xml_buffer, encoding="utf-8", xml_declaration=True)
    return xml_buffer.getvalue()


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou

    gt_matched = set()
    pred_matched = set()
    ious = []

    for i, gt in enumerate(gt_boxes):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(pred_boxes):
            if j in pred_matched:
                continue
            iou_val = iou(gt, pred)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= iou_threshold:
            gt_matched.add(i)
            pred_matched.add(best_j)
            ious.append(best_iou)

    TP = len(gt_matched)
    FP = len(pred_boxes) - len(pred_matched)
    FN = len(gt_boxes) - len(gt_matched)
    mean_iou = np.mean(ious) if ious else 0
    return TP, FP, FN, mean_iou


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


def compute_confusion_matrix(gt_list, pred_list, iou_threshold=0.5):
    y_true = []
    y_pred = []
    for gt_boxes, pred_boxes in zip(gt_list, pred_list):
        matched_gt = set()
        matched_pred = set()
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(pred, gt)
                if iou >= iou_threshold and j not in matched_gt and i not in matched_pred:
                    matched_gt.add(j)
                    matched_pred.add(i)
        # GT: 1 = célula, 0 = fondo
        for j in range(len(gt_boxes)):
            y_true.append(1)
            y_pred.append(1 if j in matched_gt else 0)
        for i in range(len(pred_boxes)):
            if i not in matched_pred:
                y_true.append(0)
                y_pred.append(1)
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    return cm

# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
# from torchvision.transforms import functional as F

# def load_fastrcnn_model(weights_path, num_classes=2):
#     model = fasterrcnn_resnet50_fpn_v2(weights=None)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     model.eval()
#     model.to(device)
#     return model

# def process_image_fastrcnn(image, model, confidence_threshold=0.5):
#     # Convierte PIL a tensor
#     img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(img_tensor)[0]
#     # Filtra por umbral de confianza
#     keep = outputs['scores'] > confidence_threshold
#     boxes = outputs['boxes'][keep].cpu().numpy()
#     scores = outputs['scores'][keep].cpu().numpy()
#     # Dibuja las cajas sobre la imagen
#     im_array = np.array(image).copy()
#     for box, score in zip(boxes, scores):
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(im_array, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(im_array, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#     im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
#     return [{
#         'image': im_array,
#         'num_detections': len(boxes),
#         'detection_info': [
#             {'id': i+1, 'confidence': float(s), 'bbox': list(map(int, b)), 'area': int((b[2]-b[0])*(b[3]-b[1]))}
#             for i, (b, s) in enumerate(zip(boxes, scores))
#         ]
#     }]
