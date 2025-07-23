import streamlit as st
import cv2
import os
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from streamlit_image_zoom import image_zoom
import xml.etree.ElementTree as ET

device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(
    page_title="🔬 Detector de Células",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache del modelo para mejorar rendimiento
@st.cache_resource
def load_model_YOLO(model_name):
    """Carga el modelo YOLO seleccionado"""
    model_paths = {
        "YOLOv12s": "models/final_model_optunav12s/weights/best.pt",
        "YOLOv11s": "models/final_model_optunav11s/weights/best.pt",
        "YOLOv10n": "models/final_model_yolov10n/weights/best.pt",
        "YOLOv9s": "models/final_model_yolov9s/weights/best.pt"
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
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        results = model(img_array, conf=confidence_threshold)
        
        processed_results = []
        for r in results:
            im_array = r.plot(labels=False, conf=True)
            
            im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            
            num_detections = len(r.boxes) if r.boxes is not None else 0
            
            detection_info = []
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detection_info.append({
                        'id': i+1,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': int((x2-x1) * (y2-y1))
                    })
            
            processed_results.append({
                'image': im_array,
                'num_detections': num_detections,
                'detection_info': detection_info
            })
        
        return processed_results
        
    except Exception as e:
        st.error(f"❌ Error procesando la imagen: {str(e)}")
        return None
    

def read_voc_xml(xml_file):
    """Lee un archivo xml en formato Pascal VOC y devuelve las bounding box Ground Truth"""
    gt_boxes=[]
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x1 = int(float(bbox.find('xmin').text))
            y1 = int(float(bbox.find('ymin').text))  
            x2 = int(float(bbox.find('xmax').text))  
            y2 = int(float(bbox.find('ymax').text))  
            gt_boxes.append([x1, y1, x2, y2]) 
    except Exception as e:
        st.warning(f"No se puede leer el xml: {e}")
    return gt_boxes

    
def draw_gt_boxes(image_array, gt_boxes, color=(255,0,0)):
    """Se dibujan las bounding boxes sobre la imagen"""
    img = image_array.copy()
    num_boxes = 0
    for box in gt_boxes:
        num_boxes += 1
        x1,y1,x2,y2 = box
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    return img, num_boxes


def sidebar_config(device):
    st.sidebar.header("⚙️ Configuración")

    model_options = [
        "YOLOv12s",
        "YOLOv11s",
        "YOLOv10n",
        "YOLOv9s"
    ]
    selected_model = st.sidebar.selectbox(
        "Selecciona el modelo",
        model_options,
        index=0
    )
    
    model = load_model_YOLO(selected_model)
    model = model.to(device)
    
    confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Ajusta la sensibilidad del detector"
    )

    enable_zoom = st.sidebar.checkbox("🔍 Zoom", value=False)
    enable_gt = st.sidebar.checkbox("Ground Truth", value=True)
    
    st.sidebar.info(f"**Dispositivo**: {device}")

    return model, confidence_threshold, enable_zoom, enable_gt


def main():
    st.markdown('<h1 class="main-header">🔬 Detector de Células</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Instrucciones:
    1. **Selecciona el modelo**
    2. **Sube una imagen** médica tipo
    3. **Ajusta el umbral** de confianza si es necesario
    4. **Visualiza los resultados** con detecciones automáticas
    """)
    
    # Barra lateral de configuración
    model, confidence_threshold, enable_zoom, enable_gt = sidebar_config(device)
    
    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        uploaded_image_file = st.file_uploader(
            "Sube una imagen",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos soportados: PNG, JPG, JPEG"
        )

    with col_upload2:
        uploaded_xml_file = st.file_uploader(
            "Sube un archivo XML (opcional)",
            type=['xml'],
            help="Ground truth en formato Pascal VOC XML"
        )

    if uploaded_image_file is not None:
        col1_image, col2_image = st.columns(2)

        gt_boxes = []
        if uploaded_xml_file is not None and enable_gt:
           gt_boxes = read_voc_xml(uploaded_xml_file)
        
        with col1_image:
            st.markdown("### Imagen Original")
            image = Image.open(uploaded_image_file)
            image_np = np.array(image)
            num_boxes = None
            if enable_gt:
                image_np, num_boxes = draw_gt_boxes(image_np, gt_boxes, color=(0,255,0))
            if enable_zoom:
                image_zoom(image_np)
            else: 
                st.image(image_np, use_container_width=True)
            
            st.markdown(f"""
            **Numero de células redondas**: {num_boxes} <br>
            **Dimensiones**: {image.size[0]} x {image.size[1]} px <br> 
            **Formato**: {image.format} <br> 
            **Modo**: {image.mode}
            """,
            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_image:
            st.markdown("### Resultados de Detección")
            
            with st.spinner("🔄 Procesando imagen..."):
                start_time = time.time()
                results = process_image_YOLO(image, model, confidence_threshold)
                process_time = time.time() - start_time
            
            if results:
                result = results[0]  # Tomar el primer resultado
                
                if enable_zoom:
                    image_zoom(result['image'])
                else:
                    st.image(result['image'], use_container_width=True)
                
                # Métricas
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                
                with col_metrics1:
                    st.metric("🔍 Celulas Detectadas", result['num_detections'])
                
                with col_metrics2:
                    st.metric("⏱️ Tiempo de Procesamiento", f"{process_time:.2f}s")
                
                with col_metrics3:
                    st.metric("🎯 Umbral Usado", f"{confidence_threshold:.2f}")
                
                # Tabla de detecciones detallada
                if result['detection_info']:
                    st.markdown("### Detalle de Detecciones")
                    
                    detection_data = []
                    for det in result['detection_info']:
                        detection_data.append({
                            'ID': det['id'],
                            'Confianza': f"{det['confidence']:.3f}",
                            'Área (px²)': det['area'],
                            'Coordenadas': f"({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})"
                        })
                    
                    st.dataframe(detection_data, use_container_width=True)
                
            else:
                st.error("❌ No se pudo procesar la imagen")


if __name__ == "__main__":
    main()


# Me gustaría añadir una checkbox para activar o desactivar las predicciones sobre la imagen y los groung truth
# comparativa de modelos 




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