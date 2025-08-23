import streamlit as st
import time
import numpy as np
import os
from PIL import Image
import torch
from streamlit_image_zoom import image_zoom
from utils import utils

import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    model, confidence_threshold, enable_zoom, enable_gt = utils.sidebar_config(device)

    tab_selected = st.radio("Selecciona vista", ["🔎 Detección individual", "📁 Carpeta completa"])

    if tab_selected == "🔎 Detección individual":
        
        st.markdown('<h1 class="main-header">🔬 Detector de Células</h1>', unsafe_allow_html=True)

        st.markdown("""
        ### Instrucciones:
        1. **Selecciona el modelo**
        2. **Sube una imagen** médica tipo
        3. **Ajusta el umbral** de confianza si es necesario
        4. **Visualiza los resultados** con detecciones automáticas
        """)

        col_upload1, col_upload2 = st.columns(2)

        with col_upload1:
            uploaded_image_file = st.file_uploader(
                "Sube una imagen",
                type=["png", "jpg", "jpeg"],
                help="Formatos soportados: PNG, JPG, JPEG",
            )

        with col_upload2:
            uploaded_xml_file = st.file_uploader(
                "Sube un archivo XML (opcional)",
                type=["xml"],
                help="Ground truth en formato Pascal VOC XML",
            )

        if uploaded_image_file is not None:
            col1_image, col2_image = st.columns(2)

            gt_boxes = []
            if uploaded_xml_file is not None and enable_gt:
                gt_boxes = utils.read_voc_xml(uploaded_xml_file)

            with col1_image:
                st.markdown("### Imagen Original")
                image = Image.open(uploaded_image_file)
                image_np = np.array(image)
                num_boxes = None
                if enable_gt:
                    image_np, num_boxes = utils.draw_gt_boxes(
                        image_np, gt_boxes, color=(0, 255, 0)
                    )
                if enable_zoom:
                    image_zoom(image_np)
                else:
                    st.image(image_np, use_container_width=True)

                mode_desc = {
                    "RGB": "Color (3 canales)",
                    "L": "Escala de grises (1 canal)",
                    "RGBA": "Color + transparencia (4 canales)",
                    "1": "Binaria",
                }

                modo_texto = mode_desc.get(image.mode, image.mode)

                st.markdown(
                    f"""
                **Numero de células redondas**: {num_boxes} <br>
                **Dimensiones**: {image.size[0]} x {image.size[1]} px <br> 
                **Formato**: {image.format} <br> 
                **Modo**: {modo_texto}
                """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with col2_image:
                st.markdown("### Resultados de Detección")

                with st.spinner("🔄 Procesando imagen..."):
                    start_time = time.time()
                    results = utils.process_image_YOLO(image, model, confidence_threshold)
                    process_time = time.time() - start_time

                    detection_info = results[0]["detection_info"]
                    image_filename = uploaded_image_file.name  
                    image_size = (image.width, image.height, 3) # o 1 si es grayscale
                    xml_bytes = utils.save_pascal_voc_xml_to_buffer(image_filename, image_size, detection_info)

                if results:
                    result = results[0]  # Tomar el primer resultado

                    if enable_zoom:
                        image_zoom(result["image"])
                    else:
                        st.image(result["image"], use_container_width=True)

                    # Métricas
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

                    with col_metrics1:
                        st.metric("🔍 Celulas Detectadas", result["num_detections"])

                    with col_metrics2:
                        st.metric("⏱️ Tiempo de Procesamiento", f"{process_time:.2f}s")

                    with col_metrics3:
                        st.metric("🎯 Umbral Usado", f"{confidence_threshold:.2f}")

                    # Tabla de detecciones detallada
                    if result["detection_info"]:

                        detection_data = []
                        for det in result["detection_info"]:
                            detection_data.append(
                                {
                                    "ID": det["id"],
                                    "Confianza": f"{det['confidence']:.3f}",
                                    "Área (px²)": det["area"],
                                    "Coordenadas": f"({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})",
                                }
                            )
                        with st.expander("Detalles de las detecciones"):
                            st.dataframe(detection_data, use_container_width=True)

                    st.download_button(
                        label="Descargar XML",
                        data=xml_bytes,
                        file_name=f"{image_filename.split('.')[0]}.xml",
                        mime="application/xml"
                    )

                else:
                    st.error("❌ No se pudo procesar la imagen")

    else:
        st.markdown('<h1 class="main-header">🔬 Detector de Células</h1>', unsafe_allow_html=True)
        st.markdown("""
        ### Instrucciones:
        1. Sube múltiples imágenes y sus archivos XML correspondientes.
        2. Asegúrate que los nombres de los archivos XML coincidan con los de las imágenes.
        3. Haz clic en 'Evaluar lote' para procesar todo el conjunto.
        """)

        col1, col2 = st.columns(2)
        with col1:
            uploaded_files = st.file_uploader(
                "Selecciona imágenes (puedes seleccionar varios archivos a la vez)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
        with col2:
            uploaded_files_xml = st.file_uploader(
                "Selecciona archivos XML (puedes seleccionar varios archivos a la vez)",
                type=["xml"],
                accept_multiple_files=True
            )

        if uploaded_files and uploaded_files_xml:
            # Crear diccionario de XML por nombre base
            xml_dict = {os.path.splitext(f.name)[0]: f for f in uploaded_files_xml}
            st.info(f"Imágenes subidas: {len(uploaded_files)} | XML subidos: {len(uploaded_files_xml)}")

            if st.button("Evaluar lote"):
                gt_list = []
                pred_list = []
                mean_ious = []
                xml_dict = {os.path.splitext(f.name)[0]: f for f in uploaded_files_xml}

                for img_file in uploaded_files:
                    image = Image.open(img_file)
                    base_name = os.path.splitext(img_file.name)[0]
                    xml_file = xml_dict.get(base_name, None)
                    gt_boxes = utils.read_voc_xml(xml_file) if xml_file else []
                    results = utils.process_image_YOLO(image, model, confidence_threshold)
                    detections = results[0]["detection_info"] if results[0]["detection_info"] else []
                    pred_boxes = [det["bbox"] for det in detections]
                    gt_list.append(gt_boxes)
                    pred_list.append(pred_boxes)
                    _, _, _, mean_iou = utils.match_boxes(gt_boxes, pred_boxes, iou_threshold=confidence_threshold)
                    mean_ious.append(mean_iou)

                cm = utils.compute_confusion_matrix(gt_list, pred_list, iou_threshold=confidence_threshold)
                TP = cm[0,0]
                FN = cm[0,1]
                FP = cm[1,0]

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                iou_promedio = np.mean(mean_ious) if mean_ious else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Precisión:** {precision:.3f}")
                    with st.expander("Matriz de confusión"):
                            fig, ax = plt.subplots(figsize=(2, 2))  # Tamaño pequeño
                            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                                        xticklabels=["Detección", "No detección"],
                                        yticklabels=["GT célula", "GT fondo"],
                                        linecolor='r', linewidths=0.5, ax=ax)
                            ax.set_xlabel("Predicción")
                            ax.set_ylabel("Ground Truth")
                            st.pyplot(fig)
                with col2:
                    st.markdown(f"**Recall:** {recall:.3f}")
                with col3:
                    st.markdown(f"**IoU promedio:** {iou_promedio:.3f}")

if __name__ == "__main__":
    main()