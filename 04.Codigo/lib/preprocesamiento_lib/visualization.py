import matplotlib.pyplot as plt
import collections
import cv2 
import os
import xml.etree.ElementTree as ET

def plot_distribution(values, labels, title, xlabel, ylabel, colors = None, figsize=(10,6)):
    """
    Gráfico de barras personalizado.

    Parameters:
    values : list
        Valores numéricos para cada barra.
    labels : list
        Etiquetas para cada barra.
    title : str
        Título del gráfico.
    xlabel : str
        Nombre del eje X.
    ylabel : str
        Nombre del eje Y.
    colors : list or None
        Lista de colores para las barras.
    figsize : tuple
        Tamaño de la figura.
    """
    if colors is None:
        colors = ['#1f77b4', "#00F7FF", "#a7f62f"]
    plt.figure(figsize=figsize)
    bars = plt.bar(labels, values, color=colors, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}',
            ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def draw_visual_boxes(base_path):

    dimension_count = collections.defaultdict(int)

    img_path = os.path.join(base_path, "original_images")
    xml_path = os.path.join(base_path, "annotations")  
    output_path = os.path.join(base_path, "visual_boxes")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]

    for image_file in image_files:
        image_number = image_file.replace('.jpg', '')
        img = cv2.imread(os.path.join(img_path, image_file))
        height, width = img.shape[:2]

        dimension_count[(width, height)] += 1

        xml_file = os.path.join(xml_path, f"{image_number}.xml")
        if os.path.exists(xml_file):
            try:
                # Parsear XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))
                        
                        xmin = max(0, min(xmin, width))
                        ymin = max(0, min(ymin, height))
                        xmax = max(0, min(xmax, width))
                        ymax = max(0, min(ymax, height))
                        
                        # Dibujar bounding box
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Guardar imagen resultante
                output_file = os.path.join(output_path, f"bbox_{image_file}")
                cv2.imwrite(output_file, img)
                
            except ET.ParseError as e:
                print(f"Error al parsear XML {xml_file}: {e}")
            except Exception as e:
                print(f"Error procesando {xml_file}: {e}")
        else:
            print(f"No se encontró XML para: {image_file}")

    total_images = sum(dimension_count.values())
    for dim, count in dimension_count.items():
        print(f"Dimension {dim}: {count} imágenes")
    print(f"Número total de imágenes: {total_images}")