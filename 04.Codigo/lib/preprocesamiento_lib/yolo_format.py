import matplotlib.pyplot as plt
import pandas as pd
import random, shutil, cv2, os
import xml.etree.ElementTree as ET

class CPreprocessing_YOLO:
    def __init__(self):
        pass


    def read_annotations(self, xml_file):
        """
        Lee las anotaciones desde un archivo XML en formato PASCAL VOC
        
        Parameters:
        -----------
        xml_file : str
            Ruta al archivo XML con las anotaciones
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con las coordenadas de los bounding boxes
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            annotations = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    # Crear fila con formato
                    annotations.append({
                        'x1': xmin,
                        'y1': ymin, 
                        'x2': xmax,
                        'y2': ymax
                    })
            
            return pd.DataFrame(annotations)
            
        except ET.ParseError as e:
            print(f"Error al parsear XML {xml_file}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error procesando XML {xml_file}: {e}")
            return pd.DataFrame()
    
    
    def convert_to_yolo_format(self, image_shape, annotations):
        yolo_annotations = []
        img_height, img_width = image_shape[:2]
    
        for _, row in annotations.iterrows():
            # Ahora las coordenadas están en formato correcto (x1, y1, x2, y2)
            x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)
            yolo_annotations.append([x_center, y_center, width, height])
    
        return yolo_annotations
    
    
    def process_images(self, image_dir, xml_dir, yolo_dir):
        """
        Procesa las imágenes y sus anotaciones XML para generar archivos YOLO
        
        Parameters:
        -----------
        image_dir : str
            Directorio con las imágenes
        xml_dir : str
            Directorio con las anotaciones XML
        yolo_dir : str
            Directorio donde guardar las etiquetas YOLO
        """
        empty_xml_images = []
        contador_img_OK = 0
        contador_img_no_OK = 0
        
        for image_file in os.listdir(image_dir):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(image_dir, image_file)
                image = cv2.imread(image_path)
    
                xml_file = os.path.splitext(image_file)[0] + '.xml'
                xml_path = os.path.join(xml_dir, xml_file)
                
                if os.path.exists(xml_path):
                    annotations = self.read_annotations(xml_path)
                    if not annotations.empty:
                        yolo_annotations = self.convert_to_yolo_format(image.shape, annotations)
                        yolo_file = os.path.splitext(image_file)[0] + '.txt'
                        yolo_path = os.path.join(yolo_dir, yolo_file)
                        with open(yolo_path, 'w') as f:
                            for annotation in yolo_annotations:
                                f.write(f"0 {annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]}\n")
                        contador_img_OK += 1
                    else:
                        # print(f'El archivo XML está vacío para la imagen: {image_file}')
                        contador_img_no_OK += 1
                        empty_xml_images.append(image_file)
                else:
                    print(f'No se encontró el archivo XML para la imagen: {image_file}')
                    contador_img_no_OK += 1
                    empty_xml_images.append(image_file)
                    
        print(f"Tenemos {contador_img_OK} imágenes con etiquetas y {contador_img_no_OK} que no estan etiquetadas")
        return empty_xml_images
    
    
    def copy_images(self, src_dir, dst_dir, empty_xml_images):
        """
        Copia las imágenes que tienen anotaciones válidas
        
        Parameters:
        -----------
        src_dir : str
            Directorio fuente de las imágenes
        dst_dir : str
            Directorio destino
        empty_xml_images : list
            Lista de imágenes sin anotaciones válidas
        """
        os.makedirs(dst_dir, exist_ok=True)
        for file_name in os.listdir(src_dir):
            if file_name.endswith('.jpg') and file_name not in empty_xml_images:
                src_path = os.path.join(src_dir, file_name)
                dst_path = os.path.join(dst_dir, file_name)
                shutil.copyfile(src_path, dst_path)
    
    
    def split_dataset(self, images_path, labels_path, output_dir, train_ratio=0.8):
        if (os.path.exists(output_dir)):
            shutil.rmtree(output_dir)
            print(f"Directorio {output_dir} eliminado y será recreado")

        train_img_dir = os.path.join(output_dir, 'train', 'images')
        train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
        test_img_dir  = os.path.join(output_dir, 'val',  'images')
        test_lbl_dir  = os.path.join(output_dir, 'val',  'labels')
    
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(test_img_dir,  exist_ok=True)
        os.makedirs(test_lbl_dir,  exist_ok=True)
    
        images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        random.shuffle(images)
    
        train_count = int(len(images) * train_ratio)
        train_files = images[:train_count]
        test_files  = images[train_count:]
    
        def copy_file(src_folder, dst_folder, filename):
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
    
        for img_file in train_files:
            copy_file(images_path, train_img_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt')
            copy_file(labels_path, train_lbl_dir, label_file)
    
        for img_file in test_files:
            copy_file(images_path, test_img_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt')
            copy_file(labels_path, test_lbl_dir, label_file)
    
        print(f"\nConjunto split de train: {len(train_files)} imágenes")
        print(f"Conjunto split de val:  {len(test_files)} imágenes")
    
    
    def plot_image_with_boxes(self, image_path, label_path):
        """
        Plot an image with its corresponding YOLO bounding boxes
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, box_width, box_height = map(float, line.split())
                
                x_center = x_center * width
                y_center = y_center * height
                box_width = box_width * width
                box_height = box_height * height
                
                x1 = x_center - (box_width / 2)
                y1 = y_center - (box_height / 2)
                
                rect = plt.Rectangle(
                    (x1, y1),
                    box_width,
                    box_height,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
        
        plt.axis('off')
        plt.show()