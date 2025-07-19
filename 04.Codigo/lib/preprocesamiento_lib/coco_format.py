import  cv2
import json
import os
from pathlib import Path

class CYoloToCoco:
    def __init__(self, classes):
        self.classes = classes
        self.categories = [{"id": i+1, "name": n} for i,n in enumerate(classes)]

    def convert_yolo_to_coco(self, img_dir, lbl_dir, out_json, dataset):
        """
        Convert annotations from YOLO format to COCO format.

        This method processes images and their corresponding YOLO format annotation
        files and converts them to a single COCO format JSON file. Each bounding box
        in YOLO format (class, x_center, y_center, width, height - normalized) is 
        converted to COCO format (x, y, width, height - in pixels).

        Parameters:
        -----------
        img_dir : str or Path
            Directory containing the image files (.jpg)
        lbl_dir : str or Path
            Directory containing the YOLO format annotation files (.txt)
        out_json : str or Path
            Path where the output COCO format JSON file will be saved
        """
        images, annots, ann_id = [], [], 1
        for img_id, fn in enumerate(sorted(os.listdir(img_dir)), 1):
            if not fn.endswith(".jpg"): continue
            path = os.path.join(img_dir, fn)
            h, w = cv2.imread(path).shape[:2]
            images.append({"id": img_id, "file_name": fn, "height": h, "width": w})
            txt = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
            if not os.path.exists(txt): continue
            for L in open(txt):
                cls, xc,yc,bw,bh = map(float, L.split())
                cx,cy,bw,bh = xc*w, yc*h, bw*w, bh*h
                x0,y0 = cx-bw/2, cy-bh/2
                annots.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": int(cls)+1,
                    "bbox": [round(x0,2), round(y0,2), round(bw,2), round(bh,2)],
                    "area": round(bw*bh,2), "iscrowd": 0
                })
                ann_id += 1

        coco = {"images": images, "annotations": annots, "categories": self.categories}
        Path(out_json).parent.mkdir(exist_ok=True, parents=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"→ {dataset}: {len(images)} imgs, {len(annots)} bbox")
        
        if (dataset == "train"):
            num_instances = len(annots)
        elif (dataset == "val"):
            num_instances = len(annots)
        elif (dataset == "test"):
            num_instances = len(annots)

        return num_instances