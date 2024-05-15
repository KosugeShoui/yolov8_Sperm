# yolo_to_coco.py
import os
import json
import torch
from PIL import Image


def convert_yolo_to_coco(yolo_folder, output_json_path):
    # COCOフォーマットのJSONデータを格納する辞書
    coco_data = {"images": [], "annotations": [], "categories": []}
    annotation_id = 1  # アノテーションのIDを管理する変数

    image_files = [img for img in os.listdir(yolo_folder) if img.endswith(('.jpg', '.png'))]
    
    for i, image_file in enumerate(image_files):
        image_file_path = os.path.join(yolo_folder, image_file)
        txt_file_path = image_file_path.rsplit('.', 1)[0] + '.txt'

        image = Image.open(image_file_path)
        image_width, image_height = image.size
        coco_data["images"].append({
            "id": i + 1,
            "width": image_width,
            "height": image_height,
            "file_name": os.path.basename(image_file_path),
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None
        })

        # ラベルファイルが存在し、かつ空でない場合のみアノテーション情報を追加
        if os.path.exists(txt_file_path) and os.path.getsize(txt_file_path) > 0:
            with open(txt_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if not line:  # 空行をスキップ
                        continue
                    category_id = int(line[0]) + 1
                    bbox = list(map(float, line[1:]))
                    x, y, w, h = bbox
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": i + 1,
                        "category_id": category_id,
                        "bbox": [(x - w/2) * image_width, (y - h/2) * image_height, w * image_width, h * image_height],
                        "area": w * image_width * h * image_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1  # アノテーションが追加されるたびにIDをインクリメント

    coco_data["categories"].append({"id": 1, "name": "sperm", "supercategory": "object"})
    coco_data["categories"].append({"id": 2, "name": "pin-head", "supercategory": "object"})
    coco_data["categories"].append({"id": 3, "name": "cluster", "supercategory": "object"})
    
    if not os.path.exists(os.path.dirname(output_json_path)):
        os.makedirs(os.path.dirname(output_json_path))
    
    with open(output_json_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

if __name__ == "__main__":
    # trainデータを変換
    convert_yolo_to_coco("./visem_detr/train", "./visem_detr/annotations/detection_train_sperm.json")
    # valデータを変換
    convert_yolo_to_coco("./visem_detr/val", "./visem_detr/annotations/detection_val_sperm.json")


