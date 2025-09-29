import os 
import shutil

root = 'dataset'

raw_images = os.path.join(root, 'all', 'images')
raw_annotations = os.path.join(root, 'all', 'annotations')

img_exts = [".jpg", ".jpeg", ".png"]
ann_exts = [".txt"]

os.makedirs(raw_images, exist_ok=True)
os.makedirs(raw_annotations, exist_ok=True)

for dataset in os.listdir(root):
    dataset_path = os.path.join(root, dataset)
    if not os.path.isdir(dataset_path) or dataset == 'all':
        continue

    images_path = os.path.join(dataset_path, "images")
    ann_path = os.path.join(dataset_path, "annotations")

    # Copy images
    if os.path.exists(images_path):
        for img_file in os.listdir(images_path):
            ext = os.path.splitext(img_file)[1].lower()
            if ext in img_exts:
                src = os.path.join(images_path, img_file)
                dst = os.path.join(raw_images, f"{img_file}")
                shutil.copy2(src, dst)

    # Copy annotations
    if os.path.exists(ann_path):
        for ann_file in os.listdir(ann_path):
            ext = os.path.splitext(ann_file)[1].lower()
            if ext in ann_exts:
                src = os.path.join(ann_path, ann_file)
                dst = os.path.join(raw_annotations, f"{ann_file}")
                shutil.copy2(src, dst)