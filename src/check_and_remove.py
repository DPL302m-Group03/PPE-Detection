import os

ann_dir = 'dataset/all/annotations'
img_dir = 'dataset/all/images'

img_exts = ('.jpg', '.jpeg', '.png')
ann_ext = '.txt'

num_annotations = len([f for f in os.listdir(ann_dir) if f.endswith('.txt')])
num_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Số lượng annotation: {num_annotations}")
print(f"Số lượng images: {num_images}")

# Lấy tên file (không phần mở rộng)
img_files = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
ann_files = set(os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith('.txt'))

missing_images = ann_files - img_files
missing_annotations = img_files - ann_files

print("Các annotation không có ảnh tương ứng:")
for name in sorted(missing_images):
    print(f"{name}.txt")

print("Các ảnh không có annotation tương ứng:")
for name in sorted(missing_annotations):
    print(f"{name}.jpg/.jpeg/.png")

img_exts = ('.jpg', '.jpeg', '.png')
ann_ext = '.txt'

# Tạo dict: {tên không đuôi: tên đầy đủ}
img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)}
ann_files = {os.path.splitext(f)[0]: f for f in os.listdir(ann_dir) if f.endswith(ann_ext)}

# Annotation không có ảnh
missing_images = set(ann_files.keys()) - set(img_files.keys())
for name in missing_images:
    path = os.path.join(ann_dir, ann_files[name])
    os.remove(path)
    print(f"Đã xoá annotation không có ảnh: {ann_files[name]}")

# Ảnh không có annotation
missing_annotations = set(img_files.keys()) - set(ann_files.keys())
for name in missing_annotations:
    path = os.path.join(img_dir, img_files[name])
    os.remove(path)
    print(f"Đã xoá ảnh không có annotation: {img_files[name]}")