import os

root = "dataset"  # thư mục chứa các dataset con

for dataset in os.listdir(root):
    dataset_path = os.path.join(root, dataset)
    img_dir = os.path.join(dataset_path, "images")
    ann_dir = os.path.join(dataset_path, "annotations")

    if not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
        continue

    # lấy list ảnh (sort để thứ tự cố định)
    img_files = sorted(os.listdir(img_dir))
    
    for idx, img_file in enumerate(img_files, start=1):
        base_name = f"{dataset}_{idx:05d}"  # ví dụ CHVG_00001
        ext = os.path.splitext(img_file)[1]  # giữ nguyên đuôi ảnh (.png, .jpg, ...)
        
        # rename ảnh
        old_img_path = os.path.join(img_dir, img_file)
        new_img_path = os.path.join(img_dir, base_name + ext)
        os.rename(old_img_path, new_img_path)

        # rename annotation (giả sử annotation trùng tên với ảnh ban đầu)
        old_ann_name = os.path.splitext(img_file)[0]
        old_ann_path = os.path.join(ann_dir, old_ann_name + ".txt")
        if os.path.exists(old_ann_path):
            new_ann_path = os.path.join(ann_dir, base_name + ".txt")
            os.rename(old_ann_path, new_ann_path)
