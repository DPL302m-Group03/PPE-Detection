import os
import shutil

root = "dataset"

for dataset in os.listdir(root):
    ann_dir = os.path.join(root, dataset, "annotations")
    if not os.path.isdir(ann_dir):
        continue

    for file in os.listdir(ann_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(ann_dir, file)
            txt_path = os.path.join(ann_dir, os.path.splitext(file)[0] + ".txt")

            # copy nội dung từ xml sang txt
            with open(xml_path, "r", encoding="utf-8") as f_xml, \
                 open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write(f_xml.read())

            os.remove(xml_path)  # xóa file xml gốc
