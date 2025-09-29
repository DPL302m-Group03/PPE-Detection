# Report Dataset PPE Detection

## [T1] Hoàn thiện tài liệu mô tả dataset, xác định loại dữ liệu cần sử dụng  
## [T2] Đề xuất & thống nhất phương pháp thu thập/tổng hợp dataset từ nhiều nguồn  

---

| **Dataset Folder**   | **[T1] Mô tả & Loại dữ liệu**                                                                                         | **[T2] Phương pháp thu thập/tổng hợp**                                                                 |
|-----------------------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **CHVG**             | Gồm ảnh và annotation định dạng `.txt` (YOLO format). Loại dữ liệu: ảnh giám sát công trình.                           | Thu thập từ nguồn CHVG, chuẩn hóa tên file `CHVG_0001.*`, đồng bộ ảnh và annotation.                     |
| **DeteksiAPD**       | Ảnh + annotation `.txt`. Loại dữ liệu: ảnh công nhân đội mũ bảo hộ (APD).                                               | Dataset open-source, rename thống nhất `DeteksiAPD_0001.*`, convert annotation sang `.txt` đồng bộ.      |
| **HardHatDetection** | Ảnh công nhân với mũ bảo hộ + bbox. Annotation dạng `.txt`.                                                            | Open dataset trên GitHub/Kaggle. Đã tiền xử lý, thống nhất cấu trúc folder.                             |
| **HarvardPPE**       | Ảnh PPE (helmet, vest) từ Harvard. Annotation `.txt`.                                                                 | Dữ liệu học thuật, đã được chuẩn hóa về cấu trúc `images/` – `annotations/`.                            |
| **HuggingFacePPE**   | Dữ liệu từ HuggingFace Hub. Loại: ảnh + txt labels.                                                                    | Import từ HuggingFace Datasets API, chuẩn hóa format annotation.                                        |
| **MendeleyPPE**      | Ảnh PPE lấy từ Mendeley Data. Annotation VOC → chuyển sang `.txt`.                                                     | Download từ Mendeley, convert XML → TXT theo YOLO format.                                               |
| **PPEKitDetection**  | Ảnh PPE kit (mũ, áo, găng tay, kính). Annotation dạng `.txt`.                                                          | Dataset tổng hợp PPE Kit, rename chuẩn `PPEKitDetection_0001.*`, unify annotation format.               |
| **SH17PPE**          | Ảnh từ dự án SH17 về nhận diện PPE. Annotation `.txt`.                                                                 | Tích hợp từ SH17 dataset, chuẩn hóa cấu trúc thư mục.                                                   |
| **SoDaContrucsion**  | Ảnh công trình xây dựng (PPE). Annotation VOC `.xml` → đã convert sang `.txt`.                                         | Crawl + public dataset. Thống nhất naming convention + convert format.                                  |

---

## Tổng kết
- **[T1]**: Dataset hiện gồm 9 nguồn, tất cả đã được chuẩn hóa về ảnh (`.jpg/.png`) và annotation (`.txt`). Các class chính: *helmet, vest, gloves, boots, mask*.  
- **[T2]**: Phương pháp thu thập & tổng hợp:  
  1. Lấy từ nhiều nguồn open dataset (GitHub, Kaggle, HuggingFace, Mendeley).  
  2. Chuẩn hóa **naming convention** theo tên folder + index (VD: `CHVG_00001.jpg`, `CHVG_00001.txt`).  
  3. Convert annotation về định dạng `.txt` thống nhất (YOLO).  
  4. Đồng bộ ảnh và annotation 1–1 (ảnh nào cũng có file label tương ứng).  
  5. Tạo pipeline tiền xử lý: rename, format conversion, kiểm tra tính nhất quán.  
