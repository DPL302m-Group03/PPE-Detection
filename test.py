import cv2
from ultralytics import YOLO

# === Cấu hình các lớp ===
LABELS = {
    0: 'worker',
    1: 'helmet',
    2: 'vest',
    3: 'gloves',
    4: 'boots',
    5: 'no_helmet',
    6: 'no_vest',
    7: 'no_gloves',
    8: 'no_boots'
}

# === Các label bảo hộ mà bạn muốn kiểm tra (tùy chọn) ===
REQUIRED_ITEMS = ['helmet']

# === Hàm kiểm tra xem box A có nằm trong box B không ===
def inside(box_a, box_b, threshold=0.1):
    """Trả về True nếu box_a nằm trong box_b (theo phần giao diện tích)."""
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b

    # Kiểm tra phần giao nhau
    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box_a_area = (x2a - x1a) * (y2a - y1a)
    return inter_area / box_a_area > threshold


# === Load YOLO model ===
model = YOLO("weights/ppe-8m.pt")  # thay bằng model của bạn

# === Mở video (hoặc webcam: cv2.VideoCapture(0)) ===
cap = cv2.VideoCapture("videos\hardhat.mp4")

# (Tùy chọn) Lưu kết quả video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Chạy YOLO detect trên frame ===
    results = model(frame, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    workers = []
    items = []

    # Gom nhóm worker & items
    for box, cls_id in zip(boxes, class_ids):
        label = names[cls_id]
        if label == 'worker':
            workers.append({'box': box, 'items': set()})
        elif label in LABELS.values():
            items.append({'box': box, 'label': label})

    # Kiểm tra trang bị của từng worker
    for worker in workers:
        wbox = worker['box']
        for item in items:
            if item['label'] in REQUIRED_ITEMS and inside(item['box'], wbox):
                worker['items'].add(item['label'])

    # Vẽ lên frame
    for worker in workers:
        x1, y1, x2, y2 = map(int, worker['box'])
        has_all = all(req in worker['items'] for req in REQUIRED_ITEMS)
        color = (0, 255, 0) if has_all else (0, 0, 255)
        label_text = f"Worker ({'Safe' if has_all else 'Unsafe'})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === Khởi tạo writer nếu cần lưu video ===
    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("output_checked.mp4", fourcc, 20.0, (w, h))

    out.write(frame)  # ghi ra file
    cv2.imshow("PPE Detection", frame)

    # Bấm Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Giải phóng tài nguyên ===
cap.release()
if out: out.release()
cv2.destroyAllWindows()

print("✅ Done! Video kết quả đã lưu tại output_checked.mp4")
