def inside(box_a, box_b, threshold=0.2):
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