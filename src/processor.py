def yolo_type(type):
    if type == 'n':
        d, w, r = 1/3, 1/4, 2.0
    elif type == 's':
        d, w, r = 1/3, 1/2, 2.0
    elif type == 'm':
        d, w, r = 2/3, 3/4, 1.5
    elif type == 'l':
        d, w, r = 1.0, 1.0, 1.0
    elif type == 'x':
        d, w, r = 1.0, 1.25, 1.0
    else:
        raise Exception("Unsupported model type")
    return d, w, r