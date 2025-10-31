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

def get_color(label):
    COLORS = {
        'helmet': (255, 225, 25),
        'vest': (0, 130, 200),
        'gloves': (245, 130, 48),
        'boots': (145, 30, 180),
        'no_helmet': (230, 25, 75),
        'no_vest': (128, 128, 128),
        'no_gloves': (70, 240, 240),
        'no_boots': (210, 245, 60),
    }

    return COLORS[label]

