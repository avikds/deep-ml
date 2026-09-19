import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    try:
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
    except Exception:
        return -1

    if not np.isscalar(iou_threshold) or not (0 <= iou_threshold <= 1):
        return -1

    # Handle empty inputs before shape validation.
    if boxes.size == 0 and scores.size == 0:
        return []

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return -1

    if scores.ndim != 1 or len(boxes) != len(scores):
        return -1

    try:
        boxes = boxes.astype(float)
        scores = scores.astype(float)
    except Exception:
        return -1

    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
        return -1

    n = len(boxes)
    if n == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    widths = np.maximum(0.0, x2 - x1)
    heights = np.maximum(0.0, y2 - y1)
    areas = widths * heights

    # Descending score order; stable tie-breaking by index.
    order = np.argsort(-scores, kind="stable")

    keep = []

    while len(order) > 0:
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter

        iou = np.zeros_like(inter, dtype=float)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]

        order = rest[iou <= iou_threshold]

    return keep