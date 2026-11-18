import numpy as np

def iou(box_a, box_b):
    """Calculate intersection over union of two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def mean_average_precision(preds, gts, iou_thresh=0.5):
    """Calculate mean AP using all-point interpolated precision."""
    classes = sorted(set(gt[0] for gt in gts))

    if not classes:
        return 0.0

    aps = []

    for cls in classes:
        # Ground-truth boxes for this class.
        gt_boxes = [gt[1] for gt in gts if gt[0] == cls]
        n_gt = len(gt_boxes)

        # Predictions for this class, sorted by confidence.
        cls_preds = [
            (score, box)
            for class_id, score, box in preds
            if class_id == cls
        ]
        cls_preds.sort(key=lambda item: item[0], reverse=True)

        if not cls_preds:
            aps.append(0.0)
            continue

        used = [False] * n_gt
        tp = []
        fp = []

        # Greedy one-to-one matching.
        for score, pred_box in cls_preds:
            best_iou = -1.0
            best_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if used[j]:
                    continue

                overlap = iou(pred_box, gt_box)

                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = j

            if best_idx >= 0 and best_iou >= iou_thresh:
                used[best_idx] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        tp = np.asarray(tp)
        fp = np.asarray(fp)

        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)

        recall = cumulative_tp / n_gt
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, 1e-12
        )

        # Monotonic precision envelope.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # All-point interpolated AP.
        previous_recall = np.concatenate(([0.0], recall[:-1]))
        ap = np.sum(
            (recall - previous_recall) * precision
        )

        aps.append(float(ap))

    return round(float(np.mean(aps)), 4)