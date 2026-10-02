import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.
    """
    anchors = np.asarray(anchors, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)

    N = anchors.shape[0]
    M = gt_boxes.shape[0]

    if N == 0:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int)
        )

    if M == 0:
        return (
            np.zeros(N, dtype=int),
            np.full(N, -1, dtype=int)
        )

    # Pairwise intersection.
    x1 = np.maximum(anchors[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(anchors[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(anchors[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(anchors[:, None, 3], gt_boxes[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    # Zero-area boxes have IoU 0.
    anchor_w = np.maximum(0.0, anchors[:, 2] - anchors[:, 0])
    anchor_h = np.maximum(0.0, anchors[:, 3] - anchors[:, 1])
    anchor_area = anchor_w * anchor_h

    gt_w = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0])
    gt_h = np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    gt_area = gt_w * gt_h

    union = anchor_area[:, None] + gt_area[None, :] - inter

    iou = np.zeros((N, M), dtype=float)
    valid = union > 0
    iou[valid] = inter[valid] / union[valid]

    # Best GT for each anchor.
    best_gt = np.argmax(iou, axis=1)
    best_iou = iou[np.arange(N), best_gt]

    labels = np.full(N, -1, dtype=int)
    matched_gt = np.full(N, -1, dtype=int)

    # Threshold-based assignment.
    pos = best_iou >= pos_threshold
    neg = best_iou < neg_threshold

    labels[neg] = 0
    labels[pos] = 1
    matched_gt[pos] = best_gt[pos]

    # Force every GT to have a positive anchor.
    # Iterating in GT order means later GTs overwrite earlier ones
    # when they choose the same anchor.
    for j in range(M):
        anchor_idx = np.argmax(iou[:, j])
        labels[anchor_idx] = 1
        matched_gt[anchor_idx] = j

    return labels, matched_gt