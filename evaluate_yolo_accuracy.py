import os
import glob
import cv2
import numpy as np
from collections import defaultdict

# ============== YOLO SETTINGS ==============
YOLO_DIR = "yolo"
CFG_PATH = os.path.join(YOLO_DIR, "yolov3-tiny.cfg")
WEIGHTS_PATH = os.path.join(YOLO_DIR, "yolov3-tiny.weights")
NAMES_PATH = os.path.join(YOLO_DIR, "coco.names")

YOLO_INPUT_SIZE = (320, 320)

# 🔥 STRICter confidence = fewer FPs, more precision
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

TARGET_CLASS_NAME = "person"

# Folder where your images AND .txt labels are
IMG_DIR = "captured_frames"   # <-- set this to your actual folder

VERBOSE_IOU_LOGS = False
# ==========================================


def load_yolo():
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    out_layers = net.getUnconnectedOutLayersNames()
    return net, classes, out_layers


def run_yolo(net, out_layers, image):
    H, W = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, YOLO_INPUT_SIZE,
        swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    confidences = []
    class_ids = []

    # min box area filter (ignore tiny detections)
    min_area = 0.01 * W * H  # 1% of image area – tune if needed

    for output in outputs:
        for d in output:
            scores = d[5:]
            class_id = int(np.argmax(scores))
            conf = scores[class_id]

            if conf <= CONF_THRESHOLD:
                continue

            cx, cy, w, h = d[0:4] * np.array([W, H, W, H])
            x = int(cx - w / 2)
            y = int(cy - h / 2)

            # Filter tiny boxes
            box_area = w * h
            if box_area < min_area:
                continue

            # Aspect ratio filter: person should be taller than wide (approx)
            aspect_ratio = h / float(w + 1e-6)  # height / width
            if aspect_ratio < 0.8:
                continue

            # Optional: ignore detections too high in the frame (like posters, etc.)
            # y_center = cy
            # if y_center < H * 0.3:  # ignore top 30% of image
            #     continue

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(conf))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            detections.append((x, y, x + w, y + h, class_ids[i], confidences[i]))

    return detections


def load_gt(label_path, W, H):
    """
    Load YOLO txt labels and convert to pixel coords:
    returns list of (x1, y1, x2, y2, class_id)
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                continue

            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)

            gt_boxes.append((x1, y1, x2, y2, cid))
    return gt_boxes


def IoU(A, B):
    x1 = max(A[0], B[0])
    y1 = max(A[1], B[1])
    x2 = min(A[2], B[2])
    y2 = min(A[3], B[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    areaA = max(0, A[2] - A[0]) * max(0, A[3] - A[1])
    areaB = max(0, B[2] - B[0]) * max(0, B[3] - B[1])
    union = areaA + areaB - inter

    if union <= 0:
        return 0.0
    return inter / union


def main():
    # Load YOLO + classes
    net, classes, out_layers = load_yolo()
    if TARGET_CLASS_NAME not in classes:
        print(f"[ERROR] '{TARGET_CLASS_NAME}' not found in coco.names!")
        return
    target_id = classes.index(TARGET_CLASS_NAME)

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not img_paths:
        print(f"[ERROR] No images found in {IMG_DIR}")
        return

    print(f"[INFO] Found {len(img_paths)} images in {IMG_DIR}")
    print(f"[INFO] Evaluating YOLO class '{TARGET_CLASS_NAME}' (ID={target_id})")
    print(f"[INFO] CONF_THRESHOLD = {CONF_THRESHOLD}")

    # ---------- FIRST PASS: GT sanity check ----------
    total_gt_all = 0
    class_hist = defaultdict(int)
    label_files_found = set()

    for img_path in img_paths:
        name = os.path.basename(img_path).rsplit(".", 1)[0]
        label_path = os.path.join(IMG_DIR, f"{name}.txt")
        if os.path.exists(label_path):
            label_files_found.add(label_path)

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_all = load_gt(label_path, W, H)
        total_gt_all += len(gt_all)
        for _, _, _, _, cid in gt_all:
            class_hist[cid] += 1

    print("\n===== GROUND TRUTH SANITY CHECK =====")
    print(f"Total label files found: {len(label_files_found)}")
    print(f"Total GT boxes (all classes): {total_gt_all}")
    if class_hist:
        print("GT class id histogram (cid: count):")
        for cid, cnt in sorted(class_hist.items()):
            print(f"  {cid}: {cnt}")
    else:
        print("No valid GT boxes parsed from any label file.")
        return

    if class_hist.get(target_id, 0) == 0:
        print(f"[WARNING] No GT boxes with class_id={target_id} ({TARGET_CLASS_NAME}).")
        print("Metrics for this class will be zero.")
    # --------------------------------------------------

    # ---------- SECOND PASS: EVALUATION ----------
    total_TP = total_FP = total_FN = 0

    for img_path in img_paths:
        name = os.path.basename(img_path).rsplit(".", 1)[0]
        label_path = os.path.join(IMG_DIR, f"{name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue
        H, W = img.shape[:2]

        gt_all = load_gt(label_path, W, H)
        gt = [g for g in gt_all if g[4] == target_id]

        preds_all = run_yolo(net, out_layers, img)
        preds = [p for p in preds_all if p[4] == target_id]

        matched_gt = set()
        matched_pred = set()
        best_iou_for_pred = [0.0] * len(preds)

        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gt):
                if gi in matched_gt:
                    continue
                iou = IoU(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            best_iou_for_pred[pi] = best_iou

            if best_iou >= IOU_THRESHOLD and best_gi != -1:
                matched_pred.add(pi)
                matched_gt.add(best_gi)

        TP = len(matched_pred)
        FP = len(preds) - TP
        FN = len(gt) - TP

        total_TP += TP
        total_FP += FP
        total_FN += FN

        print(f"{name}: TP={TP}, FP={FP}, FN={FN}")

        if VERBOSE_IOU_LOGS:
            for pi, p in enumerate(preds):
                status = "TP" if pi in matched_pred else "FP"
                print(f"  pred#{pi}: best IoU={best_iou_for_pred[pi]:.2f} -> {status}")
            for gi, g in enumerate(gt):
                if gi not in matched_gt:
                    print(f"  gt#{gi}: FN (unmatched {TARGET_CLASS_NAME})")

    # ---------- METRICS ----------
    precision = total_FP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_FP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n===== FINAL METRICS (PERSON ONLY) =====")
    print(f"TP: {total_TP}")
    print(f"FP: {total_FP}")
    print(f"FN: {total_FN}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1:.3f}")

    TN = 0
    print("\n===== CONFUSION MATRIX (PERSON) =====")
    print("            Pred:Person   Pred:Background")
    print(f"GT:Person    {total_TP:8d}       {total_FN:8d}")
    print(f"GT:Bg        {total_FP:8d}       {TN:8d}   (TN not meaningful here)")


if __name__ == "__main__":
    main()
