import cv2
import numpy as np
import time
import os

# ================== USER SETTINGS ==================

# IP Webcam URL (MJPEG) - CHANGE THIS to your phone's URL (same Wi-Fi)
# Example for IP Webcam Android: http://192.168.0.23:8080/video
PHONE_STREAM_URL = "http://172.18.233.79:8080/video"  # <-- change this if needed

# YOLO paths
YOLO_DIR = "yolo"
CFG_PATH = os.path.join(YOLO_DIR, "yolov3-tiny.cfg")
WEIGHTS_PATH = os.path.join(YOLO_DIR, "yolov3-tiny.weights")
NAMES_PATH = os.path.join(YOLO_DIR, "coco.names")

# Confidence and NMS thresholds
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

# Classes treated as obstacles (COCO names)
# Extended to include more indoor objects like tables etc.
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorbike", "bus", "truck",
    "bench", "chair", "sofa", "diningtable", "bed", "pottedplant",
    "tvmonitor"
}

# Run YOLO only every N frames (higher N = higher FPS, slower updates)
YOLO_EVERY_N = 5

# Fraction of frame height from which we start the "path" ROI (bottom part)
ROI_START_FRACTION = 0.4  # use bottom 60% of frame for YOLO + wall check

# YOLO input size (smaller = faster, less accurate)
YOLO_INPUT_SIZE = (320, 320)

# Target display / processing resolution (square)
TARGET_SIZE = (720, 720)

# Edge density threshold for wall/close-object detection
EDGE_DENSITY_THRESHOLD = 0.02  # tune if needed

# ========= IMAGE CAPTURE SETTINGS =========
SAVE_IMAGES = True               # Set False if you don't want to save
NUM_IMAGES_TO_SAVE = 200         # how many frames to capture
SAVE_EVERY_N_FRAMES = 20          # save every Nth frame
SAVE_DIR = "captured_frames"     # folder where images will be stored
# =========================================


def load_yolo():
    """Load YOLOv3-tiny network and class names."""
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    out_layers = net.getUnconnectedOutLayersNames()
    return net, classes, out_layers


def run_yolo_on_roi(net, out_layers, roi):
    """
    Run YOLO on a region-of-interest (roi) frame.
    Returns detections in ROI coords: (x, y, w, h, class_id, conf).
    """
    (H, W) = roi.shape[:2]

    blob = cv2.dnn.blobFromImage(
        roi, 1 / 255.0, YOLO_INPUT_SIZE,
        swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                cx, cy, w, h = box.astype("int")

                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            detections.append((x, y, w, h, class_ids[i], confidences[i]))

    return detections


def wall_like_blocking_from_edges(frame):
    """
    Use Canny edges in the bottom ROI to detect 'wall-like' or 'big close object'
    in each of the three regions (left, center, right).
    Returns dict region_blocked_wall[region] = bool
    """
    H, W = frame.shape[:2]
    roi_y_start = int(H * ROI_START_FRACTION)
    roi = frame[roi_y_start:H, :]

    # Convert ROI to grayscale, blur, and Canny edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blur, 50, 150)

    region_blocked_wall = {"left": False, "center": False, "right": False}

    third = W // 3

    for idx, region_name in enumerate(["left", "center", "right"]):
        x_start = idx * third
        x_end = W if idx == 2 else (idx + 1) * third  # ensure cover full width
        region_edges = edges[:, x_start:x_end]

        # focus on bottom part of ROI for very close obstacles
        h_r = region_edges.shape[0]
        bottom_slice = region_edges[int(h_r * 0.5):, :]

        total_pixels = bottom_slice.size
        edge_pixels = np.count_nonzero(bottom_slice)

        if total_pixels > 0:
            density = edge_pixels / float(total_pixels)
        else:
            density = 0.0

        # If edge density is high, treat as "wall/blocked"
        if density > EDGE_DENSITY_THRESHOLD:
            region_blocked_wall[region_name] = True

    return region_blocked_wall


def compute_free_path_and_haptics(frame, detections, classes):
    """
    Uses YOLO detections + edge-based wall detection to:
      - decide left/forward/right/no-path
      - estimate approximate distance to nearest person
      - decide a logical 'haptic action' (LEFT/RIGHT/BOTH/NONE)

    Returns:
      frame (with overlays),
      decision (string),
      haptic_action (string),
      approx_distance_m (float or None)
    """
    H, W = frame.shape[:2]
    third = W // 3

    # Start with YOLO-based blocking
    region_blocked = {"left": False, "center": False, "right": False}

    path_y_min = int(H * ROI_START_FRACTION)  # lower region is "path"

    nearest_person_distance_m = None
    largest_person_height = 0

    # --- Process YOLO detections ---
    for (x, y, w, h, class_id, conf) in detections:
        class_name = classes[class_id]

        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, W - 1), min(y + h, H - 1)

        # Draw all detections (obstacles in red, others in blue)
        if class_name in OBSTACLE_CLASSES:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Only treat as navigation obstacle if it touches the path area
        if y2 < path_y_min:
            continue

        # Mark region blocked based on horizontal position
        cx = (x1 + x2) // 2
        if cx < third:
            region_blocked["left"] = True
        elif cx < 2 * third:
            region_blocked["center"] = True
        else:
            region_blocked["right"] = True

        # Distance estimation only for "person"
        if class_name == "person":
            box_height = y2 - y1
            if box_height > largest_person_height:
                largest_person_height = box_height

    # --- Approximate distance to nearest person ---
    if largest_person_height > 0:
        # Very rough constant: tune by experiment
        K = 500.0
        nearest_person_distance_m = K / float(largest_person_height)
    else:
        nearest_person_distance_m = None

    # --- Edge-based wall detection ---
    region_blocked_wall = wall_like_blocking_from_edges(frame)

    # Combine YOLO + wall/edge blocking
    for region in region_blocked.keys():
        if region_blocked_wall[region]:
            region_blocked[region] = True

    # --- Direction decision from blocked regions ---
    if not region_blocked["center"]:
        decision = "GO FORWARD"
    elif not region_blocked["left"]:
        decision = "GO LEFT"
    elif not region_blocked["right"]:
        decision = "GO RIGHT"
    else:
        decision = "NO CLEAR PATH"

    # --- Logical "haptic" action (no hardware here) ---
    if decision == "GO LEFT":
        haptic_action = "LEFT"    # would buzz left motor
    elif decision == "GO RIGHT":
        haptic_action = "RIGHT"   # would buzz right motor
    elif decision == "NO CLEAR PATH":
        haptic_action = "BOTH"    # would buzz both motors
    else:
        haptic_action = "NONE"    # no vibration

    # If nearest person < 1m -> override to danger
    if nearest_person_distance_m is not None and nearest_person_distance_m < 1.0:
        haptic_action = "BOTH"
        decision = "OBSTACLE < 1m"

    # --- Hollow Path Box (no overlay fill) ---
    top_y = int(H * ROI_START_FRACTION)
    bottom_y = H
    left_x = int(W * 0.20)
    right_x = int(W * 0.80)

    if "FORWARD" in decision:
        box_color = (0, 255, 0)   # green if forward is free
    else:
        box_color = (0, 0, 255)   # red otherwise

    thickness = 3
    cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), box_color, thickness)

    # --- Bottom bars for L/F/R ---
    bar_overlay = frame.copy()
    left_color = (0, 255, 0) if not region_blocked["left"] else (0, 0, 255)
    center_color = (0, 255, 0) if not region_blocked["center"] else (0, 0, 255)
    right_color = (0, 255, 0) if not region_blocked["right"] else (0, 0, 255)

    cv2.rectangle(bar_overlay, (0, H - 25), (third, H), left_color, -1)
    cv2.rectangle(bar_overlay, (third, H - 25), (2 * third, H), center_color, -1)
    cv2.rectangle(bar_overlay, (2 * third, H - 25), (W, H), right_color, -1)

    frame = cv2.addWeighted(bar_overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, "L", (third // 2 - 5, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "F", (third + third // 2 - 5, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "R", (2 * third + third // 2 - 5, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Decision + distance text
    cv2.putText(frame, decision, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if nearest_person_distance_m is not None:
        txt = f"Nearest person ~ {nearest_person_distance_m:.1f} m"
        cv2.putText(frame, txt, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, decision, haptic_action, nearest_person_distance_m


def main():
    print("[INFO] Loading YOLO...")
    net, classes, out_layers = load_yolo()

    print(f"[INFO] Opening IP stream: {PHONE_STREAM_URL}")
    cap = cv2.VideoCapture(PHONE_STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: could not open IP camera stream.")
        return

    # ---- NEW: create folder + counters for saving frames ----
    if SAVE_IMAGES:
        os.makedirs(SAVE_DIR, exist_ok=True)
    saved_count = 0
    # ---------------------------------------------------------

    prev_time = time.time()
    frame_idx = 0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame, reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(PHONE_STREAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_idx += 1

        # Resize to 720x720 for processing and display
        frame = cv2.resize(frame, TARGET_SIZE)
        H, W = frame.shape[:2]

        # ---- NEW: save frames automatically ----
        if SAVE_IMAGES and (frame_idx % SAVE_EVERY_N_FRAMES == 0) and saved_count < NUM_IMAGES_TO_SAVE:
            img_path = os.path.join(SAVE_DIR, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1
            print(f"[SAVE] {img_path}")

            # Optional: stop capture once enough images are saved
            if saved_count >= NUM_IMAGES_TO_SAVE:
                print("[INFO] Captured required images. Stopping capture loop.")
                break
        # ----------------------------------------

        # ROI for YOLO (bottom part only)
        roi_y_start = int(H * ROI_START_FRACTION)
        roi = frame[roi_y_start:H, :]

        # Run YOLO only every N frames
        if frame_idx % YOLO_EVERY_N == 0:
            roi_detections = run_yolo_on_roi(net, out_layers, roi)
            # Convert ROI detections to full-frame coordinates (add y offset)
            last_detections = []
            for (x, y, w, h, cid, conf) in roi_detections:
                full_y = y + roi_y_start
                last_detections.append((x, full_y, w, h, cid, conf))

        detections = last_detections

        frame, decision, haptic_action, dist_m = compute_free_path_and_haptics(
            frame, detections, classes
        )

        # FPS display
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (W - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Optional: print to console
        print(f"Decision: {decision:15} | Haptic: {haptic_action:5} | Dist: {dist_m}")

        cv2.imshow("YOLO Path Navigation - Phone Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
