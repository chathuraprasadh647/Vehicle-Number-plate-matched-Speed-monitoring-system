# Standard Library Imports
import os
import math
from collections import OrderedDict

# Third-Party Imports
import cv2
import numpy as np

# --- Dependency Installation Notes ---
# IMPORTANT: Ensure all required dependencies are installed.
#
# 1. Tesseract OCR Engine:
#    The Tesseract OCR engine must be installed on your system for pytesseract to work.
#    - On Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y tesseract-ocr
#    - On Fedora: sudo dnf install -y tesseract
#    - On macOS (using Homebrew): brew install tesseract
#    The 'pytesseract' Python package is also required: pip install pytesseract
#
# 2. Ultralytics YOLO:
#    For object detection: pip install ultralytics
#    This package includes PyTorch and other dependencies. If you have CUDA, it might use the GPU.
#
# 3. OpenCV and NumPy:
#    Usually installed as dependencies of the above or via: pip install opencv-python numpy

try:
    import pytesseract
except ImportError:
    print("--------------------------------------------------------------------")
    print("WARNING: `pytesseract` library not found or Tesseract OCR not installed.")
    print("OCR functionality will be disabled.")
    print("Please install Tesseract OCR engine on your system and the `pytesseract` Python package.")
    print("See installation notes at the top of this script.")
    print("--------------------------------------------------------------------")
    pytesseract = None

try:
    from ultralytics import YOLO
except ImportError:
    print("--------------------------------------------------------------------")
    print("ERROR: `ultralytics` library not found.")
    print("YOLO object detection cannot proceed. Please install it: pip install ultralytics")
    print("See installation notes at the top of this script.")
    print("--------------------------------------------------------------------")
    # Exit if YOLO is not available, as it's core to the script's function
    exit()


# --- Configuration Constants ---
# File Paths
VIDEO_PATH = 'test_video.mp4'         # Input video file
OUTPUT_VIDEO_PATH = "output_video.mp4" # Output video file
PLATE_OCR_IMAGE_PATH = "plate_for_ocr.jpg" # Temporary image for OCR processing (overwritten)

# YOLO Model
MODEL_NAME = 'yolov8n.pt'             # Pre-trained YOLO model

# Object Detection
VEHICLE_CLASS_IDS = [2, 3, 5, 7]      # COCO class IDs for car, motorcycle, bus, truck

# License Plate Detection Parameters
LP_ASPECT_RATIO_MIN = 1.8             # Min aspect ratio for a license plate
LP_ASPECT_RATIO_MAX = 5.0             # Max aspect ratio for a license plate
LP_MIN_AREA = 150                     # Min area of a license plate contour
LP_MAX_AREA = 4000                    # Max area of a license plate contour
LP_ROI_PADDING = 5                    # Padding around the detected plate for OCR image

# OCR Configuration
OCR_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' # Allowed characters for OCR
OCR_LANG = 'eng'
# PSM 7: Treat as single text line. PSM 8: Treat as single word. PSM 13: Raw line.
OCR_PSM_CONFIG = '8'
OCR_CONFIG_STRING = f'--oem 3 --psm {OCR_PSM_CONFIG} -l {OCR_LANG} -c tessedit_char_whitelist={OCR_WHITELIST}'


# Vehicle Tracking Parameters
MAX_DISAPPEARED_FRAMES = 30           # Max frames to keep a track alive without new detection
MAX_TRACKING_DISTANCE = 75            # Max distance (pixels) to match centroids for tracking
MIN_CONSECUTIVE_FRAMES_FOR_SPEED = 2  # Min frames an object must be tracked to calculate speed

# Display Options
FONT_SCALE_INFO = 0.5                 # Font scale for ID, Speed, OCR text
FONT_THICKNESS = 1
BOX_THICKNESS = 2
VEHICLE_BOX_COLOR = (0, 255, 0)       # Green
PLATE_BOX_COLOR = (255, 0, 255)       # Magenta
ID_SPEED_TEXT_COLOR = (255, 0, 0)     # Blue
OCR_TEXT_COLOR = (0, 0, 255)          # Red
TRACKING_INFO_COLOR = (0, 255, 255)   # Cyan (for lost tracks)

# --- Vehicle Tracker Class ---
class VehicleTracker:
    def __init__(self):
        self.next_object_id = 0
        # self.tracked_objects stores:
        # {id: {"centroid": (cx, cy), "prev_centroid": (cx, cy),
        #       "bbox": (x1,y1,x2,y2), "class_name": str, "confidence": float,
        #       "frame_last_seen": int, "consecutive_hits": int}}
        self.tracked_objects = OrderedDict()
        self.disappeared_frames = OrderedDict() # {id: num_frames_disappeared}

    def register(self, centroid, bbox, class_name, confidence, frame_num):
        self.tracked_objects[self.next_object_id] = {
            "centroid": centroid, "prev_centroid": None, "bbox": bbox,
            "class_name": class_name, "confidence": confidence,
            "frame_last_seen": frame_num, "consecutive_hits": 1
        }
        self.disappeared_frames[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.tracked_objects[object_id]
        del self.disappeared_frames[object_id]

    def update(self, current_detections, frame_num):
        # current_detections: list of tuples [( (x1,y1,x2,y2), class_name, confidence ), ...]

        if not current_detections: # No detections in this frame
            ids_to_deregister = []
            for object_id in list(self.disappeared_frames.keys()): # Use list for safe iteration
                self.disappeared_frames[object_id] += 1
                self.tracked_objects[object_id]["consecutive_hits"] = 0
                if self.disappeared_frames[object_id] > MAX_DISAPPEARED_FRAMES:
                    ids_to_deregister.append(object_id)
            for object_id in ids_to_deregister:
                self.deregister(object_id)
            return self.tracked_objects

        input_centroids = np.zeros((len(current_detections), 2), dtype="int")
        input_bboxes = [det[0] for det in current_detections]
        input_classes = [det[1] for det in current_detections]
        input_confidences = [det[2] for det in current_detections]

        for i, bbox in enumerate(input_bboxes):
            x1, y1, x2, y2 = bbox
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if not self.tracked_objects: # No objects currently being tracked
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_classes[i], input_confidences[i], frame_num)
        else:
            object_ids = list(self.tracked_objects.keys())
            object_stored_centroids = np.array([obj["centroid"] for obj in self.tracked_objects.values()])

            dist = np.cdist(object_stored_centroids, input_centroids)
            rows = dist.min(axis=1).argsort() # Indices of tracked objects, sorted by their min dist to a new detection
            cols = dist.argmin(axis=1)[rows]  # Indices of new detections best matching these sorted tracked objects

            used_rows = set()
            used_cols = set()

            for (row_idx, col_idx) in zip(rows, cols):
                if row_idx in used_rows or col_idx in used_cols:
                    continue
                if dist[row_idx, col_idx] > MAX_TRACKING_DISTANCE:
                    continue

                object_id = object_ids[row_idx]
                self.tracked_objects[object_id]["prev_centroid"] = self.tracked_objects[object_id]["centroid"]
                self.tracked_objects[object_id]["centroid"] = input_centroids[col_idx]
                self.tracked_objects[object_id]["bbox"] = input_bboxes[col_idx] # Update bbox
                self.tracked_objects[object_id]["class_name"] = input_classes[col_idx] # Update class (could change if model flickers)
                self.tracked_objects[object_id]["confidence"] = input_confidences[col_idx] # Update confidence
                self.tracked_objects[object_id]["frame_last_seen"] = frame_num
                self.tracked_objects[object_id]["consecutive_hits"] += 1
                self.disappeared_frames[object_id] = 0
                used_rows.add(row_idx)
                used_cols.add(col_idx)

            unused_rows = set(range(dist.shape[0])).difference(used_rows)
            for row_idx in unused_rows:
                object_id = object_ids[row_idx]
                self.disappeared_frames[object_id] += 1
                self.tracked_objects[object_id]["consecutive_hits"] = 0
                if self.disappeared_frames[object_id] > MAX_DISAPPEARED_FRAMES:
                    self.deregister(object_id)

            unused_cols = set(range(dist.shape[1])).difference(used_cols)
            for col_idx in unused_cols:
                self.register(input_centroids[col_idx], input_bboxes[col_idx],
                              input_classes[col_idx], input_confidences[col_idx], frame_num)

        # Ensure all tracked objects have frame_last_seen updated (mainly for those not matched this cycle but not yet deregistered)
        for obj_id in self.tracked_objects:
            if self.tracked_objects[obj_id]["frame_last_seen"] != frame_num and obj_id not in self.disappeared_frames:
                 # This case should ideally be handled by disappeared logic if an object is truly missed
                 pass # Or self.disappeared_frames[obj_id]+=1 if not handled above already for unmatched.
                      # The current logic seems to cover unmatche existing tracks under unused_rows.

        return self.tracked_objects

# --- Image Processing Functions ---
def find_license_plate_in_roi(vehicle_roi_color):
    if vehicle_roi_color is None or vehicle_roi_color.size == 0:
        return None, None # No ROI or empty ROI

    gray_roi = cv2.cvtColor(vehicle_roi_color, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Morphological operations to enhance plate-like regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blurred_roi, cv2.MORPH_BLACKHAT, kernel)

    # Edge detection
    sobel_x = cv2.Sobel(blackhat, cv2.CV_8U, 1, 0, ksize=3)
    _, threshold_sobel = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing operation to connect disparate parts of characters/plate boundaries
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed_img = cv2.morphologyEx(threshold_sobel, cv2.MORPH_CLOSE, closing_kernel, iterations=2) # Increased iterations

    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None

    potential_plates = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if w_c == 0 or h_c == 0: continue
        aspect_ratio = float(w_c) / h_c
        area = cv2.contourArea(cnt)

        if LP_ASPECT_RATIO_MIN < aspect_ratio < LP_ASPECT_RATIO_MAX and \
           LP_MIN_AREA < area < LP_MAX_AREA:
            # Check if contour is somewhat rectangular
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True) # Adjusted epsilon slightly
            if len(approx) >= 4 and len(approx) <=6 : # Allow for slight imperfections beyond perfect 4
                potential_plates.append(approx)

    if not potential_plates: return None, None

    largest_plate_contour = max(potential_plates, key=cv2.contourArea)
    x_lp, y_lp, w_lp, h_lp = cv2.boundingRect(largest_plate_contour)

    # Extract plate ROI with padding
    y_start = max(0, y_lp - LP_ROI_PADDING)
    y_end = min(vehicle_roi_color.shape[0], y_lp + h_lp + LP_ROI_PADDING)
    x_start = max(0, x_lp - LP_ROI_PADDING)
    x_end = min(vehicle_roi_color.shape[1], x_lp + w_lp + LP_ROI_PADDING)
    extracted_plate_roi = vehicle_roi_color[y_start:y_end, x_start:x_end]

    return largest_plate_contour, extracted_plate_roi

def perform_ocr_on_plate(plate_image_path):
    if pytesseract is None:
        return "OCR N/A (Pytesseract not configured)"
    if not os.path.exists(plate_image_path):
        return "OCR Error: Plate image not found"

    try:
        plate_image = cv2.imread(plate_image_path)
        if plate_image is None:
            return "OCR Error: Failed to read plate image"

        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Resize for potentially better OCR (especially for small plates)
        # Target height of ~50-70px can be good.
        current_h, current_w = gray_plate.shape[:2]
        scale_factor = max(1.0, 70.0 / current_h if current_h > 0 else 1.0) # Avoid scaling down too much, aim for height of 70
        if scale_factor > 3: scale_factor = 3 # Cap max scaling

        if scale_factor != 1.0:
             interp_method = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
             resized_plate = cv2.resize(gray_plate, (0,0), fx=scale_factor, fy=scale_factor, interpolation=interp_method)
        else:
             resized_plate = gray_plate

        # Adaptive thresholding
        thresh_plate = cv2.adaptiveThreshold(resized_plate, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 19, 9) # Adjusted blocksize and C

        # Optional: A bit of dilation/erosion or opening/closing if characters are too thin/thick
        # kernel_ocr = np.ones((2,1), np.uint8) # Thin vertical kernel
        # thresh_plate = cv2.dilate(thresh_plate, kernel_ocr, iterations=1)


        text = pytesseract.image_to_string(thresh_plate, config=OCR_CONFIG_STRING)
        cleaned_text = ''.join(char for char in text if char.isalnum() or char == '-').upper()

        return cleaned_text if cleaned_text else "No Text Found"
    except Exception as e:
        print(f"ERROR: OCR processing failed: {e}")
        return "OCR Exception"

# --- Main Processing Function ---
def main():
    print_initial_guidance()

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file '{VIDEO_PATH}' not found. Please place it in the script's directory.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{VIDEO_PATH}'. Check file integrity and OpenCV setup.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print("WARNING: Video FPS is 0. Using a default of 25. Speed calculations may be inaccurate.")
        video_fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_width, frame_height))

    try:
        yolo_model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model '{MODEL_NAME}': {e}")
        cap.release()
        out_video.release()
        return

    vehicle_tracker = VehicleTracker()
    current_frame_num = 0

    print_speed_calculation_guidance(video_fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("INFO: End of video or error reading frame.")
            break

        current_frame_num += 1
        output_frame = frame.copy()

        try:
            yolo_results = yolo_model(frame, verbose=False)[0] # verbose=False to reduce console spam
        except Exception as e:
            print(f"ERROR: YOLO prediction failed on frame {current_frame_num}: {e}")
            continue

        detections_for_tracker = []
        for box_data in yolo_results.boxes:
            class_id = int(box_data.cls)
            if class_id in VEHICLE_CLASS_IDS:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                confidence = float(box_data.conf)
                class_name = yolo_model.names[class_id]
                detections_for_tracker.append(((x1, y1, x2, y2), class_name, confidence))

        tracked_vehicles = vehicle_tracker.update(detections_for_tracker, current_frame_num)

        for obj_id, data in tracked_vehicles.items():
            x1, y1, x2, y2 = data["bbox"]
            centroid = data["centroid"]
            prev_centroid = data["prev_centroid"]

            # Draw vehicle bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), VEHICLE_BOX_COLOR, BOX_THICKNESS)

            # Calculate and display speed
            speed_px_per_frame = 0
            if prev_centroid is not None and data["consecutive_hits"] >= MIN_CONSECUTIVE_FRAMES_FOR_SPEED:
                displacement = math.sqrt((centroid[0] - prev_centroid[0])**2 +
                                         (centroid[1] - prev_centroid[1])**2)
                speed_px_per_frame = displacement

            info_text_line1 = f"ID: {obj_id}"
            info_text_line2 = f"Speed: {speed_px_per_frame:.1f} px/f"

            # Position text above the bounding box
            cv2.putText(output_frame, info_text_line1, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, ID_SPEED_TEXT_COLOR, FONT_THICKNESS)
            cv2.putText(output_frame, info_text_line2, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, ID_SPEED_TEXT_COLOR, FONT_THICKNESS)

            # License Plate Detection and OCR for this tracked vehicle
            vehicle_roi = frame[y1:y2, x1:x2]
            plate_contour, plate_img = find_license_plate_in_roi(vehicle_roi)

            if plate_contour is not None and plate_img is not None:
                # Adjust plate contour to original image coordinates
                plate_contour_orig = plate_contour.copy()
                plate_contour_orig[:, :, 0] += x1
                plate_contour_orig[:, :, 1] += y1
                cv2.drawContours(output_frame, [plate_contour_orig], -1, PLATE_BOX_COLOR, BOX_THICKNESS)

                try:
                    cv2.imwrite(PLATE_OCR_IMAGE_PATH, plate_img) # Overwritten for each plate
                    ocr_text = perform_ocr_on_plate(PLATE_OCR_IMAGE_PATH)

                    # Position OCR text near the plate
                    (lp_x, lp_y, _, lp_h) = cv2.boundingRect(plate_contour_orig)
                    ocr_text_y_pos = lp_y + lp_h + 15 # Below the plate
                    if ocr_text_y_pos + 10 > frame_height : ocr_text_y_pos = lp_y - 10 # Above if too low

                    cv2.putText(output_frame, ocr_text, (lp_x, ocr_text_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, OCR_TEXT_COLOR, FONT_THICKNESS)
                except Exception as e:
                    print(f"ERROR: Plate processing/OCR for ID {obj_id}, Frame {current_frame_num}: {e}")

        # Draw info for tracks that are being tracked but weren't matched to a current detection box this frame
        # (e.g. if YOLO missed it but tracker keeps it alive)
        for obj_id, data in vehicle_tracker.tracked_objects.items(): # Iterate through all, not just those updated this cycle
            if data["frame_last_seen"] == current_frame_num and data["consecutive_hits"] == 0 : # Object being coasted
                 centroid = data["centroid"]
                 cv2.putText(output_frame, f"ID:{obj_id} (Coasting)", (centroid[0] + 8, centroid[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO - 0.1, TRACKING_INFO_COLOR, FONT_THICKNESS)


        out_video.write(output_frame)
        if current_frame_num % 100 == 0: # Log progress every 100 frames
            print(f"INFO: Processed frame {current_frame_num}...")

    cleanup(cap, out_video)
    print(f"INFO: Video processing complete. Output saved to '{OUTPUT_VIDEO_PATH}'.")

def print_initial_guidance():
    print("====================================================================")
    print("Starting Vehicle Detection, Tracking, and OCR Script")
    print("Ensure all dependencies (YOLO/Ultralytics, Tesseract, OpenCV) are installed.")
    print("Refer to comments at the top of 'main.py' for installation help.")
    print("====================================================================")

def print_speed_calculation_guidance(fps):
    print("\n--- IMPORTANT: SPEED CALCULATION ---")
    print("The speed displayed is in 'pixels per frame'.")
    print("To convert this to real-world units (e.g., km/h or mph):")
    print("1. Determine 'pixels_per_meter': This requires camera calibration or a known object size in the video.")
    print(f"2. Video FPS (Frames Per Second): This video reports/is set to {fps:.2f} FPS.")
    print("3. Formula: real_speed_mps = (pixels_per_frame * video_fps) / pixels_per_meter")
    print("   real_speed_kmh = real_speed_mps * 3.6")
    print("   real_speed_mph = real_speed_mps * 2.23694")
    print("This script DOES NOT perform this real-world conversion as 'pixels_per_meter' is scene-dependent.")
    print("-------------------------------------\n")

def cleanup(cap, out_video):
    """Releases video capture and writer objects."""
    if cap: cap.release()
    if out_video: out_video.release()
    cv2.destroyAllWindows() # Close any OpenCV windows if they were used

if __name__ == "__main__":
    main()
