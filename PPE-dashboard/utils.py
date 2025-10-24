import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import threading
import queue
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
import csv
import base64
from collections import deque

# Import constants from our new config file
from config import *

# --- Optional Imports ---
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    logging.error("Failed to import YOLO. `ultralytics` package may not be installed.")

# --- Worker Classes ---

class WorkerTracker:
    """Tracks workers using IoU and manages their PPE state."""
    def __init__(self, iou_threshold=0.4, max_disappeared=300):
        self.next_worker_id = 1
        self.workers = {}
        self.iou_threshold = iou_threshold # This is now set from the Streamlit slider
        self.max_disappeared = max_disappeared
        logging.info(f"WorkerTracker initialized with IoU threshold: {self.iou_threshold}") # Log the threshold
        
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        worker_boxes = [d for d in detections if d['label'] == 'worker']
        
        # Mark all as disappeared if no workers detected
        if len(worker_boxes) == 0:
            for wid in list(self.workers.keys()):
                self.workers[wid]['disappeared'] += 1
                if self.workers[wid]['disappeared'] > self.max_disappeared:
                    del self.workers[wid]
            return []
        
        current_ids = list(self.workers.keys())
        matched = set()
        new_workers = []
        
        # Match detected workers to existing workers
        for worker_box in worker_boxes:
            best_iou = 0
            best_id = None
            
            for wid in current_ids:
                if wid in matched:
                    continue
                iou = self.calculate_iou(worker_box['bbox'], self.workers[wid]['bbox'])
                # Use the class's iou_threshold
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_id = wid
            
            if best_id is not None: # Matched existing worker
                self.workers[best_id]['bbox'] = worker_box['bbox']
                self.workers[best_id]['disappeared'] = 0
                self.workers[best_id]['confidence'] = worker_box['confidence']
                matched.add(best_id)
                new_workers.append({'id': best_id, 'bbox': worker_box['bbox'], 'confidence': worker_box['confidence']})
            else: # New worker detected
                new_id = self.next_worker_id
                self.next_worker_id += 1
                self.workers[new_id] = {
                    'bbox': worker_box['bbox'],
                    'disappeared': 0,
                    'confidence': worker_box['confidence'],
                    'ppe': {'helmet': False, 'vest': False, 'gloves': False, 'boots': False}
                }
                new_workers.append({'id': new_id, 'bbox': worker_box['bbox'], 'confidence': worker_box['confidence']})
        
        # Mark unmatched workers as disappeared
        for wid in current_ids:
            if wid not in matched:
                self.workers[wid]['disappeared'] += 1
                if self.workers[wid]['disappeared'] > self.max_disappeared:
                    del self.workers[wid]
        
        # Associate PPE with each worker
        for worker in new_workers:
            wid = worker['id']
            wx1, wy1, wx2, wy2 = worker['bbox']
            
            self.workers[wid]['ppe'] = {'helmet': False, 'vest': False, 'gloves': False, 'boots': False}
            
            for det in detections:
                if det['label'] in ['helmet', 'vest', 'gloves', 'boots']:
                    px1, py1, px2, py2 = det['bbox']
                    # Check for simple overlap
                    if not (px2 < wx1 or px1 > wx2 or py2 < wy1 or py1 > wy2):
                        self.workers[wid]['ppe'][det['label']] = True
            
            worker['ppe'] = self.workers[wid]['ppe'].copy()
        
        return new_workers

class CameraWorker(threading.Thread):
    """Thread for capturing frames from the camera."""
    def __init__(self, cam_index, frame_q, stop_event):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.frame_q = frame_q
        self.stop_event = stop_event
        self.cap = None

    def run(self):
        try:
            try:
                # Try DirectShow backend for Windows
                self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(self.cam_index)

            if not self.cap or not self.cap.isOpened():
                logging.warning(f"CameraWorker: failed to open camera index {self.cam_index}")
                return
            
            # Set desired resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("CameraWorker: Failed to read frame.")
                    break
                try:
                    # Put frame into the queue
                    self.frame_q.put_nowait(frame)
                except queue.Full:
                    # If queue is full, drop the oldest frame and add the new one
                    try:
                        _ = self.frame_q.get_nowait()
                        self.frame_q.put_nowait(frame)
                    except queue.Empty:
                        pass
                time.sleep(0.001) # Small sleep to prevent busy-waiting
        finally:
            if self.cap:
                self.cap.release()
            logging.info("CameraWorker stopped.")

class InferenceWorker(threading.Thread):
    """Thread for running model inference."""
    def __init__(self, frame_q, result_q, model_object, conf_thresh_getter, 
                 ppe_monitor_getter, stop_event, tracker, violation_buffer):
        super().__init__(daemon=True)
        self.frame_q = frame_q
        self.result_q = result_q
        self.model = model_object 
        self.get_conf = conf_thresh_getter
        self.get_ppe_to_monitor = ppe_monitor_getter
        self.stop_event = stop_event
        self.tracker = tracker
        self.violation_buffer = violation_buffer
        self.frame_count = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue # No frame available, loop again

            self.frame_count += 1
            start_time = time.time()
            
            # Prepare frame: BGR to RGB, resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            scale = 960 / max(h, w) # Resize to max 960px
            frame_rgb_resized = cv2.resize(frame_rgb, (int(w*scale), int(h*scale)))

            model = self.model
            
            conf = self.get_conf()
            required_ppe = self.get_ppe_to_monitor()
            
            dets = []
            workers = []
            
            if model is not None and YOLO is not None:
                try:
                    # Run inference
                    res = model(frame_rgb_resized, conf=conf, verbose=False)
                    # Get basic detections
                    annotated_resized, dets = annotate_frame(frame_rgb_resized, res, conf_thresh=conf)
                    # Track workers
                    workers = self.tracker.update(dets)
                    # Draw worker-specific boxes
                    annotated = annotate_frame_with_workers(frame_rgb_resized.copy(), workers, required_ppe)
                    
                except Exception as e:
                    logging.error(f"Inference error: {e}", exc_info=True)
                    annotated = frame_rgb_resized # Failsafe on error
            else:
                # Draw error message directly on the frame if model isn't ready
                annotated = frame_rgb_resized.copy()
                cv2.putText(annotated, "MODEL NOT LOADED or YOLO NOT INSTALLED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                logging.warning("Inference skipped: Model or YOLO not available.")
            
            # Add to violation buffer
            frame_rgb_to_buffer = cv2.resize(frame_rgb, (annotated.shape[1], annotated.shape[0]))
            self.violation_buffer.append(frame_rgb_to_buffer.copy())
            
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Send results to main thread
            payload = {
                'annotated': annotated, 'dets': dets, 'workers': workers,
                'fps': fps, 'frame_count': self.frame_count,
                'ts': datetime.now().isoformat(), 'required_ppe': required_ppe
            }
            
            try:
                self.result_q.put_nowait(payload)
            except queue.Full:
                try:
                    _ = self.result_q.get_nowait()
                    self.result_q.put_nowait(payload)
                except queue.Empty:
                    pass
            
            time.sleep(0.001) # Small sleep
        
        logging.info("InferenceWorker stopped.")

# --- Helper Functions ---

def list_models():
    """Finds all .pt models in the MODELS_DIR."""
    models = {}
    if MODELS_DIR.exists():
        for sub in MODELS_DIR.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.suffix == ".pt":
                        models[f"{sub.name}/{f.name}"] = str(f)
    return models

def load_model(path: str):
    """Loads a YOLO model, returns None on failure."""
    if YOLO is None:
        logging.error("ultralytics not installed. Install `ultralytics` and compatible `torch`.")
        return None
    try:
        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        model = YOLO(path)
        if torch is not None:
            model.to(device)
        logging.info(f"Model loaded successfully from {path} on {device}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {path}: {e}", exc_info=True)
        return None

def annotate_frame(frame_rgb, results, conf_thresh=0.50):
    """Draws basic model detections (all classes)."""
    img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    dets = []
    try:
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        names = getattr(r, 'names', None) or {}
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy[0][:4])
                conf = float(box.conf[0])
                cls_key = int(box.cls[0])
                label = LABEL_MAP.get(cls_key, names.get(cls_key, str(cls_key)))
                
                if conf < conf_thresh:
                    continue
                
                rgb = COLORS.get(label, (0,255,0))
                color_bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(img_bgr, f"{label} {conf:.2f}", (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                dets.append({'label': label, 'confidence': conf, 'bbox': (x1,y1,x2,y2)})
    except Exception as e:
        logging.warning(f"Error in annotate_frame: {e}")
        pass
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), dets

def annotate_frame_with_workers(frame_rgb, workers, required_ppe):
    """Draws worker-specific boxes with compliance status."""
    img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    for worker in workers:
        wid = worker['id']
        x1, y1, x2, y2 = worker['bbox']
        ppe = worker['ppe']
        
        # Check compliance
        missing = [p for p in required_ppe if not ppe.get(p, False)]
        is_compliant = len(missing) == 0
        
        # Set color and thickness based on compliance
        if is_compliant:
            color = (34, 197, 94)  # Green
            thickness = 3
        else:
            # Pulsing red for violations
            pulse = abs(math.sin(time.time() * 3))
            thickness = int(3 + 3 * pulse)
            color = (239, 68, 68)  # Red
        
        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
        
        # Draw status text
        status = "✓ Compliant" if is_compliant else "⚠ VIOLATION"
        text = f"Worker #{wid} {status}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
        cv2.putText(img_bgr, text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Show missing PPE if non-compliant
        if not is_compliant:
            missing_text = f"Missing: {', '.join([p.upper() for p in missing])}"
            cv2.putText(img_bgr, missing_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw PPE indicators (small icons)
        indicator_y = y1 + 25
        for idx, (ppe_item, icon) in enumerate([('helmet', 'H'), ('vest', 'V'), ('gloves', 'G'), ('boots', 'B')]):
            indicator_x = x1 + idx * 25
            ppe_color = (34, 197, 94) if ppe.get(ppe_item, False) else (239, 68, 68)
            cv2.circle(img_bgr, (indicator_x + 10, indicator_y), 10, ppe_color, -1)
            cv2.putText(img_bgr, icon, (indicator_x + 5, indicator_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_violation_clip(buffer, worker_id, missing_ppe):
    """Saves a clip, using OUT_VIOLATIONS from config."""
    try:
        if len(buffer) == 0:
            return False
        
        # Define paths
        output_path = OUT_VIOLATIONS / f"violation_worker{worker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        metadata_path = output_path.with_suffix('.json')
        
        # Get frame size from buffer
        h, w = buffer[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (w, h))
        
        # Write frames
        for frame in buffer:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        
        # Write metadata
        import json
        metadata = {
            'worker_id': worker_id, 'missing_ppe': missing_ppe,
            'timestamp': datetime.now().isoformat(), 'frames': len(buffer)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Saved violation clip: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving violation clip: {e}")
        return False

def save_detection(rec: dict):
    """Saves a detection record, using OUT_DIR from config."""
    csv_path = OUT_DIR / "detections.csv" # Moved from /output/violations
    headers = ['timestamp', 'session_id', 'worker_id', 'helmet', 'vest', 'gloves', 'boots', 
               'compliance_status', 'frame_number']
    write_header = not csv_path.exists()
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(rec)
    except Exception as e:
        logging.warning(f"Error saving detection to CSV: {e}")

# --- Analytics Functions ---

def create_gauge_chart(value, title, icon):
    """Creates a Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{icon} {title}", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1e40af" if value > 90 else "#f97316" if value > 70 else "#ef4444"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cccccc",
            'steps': [
                {'range': [0, 70], 'color': '#fee2e2'},
                {'range': [70, 90], 'color': '#ffedd5'},
                {'range': [90, 100], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 'value': 90
            }
        }
    ))
    fig.update_layout(
        height=200, 
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
        paper_bgcolor="white",
        font_color="#1e40af"
    )
    return fig

def generate_session_report(stats, violations_df):
    """Generates a Markdown string for the session report."""
    start_time_obj = stats.get('start_time')
    if start_time_obj is None:
        return "# 🦺 PPE Detection Session Report\n\nERROR: Session not started."

    start_time = start_time_obj.strftime('%Y-%m-%d %H:%M:%S')
    duration = datetime.now() - start_time_obj
    duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m {duration.seconds % 60}s"
    
    total_det = stats.get('total_detections', 0)
    total_viol = stats.get('total_violations', 0)
    compliance_rate = ((total_det - total_viol) / total_det * 100) if total_det > 0 else 100
    
    report = f"""
# 🦺 PPE Detection Session Report
- **Session Start:** {start_time}
- **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Session Duration:** {duration_str}
---
## 📊 Session Summary
| Metric | Value |
| :--- | :--- |
| Total Detections (Worker-Frames) | {total_det} |
| Total Violations | {total_viol} |
| **Overall Compliance Rate** | **{compliance_rate:.1f}%** |
---
## 📋 Violation Log
"""
    if violations_df.empty:
        report += "✅ No violations recorded during this session."
    else:
        # Create a clean copy for manipulation
        df_copy = violations_df.copy()
        # Convert list of missing PPE to a comma-separated string
        df_copy['Missing PPE'] = df_copy['Missing PPE'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        # Get violation counts
        violation_counts = df_copy.groupby('Missing PPE').size().reset_index(name='Count')
        violation_counts = violation_counts.sort_values(by='Count', ascending=False)
        
        report += "### Violations by Type\n\n"
        report += violation_counts.to_markdown(index=False)
        report += "\n\n### Detailed Log (Last 20 Violations)\n\n"
        # Show last 20 violations
        report += df_copy.tail(20).to_markdown(index=False)

    return report

def get_table_download_link(content, filename, text):
    """Generates a download link for a text file."""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
