"""
AI Employee Monitoring - core engine
====================================
Detects persons + cell phones with YOLOv8, tracks each person with an ID,
and classifies status:
    WORKING      -> person is inside his desk zone (green)
    PHONE USAGE  -> a cell phone detected inside the person box (red)
    ABSENT       -> person ID disappeared from camera for N frames (red, last pos)
    IDLE         -> person visible but outside any zone (orange)

Draws the same style overlay as reference images:
    - colored boxes with "ID x - STATUS" labels
    - yellow box on detected phone ("phone")
    - top-left black panel listing every ID + status
    - "Working: N   Alerts: M" counter
"""

import cv2
import json
import numpy as np

# ------------------------------------------------------------------ colors (BGR)
GREEN  = (80, 200, 90)
RED    = (60, 60, 230)
YELLOW = (60, 200, 235)
ORANGE = (60, 160, 240)
WHITE  = (255, 255, 255)
BLACK  = (20, 20, 20)

STATUS_COLORS = {
    "WORKING": GREEN,
    "PHONE USAGE": RED,
    "ABSENT": RED,
    "IDLE": ORANGE,
}

PERSON_CLS = 0        # COCO: person
PHONE_CLS = 67        # COCO: cell phone


def _center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _point_in_poly(pt, poly):
    """poly = list of [x, y] pixel points."""
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0


def _label(frame, text, x1, y1, color):
    """Filled label bar above a box."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y = max(y1 - 8, th + 6)
    cv2.rectangle(frame, (x1, y - th - 8), (x1 + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x1 + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)


class MonitorEngine:
    def __init__(self, model, config_path="config.json"):
        """
        model: already loaded YOLO model (ultralytics)
        """
        self.model = model
        self.cfg = self._load_config(config_path)

        self.absent_after = self.cfg.get("absent_after_frames", 60)
        self.remove_after = self.cfg.get("remove_after_frames", 240)
        self.conf = self.cfg.get("conf", 0.3)
        # zones stored as normalized points 0..1
        self.zones = self.cfg.get("zones", [])

        # id -> {"box": (x1,y1,x2,y2), "status": str, "missing": int}
        self.memory = {}
        self.frame_count = 0

    # ------------------------------------------------------------------ config
    @staticmethod
    def _load_config(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def reload_config(self, config_path="config.json"):
        self.cfg = self._load_config(config_path)
        self.zones = self.cfg.get("zones", [])
        self.absent_after = self.cfg.get("absent_after_frames", 60)
        self.remove_after = self.cfg.get("remove_after_frames", 240)
        self.conf = self.cfg.get("conf", 0.3)

    # ------------------------------------------------------------------ main
    def process(self, frame):
        """Returns (annotated_frame, statuses_dict, alerts_count)."""
        self.frame_count += 1
        h, w = frame.shape[:2]

        res = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLS, PHONE_CLS],
            conf=self.conf,
            verbose=False,
        )[0]

        persons, phones = [], []
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls[0])
                box = tuple(float(v) for v in b.xyxy[0])
                tid = int(b.id[0]) if b.id is not None else None
                if cls == PERSON_CLS:
                    persons.append((tid, box))
                elif cls == PHONE_CLS:
                    phones.append(box)

        # ---------------- which person has a phone?
        phone_users = set()
        phone_boxes = []
        for pbox in phones:
            pc = _center(pbox)
            owner = None
            for tid, box in persons:
                x1, y1, x2, y2 = box
                if x1 <= pc[0] <= x2 and y1 <= pc[1] <= y2:
                    owner = tid
                    break
            if owner is not None:
                phone_users.add(owner)
            phone_boxes.append(pbox)

        # ---------------- pixel zones for this frame size
        zones_px = []
        for z in self.zones:
            pts = [[int(p[0] * w), int(p[1] * h)] for p in z["points"]]
            zones_px.append(pts)

        # ---------------- update memory for visible persons
        seen = set()
        for tid, box in persons:
            if tid is None:
                tid = -1  # unidentified person
            seen.add(tid)
            cx, cy = _center(box)
            in_zone = any(_point_in_poly((cx, cy), zp) for zp in zones_px)
            if tid in phone_users:
                status = "PHONE USAGE"
            elif in_zone or not zones_px:
                status = "WORKING"
            else:
                status = "IDLE"
            self.memory[tid] = {"box": box, "status": status, "missing": 0}

        # ---------------- handle missing persons (ABSENT)
        for tid in list(self.memory.keys()):
            if tid in seen:
                continue
            m = self.memory[tid]
            m["missing"] += 1
            if m["missing"] > self.remove_after:
                del self.memory[tid]
            elif m["missing"] > self.absent_after:
                m["status"] = "ABSENT"

        # ---------------- draw
        annotated = frame.copy()
        self._draw_zones(annotated, zones_px)
        for pbox in phone_boxes:
            self._draw_phone(annotated, pbox)

        statuses = {}
        for tid in sorted(self.memory.keys()):
            m = self.memory[tid]
            statuses[tid] = m["status"]
            self._draw_person(annotated, tid, m)

        alerts = sum(1 for s in statuses.values() if s in ("PHONE USAGE", "ABSENT"))
        working = sum(1 for s in statuses.values() if s == "WORKING")

        self._draw_panel(annotated, statuses, working, alerts)
        return annotated, statuses, alerts

    # ------------------------------------------------------------------ drawing
    def _draw_zones(self, frame, zones_px):
        for i, zp in enumerate(zones_px):
            pts = np.array(zp, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], GREEN)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], True, GREEN, 2)

    def _draw_person(self, frame, tid, m):
        status = m["status"]
        color = STATUS_COLORS.get(status, WHITE)
        x1, y1, x2, y2 = [int(v) for v in m["box"]]

        # transparent fill like reference images
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid} - {status}" if tid >= 0 else f"PERSON - {status}"
        _label(frame, label, x1, y1, color)

    def _draw_phone(self, frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)
        _label(frame, "phone", x1, y1, YELLOW)

    def _draw_panel(self, frame, statuses, working, alerts):
        """Top-left info panel, same style as reference images."""
        h, w = frame.shape[:2]
        panel_w = min(280, w // 3)
        rows = min(len(statuses), 10)
        panel_h = 78 + rows * 26

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), BLACK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "AI EMPLOYEE MONITORING",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps_text}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        y = 66
        for tid in sorted(statuses.keys()):
            if tid < 0:
                continue
            status = statuses[tid]
            color = STATUS_COLORS.get(status, WHITE)
            cv2.circle(frame, (16, y - 5), 5, color, -1)
            cv2.putText(frame, f"ID {tid} - {status}",
                        (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
            y += 26

        cv2.putText(frame, f"Working: {working}   Alerts: {alerts}",
                    (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 220, 120), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ misc
    fps_text = "--"

    def set_fps(self, fps):
        self.fps_text = f"{fps:.1f}" if fps else "--"
