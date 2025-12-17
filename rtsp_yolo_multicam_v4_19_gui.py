import cv2
import time
import torch
import threading
from ultralytics import YOLO
import numpy as np
import os
import csv
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import configparser
import psutil
import GPUtil
from bytetrack import ByteTracker
import mysql.connector
from mysql.connector import Error





CONFIG_FILE = "settings.ini"
CAMERA_FILE = "cameras.txt"

def format_bytes(value):
    """ Pretvori byte/s v spremenljive enote """
    if value < 1024:
        return f"{value:.0f} B/s"
    elif value < 1024**2:
        return f"{value/1024:.1f} kB/s"
    elif value < 1024**3:
        return f"{value/(1024**2):.2f} MB/s"
    else:
        return f"{value/(1024**3):.2f} GB/s"


# global buffer za mrežni promet
_last_net = None
_last_time = None

# Če settings.ini ne obstaja, ustvarimo osnovnega
if not os.path.exists(CONFIG_FILE):
    default_ini = """[general]
confidence_threshold = 0.60
log_file = logs/detections.csv
log_cooldown_seconds = 2

[filters]
classes = person,car
"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(default_ini)

# Preberemo konfiguracijo
config = configparser.ConfigParser()
config.read(CONFIG_FILE, encoding="utf-8")

CONFIDENCE_THRESHOLD = float(config["general"]["confidence_threshold"])
LOG_FILE = config["general"]["log_file"]
COOLDOWN_SECONDS = float(config["general"]["log_cooldown_seconds"])
YOLO_LOGGING = config["general"].get("yolo_logging", "True").strip().lower() in ("true", "1", "yes")


FILTER_CLASSES_RAW = config["filters"]["classes"].strip()
FILTER_CLASSES = [c.strip() for c in FILTER_CLASSES_RAW.split(",") if c.strip()] if FILTER_CLASSES_RAW else []

STREAM_FPS = float(config["general"].get("stream_fps", "0"))

# Poskrbimo za direktorij logov
LOG_DIR = os.path.dirname(LOG_FILE)
if LOG_DIR and not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception as e:
        print("⚠ Napaka pri ustvarjanju log direktorija:", e)

torch.backends.cudnn.benchmark = True


# ============================================================
# CameraThread – tu se zgodi YOLO + logika
# ============================================================

class CameraThread(threading.Thread):
    def __init__(self, name, url, model, device, update_callback, log_callback,
             parent, render_boxes=True, render_labels=True, render_conf=True):
        super().__init__(daemon=True)
        self.name = name
        self.url = url
        self.model = model
        self.device = device
        self.update_callback = update_callback  # klic v GUI
        self.log_callback = log_callback        # klic za CSV
        self.tracker_log_callback = parent.log_tracker_detection   # <-- NOV TRACKER LOG
        self.tracker_printscreen_callback = parent.save_tracker_printscreen # <-- NOV PRINTSCREEN LOG
        self.render_boxes = render_boxes
        self.render_labels = render_labels
        self.render_conf = render_conf

        self.stop_event = threading.Event()
        self.show_stream = True

        self.reconnecting = False
        self.reconnect_timer = 0
        self.fps = 0.0
        self.bitrate = 0.0
        self.detections = 0
        self.last_connect = "Ni povezave"
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cap = None

        self.last_logged = {}        # za cooldown po razredih
        self.logged_track_ids = set()
        self.last_frame_time = time.time()

        # zgodovina za 5s povprečja
        self.fps_history = []
        self.bitrate_history = []
        self.history_window_seconds = 5
        
        self.parent = parent
    
        # inicializacija ByteTrack s konfiguracijo iz GUI (settings.ini)
        if parent.tracker_enabled:
            self.tracker = ByteTracker(
                track_thresh=parent.tracker_track_thresh,
                match_thresh=parent.tracker_match_thresh,
                track_buffer=parent.tracker_track_buffer,
                frame_rate= parent.tracker_frame_rate
            )
        else:
            self.tracker = None


    def run(self):
        frame_id = 0
        start = time.time()
        while not self.stop_event.is_set():
            frame_id += 1

            # Auto-reconnect
            # Kamera ni povezana
            if self.cap is None or not self.cap.isOpened():

                # Posodobi GUI SAMO 1×, ne spamaj
                if not self.reconnecting:
                    self.reconnecting = True

                    # Pošlji placeholder frame, ampak varno!
                    try:
                        self.update_callback(self.name, self.frame, self.fps,
                                             self.detections, self.bitrate,
                                             True, self.last_connect)
                    except Exception:
                        pass  # ne crashaj GUI

                # počakaj 15 sekund pred ponovnim poskusom
                self.reconnect_timer += 1
                time.sleep(1)

                if self.reconnect_timer < 15:
                    continue

                self.reconnect_timer = 0

                # poskusi ponovno
                self.cap = cv2.VideoCapture(self.url)

                if self.cap.isOpened():
                    self.reconnecting = False
                    self.last_connect = datetime.now().strftime("%H:%M:%S")
                    print(f"[{self.name}] Kamera ponovno povezana!")
                else:
                    print(f"[{self.name}] Kamera še vedno offline...")

                continue


            # Branje frame-a
            ret, frame = self.cap.read()
            now = time.time()

            if not ret:
                self.cap.release()
                self.cap = None
                continue
                
            if STREAM_FPS > 0:
                processing_time = time.time() - start
                delay = (1.0 / STREAM_FPS) - processing_time
                if delay > 0:
                    time.sleep(delay)

            # Izračun pretoka
            dt = now - self.last_frame_time
            self.last_frame_time = now
            if dt > 0:
                current_bitrate = frame.nbytes / dt
            else:
                current_bitrate = 0.0

            # YOLO detekcija + DeepSORT tracking – DELAJ NA BGR
            start = time.time()
            results = self.model(frame, verbose=False)  # ali self.model.predict(frame, verbose=False, device=self.device)
            boxes = results[0].boxes
            names = results[0].names
            self.detections = len(boxes)

            # osnovni YOLO anotirani frame (če je omogočeno)
            annotated = results[0].plot(
                boxes=self.render_boxes,
                labels=self.render_labels,
                conf=self.render_conf
            )

            # pripravimo detections za ByteTrack (filtrirani po dovoljenih razredih)
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names[cls_id]

                if FILTER_CLASSES and label not in FILTER_CLASSES:
                    continue

                detections.append({
                    "bbox": np.array([x1, y1, x2, y2], dtype=float),
                    "score": conf,
                    "label": label,
                    "cls_id": cls_id,
                })

            # posodobitev trackerja
            tracks = []
            if self.tracker:
                tracks = self.tracker.update(detections)
                
             # RISANJE TRACKER OKVIRJEV IN ID-jev  <----- TUKAJ!!!
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                if self.parent.show_tracker_boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if self.parent.show_tracker_ids:
                    cv2.putText(
                        annotated,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2
                    )               
                
            # frame, na katerega rišemo tracker (YOLO anotirani)
            draw_frame = annotated

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # risanje DeepSORT okvirjev
                if self.parent.show_tracker_boxes:
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # risanje DeepSORT ID-jev
                if self.parent.show_tracker_ids:
                    cv2.putText(
                        draw_frame,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            # TRACKER LOGGING — popolnoma ločeno od YOLO logiranja
            # TRACKER LOGGING — vsak track_id se zapiše samo ENKRAT
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                # če smo ta track_id že zapisali, NE DELAJ NIČ
                if track_id in self.logged_track_ids:
                    continue

                # vzemi YOLO podatke, ki jih DeepSORT pripne tracku
                det_class = getattr(track, "label", None)
                det_conf = getattr(track, "score", None)

                if det_class is None or det_conf is None:
                    continue

                # preveri YOLO filter pogoje
                if det_conf < CONFIDENCE_THRESHOLD:
                    continue

                if FILTER_CLASSES and det_class not in FILTER_CLASSES:
                    continue

                # PRINTSCREEN + filename
                filename = None
                try:
                    filename = self.tracker_printscreen_callback(
                        self.name, frame, det_class, track.to_ltrb()
                    )
                except Exception as e:
                    print("⚠ Napaka pri printscreen callbacku:", e)

                # ZAPIS V CSV (samo 1× za celoten lifetime tracka)
                self.tracker_log_callback(
                    self.name, track_id, det_class, det_conf, filename
                )

               
                
                # označi track_id kot že zapisan
                self.logged_track_ids.add(track_id)



            end = time.time()


            current_fps = 1.0 / (end - start) if (end - start) > 0 else 0.0

            # Zgodovina za povprečje 5s
            self.fps_history.append((now, current_fps))
            self.bitrate_history.append((now, current_bitrate))

            self.fps_history = [(t, v) for (t, v) in self.fps_history
                                if now - t <= self.history_window_seconds]
            self.bitrate_history = [(t, v) for (t, v) in self.bitrate_history
                                    if now - t <= self.history_window_seconds]

            if self.fps_history:
                self.fps = sum(v for (_, v) in self.fps_history) / len(self.fps_history)
            else:
                self.fps = 0.0

            if self.bitrate_history:
                self.bitrate = sum(v for (_, v) in self.bitrate_history) / len(self.bitrate_history)
            else:
                self.bitrate = 0.0

            # LOGIRANJE – tu se upošteva:
            # - CONFIDENCE_THRESHOLD
            # - FILTER_CLASSES
            # - COOLDOWN_SECONDS
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names[cls_id]

                if self.should_log_detection(label, conf, now):
                    self.log_callback(self.name, label, conf)

            # Prikaz ali skrito
            if self.show_stream:
                self.frame = draw_frame.copy()
            else:
                self.frame = np.zeros_like(frame)

            # Posodobitev GUI
            self.update_callback(self.name, self.frame, self.fps,
                                 self.detections, self.bitrate,
                                 self.reconnecting, self.last_connect)

        if self.cap:
            self.cap.release()

    def should_log_detection(self, label, conf, now):
        # 1) Confidence filter
        if conf < CONFIDENCE_THRESHOLD:
            return False

        # 2) Class filter
        if FILTER_CLASSES and label not in FILTER_CLASSES:
            return False

        # 3) Cooldown filter (per class)
        last_t = self.last_logged.get(label, 0)
        if now - last_t < COOLDOWN_SECONDS:
            return False

        # Posodobimo timestamp (šele tukaj!)
        self.last_logged[label] = now
        return True



    def stop(self):
        self.stop_event.set()


# ============================================================
# GUI
# ============================================================

class YoloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP YOLO Multicam – v4.16 (fixed)")
        self.root.configure(bg="#1e1e1e")

        self.cameras = []
        self.threads = {}
        self.active_labels = set()

        self.model = None
        self.device = self.select_device()
        self.yolo_model_name = config["general"].get("YOLO_MODEL", "yolov8m.pt").strip()

        self.frames = {}
        self.led_labels = {}
        self.image_labels = {}
        self.stats_labels = {}
        self.show_buttons = {}

        self.fullscreen_camera = None

        self.main_frame = tk.Frame(root, bg="#1e1e1e")
        self.main_frame.pack(fill="both", expand=True)

        self.status_label = tk.Label(
            root, text="Inicializacija...",
            anchor="w", bg="#252526", fg="#f5f5f5", font=("Segoe UI", 9)
        )
        self.status_label.pack(side="bottom", fill="x")

        self.load_model()
        self.init_mysql()
        self.load_cameras()
        self.build_grid()
        self.update_status_bar()

        root.protocol("WM_DELETE_WINDOW", self.exit_app)

        # Ustvarimo CSV če ne obstaja
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["čas", "kamera", "objekt", "verjetnost"])

    # ---------------- Device ----------------
    def select_device(self):
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            return "cuda"
        print("CPU")
        return "cpu"

    # ---------------- Model ----------------
    def load_model(self):
        # Uporabi že prebrano ime modela iz __init__()
        model_path = os.path.join("models", self.yolo_model_name)

        # Naloži model ali uporabi fallback
        if not os.path.exists(model_path):
            print(f"⚠ Model '{model_path}' ne obstaja! Uporabljam privzetega: yolov8m.pt")
            model_path = os.path.join("models", "yolov8m.pt")

        print(f"🔍 Nalagam YOLO model: {model_path}")

        # YOLO model → GPU ali CPU
        self.model = YOLO(model_path).to(self.device)

        # Preberi prikazne nastavitve iz settings.ini
        # Pretvorimo string -> bool
        def str_to_bool(v):
            return str(v).strip().lower() in ("true", "1", "yes", "on")

        self.render_boxes = str_to_bool(config["general"].get("SHOW_BOXES", "True"))
        self.render_labels = str_to_bool(config["general"].get("SHOW_LABELS", "True"))
        self.render_conf = str_to_bool(config["general"].get("SHOW_CONFIDENCE", "True"))
        self.show_tracker_boxes = str_to_bool(config["general"].get("SHOW_TRACKER_BOXES", "True"))
        self.show_tracker_ids = str_to_bool(config["general"].get("SHOW_TRACKER_IDS", "True"))
        
        
        # ================================
        # TRACKER nastavitve (ByteTrack)
        # ================================
        tr = config["tracker"]

        self.tracker_enabled = str_to_bool(tr.get("enabled", "True"))
        self.tracker_boxes = str_to_bool(tr.get("draw_boxes", "True"))
        self.tracker_ids = str_to_bool(tr.get("draw_ids", "True"))

        self.tracker_frame_rate = float(tr.get("frame_rate", "30"))
        self.tracker_track_thresh = float(tr.get("track_thresh", "0.35"))
        self.tracker_match_thresh = float(tr.get("match_thresh", "0.7"))
        self.tracker_track_buffer = int(tr.get("track_buffer", "30"))

        
        # TRACKER LOGGING SETTINGS
        self.tracker_logging = str_to_bool(tr.get("tracker_logging", "True"))
        self.tracker_log_file = tr.get("tracker_log_file", "logs/tracker_detections.csv")

        # poskrbi, da direktorij obstaja
        log_dir = os.path.dirname(self.tracker_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # ustvari CSV, če manjka
        if self.tracker_logging and not os.path.exists(self.tracker_log_file):
            with open(self.tracker_log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["čas", "kamera", "track_id", "yolo_class", "confidence", "filename"])
                
        # Tracker printscreen settings
        self.tracker_printscreen = str_to_bool(tr.get("tracker_printscreen", "True"))
        self.tracker_printscreen_box_dir = tr.get("tracker_printscreen_box_dir", "tracker_printscreen_box")
        self.tracker_printscreen_full_dir = tr.get("tracker_printscreen_full_dir", "tracker_printscreen_full")
        self.tracker_printscreen_full_image = str_to_bool(tr.get("tracker_printscreen_full_image", "True"))
        self.tracker_printscreen_box_image = str_to_bool(tr.get("tracker_printscreen_box_image", "True"))


        # ustvari direktorij, če ne obstaja
        if not os.path.exists(self.tracker_printscreen_box_dir):
            try:
                os.makedirs(self.tracker_printscreen_box_dir, exist_ok=True)
            except Exception as e:
                print("⚠ Napaka pri ustvarjanju direktorija za box printscreene:", e)
                
        if not os.path.exists(self.tracker_printscreen_full_dir):
            try:
                os.makedirs(self.tracker_printscreen_full_dir, exist_ok=True)
            except Exception as e:
                print("⚠ Napaka pri ustvarjanju direktorija za full printscreene:", e)                
                
                
        print(
            f"🎨 YOLO prikaz: boxes={self.render_boxes}, labels={self.render_labels}, conf={self.render_conf}"
        )
        print(
            f"🎨 ByteTrack prikaz: boxes={self.tracker_boxes}, ids={self.tracker_ids}"
        )

    # ---------------- Cameras --------------
    def load_cameras(self):
        if not os.path.exists(CAMERA_FILE):
            print(f"Datoteka {CAMERA_FILE} manjka!")
            return

        with open(CAMERA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue

                if "|" in line:
                    name, url = line.split("|", 1)
                else:
                    name = f"Kamera {len(self.cameras)+1}"
                    url = line

                self.cameras.append((name.strip(), url.strip()))

    # ---------------- Logging ---------------
    def log_detection(self, camera_name, label, conf):
        if not YOLO_LOGGING:
            return  # logiranje je izklopljeno

        try:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    camera_name,
                    label,
                    f"{conf:.2f}"
                ])
        except PermissionError:
            print("⚠ CSV datoteka je zaklenjena (Excel?).")
        except Exception as e:
            print("⚠ Napaka pri pisanju CSV:", e)

    # ---------------- Deeptracker Logging ------------------
    def log_tracker_detection(self, camera_name, track_id, label, conf, filename):
        """Logiranje v CSV + (opcijsko) v MySQL tabelo YOLO_DT z retry sistemom."""

        # 1) CSV log
        if self.tracker_logging:
            try:
                with open(self.tracker_log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        camera_name,
                        track_id,
                        label,
                        f"{conf:.2f}",
                        filename or ""
                    ])
            except Exception as e:
                print("⚠ Napaka pri zapisovanju tracker CSV:", e)

        # 2) MySQL log
        if not self.mysql_enabled:
            return

        if self.mysql_conn is None:
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sql = """
            INSERT INTO YOLO_DT
                (ts, camera_name, track_id, yolo_class, confidence, filename, plate_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        data = (ts, camera_name, int(track_id), label, float(conf), filename or "", None)

        # --- retry logika (max 3 poskusi) ---
        attempts = 3

        for attempt in range(1, attempts + 1):
            try:
                cursor = self.mysql_conn.cursor()
                cursor.execute(sql, data)
                cursor.close()

                # OK, imamo success → prekini retry
                return

            except Exception as e:
                print(f"⚠ MySQL log napaka (poskus {attempt}/{attempts}): {e}")

                # Če smo izčrpali poskuse → končaj
                if attempt == attempts:
                    print("❌ MySQL log FAILED po 3 poskusih.\n")
                    return

                # Poskusi reconnect
                print("↻ Poskus ponovne vzpostavitve MySQL povezave...")
                self.reconnect_mysql()
                time.sleep(0.5)  # malo počakaj




    # ---------------- MySQL logging ------------------ 
    def init_mysql(self):
        """Inicializacija MySQL povezave na osnovi settings.ini [mysql]"""
        if "mysql" not in config:
            print("MySQL: [mysql] sekcija ni definirana v settings.ini – MySQL logiranje izklopljeno.")
            self.mysql_enabled = False
            return

        mysql_cfg = config["mysql"]
        enabled = mysql_cfg.get("enabled", "False").strip().lower() in ("true", "1", "yes")

        if not enabled:
            print("MySQL logiranje je izklopljeno (enabled = False).")
            self.mysql_enabled = False
            return

        host = mysql_cfg.get("host", "127.0.0.1")
        port = int(mysql_cfg.get("port", "3306"))
        user = mysql_cfg.get("user", "")
        password = mysql_cfg.get("password", "")
        database = mysql_cfg.get("database", "YOLO_DB")

        try:
            self.mysql_conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                autocommit=True
            )
            if self.mysql_conn.is_connected():
                self.mysql_enabled = True
                print(f"MySQL: Povezava uspešna na {host}:{port}, baza {database}")
            else:
                self.mysql_enabled = False
                print("MySQL: Povezava ni uspela.")
        except Error as e:
            self.mysql_enabled = False
            self.mysql_conn = None
            print(f"MySQL: napaka pri povezavi: {e}")
            
            
    def reconnect_mysql(self):
        """Poskusi ponovno vzpostaviti MySQL povezavo."""
        if not self.mysql_enabled:
            return

        if "mysql" not in config:
            return

        mysql_cfg = config["mysql"]

        host = mysql_cfg.get("host", "127.0.0.1")
        port = int(mysql_cfg.get("port", "3306"))
        user = mysql_cfg.get("user", "")
        password = mysql_cfg.get("password", "")
        database = mysql_cfg.get("database", "YOLO_DB")

        try:
            self.mysql_conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                autocommit=True
            )

            if self.mysql_conn.is_connected():
                print("✔ MySQL ponovno povezan!")
                return True

        except Exception as e:
            print(f"❌ MySQL reconnect error: {e}")

        # fallback
        self.mysql_conn = None
        return False
                
            


    # ---------------- Printscreen ------------------           
    def save_tracker_printscreen(self, camera_name, frame, det_class, bbox=None):
        if not self.tracker_printscreen:
            return None

        if frame is None or frame.size == 0:
            print("⚠ Frame je prazen, printscreen preskočen.")
            return None

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        def sanitize(s):
            s = s.strip()
            for ch in [" ", ":", "/", "\\"]:
                s = s.replace(ch, "_")
            return s

        safe_cam = sanitize(camera_name)
        safe_class = sanitize(det_class)

        safe_cam = "_".join(filter(None, safe_cam.split("_")))
        safe_class = "_".join(filter(None, safe_class.split("_")))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        filename = None  # <- privzeta vrednost

        # FULL
        if self.tracker_printscreen_full_image:
            filename = f"{timestamp}_{safe_cam}_{safe_class}.jpg"
            path = os.path.join(self.tracker_printscreen_full_dir, filename)
            try:
                img.save(path, format="JPEG", quality=90, optimize=True)
                print(f"[Printscreen FULL] Saved: {path}")
            except:
                pass

        # BOX
        if self.tracker_printscreen_box_image and bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame[y1:y2, x1:x2]

            if crop is not None and crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_img = Image.fromarray(crop_rgb)

                filename = f"{timestamp}_{safe_cam}_{safe_class}_box.jpg"
                path = os.path.join(self.tracker_printscreen_box_dir, filename)
                try:
                    crop_img.save(path, format="JPEG", quality=90, optimize=True)
                    print(f"[Printscreen BOX] Saved: {path}")
                except:
                    pass

        return filename

                    

    # ---------------- Grid ------------------
    def build_grid(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        cols = 3

        style = ttk.Style()
        style.configure(
            "Blue.TButton",
            background="#0078D7",
            foreground="#000000",
            font=("Segoe UI", 9, "bold")
        )

        for i, (name, url) in enumerate(self.cameras):

            frame = tk.Frame(self.main_frame, bg="#252526")
            frame.grid(row=i // cols, column=i % cols, padx=5, pady=5, sticky="nsew")

            led = tk.Canvas(frame, width=15, height=15, bg="#252526", highlightthickness=0)
            led.pack(anchor="ne", padx=5, pady=2)
            led_id = led.create_oval(2, 2, 13, 13, fill="red")
            self.led_labels[name] = (led, led_id)

            tk.Label(frame, text=name, bg="#252526", fg="#dcdcdc",
                     font=("Segoe UI", 10, "bold")).pack(anchor="n")

            img_label = tk.Label(frame, bg="#000")
            img_label.pack(fill="both", expand=True)

            self.active_labels.add(name)
            self.image_labels[name] = img_label

            btn_frame = tk.Frame(frame, bg="#252526")
            btn_frame.pack(fill="x")

            btn_toggle = ttk.Button(
                btn_frame, text="Skrij",
                command=lambda n=name: self.toggle_stream(n),
                style="Blue.TButton"
            )
            btn_toggle.pack(side="left", expand=True, fill="x", padx=2, pady=2)

            btn_full = ttk.Button(
                btn_frame, text="Povečaj",
                command=lambda n=name: self.fullscreen_view(n),
                style="Blue.TButton"
            )
            btn_full.pack(side="left", expand=True, fill="x", padx=2, pady=2)

            self.show_buttons[name] = btn_toggle

            stats = tk.Label(
                frame,
                text="FPS: -  |  Pretok: -\nDetekcije: -\nZadnja povezava: -",
                bg="#1e1e1e", fg="#f5f5f5", font=("Consolas", 9)
            )
            stats.pack(fill="x", pady=(2, 5))

            self.stats_labels[name] = stats

            if name not in self.threads:
                t = CameraThread(
                name,
                url,
                self.model,
                self.device,
                self.update_frame,
                self.log_detection,
                parent=self,
                render_boxes=self.render_boxes,
                render_labels=self.render_labels,
                render_conf=self.render_conf
            )
                self.threads[name] = t
                t.start()

        exit_btn = ttk.Button(
            self.main_frame, text="Izhod",
            command=self.exit_app, style="Blue.TButton"
        )
        exit_btn.grid(row=(len(self.cameras) // cols) + 1, column=1, pady=10)

        for i in range(cols):
            self.main_frame.columnconfigure(i, weight=1)
        for j in range((len(self.cameras) // cols) + 2):
            self.main_frame.rowconfigure(j, weight=1)

    # ---------------- Update frame ----------
    def update_frame(self, name, frame, fps, detections, bitrate,
                     reconnecting, last_connect):

        # Če widgeti za to kamero trenutno ne obstajajo, ne delaj nič
        if name not in self.image_labels:
            return

        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img = Image.fromarray(img)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        if self.fullscreen_camera == name:
            img = img.resize((1440, 900), Image.LANCZOS)
        else:
            img = img.resize((400, 300), Image.LANCZOS)

        bitrate_str = format_bytes(bitrate)

        def update_gui():

            # --- preveri, če widget sploh še obstaja ---
            if name not in self.image_labels:
                return

            lbl = self.image_labels[name]

            try:
                if not lbl.winfo_exists():
                    return
            except tk.TclError:
                return

            # --- ustvari PhotoImage v glavnem threadu ---
            try:
                imgtk = ImageTk.PhotoImage(image=img)
            except tk.TclError:
                return

            # --- posodobi sliko ---
            try:
                lbl.configure(image=imgtk)
                lbl.image = imgtk
            except tk.TclError:
                return

            # --- STATISTIKA (če obstaja) ---
            if name in self.stats_labels:
                stats_lbl = self.stats_labels[name]

                try:
                    if stats_lbl.winfo_exists():
                        stats_lbl.config(
                            text=f"FPS: {fps:.1f}  |  Pretok: {bitrate_str}\n"
                                 f"Detekcije: {detections}\n"
                                 f"Zadnja povezava: {last_connect}"
                        )
                except tk.TclError:
                    pass

            # --- LED indikator (če obstaja) ---
            if name in self.led_labels:
                led, led_id = self.led_labels[name]
                try:
                    if led.winfo_exists():
                        thread = self.threads.get(name)
                        color = self.get_led_color(thread, reconnecting)
                        led.itemconfig(led_id, fill=color)
                except tk.TclError:
                    pass

        # schedule safe update in main thread
        self.root.after(0, update_gui)


    # ---------------- Toggle stream ----------
    def toggle_stream(self, name):
        t = self.threads[name]
        t.show_stream = not t.show_stream
        self.show_buttons[name].config(text="Skrij" if t.show_stream else "Prikaži")

    # ---------------- Fullscreen ------------
    def fullscreen_view(self, name):
        self.image_labels.clear()
        self.stats_labels.clear()
        self.led_labels.clear()
        self.fullscreen_camera = name
        self.active_labels.clear()

        for widget in self.main_frame.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        frame.pack(fill="both", expand=True)

        img_label = tk.Label(frame, bg="#000")
        img_label.pack(fill="both", expand=True)

        self.image_labels[name] = img_label
        self.active_labels.add(name)

        back = ttk.Button(
            frame, text="Nazaj",
            command=self.exit_fullscreen, style="Blue.TButton"
        )
        back.pack(pady=10)

    def exit_fullscreen(self):
        self.fullscreen_camera = None
        self.build_grid()

    # ---------------- Status bar ------------
    def update_status_bar(self):
        global _last_net, _last_time

        # ---------------- CPU ----------------
        cpu_usage = psutil.cpu_percent(interval=None)

        # ---------------- RAM ----------------
        ram = psutil.virtual_memory()
        ram_usage = ram.percent

        # ---------------- GPU ----------------
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_load = gpu.load * 100
            gpu_mem = gpu.memoryUtil * 100
            gpu_temp = gpu.temperature
        else:
            gpu_load = 0
            gpu_mem = 0
            gpu_temp = 0

        # ---------------- Internetni pretok ----------------
        net = psutil.net_io_counters()
        now = time.time()

        if _last_net is None:
            up_speed = 0
            down_speed = 0
        else:
            dt = now - _last_time
            if dt <= 0:
                dt = 1e-6
            up_speed = (net.bytes_sent - _last_net.bytes_sent) / dt
            down_speed = (net.bytes_recv - _last_net.bytes_recv) / dt

        _last_net = net
        _last_time = now

        up_str = format_bytes(up_speed)
        down_str = format_bytes(down_speed)

        # ---------------- Filtri YOLO ----------------
        filtered = ", ".join(FILTER_CLASSES) if FILTER_CLASSES else "Vse"

        # ---------------- Končni prikaz ----------------
        self.status_label.config(
            text=(
                f"🧠 Naprava: {self.device.upper()}  |  "
                f"CPU: {cpu_usage:.1f}%  |  "
                f"RAM: {ram_usage:.1f}%  |  "
                f"GPU: {gpu_load:.1f}%  |  "
                f"VRAM: {gpu_mem:.1f}%  |  "
                f"Temp: {gpu_temp}°C  |  "
                f"↑ {up_str}  ↓ {down_str}  |  "
                f"YOLO model: {self.yolo_model_name}  |  "
                f"Filtrirani razredi: {filtered}  |  "
                f"{datetime.now().strftime('%H:%M:%S')}"
            )
        )
        # osveži vsakih 1000 ms
        self.root.after(1000, self.update_status_bar)
        
    def get_led_color(self, thread, reconnecting):
        if reconnecting:
            return "orange"
        if not thread.cap or not thread.cap.isOpened():
            return "red"
        return "green"

    # ---------------- Exit -----------------
    def exit_app(self):
        for t in self.threads.values():
            t.stop()
        self.root.destroy()
        print("Program zaključen.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    gui = YoloGUI(root)
    root.state("zoomed")
    root.mainloop()