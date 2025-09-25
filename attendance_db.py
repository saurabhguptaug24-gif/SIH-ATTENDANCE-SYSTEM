"""
 Attendance API Backend (MongoDB Version)
- Unified Face Recognition + OCR + RFID attendance system
- Dedicated admin endpoints for student enrollment
- Persistent student database via MongoDB
- Better defaults (FR threshold 0.60), clamped confidences, robust OCR parsing
- Configurable via Config dataclass and optional request params
- Safer exception handling and clearer logging with Request IDs
- Added concurrency safety via MongoDB's atomic operations
- Added basic security with API Key for admin endpoints
- Enabled CORS for cross-domain communication
- Face Recognition and OCR are now optional dependencies.
- Added /api/features endpoint for frontend to check available features.
- Ready for production deployment with Gunicorn, using environment variables for configuration.

Requirements (pip):
Flask, Flask-PyMongo, opencv-python, numpy, pandas, face-recognition, easyocr, pytesseract, fuzzywuzzy, python-Levenshtein, tqdm, python-dotenv, gunicorn, Flask-Cors
"""

import csv
import json
import logging
import os
import re
import sys
import tempfile
import uuid
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, abort, g
from fuzzywuzzy import fuzz
from flask_pymongo import PyMongo, ObjectId
from pymongo.errors import ConnectionFailure, DuplicateKeyError, OperationFailure
from pymongo import ASCENDING
from flask_cors import CORS

# Try imports that may not be available in some test envs
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: face_recognition not available. Face recognition disabled.")
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available. OCR functionality may be limited.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR functionality may be limited.")

OCR_AVAILABLE = EASYOCR_AVAILABLE or PYTESSERACT_AVAILABLE

try:
    from filelock import FileLock # Not used in this version but kept for consistency
except ImportError:
    pass

warnings.filterwarnings('ignore', category=UserWarning)

# ---------------- CONFIG ----------------
@dataclass
class Config:
    """
    Configuration settings for the Attendance API.
    """
    # General
    mongodb_uri: str = os.getenv("MONGO_URI")
    mongodb_collection_students: str = "students"
    mongodb_collection_attendance: str = "attendance"
    workers: int = 4
    max_width: int = 1600
    
    # Security
    api_key: str = os.getenv("API_KEY")
    
    # Face recognition
    fr_threshold: float = 0.60
    min_votes: int = 1

    # OCR
    ocr_languages: List[str] = None
    use_gpu_easyocr: bool = False
    ocr_confidence_threshold: float = 0.3
    fuzzy_match_threshold: int = 80
    min_cell_width: int = 40
    min_cell_height: int = 20
    vertical_kernel_ratio: int = 50
    horizontal_kernel_ratio: int = 50
    row_threshold: int = 15

    # Preprocessing toggles
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_contrast_enhancement: bool = True

    # Runtime
    debug: bool = False

    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ["en"]


# ---------------- LOGGER ----------------
logger = logging.getLogger("attendance_api")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (ReqID:%(req_id)s) %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.req_id = getattr(g, 'request_id', 'N/A')
        return True

logger.addFilter(RequestIdFilter())


# ---------------- Utilities ----------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamps a float value between a lower and upper bound."""
    return max(lo, min(hi, float(x)))

def resize_image(image: np.ndarray, max_width: int) -> np.ndarray:
    """Resizes an image to a maximum width while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def make_error_response(message: str, error_type: str, status_code: int) -> Tuple[Any, int]:
    """Standardizes error response structure."""
    return jsonify({
        "status": "error",
        "error_type": error_type,
        "message": message,
        "req_id": g.request_id
    }), status_code

def validate_student_id(sid: str) -> bool:
    """Validates student ID as alphanumeric only (A-Z, a-z, 0-9)."""
    return bool(re.fullmatch(r'^[a-zA-Z0-9]+$', sid))

def validate_rfid_tag(tag: str) -> bool:
    """Validates RFID tag as fixed length 10 hex characters (example)."""
    return bool(re.fullmatch(r'^[0-9a-fA-F]{10}$', tag))

# ---------------- Database Connection ----------------
app = Flask(__name__)
# Enable CORS for all routes, allowing cross-origin requests from the Vercel frontend.
CORS(app) 
config = Config()

if not config.mongodb_uri:
    logger.critical("MONGO_URI environment variable is not set. Exiting.")
    sys.exit(1)
if not config.api_key:
    logger.critical("API_KEY environment variable is not set. Exiting.")
    sys.exit(1)

app.config["MONGO_URI"] = config.mongodb_uri
mongo = PyMongo(app)

# ---------------- API Core Logic (Updated for MongoDB) ----------------
class AttendanceManager:
    def __init__(self, mongo_client):
        self.attendance_collection = mongo_client.db[config.mongodb_collection_attendance]

    def ensure_day(self, date_str: str, names: List[str]):
        """
        Ensures a day's attendance record exists and includes all students.
        This operation is atomic and won't overwrite existing 'Present' statuses.
        """
        # Create an update document for all names, setting them to Absent if they don't exist
        initial_data = {
            f"data.{n}": {"status": "Absent", "method": "none", "timestamp": None} 
            for n in names
        }

        self.attendance_collection.update_one(
            {"date": date_str},
            {"$set": initial_data},
            upsert=True
        )
        logger.debug(f"Ensured attendance record for {date_str} is complete.")

    def mark(self, date_str: str, student_name: str, method: str = "manual"):
        """Marks a student as 'Present' for a given day using MongoDB's atomic operations."""
        ts = datetime.now().isoformat()
        
        # Atomically update the student's status only if it's currently 'Absent'
        self.attendance_collection.update_one(
            {"date": date_str, f"data.{student_name}.status": "Absent"},
            {"$set": {
                f"data.{student_name}.status": "Present",
                f"data.{student_name}.method": method,
                f"data.{student_name}.timestamp": ts,
            }}
        )
        logger.debug(f"Attempted to mark {student_name} present by {method} for {date_str}.")

    def update_manual(self, date_str: str, updates: List[Dict[str, str]]):
        """
        Updates attendance records for a list of students, overriding any existing status.
        This is for manual overrides by a teacher.
        """
        ts = datetime.now().isoformat()
        # Create a single update document with all student updates
        update_doc = {
            f"data.{u['name']}": {
                "status": u["status"],
                "method": "manual",
                "timestamp": ts
            }
            for u in updates
        }

        self.attendance_collection.update_one(
            {"date": date_str},
            {"$set": update_doc}
        )
        logger.info(f"Manual attendance updates for {len(updates)} students applied for {date_str}.")

    def get_day(self, date_str: str) -> Dict[str, Dict[str, Any]]:
        """Retrieves attendance data for a specific day."""
        doc = self.attendance_collection.find_one({"date": date_str})
        return doc.get("data", {}) if doc else {}

    def get_summary(self, date_str: str) -> Dict[str, int]:
        """Calculates and returns summary statistics for a given day."""
        all_students = list(student_db.get_all_students().keys())
        total_students = len(all_students)
        
        doc = self.attendance_collection.find_one({"date": date_str})
        
        if not doc:
            return {
                "total_students": total_students,
                "total_present": 0,
                "total_absent": total_students,
            }
            
        attendance_data = doc.get("data", {})
        present_count = sum(1 for status in attendance_data.values() if status.get("status") == "Present")
        absent_count = total_students - present_count
        
        return {
            "total_students": total_students,
            "total_present": present_count,
            "total_absent": absent_count,
        }

class StudentDBManager:
    def __init__(self, mongo_client):
        self.students_collection = mongo_client.db[config.mongodb_collection_students]
        
    def add_or_update(self, student_id: str, data: Dict[str, Any]):
        """Adds or updates a student record by student_id."""
        self.students_collection.update_one({"student_id": student_id}, {"$set": data}, upsert=True)

    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single student record by ID."""
        return self.students_collection.find_one({"student_id": student_id})

    def get_student_by_rfid(self, rfid: str) -> Optional[Dict[str, Any]]:
        """Retrieves a student record by RFID tag."""
        return self.students_collection.find_one({"rfid": rfid})

    def get_all_students(self) -> Dict[str, Any]:
        """Returns the entire student database."""
        students = self.students_collection.find({})
        return {s["student_id"]: s for s in students}
    
    def get_all_student_names(self) -> List[str]:
        """Returns a list of all student names."""
        return [s.get("name") for s in self.students_collection.find({}, {"name": 1}) if s.get("name")]

    def get_all_encodings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Returns all face encodings and their corresponding student IDs."""
        known_encs = []
        known_sids = []
        for s in self.students_collection.find({}, {"face_encodings": 1, "student_id": 1}):
            encs = s.get("face_encodings", [])
            for enc_list in encs:
                known_encs.append(np.array(enc_list))
                known_sids.append(s["student_id"])
        return known_encs, known_sids

    def add_face_encodings(self, student_id: str, encodings: List[np.ndarray]):
        """Adds new face encodings to an existing student record."""
        # Convert numpy arrays to lists for JSON serialization
        enc_lists = [e.tolist() for e in encodings]
        
        result = self.students_collection.update_one(
            {"student_id": student_id},
            {"$push": {"face_encodings": {"$each": enc_lists}}}
        )
        return result.modified_count > 0


# ---------------- Face Recognition ----------------
# (run_face_rec_api and its helpers remain unchanged)
def match_encoding(enc: np.ndarray, known_encs: List[np.ndarray], known_sids: List[str], threshold: float) -> Optional[Tuple[str, float]]:
    if not FACE_RECOGNITION_AVAILABLE or not known_encs:
        return None
    distances = face_recognition.face_distance(known_encs, enc)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    sid = known_sids[best_idx]
    if best_dist <= threshold:
        confidence = clamp(1.0 - (best_dist / (threshold * 1.5)))
        return sid, confidence
    return None

def process_single_photo_fr(img_path: Path, known_encs: List[np.ndarray], known_sids: List[str], config: Config) -> Set[Tuple[str, float]]:
    if not FACE_RECOGNITION_AVAILABLE:
        return set()
    present: Set[Tuple[str, float]] = set()
    img = cv2.imread(str(img_path))
    if img is None:
        logger.debug(f"Could not read input image {img_path}")
        return present
    img = resize_image(img, config.max_width)
    rgb = img[:, :, ::-1]
    try:
        face_locs = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locs)
    except Exception as e:
        logger.exception(f"Error extracting faces for {img_path}: {e}")
        return present
    for enc in face_encs:
        match = match_encoding(enc, known_encs, known_sids, config.fr_threshold)
        if match:
            sid, conf = match
            present.add((sid, float(clamp(conf))))
    return present

def aggregate_attendance_face(results: List[Set[Tuple[str, float]]], min_votes: int) -> Dict[str, Dict[str, Any]]:
    counts: Dict[str, List[float]] = defaultdict(list)
    for s in results:
        for sid, conf in s:
            counts[sid].append(conf)
    final: Dict[str, Dict[str, Any]] = {}
    for sid, confs in counts.items():
        if len(confs) >= min_votes:
            final[sid] = {
                "presence_count": len(confs),
                "confidence": float(np.mean(confs))
            }
    return final

def run_face_rec_api(students_db: StudentDBManager, input_dir: Path, config: Config, attendance_mgr: AttendanceManager) -> pd.DataFrame:
    if not FACE_RECOGNITION_AVAILABLE:
        return pd.DataFrame()
    known_encs, known_sids = students_db.get_all_encodings()
    student_records = students_db.get_all_students()
    
    if not known_encs:
        logger.warning("No enrolled encodings found; all students will be marked absent.")
        return pd.DataFrame([{"student_id": sid, "name": s.get("name"), "status": "Absent"} for sid, s in student_records.items()])

    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if not input_images:
        logger.warning("No input images provided. All students will be marked absent.")
        return pd.DataFrame([{"student_id": sid, "name": s.get("name"), "status": "Absent"} for sid, s in student_records.items()])

    results = []
    with ProcessPoolExecutor(max_workers=max(1, config.workers)) as exe:
        futures = {exe.submit(process_single_photo_fr, p, known_encs, known_sids, config): p for p in input_images}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                logger.exception(f"Face processing failed for {futures[fut]}: {e}")

    agg = aggregate_attendance_face(results, config.min_votes)
    date_str = str(date.today())
    all_names = [s.get("name") for s in student_records.values() if s.get("name")]
    attendance_mgr.ensure_day(date_str, all_names)

    rows = []
    for sid, s_data in student_records.items():
        name = s_data.get("name", sid)
        rec = agg.get(sid)
        if rec:
            status = 'Present'
            if name: 
                attendance_mgr.mark(date_str, name, method='face')
            conf = rec['confidence']
            cnt = rec['presence_count']
        else:
            status = 'Absent'
            conf = 0.0
            cnt = 0
        rows.append({"student_id": sid, "name": name, "status": status, "presence_count": cnt, "confidence": float(clamp(conf))})

    logger.info(f"Successfully processed face recognition. Marked {len(agg)} students as present.")
    return pd.DataFrame(rows)


# ---------------- OCR ----------------
# (run_ocr_api and its helpers remain unchanged)
@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    method: str

class ImagePreprocessor:
    @staticmethod
    def deskew(gray: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 10:
            return gray
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45: angle = 90 + angle
        if abs(angle) < 0.5: return gray
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    @staticmethod
    def denoise(gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    @staticmethod
    def enhance_contrast(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    def preprocess(self, gray: np.ndarray, config: Config) -> np.ndarray:
        img = gray.copy()
        if config.enable_denoise: img = self.denoise(img)
        if config.enable_contrast_enhancement: img = self.enhance_contrast(img)
        if config.enable_deskew: img = self.deskew(img)
        return img

class OCREngine:
    def __init__(self, config: Config):
        self.config = config
        self.easy_reader = None
        if easyocr is not None:
            try:
                self.easy_reader = easyocr.Reader(self.config.ocr_languages, gpu=self.config.use_gpu_easyocr)
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
    def ocr_easy(self, img: np.ndarray) -> Optional[OCRResult]:
        if not easyocr or not self.easy_reader: return None
        try:
            results = self.easy_reader.readtext(img, detail=1, paragraph=True)
            if not results: return None
            texts = [r[1] for r in results if r[1].strip()]
            confs = [r[2] for r in results if isinstance(r[2], (int, float))]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return OCRResult(text=" ".join(texts).strip(), confidence=avg_conf, bbox=(0, 0, img.shape[1], img.shape[0]), method="easyocr")
        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")
            return None
    def ocr_tesseract(self, img: np.ndarray) -> Optional[OCRResult]:
        if pytesseract is None: return None
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
            texts, confs = [], []
            for i, t in enumerate(data.get('text', [])):
                txt, conf_str = str(t).strip(), data.get('conf', [])[i]
                if not txt: continue
                try: conf = float(conf_str);
                except Exception: continue
                if conf >= 0: confs.append(conf / 100.0); texts.append(txt)
            if not texts: return None
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return OCRResult(text=" ".join(texts), confidence=avg_conf, bbox=(0, 0, img.shape[1], img.shape[0]), method="pytesseract")
        except Exception as e:
            logger.debug(f"pytesseract error: {e}")
            return None
    def ocr_cell(self, img: np.ndarray) -> OCRResult:
        res = None
        res_easy = self.ocr_easy(img)
        if res_easy and res_easy.confidence >= self.config.ocr_confidence_threshold: res = res_easy
        else:
            res_tess = self.ocr_tesseract(img)
            if res_tess and (not res_easy or res_tess.confidence >= res_easy.confidence): res = res_tess
            elif res_easy: res = res_easy
        if res is None: res = OCRResult(text="", confidence=0.0, bbox=(0, 0, 0, 0), method="none")
        return res
    def ocr_full_image(self, img: np.ndarray) -> List[OCRResult]:
        res_easy, res_tess = self.ocr_easy(img), self.ocr_tesseract(img)
        results = []
        if res_easy and res_easy.confidence >= self.config.ocr_confidence_threshold: results.append(res_easy)
        if res_tess and res_tess.confidence >= self.config.ocr_confidence_threshold:
            if not res_easy or res_tess.confidence > res_easy.confidence: results.append(res_tess)
        return results if results else [OCRResult(text="", confidence=0.0, bbox=(0, 0, 0, 0), method="none")]

class TableExtractor:
    def __init__(self, config: Config): self.config = config
    def detect_table_structure(self, gray: np.ndarray) -> np.ndarray:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        v_ratio = max(1, gray.shape[0] // self.config.vertical_kernel_ratio)
        h_ratio = max(1, gray.shape[1] // self.config.horizontal_kernel_ratio)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_ratio))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_ratio, 1))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        v_lines = cv2.dilate(v_lines, vertical_kernel, iterations=1)
        h_lines = cv2.dilate(h_lines, horizontal_kernel, iterations=1)
        mask = cv2.add(v_lines, h_lines)
        return mask
    def extract_cells(self, gray: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cells: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= self.config.min_cell_width and h >= self.config.min_cell_height: cells.append((x, y, w, h))
        return sorted(cells, key=lambda b: (b[1], b[0]))

class AttendanceProcessor:
    def __init__(self, config: Config): self.config = config
    def build_table_rows(self, ocr_results: List[OCRResult], cells: List[Tuple[int, int, int, int]]) -> List[List[str]]:
        rows: List[List[str]] = []
        if not ocr_results or not cells or len(ocr_results) != len(cells): return rows
        current_y, current_row = None, []
        for res, (x, y, w, h) in zip(ocr_results, cells):
            if current_y is None: current_y = y
            if abs(y - current_y) > self.config.row_threshold:
                if current_row: rows.append(current_row)
                current_row, current_y = [], y
            text_to_add = res.text.strip() if res.confidence >= self.config.ocr_confidence_threshold else ""
            current_row.append(text_to_add)
        if current_row: rows.append(current_row)
        return rows
    def identify_name_column(self, table_rows: List[List[str]], db_df: pd.DataFrame) -> int:
        if not table_rows or 'Student Name' not in db_df.columns: return -1
        db_names = [str(n).strip().lower() for n in db_df['Student Name'].dropna()]
        if not db_names: return -1
        num_cols, scores = max((len(r) for r in table_rows), default=0), []
        for col in range(num_cols):
            col_items = [r[col].strip().lower() for r in table_rows if col < len(r) and r[col].strip()]
            if not col_items: scores.append(0); continue
            total_ratio, matches_count = 0, 0
            for item in col_items:
                best_ratio = max((fuzz.token_sort_ratio(item, dn) for dn in db_names), default=0)
                total_ratio += best_ratio
                if best_ratio >= self.config.fuzzy_match_threshold: matches_count += 1
            avg_ratio, coverage_factor = total_ratio / len(col_items), matches_count / len(col_items)
            scores.append(avg_ratio * coverage_factor)
        if scores:
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > 50 * (1 / num_cols if num_cols > 0 else 1): return best_idx
        return -1
    def match_and_mark(self, ocr_names: List[str], db_df: pd.DataFrame) -> pd.DataFrame:
        records, cleaned_ocr_names = [], {n.strip().lower() for n in ocr_names if n.strip()}
        for _, row in db_df.iterrows():
            db_name, status, best_score, best_match = str(row.get('Student Name', '')).strip(), 'Absent', 0, ''
            for o_name in cleaned_ocr_names:
                s = fuzz.token_sort_ratio(db_name.lower(), o_name)
                if s > best_score: best_score, best_match = s, o_name
            if best_score >= self.config.fuzzy_match_threshold: status = 'Present'
            records.append({
                'Roll No.': row.get('Roll No.', ''), 'Student Name': db_name, 'Class': row.get('Class', ''), 'Status': status, 'Match Score': best_score,
                'Matched OCR Text': best_match if status == 'Present' else ''
            })
        return pd.DataFrame(records)

def run_ocr_api(students_db: StudentDBManager, image_path: Path, config: Config, attendance_mgr: AttendanceManager) -> pd.DataFrame:
    pre, engine, extractor, proc = ImagePreprocessor(), OCREngine(config), TableExtractor(config), AttendanceProcessor(config)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None: raise FileNotFoundError("Attendance sheet image could not be read")
    processed, mask, cells = pre.preprocess(gray, config), extractor.detect_table_structure(processed), extractor.extract_cells(processed, mask)
    ocr_results = []
    if cells:
        with ThreadPoolExecutor(max_workers=max(1, config.workers)) as exe:
            futures = [exe.submit(engine.ocr_cell, processed[y:y+h, x:x+w]) for (x, y, w, h) in cells]
            for fut in as_completed(futures):
                try: ocr_results.append(fut.result())
                except Exception as e: logger.exception(f"OCR cell failed: {e}")
        table_rows = proc.build_table_rows(ocr_results, cells)
    else:
        logger.warning("No table cells detected. Falling back to full image OCR.")
        ocr_results = engine.ocr_full_image(processed)
        table_rows = []
    student_data = students_db.get_all_students()
    if not student_data: logger.error("Student database is empty."); return pd.DataFrame()
    db_df = pd.DataFrame([{"student_id": sid, "Roll No.": sid, "Student Name": s.get("name"), "Class": s.get("class")} for sid, s in student_data.items()])
    ocr_names, name_col = [], -1
    if table_rows: name_col = proc.identify_name_column(table_rows, db_df)
    if name_col != -1 and table_rows: ocr_names = [row[name_col] for row in table_rows if name_col < len(row) and row[name_col].strip()]
    else: ocr_names = [r.text for r in ocr_results if r.confidence >= config.ocr_confidence_threshold and r.text.strip()]
    if not ocr_names:
        logger.warning("No high-confidence text extracted by OCR for matching.")
        return pd.DataFrame([{'Roll No.': s.get('student_id'), 'Student Name': s.get('name'), 'Class': s.get('class'), 'Status': 'Absent', 'Match Score': 0, 'Matched OCR Text': ''} for s in student_data.values()])
    attendance_df = proc.match_and_mark(ocr_names, db_df)
    date_str = str(date.today())
    all_names = [s.get("name") for s in student_data.values() if s.get("name")]
    attendance_mgr.ensure_day(date_str, all_names)
    for _, r in attendance_df.iterrows():
        if r['Status'] == 'Present': attendance_mgr.mark(date_str, r['Student Name'], method='ocr')
    attendance_df['Date'] = date_str
    logger.info(f"Successfully processed OCR attendance. Marked {len(attendance_df[attendance_df['Status'] == 'Present'])} students as present.")
    return attendance_df


# Instantiate the DB and Attendance Managers here
student_db = StudentDBManager(mongo)
attendance_mgr = AttendanceManager(mongo)

# ---------------- API Endpoints ----------------
@app.before_request
def assign_request_id():
    g.request_id = str(uuid.uuid4())[:8]
    logger.info(f"Incoming {request.method} {request.path}")

def require_api_key(f):
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if key != config.api_key:
            logger.warning("Access Forbidden: Invalid API Key provided.")
            return make_error_response("Forbidden: Invalid API Key", "AuthenticationError", 403)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route("/", methods=["GET"])
def home():
    try:
        if mongo.cx:
            mongo.cx.server_info() # Check connection
        db_status = "connected"
    except ConnectionFailure:
        db_status = "failed"
    return jsonify({"status": "ok", "message": "Attendance API running", "database_status": db_status, "req_id": g.request_id})

@app.route("/api/attendance/summary", methods=["GET"])
def api_get_attendance_summary():
    """API endpoint to get summary statistics for a given day."""
    date_str = request.args.get('date', str(date.today()))
    try:
        summary = attendance_mgr.get_summary(date_str)
        return jsonify({"status": "success", "date": date_str, "data": summary, "req_id": g.request_id})
    except Exception as e:
        logger.exception("Error getting attendance summary.")
        return make_error_response(f"Internal error retrieving attendance summary: {e}", "InternalError", 500)

@app.route("/api/attendance/export", methods=["GET"])
def api_export_attendance():
    """API endpoint to export attendance data as a CSV file."""
    date_str = request.args.get('date', str(date.today()))
    
    attendance_data = attendance_mgr.get_day(date_str)
    if not attendance_data:
        return jsonify({"status": "warning", "message": f"No attendance data found for {date_str} to export."}), 404

    df_data = []
    for name, record in attendance_data.items():
        df_data.append({
            "name": name,
            "status": record.get("status"),
            "method": record.get("method"),
            "timestamp": record.get("timestamp"),
            "date": date_str
        })
    df = pd.DataFrame(df_data)
    
    # Create an in-memory CSV file
    csv_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
    df.to_csv(csv_file.name, index=False)
    csv_file.close()
    
    # Return the file as a response
    try:
        from flask import send_file
        return send_file(
            csv_file.name,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"attendance_{date_str}.csv"
        )
    except ImportError:
        logger.error("send_file is not available. Check your Flask installation.")
        return make_error_response("File export not supported.", "InternalError", 500)
    finally:
        os.unlink(csv_file.name)


@app.route("/api/get_attendance", methods=["GET"])
def api_get_attendance():
    date_str = request.args.get('date', str(date.today()))
    try:
        attendance_data = attendance_mgr.get_day(date_str)
        if not attendance_data: return jsonify({"status": "warning", "message": f"No attendance data found for {date_str}", "req_id": g.request_id}), 404
        return jsonify({"status": "success", "date": date_str, "data": attendance_data, "req_id": g.request_id})
    except Exception as e:
        logger.exception("Error getting attendance data")
        return make_error_response(f"Internal error retrieving attendance: {e}", "InternalError", 500)
        
@app.route("/api/students/names", methods=["GET"])
@require_api_key
def api_get_student_names():
    """Returns a list of all student names for the frontend to render manual attendance forms."""
    try:
        names = student_db.get_all_student_names()
        return jsonify({"status": "success", "data": names, "req_id": g.request_id}), 200
    except Exception as e:
        logger.exception("Error getting student names")
        return make_error_response(f"Internal error retrieving student names: {e}", "InternalError", 500)

@app.route("/api/students", methods=["GET"])
@require_api_key
def api_get_students():
    try:
        return jsonify({"status": "success", "data": student_db.get_all_students(), "req_id": g.request_id}), 200
    except Exception as e:
        logger.exception("Error getting student data")
        return make_error_response(f"Internal error retrieving student data: {e}", "InternalError", 500)

@app.route("/api/enroll/rfid", methods=["POST"])
@require_api_key
def api_enroll_rfid():
    data = request.json
    if not all(field in data for field in ["student_id", "name", "rfid"]): return make_error_response("Missing required fields: student_id, name, and rfid", "MissingField", 400)
    student_id, name, rfid = str(data.get("student_id")).strip(), str(data.get("name")).strip(), str(data.get("rfid")).strip()
    if not validate_student_id(student_id): return make_error_response("Invalid student_id. Must be alphanumeric.", "ValidationError", 400)
    if not validate_rfid_tag(rfid): return make_error_response("Invalid rfid tag format. Must be a 10-char hex string.", "ValidationError", 400)
    student_record = {"student_id": student_id, "name": name, "rfid": rfid, "face_encodings": []}
    try:
        student_db.add_or_update(student_id, student_record)
        return jsonify({"status": "success", "message": "Student record created/updated successfully", "req_id": g.request_id}), 201
    except Exception as e:
        logger.exception("RFID enrollment failed")
        return make_error_response(f"Internal error during RFID enrollment: {e}", "InternalError", 500)

@app.route("/api/enroll/face", methods=["POST"])
@require_api_key
def api_enroll_face():
    student_id = request.form.get("student_id")
    if not FACE_RECOGNITION_AVAILABLE:
        return make_error_response("Face recognition not available on this server.", "FeatureUnavailable", 503)
    if not student_id: return make_error_response("Missing student_id in form data", "MissingField", 400)
    if not validate_student_id(student_id): return make_error_response("Invalid student_id. Must be alphanumeric.", "ValidationError", 400)
    if not student_db.get_student_by_id(student_id): return make_error_response(f"Student ID {student_id} not found in database. Enroll via RFID/initial DB first.", "NotFoundError", 404)
    if "photo" not in request.files: return make_error_response("Missing 'photo' file", "MissingField", 400)
    photo_file = request.files["photo"]
    
    # Secure file saving to prevent path traversal
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / (uuid.uuid4().hex + Path(photo_file.filename).suffix)
        photo_file.save(tmp_path)
        
        try:
            img = cv2.imread(str(tmp_path))
            if img is None: return make_error_response("Could not read the uploaded image", "FileError", 400)
            rgb_image, face_locations = img[:, :, ::-1], face_recognition.face_locations(img[:, :, ::-1])
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if len(face_encodings) != 1:
                msg = f"{len(face_encodings)} faces detected in photo. Enrollment requires exactly one face."
                return make_error_response(msg, "ValidationError", 400)
            student_db.add_face_encodings(student_id, face_encodings)
            return jsonify({"status": "success", "message": f"Added {len(face_encodings)} face encodings for student {student_id}", "req_id": g.request_id}), 200
        except Exception as e:
            logger.exception("Face encoding failed for enrollment")
            return make_error_response(f"Internal error during face enrollment: {e}", "InternalError", 500)

@app.route("/api/process", methods=["POST"])
@require_api_key
def api_process():
    if not FACE_RECOGNITION_AVAILABLE:
        return make_error_response("Face recognition not available on this server.", "FeatureUnavailable", 503)
    input_photos_files = request.files.getlist('input_photos')
    if not input_photos_files: return make_error_response("Missing required file: input_photos", "MissingFile", 400)
    try:
        fr_threshold = clamp(float(request.form.get('fr_threshold', config.fr_threshold)), 0.0, 1.0)
        min_votes = max(1, int(request.form.get('min_votes', config.min_votes)))
    except (ValueError, TypeError): return make_error_response("fr_threshold must be a number and min_votes must be an integer", "ValidationError", 400)
    current_fr_threshold, current_min_votes = config.fr_threshold, config.min_votes
    config.fr_threshold, config.min_votes = fr_threshold, min_votes
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / 'input_photos'
            input_dir.mkdir(parents=True, exist_ok=True)
            for f in input_photos_files: 
                # Secure file saving
                filename = uuid.uuid4().hex + Path(f.filename).suffix
                f.save(input_dir / filename)
            attendance_df = run_face_rec_api(student_db, input_dir, config, attendance_mgr)
            return jsonify({"status": "success", "data": attendance_df.to_dict(orient='records'), "req_id": g.request_id}), 200
    except Exception as e:
        logger.exception("/api/process failed")
        return make_error_response(f"Internal error during face recognition process: {e}", "InternalError", 500)
    finally:
        config.fr_threshold, config.min_votes = current_fr_threshold, current_min_votes

@app.route("/api/rfid_scan", methods=["POST"])
def api_rfid():
    rfid_tag = request.form.get('rfid_tag')
    if not rfid_tag: return make_error_response("RFID tag not provided in form data.", "MissingField", 400)
    if not validate_rfid_tag(rfid_tag): return make_error_response("Invalid rfid tag format.", "ValidationError", 400)
    try:
        student = student_db.get_student_by_rfid(rfid_tag)
        if student:
            date_str = str(date.today())
            all_names = [s.get("name") for s in student_db.get_all_students().values() if s.get("name")]
            attendance_mgr.ensure_day(date_str, all_names)
            attendance_mgr.mark(date_str, student['name'], method='rfid')
            return jsonify({"status": "success", "student_name": student['name'], "message": f"Attendance logged for {student['name']}", "req_id": g.request_id}), 200
        else:
            logger.warning(f"Unknown RFID tag scanned: {rfid_tag}")
            return jsonify({"status": "warning", "message": f"Unknown RFID tag {rfid_tag}", "req_id": g.request_id}), 404
    except Exception as e:
        logger.exception("/api/rfid_scan failed")
        return make_error_response(f"Internal error during RFID scan: {e}", "InternalError", 500)

@app.route("/api/ocr_process", methods=["POST"])
@require_api_key
def api_ocr():
    if not OCR_AVAILABLE:
        return make_error_response("OCR not available on this server.", "FeatureUnavailable", 503)
    if 'attendance_sheet' not in request.files: return make_error_response("Missing required file: attendance_sheet", "MissingFile", 400)
    attendance_sheet = request.files['attendance_sheet']
    try:
        fuzzy_thr = int(request.form.get('fuzzy_threshold', config.fuzzy_match_threshold))
        if not 0 <= fuzzy_thr <= 100: return make_error_response("fuzzy_threshold must be an integer between 0 and 100", "ValidationError", 400)
    except ValueError: return make_error_response("fuzzy_threshold must be an integer", "ValidationError", 400)
    current_fuzzy_thr = config.fuzzy_match_threshold
    config.fuzzy_match_threshold = fuzzy_thr
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sheet_path = Path(tmp_dir) / (uuid.uuid4().hex + Path(attendance_sheet.filename).suffix)
            attendance_sheet.save(sheet_path)
            attendance_df = run_ocr_api(student_db, sheet_path, config, attendance_mgr)
            if attendance_df.empty: return jsonify({"status": "warning", "message": "OCR process completed but no attendance data was generated.", "req_id": g.request_id}), 200
            return jsonify({"status": "success", "data": attendance_df.to_dict(orient='records'), "req_id": g.request_id}), 200
    except Exception as e:
        logger.exception("/api/ocr_process failed")
        return make_error_response(f"Internal error during OCR process: {e}", "InternalError", 500)
    finally: config.fuzzy_match_threshold = current_fuzzy_thr

@app.route("/api/manual_attendance", methods=["POST"])
@require_api_key
def api_manual_attendance():
    data = request.json
    required_fields = ["date", "updates"]
    if not all(field in data for field in required_fields):
        return make_error_response("Missing required fields: date and updates", "MissingField", 400)
    
    date_str = data.get("date")
    updates = data.get("updates")
    
    if not isinstance(updates, list):
        return make_error_response("Updates must be a list of objects.", "ValidationError", 400)

    try:
        attendance_mgr.update_manual(date_str, updates)
        return jsonify({"status": "success", "message": f"Successfully updated manual attendance for {len(updates)} students.", "req_id": g.request_id}), 200
    except Exception as e:
        logger.exception("Manual attendance update failed.")
        return make_error_response(f"Internal error updating manual attendance: {e}", "InternalError", 500)
    
@app.route("/api/features", methods=["GET"])
def api_get_features():
    """Returns a list of available features."""
    return jsonify({
        "status": "success",
        "data": {
            "face_recognition": FACE_RECOGNITION_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
            "pytesseract": PYTESSERACT_AVAILABLE,
            "ocr": OCR_AVAILABLE,
        },
        "req_id": g.request_id
    })


if __name__ == '__main__':
    logger.info("Starting Attendance API")
    if config.debug: logger.setLevel(logging.DEBUG)
    
    # Check if a MongoDB connection is possible at startup
    try:
        mongo.cx.server_info()
        logger.info("Successfully connected to MongoDB.")
        
        # Ensure collections and indexes exist
        students_collection = mongo.db[config.mongodb_collection_students]
        attendance_collection = mongo.db[config.mongodb_collection_attendance]
        
        # Create unique index to prevent duplicate student_id entries
        students_collection.create_index("student_id", unique=True)
        students_collection.create_index("rfid", unique=True, sparse=True) # RFID is optional, so we use sparse
        
        # Create a unique index for attendance to ensure only one document per day
        attendance_collection.create_index("date", unique=True)
        
    except ConnectionFailure:
        logger.critical("Failed to connect to MongoDB at startup.")
        sys.exit(1)
    except OperationFailure as e:
        logger.warning(f"Failed to create MongoDB index: {e}")
        
    app.run(host='0.0.0.0', port=5000, debug=config.debug)

