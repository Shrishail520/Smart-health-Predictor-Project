from flask import Flask, render_template, request, redirect, url_for, session, send_file, send_from_directory, abort, jsonify
from datetime import datetime
import os
import secrets
import csv
import smtplib
import ssl
import urllib.request
import urllib.error
from email.message import EmailMessage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
import io
import json
import sqlite3
import uuid
import base64
import re
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

app = Flask(__name__)
secret_key = os.getenv("FLASK_SECRET_KEY")
if not secret_key:
    secret_key = secrets.token_hex(32)
    print("WARNING: FLASK_SECRET_KEY not set. Using a temporary secret key for this run.")
app.config["SECRET_KEY"] = secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_ROOT, "smart_health.db")
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_IMAGE_FILE_SIZE = 3 * 1024 * 1024
MAX_CAMERA_IMAGE_SIZE = 3 * 1024 * 1024
DATASET_PATH = os.path.join(APP_ROOT, "symptom_dataset.csv")
SYMPTOM_FEATURES = [
    "fever", "fatigue", "bodypain", "chills", "sweating", "weightloss",
    "cough", "cold", "breath", "chest", "sorethroat", "wheezing",
    "nausea", "vomiting", "diarrhea", "abdominal", "acidity", "appetite",
    "headache", "dizziness", "blurred", "confusion", "sleep",
    "jointpain", "swelling", "stiffness", "rash", "itching", "dryskin",
]

notifications = {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ML_MODEL = None

def write_default_dataset_if_missing():
    if os.path.exists(DATASET_PATH):
        return
    rows = [
        {"disease": "Respiratory Infection", "fever": 1, "cough": 1, "breath": 1, "sorethroat": 1, "wheezing": 1, "chills": 1},
        {"disease": "Respiratory Infection", "fever": 1, "cough": 1, "breath": 1, "chills": 1},
        {"disease": "Respiratory Infection", "fever": 1, "cough": 1, "breath": 1, "sweating": 1},
        {"disease": "Common Cold", "cold": 1, "cough": 1, "sorethroat": 1, "headache": 1},
        {"disease": "Common Cold", "cold": 1, "cough": 1, "fatigue": 1},
        {"disease": "Common Cold", "cold": 1, "cough": 1, "chills": 1},
        {"disease": "Heart Related Issue", "chest": 1, "breath": 1, "sweating": 1, "dizziness": 1},
        {"disease": "Heart Related Issue", "chest": 1, "breath": 1, "fatigue": 1},
        {"disease": "Heart Related Issue", "chest": 1, "dizziness": 1, "sweating": 1},
        {"disease": "Digestive Disorder", "nausea": 1, "vomiting": 1, "diarrhea": 1, "abdominal": 1},
        {"disease": "Digestive Disorder", "acidity": 1, "abdominal": 1, "appetite": 1},
        {"disease": "Digestive Disorder", "nausea": 1, "abdominal": 1, "diarrhea": 1},
        {"disease": "Neurological Issue", "headache": 1, "dizziness": 1, "blurred": 1, "confusion": 1},
        {"disease": "Neurological Issue", "headache": 1, "sleep": 1, "dizziness": 1},
        {"disease": "Neurological Issue", "confusion": 1, "blurred": 1, "headache": 1},
        {"disease": "Skin Allergy", "rash": 1, "itching": 1, "dryskin": 1},
        {"disease": "Skin Allergy", "rash": 1, "itching": 1},
        {"disease": "Skin Allergy", "itching": 1, "dryskin": 1},
        {"disease": "General Checkup Recommended", "fatigue": 1, "bodypain": 1},
        {"disease": "General Checkup Recommended", "sleep": 1, "fatigue": 1},
        {"disease": "General Checkup Recommended", "jointpain": 1, "stiffness": 1, "swelling": 1},
    ]
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SYMPTOM_FEATURES + ["disease"])
        writer.writeheader()
        for row in rows:
            full_row = {feature: 0 for feature in SYMPTOM_FEATURES}
            full_row.update({k: int(v) for k, v in row.items() if k in SYMPTOM_FEATURES})
            full_row["disease"] = row["disease"]
            writer.writerow(full_row)

def load_ml_training_data():
    X = []
    y = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = set(SYMPTOM_FEATURES + ["disease"])
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError("symptom_dataset.csv has missing columns.")
        for row in reader:
            X.append([int(row.get(feature, 0) or 0) for feature in SYMPTOM_FEATURES])
            y.append((row.get("disease") or "").strip())
    if len(set(y)) < 2:
        raise ValueError("Dataset must contain at least two diseases.")
    return X, y

def init_ml_model():
    global ML_MODEL
    if LogisticRegression is None:
        print("ML model disabled: scikit-learn is not installed.")
        return
    try:
        write_default_dataset_if_missing()
        X, y = load_ml_training_data()
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        ML_MODEL = model
        print(f"ML model ready using dataset: {os.path.basename(DATASET_PATH)}")
    except Exception as e:
        ML_MODEL = None
        print(f"ML model disabled: {e}")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                role TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                name TEXT,
                age INTEGER,
                contact TEXT,
                email TEXT,
                symptoms TEXT,
                disease TEXT,
                remarks TEXT,
                updated_at TEXT,
                prescription TEXT,
                prescribed_at TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                filename TEXT,
                uploaded_at TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                message TEXT,
                created_at TEXT
            )
            """
        )

init_db()

def migrate_users_email():
    with get_db() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "email" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email)")

def seed_demo_users():
    demo_users = [
        ("doctor1", "", "doctor123", "admin"),
        ("patient1", "patient1@example.com", "patient123", "user"),
    ]
    with get_db() as conn:
        for username, email, password, role in demo_users:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE users SET email = ?, password_hash = ?, role = ? WHERE id = ?",
                    (email, generate_password_hash(password), role, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (username, email, generate_password_hash(password), role),
                )

migrate_users_email()
seed_demo_users()
init_ml_model()

def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))

def is_valid_contact(contact):
    return bool(re.match(r"^[0-9+\-\s]{7,15}$", contact or ""))

def is_allowed_image(filename):
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def file_size_bytes(file_storage):
    try:
        pos = file_storage.stream.tell()
        file_storage.stream.seek(0, os.SEEK_END)
        size = file_storage.stream.tell()
        file_storage.stream.seek(pos)
        return size
    except Exception:
        return None

def validate_upload(file_storage):
    if not file_storage or not file_storage.filename:
        return None
    safe_name = secure_filename(file_storage.filename)
    if not safe_name or not is_allowed_image(safe_name):
        return "Only PNG, JPG, JPEG, GIF, and WEBP images are allowed."
    size = file_size_bytes(file_storage)
    if size is not None and size > MAX_IMAGE_FILE_SIZE:
        return "Each uploaded image must be 3MB or smaller."
    return None

def validate_camera_data(data_url):
    if not data_url:
        return None
    match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", data_url)
    if not match:
        return "Captured photo is not in a valid image format."
    b64_data = match.group(2)
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        return "Captured photo data is invalid."
    if len(raw) > MAX_CAMERA_IMAGE_SIZE:
        return "Captured image must be 3MB or smaller."
    return None

def save_upload(file_storage, patient_id):
    if not file_storage or not file_storage.filename:
        return None
    safe_name = secure_filename(file_storage.filename)
    if not safe_name or not is_allowed_image(safe_name):
        return None
    unique_name = f"{patient_id}_{uuid.uuid4().hex}_{safe_name}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    file_storage.save(path)
    return unique_name

def save_camera_data(data_url, patient_id):
    if not data_url:
        return None
    match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", data_url)
    if not match:
        return None
    ext = match.group(1)
    if ext == "jpeg":
        ext = "jpg"
    b64_data = match.group(2)
    try:
        raw = base64.b64decode(b64_data)
    except Exception:
        return None
    unique_name = f"{patient_id}_{uuid.uuid4().hex}_camera.{ext}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    with open(path, "wb") as f:
        f.write(raw)
    return unique_name

# ---------------- LOGIN ----------------
@app.route("/")
def login_choice():
    return render_template("login_choice.html")

def handle_login(role, template_name):
    error = None
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        with get_db() as conn:
            row = conn.execute(
                "SELECT username, password_hash, role FROM users WHERE username = ?",
                (u,),
            ).fetchone()

        if row and row["role"] == role and check_password_hash(row["password_hash"], p):
            session["role"] = role
            session["username"] = u
            return redirect(url_for("admin_dashboard" if role == "admin" else "user_dashboard"))
        else:
            error = "Invalid username or password"

    return render_template(template_name, error=error, success=request.args.get("success"))

@app.route("/login/doctor", methods=["GET", "POST"])
def login_doctor():
    return handle_login("admin", "login_doctor.html")

@app.route("/login/patient", methods=["GET", "POST"])
def login_patient():
    error = None
    success = request.args.get("success")
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        p = request.form["password"]

        if not email or not p:
            error = "Email and password are required"
            return render_template("login_patient.html", error=error)

        with get_db() as conn:
            row = conn.execute(
                "SELECT username, email, password_hash, role FROM users WHERE LOWER(email) = ?",
                (email,),
            ).fetchone()

            if row:
                if row["role"] != "user":
                    error = "This email is not a patient account"
                elif check_password_hash(row["password_hash"], p):
                    session["role"] = "user"
                    session["username"] = row["username"]
                    return redirect(url_for("user_dashboard"))
                else:
                    error = "Invalid email or password"
            else:
                error = "Account not found. Please sign up first."

    return render_template("login_patient.html", error=error, success=success)

@app.route("/signup/patient", methods=["GET", "POST"])
def signup_patient():
    error = None
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        p = request.form["password"]

        if not email or not p:
            error = "Email and password are required"
            return render_template("signup_patient.html", error=error)
        if not is_valid_email(email):
            error = "Please enter a valid email address"
            return render_template("signup_patient.html", error=error)
        if len(p) < 6:
            error = "Password must be at least 6 characters"
            return render_template("signup_patient.html", error=error)

        with get_db() as conn:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE LOWER(email) = ?",
                (email,),
            ).fetchone()
            if existing:
                error = "An account with this email already exists"
            else:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (email, email, generate_password_hash(p), "user"),
                )
                return redirect(url_for("login_patient", success=1))

    return render_template("signup_patient.html", error=error)

# ---------------- USER DASHBOARD ----------------
@app.route("/user", methods=["GET", "POST"])
def user_dashboard():
    if session.get("role") != "user":
        return redirect(url_for("login_choice"))

    notification = None
    latest_record = None
    latest_photos = []
    history_records = []
    history_photos = {}
    form_error = None

    if request.method == "POST":
        name = request.form["name"].strip()
        age_raw = request.form["age"].strip()
        contact = request.form["contact"].strip()
        email = request.form.get("email", "").strip().lower()
        symptoms = [s.strip().lower() for s in request.form.getlist("symptoms") if s.strip()]
        files = request.files.getlist("photos")
        camera_data = request.form.get("camera_photo", "")

        if not name:
            form_error = "Patient name is required."
        elif not age_raw.isdigit():
            form_error = "Age must be a valid number."
        else:
            age = int(age_raw)
            if age < 1 or age > 120:
                form_error = "Age must be between 1 and 120."
            elif not is_valid_contact(contact):
                form_error = "Contact must be 7-15 characters and contain only digits, spaces, + or -."
            elif not is_valid_email(email):
                form_error = "Please enter a valid email address."
            elif not symptoms:
                form_error = "Please select at least one symptom."
            else:
                for file_obj in files:
                    upload_error = validate_upload(file_obj)
                    if upload_error:
                        form_error = upload_error
                        break
                if not form_error:
                    form_error = validate_camera_data(camera_data)

        if not form_error:
            disease = predict_disease(symptoms)
            with get_db() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO patients
                    (user, name, age, contact, email, symptoms, disease, remarks, updated_at, prescription, prescribed_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.get("username"),
                        name,
                        int(age_raw),
                        contact,
                        email,
                        json.dumps(symptoms),
                        disease,
                        "",
                        "",
                        "",
                        "",
                        "Pending",
                    ),
                )
                patient_id = cur.lastrowid

                for f in files:
                    filename = save_upload(f, patient_id)
                    if filename:
                        conn.execute(
                            "INSERT INTO photos (patient_id, filename, uploaded_at) VALUES (?, ?, ?)",
                            (patient_id, filename, datetime.now().strftime("%d-%m-%Y %H:%M")),
                        )

                camera_filename = save_camera_data(camera_data, patient_id)
                if camera_filename:
                    conn.execute(
                        "INSERT INTO photos (patient_id, filename, uploaded_at) VALUES (?, ?, ?)",
                        (patient_id, camera_filename, datetime.now().strftime("%d-%m-%Y %H:%M")),
                    )
            session.clear()
            return redirect(url_for("login_doctor", success=1))

    username = session.get("username")
    if username:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE user = ? ORDER BY id DESC LIMIT 1",
                (username,),
            ).fetchone()
            if row:
                latest_record = dict(row)
                latest_record["symptoms"] = json.loads(latest_record.get("symptoms") or "[]")
                latest_photos = conn.execute(
                    "SELECT * FROM photos WHERE patient_id = ? ORDER BY id DESC",
                    (latest_record["id"],),
                ).fetchall()

            all_rows = conn.execute(
                "SELECT * FROM patients WHERE user = ? ORDER BY id DESC",
                (username,),
            ).fetchall()
            for r in all_rows:
                item = dict(r)
                item["symptoms"] = json.loads(item.get("symptoms") or "[]")
                history_records.append(item)

            if history_records:
                ids = [str(p["id"]) for p in history_records]
                placeholders = ",".join("?" for _ in ids)
                photo_rows = conn.execute(
                    f"SELECT * FROM photos WHERE patient_id IN ({placeholders}) ORDER BY id DESC",
                    ids,
                ).fetchall()
                for ph in photo_rows:
                    history_photos.setdefault(ph["patient_id"], []).append(ph)

    if username:
        with get_db() as conn:
            nrow = conn.execute(
                "SELECT message FROM notifications WHERE username = ? ORDER BY id DESC LIMIT 1",
                (username,),
            ).fetchone()
            if nrow:
                notification = nrow["message"]

    return render_template(
        "user_dashboard.html",
        notification=notification,
        latest_record=latest_record,
        latest_photos=latest_photos,
        history_records=history_records,
        history_photos=history_photos,
        form_error=form_error
    )

# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login_choice"))
    success = request.args.get("success")
    with get_db() as conn:
        patient_rows = conn.execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
        photo_rows = conn.execute("SELECT * FROM photos ORDER BY id DESC").fetchall()

    patients = []
    for row in patient_rows:
        item = dict(row)
        item["symptoms"] = json.loads(item.get("symptoms") or "[]")
        patients.append(item)

    photos_by_pid = {}
    for ph in photo_rows:
        photos_by_pid.setdefault(ph["patient_id"], []).append(ph)

    return render_template(
        "admin_dashboard.html",
        patients=patients,
        photos_by_pid=photos_by_pid,
        success=success
    )

# ---------------- UPDATE DISEASE ----------------
@app.route("/update_disease", methods=["POST"])
def update_disease():
    pid = int(request.form["patient_id"])
    disease = request.form["disease"]
    remarks = request.form["remarks"]
    prescription = request.form.get("prescription", "").strip()

    updated_at = datetime.now().strftime("%d-%m-%Y %H:%M")
    with get_db() as conn:
        conn.execute(
            """
            UPDATE patients
            SET disease = ?, remarks = ?, updated_at = ?, prescription = ?, prescribed_at = ?, status = ?
            WHERE id = ?
            """,
            (disease, remarks, updated_at, prescription, updated_at, "Reviewed", pid),
        )
        row = conn.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()

    if not row:
        return redirect(url_for("admin_dashboard", success=0))

    user = row["user"]
    if prescription:
        notifications[user] = (
            f"Doctor updated your diagnosis to: {disease}. "
            f"Remarks: {remarks}. Prescription: {prescription}"
        )
    else:
        notifications[user] = (
            f"Doctor updated your diagnosis to: {disease}. Remarks: {remarks}"
        )

    with get_db() as conn:
        conn.execute(
            "INSERT INTO notifications (username, message, created_at) VALUES (?, ?, ?)",
            (user, notifications[user], updated_at),
        )

    # Demo SMS / Email
    send_sms(row["contact"], notifications[user])
    send_email(row["email"], notifications[user])

    return redirect(url_for("admin_dashboard", success=1))

# ---------------- PDF REPORT ----------------
@app.route("/generate_pdf/<int:pid>")
def generate_pdf(pid):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    if not row:
        abort(404)
    p = dict(row)
    p["symptoms"] = json.loads(p.get("symptoms") or "[]")

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    left = 40
    right = page_width - 40
    inner_width = right - left
    cursor_y = page_height - 42

    def draw_wrapped_line(label, value, y_start, value_font="Helvetica", value_size=10, width=220):
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left, y_start, label)
        lines = simpleSplit((value or "N/A").strip() or "N/A", value_font, value_size, width)
        text_y = y_start
        pdf.setFont(value_font, value_size)
        for line in lines:
            pdf.drawString(left + 52, text_y, line)
            text_y -= 13
        return text_y

    def parse_prescription_rows(text):
        raw = (text or "").strip()
        if not raw:
            return []
        chunks = [c.strip() for c in re.split(r"[\n;]+", raw) if c.strip()]
        rows = []
        for item in chunks:
            parts = [p.strip() for p in item.split(",")]
            if len(parts) >= 3:
                med = parts[0]
                dose = parts[1]
                duration = ", ".join(parts[2:])
            elif len(parts) == 2:
                med = parts[0]
                dose = parts[1]
                duration = "As advised"
            else:
                med = parts[0]
                dose = "As advised"
                duration = "As advised"
            rows.append((med, dose, duration))
        return rows

    # Clinic header
    pdf.setStrokeColor(colors.HexColor("#0f172a"))
    pdf.setFillColor(colors.HexColor("#f8fafc"))
    pdf.roundRect(left, cursor_y - 68, inner_width, 68, 6, fill=1, stroke=1)

    logo_path = os.path.join(APP_ROOT, "static", "logo.png")
    if os.path.exists(logo_path):
        try:
            pdf.drawImage(logo_path, left + 10, cursor_y - 58, width=46, height=46, preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left + 64, cursor_y - 25, "SMART HEALTH CLINIC")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left + 64, cursor_y - 40, "Doctor Prescription Sheet")
    pdf.drawString(left + 64, cursor_y - 54, "24x7 Care | Phone: +91-00000-00000")

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawRightString(right - 10, cursor_y - 24, f"OPD No: SHC-{pid:05d}")
    pdf.setFont("Helvetica", 10)
    pdf.drawRightString(right - 10, cursor_y - 39, f"Date: {datetime.now().strftime('%d-%m-%Y')}")
    pdf.drawRightString(right - 10, cursor_y - 54, f"Time: {datetime.now().strftime('%H:%M')}")

    cursor_y -= 86

    # Patient details band
    pdf.setStrokeColor(colors.HexColor("#94a3b8"))
    pdf.line(left, cursor_y, right, cursor_y)
    cursor_y -= 14
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left, cursor_y, f"Patient: {p.get('name') or 'N/A'}")
    pdf.drawString(left + 230, cursor_y, f"Age: {p.get('age') or 'N/A'}")
    pdf.drawString(left + 320, cursor_y, f"Contact: {p.get('contact') or 'N/A'}")
    cursor_y -= 16
    pdf.drawString(left, cursor_y, f"Status: {p.get('status') or 'Pending'}")
    pdf.drawString(left + 230, cursor_y, f"Last Update: {p.get('updated_at') or 'N/A'}")
    cursor_y -= 14
    pdf.line(left, cursor_y, right, cursor_y)
    cursor_y -= 20

    # Clinical notes
    symptoms_text = ", ".join(p.get("symptoms") or []) or "N/A"
    cursor_y = draw_wrapped_line("Symptoms:", symptoms_text, cursor_y, width=inner_width - 70) - 6
    cursor_y = draw_wrapped_line("Diagnosis:", p.get("disease") or "General Checkup Recommended", cursor_y, width=inner_width - 70) - 6
    cursor_y = draw_wrapped_line("Remarks:", p.get("remarks") or "N/A", cursor_y, width=inner_width - 70) - 8

    # Rx section
    pdf.setFont("Helvetica-Bold", 26)
    pdf.setFillColor(colors.HexColor("#0b3a66"))
    pdf.drawString(left, cursor_y, "Rx")
    cursor_y -= 20

    rows = parse_prescription_rows(p.get("prescription"))
    if not rows:
        rows = [(p.get("prescription") or "No medicine prescribed", "As advised", "As advised")]

    table_top = cursor_y
    row_h = 20
    med_w = inner_width * 0.45
    dose_w = inner_width * 0.2
    dur_w = inner_width - med_w - dose_w
    x1 = left
    x2 = x1 + med_w
    x3 = x2 + dose_w
    x4 = right

    pdf.setFillColor(colors.HexColor("#e2e8f0"))
    pdf.rect(left, table_top - row_h, inner_width, row_h, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x1 + 6, table_top - 14, "Medicine")
    pdf.drawString(x2 + 6, table_top - 14, "Dosage")
    pdf.drawString(x3 + 6, table_top - 14, "Duration")

    y = table_top - row_h
    pdf.setStrokeColor(colors.HexColor("#94a3b8"))
    pdf.line(x1, y, x4, y)
    for med, dose, duration in rows[:10]:
        y -= row_h
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.black)
        pdf.drawString(x1 + 6, y + 6, simpleSplit(med, "Helvetica", 10, med_w - 12)[0] if med else "N/A")
        pdf.drawString(x2 + 6, y + 6, simpleSplit(dose, "Helvetica", 10, dose_w - 12)[0] if dose else "N/A")
        pdf.drawString(x3 + 6, y + 6, simpleSplit(duration, "Helvetica", 10, dur_w - 12)[0] if duration else "N/A")
        pdf.line(x1, y, x4, y)

    pdf.line(x1, table_top, x1, y)
    pdf.line(x2, table_top, x2, y)
    pdf.line(x3, table_top, x3, y)
    pdf.line(x4, table_top, x4, y)

    cursor_y = y - 22
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#334155"))
    pdf.drawString(left, cursor_y, "Follow-up: Visit after 5-7 days or earlier if symptoms worsen.")

    # Signature
    sign_y = cursor_y - 34
    pdf.setStrokeColor(colors.HexColor("#64748b"))
    pdf.line(right - 190, sign_y, right - 20, sign_y)
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(right - 186, sign_y - 14, "Dr. Vinay Hosadurga")
    pdf.setFont("Helvetica", 9)
    pdf.drawString(right - 186, sign_y - 27, "MBBS | Reg. No: KMC-000000")

    pdf.save()

    buffer.seek(0)
    safe_patient_name = (p.get("name") or "patient").strip().replace(" ", "_")
    filename = f"medical_report_{safe_patient_name}_{pid}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename)

# ---------------- PHOTO ACCESS ----------------
@app.route("/photo/<int:photo_id>")
def view_photo(photo_id):
    role = session.get("role")
    if role not in ("admin", "user"):
        return redirect(url_for("login_choice"))
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT photos.filename, patients.user
            FROM photos
            JOIN patients ON photos.patient_id = patients.id
            WHERE photos.id = ?
            """,
            (photo_id,),
        ).fetchone()
    if not row:
        abort(404)
    if role == "user" and row["user"] != session.get("username"):
        return redirect(url_for("login_choice"))
    return send_from_directory(UPLOAD_FOLDER, row["filename"])

@app.route("/download_photo/<int:photo_id>")
def download_photo(photo_id):
    if session.get("role") != "admin":
        return redirect(url_for("login_choice"))
    with get_db() as conn:
        row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, row["filename"], as_attachment=True)

# ---------------- DISEASE LOGIC ----------------
def predict_disease(symptoms):
    if ML_MODEL is not None:
        symptom_set = {s.strip().lower() for s in symptoms}
        vector = [[1 if feature in symptom_set else 0 for feature in SYMPTOM_FEATURES]]
        try:
            return str(ML_MODEL.predict(vector)[0])
        except Exception:
            pass
    if "fever" in symptoms and "breath" in symptoms:
        return "Respiratory Infection"
    if "chest" in symptoms:
        return "Heart Related Issue"
    if "cold" in symptoms and "cough" in symptoms:
        return "Common Cold"
    return "General Checkup Recommended"

def send_sms(number, message):
    if not number:
        return
    print(f"SMS SENT TO {number}: {message}")

def send_email(email, message):
    if not email:
        return
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    if not smtp_user or not smtp_pass:
        print("EMAIL NOT SENT: SMTP_USER/SMTP_PASS not set")
        return

    msg = EmailMessage()
    msg["Subject"] = "Smart Health Update"
    msg["From"] = smtp_user
    msg["To"] = email
    msg.set_content(message)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"EMAIL SENT TO {email}")
    except Exception as e:
        print(f"EMAIL NOT SENT: {e}")

def ask_chatgpt(user_message):
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY is not configured on the server."

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": "You are a concise health assistant for a demo app. Provide general guidance only. Do not diagnose with certainty.",
            },
            {"role": "user", "content": user_message},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        reply = body.get("output_text", "").strip()
        if not reply:
            return None, "No response text received from ChatGPT."
        return reply, None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        return None, f"ChatGPT API error: {err_body}"
    except Exception as e:
        return None, f"ChatGPT connection failed: {e}"

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_choice"))

# ---------------- CHATGPT API ----------------
@app.route("/api/chatgpt", methods=["POST"])
def chatgpt_api():
    if session.get("role") not in ("admin", "user"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400
    if len(message) > 1000:
        return jsonify({"error": "Message too long (max 1000 characters)"}), 400

    answer, err = ask_chatgpt(message)
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"answer": answer})

if __name__ == "__main__":
    print("SMART HEALTH APP RUNNING")
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
