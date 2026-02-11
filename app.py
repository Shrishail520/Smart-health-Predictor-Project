from flask import Flask, render_template, request, redirect, url_for, session, send_file, send_from_directory, abort
from datetime import datetime
import os
import smtplib
import ssl
from email.message import EmailMessage
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import json
import sqlite3
import uuid
import base64
import re
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "smart_health_secret"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_ROOT, "smart_health.db")
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

notifications = {}

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

def seed_demo_users():
    demo_users = [
        ("doctor1", "doctor123", "admin"),
        ("patient1", "patient123", "user"),
    ]
    with get_db() as conn:
        for username, password, role in demo_users:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), role),
                )

seed_demo_users()

def save_upload(file_storage, patient_id):
    if not file_storage or not file_storage.filename:
        return None
    safe_name = secure_filename(file_storage.filename)
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

    return render_template(template_name, error=error)

@app.route("/login/doctor", methods=["GET", "POST"])
def login_doctor():
    return handle_login("admin", "login_doctor.html")

@app.route("/login/patient", methods=["GET", "POST"])
def login_patient():
    return handle_login("user", "login_patient.html")

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

    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        contact = request.form["contact"]
        email = request.form.get("email", "")
        symptoms = request.form.getlist("symptoms")

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
                    age,
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

            files = request.files.getlist("photos")
            for f in files:
                filename = save_upload(f, patient_id)
                if filename:
                    conn.execute(
                        "INSERT INTO photos (patient_id, filename, uploaded_at) VALUES (?, ?, ?)",
                        (patient_id, filename, datetime.now().strftime("%d-%m-%Y %H:%M")),
                    )

            camera_data = request.form.get("camera_photo", "")
            camera_filename = save_camera_data(camera_data, patient_id)
            if camera_filename:
                conn.execute(
                    "INSERT INTO photos (patient_id, filename, uploaded_at) VALUES (?, ?, ?)",
                    (patient_id, camera_filename, datetime.now().strftime("%d-%m-%Y %H:%M")),
                )

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
        history_photos=history_photos
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

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, "SMART HEALTH MEDICAL REPORT")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, 760, f"Patient Name: {p['name']}")
    pdf.drawString(50, 740, f"Age: {p['age']}")
    pdf.drawString(50, 720, f"Contact: {p['contact']}")
    pdf.drawString(50, 700, f"Symptoms: {', '.join(p['symptoms'])}")
    pdf.drawString(50, 680, f"Disease: {p['disease']}")
    pdf.drawString(50, 660, f"Doctor Remarks: {p['remarks']}")
    pdf.drawString(50, 640, f"Prescription: {p['prescription'] or 'N/A'}")
    pdf.drawString(50, 620, f"Status: {p.get('status', 'Pending')}")
    pdf.drawString(50, 600, f"Updated On: {p['updated_at']}")

    pdf.setFont("Helvetica-Oblique", 14)
    pdf.drawString(350, 600, "Dr. Vinay Hosadurga")
    pdf.drawString(350, 580, "MBBS")

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="medical_report.pdf")

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

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_choice"))

if __name__ == "__main__":
    print("SMART HEALTH APP RUNNING")
    app.run(debug=True)
