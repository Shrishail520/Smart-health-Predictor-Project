# Smart Health Predictor Project

Flask-based mini project for patient symptom submission, doctor review, image upload, and PDF medical report generation.

## Features
- Doctor and patient role-based login
- Patient signup flow (separate from login)
- Symptom-based disease prediction (ML: Logistic Regression)
- Multiple photo upload + camera capture
- Doctor diagnosis, remarks, and prescription update
- Notification display for patients
- PDF report download
- SQLite local database

## Tech Stack
- Python
- Flask
- scikit-learn
- SQLite
- ReportLab
- HTML/CSS templates

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install flask reportlab werkzeug scikit-learn`
3. Set environment variables (PowerShell):
   - `$env:FLASK_SECRET_KEY="your-long-random-secret"`
   - Optional for email:
   - `$env:SMTP_USER="your-email@gmail.com"`
   - `$env:SMTP_PASS="your-app-password"`
4. Run:
   - `python app.py`
5. Open:
   - `http://127.0.0.1:5000`

## Default Demo Credentials
- Doctor login:
  - Username: `doctor1`
  - Password: `doctor123`
- Patient login:
  - Email: `patient1@example.com`
  - Password: `patient123`

## Notes
- Development debug mode is off by default.
- To enable debug mode explicitly:
  - `$env:FLASK_DEBUG="1"`
- Uploaded files are stored in `uploads/`.
- Database file is `smart_health.db`.
- ML training dataset file is `symptom_dataset.csv`.
