import os
import cv2
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
from flask import Flask, render_template, request, Response, redirect, url_for, session

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import TableStyle
from reportlab.lib.pagesizes import A4

app = Flask(__name__)
app.secret_key = "motionguard_secret_key"

# ---------------- ADMIN CONFIG ----------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "motion123"

# ---------------- INCIDENT STORAGE ----------------
INCIDENTS_FILE = "incidents.json"

if not os.path.exists(INCIDENTS_FILE):
    with open(INCIDENTS_FILE, "w") as f:
        json.dump([], f)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results", "frames")
REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")
MODEL_PATH = os.path.join(BASE_DIR, "model", "motionguard_model.h5")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
motion_model = tf.keras.models.load_model(MODEL_PATH)

current_video_path = ""
current_evidence_images = []

# =========================================================
# ================== HELPER FUNCTIONS =====================
# =========================================================

def draw_motion_and_save_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    saved_images = []
    frame_id = 0

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_found = False

        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 3)
            motion_found = True

        if motion_found:
            img_name = f"motion_{frame_id}.jpg"
            cv2.imwrite(os.path.join(RESULTS_FOLDER, img_name), frame1)
            saved_images.append(img_name)

        frame1 = frame2
        ret, frame2 = cap.read()
        frame_id += 1

    cap.release()
    return saved_images


def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


# ---------------- LIVE VIDEO STREAM ----------------
def generate_live_stream():
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 3)

            cv2.putText(frame1,
                        "INTRUSION DETECTED",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

            motion_detected = True

        # ✅ SAVE INCIDENT IF MOTION
        if motion_detected:
            print("SCREENSHOT SAVING...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"live_{timestamp}.jpg"
            img_path = os.path.join(RESULTS_FOLDER, img_name)
            print("Saving to:", img_path)  

            cv2.imwrite(img_path, frame1)

            incident = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "motion_detected": True,
                "confidence": 100,
                "evidence_count": 1,
                "threat_level": "HIGH",
                "image": img_name
            }

            with open(INCIDENTS_FILE, "r") as f:
                data = json.load(f)

            data.append(incident)

            with open(INCIDENTS_FILE, "w") as f:
                json.dump(data, f, indent=4)

        # ✅ STREAM FRAME
        _, buffer = cv2.imencode('.jpg', frame1)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
# =========================================================
# ======================== ROUTES =========================
# =========================================================

@app.route("/")
def home():
    if os.path.exists(INCIDENTS_FILE):
        with open(INCIDENTS_FILE, "r") as f:
            incidents = json.load(f)
    else:
        incidents = []

    total = len(incidents)
    high = sum(1 for i in incidents if i.get("threat_level","LOW") == "HIGH")
    medium = sum(1 for i in incidents if i.get("threat_level","LOW") == "MEDIUM")
    low = sum(1 for i in incidents if i.get("threat_level","LOW") == "LOW")

    latest_threat = incidents[-1].get("threat_level","LOW") if incidents else "LOW"

    # Get last 3 screenshots
    images = [i.get("image") for i in incidents if i.get("image")]
    latest_images = images[-3:] if images else []

    return render_template(
        "index.html",
        total=total,
        high=high,
        medium=medium,
        low=low,
        latest_threat=latest_threat,
        latest_images=latest_images[::-1]
    )

@app.route("/upload-page")
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    global current_video_path, current_evidence_images

    video = request.files.get("video")
    if not video:
        return "No video uploaded"

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    current_video_path = video_path

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    IMG_SIZE = 64
    SEQ_LEN = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 10 == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
        count += 1
    cap.release()

    if len(frames) < SEQ_LEN:
        return "Video too short"

    sequences = [frames[i:i+SEQ_LEN] for i in range(len(frames) - SEQ_LEN + 1)]
    X = np.array(sequences, dtype=np.float32)

    prediction = motion_model.predict(X)
    avg_pred = float(np.mean(prediction))

    motion_detected = avg_pred > 0.5
    confidence = avg_pred * 100

    images = []
    if motion_detected:
        images = draw_motion_and_save_frames(video_path)
        current_evidence_images = images

    threat_level = "LOW"
    if motion_detected:
        if confidence > 80:
            threat_level = "HIGH"
        elif confidence > 50:
            threat_level = "MEDIUM"

    incident = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "motion_detected": motion_detected,
        "confidence": round(confidence, 2),
        "evidence_count": len(images),
        "threat_level": threat_level
    }

    with open(INCIDENTS_FILE, "r") as f:
        data = json.load(f)

    data.append(incident)

    with open(INCIDENTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return render_template("result.html",
                           motion_detected=motion_detected,
                           confidence=round(confidence, 2),
                           images=images,
                           threat_level=threat_level)


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(current_video_path),

                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/evidence")
def evidence():
    global current_evidence_images
    return render_template("evidence.html", images=current_evidence_images)

@app.route("/live")
def live_monitor():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    return render_template("live.html")


@app.route('/live_feed')
def live_feed():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    return Response(generate_live_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USERNAME and request.form["password"] == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Invalid Credentials")
    return render_template("admin_login.html")


@app.route("/admin")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    with open(INCIDENTS_FILE, "r") as f:
        incidents = json.load(f)

    total = len(incidents)
    high = sum(1 for i in incidents if i.get("threat_level", "LOW") == "HIGH")
    medium = sum(1 for i in incidents if i.get("threat_level", "LOW") == "MEDIUM")
    low = sum(1 for i in incidents if i.get("threat_level", "LOW") == "LOW")

    per_day = {}
    for i in incidents:
        date = i["timestamp"].split(" ")[0]
        per_day[date] = per_day.get(date, 0) + 1

    dates = list(per_day.keys())
    counts = list(per_day.values())

    return render_template("admin.html",
                           incidents=incidents[::-1],
                           total=total,
                           high=high,
                           medium=medium,
                           low=low,
                           dates=dates,
                           counts=counts)


@app.route("/admin-logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))
@app.route("/download-report")
def download_report():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    with open(INCIDENTS_FILE, "r") as f:
        data = json.load(f)

    if not data:
        return "No incidents available"

    latest = data[-1]

    file_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = os.path.join(REPORTS_FOLDER, file_name)

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>MotionGuard Surveillance Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    table_data = [
        ["Date & Time", latest["timestamp"]],
        ["Motion Detected", "YES" if latest["motion_detected"] else "NO"],
        ["Confidence (%)", str(latest["confidence"])],
        ["Threat Level", latest.get("threat_level", "LOW")],
        ["Evidence Frames", str(latest["evidence_count"])]
    ]

    table = Table(table_data, colWidths=[2.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
    ]))

    elements.append(table)
    doc.build(elements)

    return redirect(f"/static/reports/{file_name}")
@app.route("/check-latest-incident")
def check_latest_incident():
    if not os.path.exists(INCIDENTS_FILE):
        return {"threat": "LOW"}

    with open(INCIDENTS_FILE, "r") as f:
        data = json.load(f)

    if not data:
        return {"threat": "LOW"}

    latest = data[-1]
    return {"threat": latest.get("threat_level", "LOW")}
@app.route("/delete-incident/<int:index>")
def delete_incident(index):
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    with open(INCIDENTS_FILE, "r") as f:
        data = json.load(f)

    # reverse index because dashboard shows reversed list
    real_index = len(data) - 1 - index

    if 0 <= real_index < len(data):
        data.pop(real_index)

        with open(INCIDENTS_FILE, "w") as f:
            json.dump(data, f, indent=4)

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)