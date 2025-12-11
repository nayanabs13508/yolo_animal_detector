# app.py
"""
Animal Detector Flask app (with upload WhatsApp alerts + optional image media)
- Upload detection + WhatsApp alerts (upload only)
- Webcam endpoints: detection, save snapshot, generate report (NO WhatsApp)
"""
# add this alongside other utils imports
from utils.telegram_alert import send_telegram_alert
import utils.alerts as alerts   # your Twilio helper (already exists)
try:
    from utils.telegram_alert import send_telegram_alert
except Exception:
    # if telegram helper missing, provide fallback to avoid crash
    def send_telegram_alert(message, image_path=None):
        return {'ok': False, 'error': 'telegram helper missing'}

import os
import io
import sqlite3
import csv
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, current_app
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# load .env
try:
    load_dotenv(override=False)
except TypeError:
    load_dotenv()

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("animal_detector")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'static', 'results')
WEBCAM_DIR = os.path.join(BASE_DIR, 'static', 'webcam_alerts')
REPORT_DIR = os.path.join(BASE_DIR, 'static', 'reports')
DB_PATH = os.path.join(BASE_DIR, 'detections.db')
LOG_CSV = os.path.join(BASE_DIR, 'detection_log.csv')
LAST_ALERT_FILE = os.path.join(BASE_DIR, 'last_alert.txt')

# ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEBCAM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

ALERT_COOLDOWN_SECONDS = int(os.getenv('ALERT_COOLDOWN_SECONDS', '30'))

from ultralytics import YOLO

# utils
from utils.yolo_helper import run_inference_on_image, ensure_dirs
from utils import alerts
from utils.telegram_alert import send_telegram_alert

# optional CLIP helpers
try:
    from utils.clip_check import is_clip_enabled, clip_rerank
except Exception:
    def is_clip_enabled(): return False
    def clip_rerank(*a, **k): return None

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['RESULTS_FOLDER'] = RESULTS_DIR
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


MODEL_NAME = os.getenv('MODEL_NAME', 'yolov8n.pt')
logger.info("Loading YOLO model: %s", MODEL_NAME)
MODEL = YOLO(MODEL_NAME)

DETECT_CONF = float(os.getenv('DETECT_CONF', '0.50'))
DETECT_IMGSZ = int(os.getenv('DETECT_IMGSZ', '640'))


def init_db():
    """Ensure DB exists and tables created."""
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            class_name TEXT,
            confidence REAL,
            image_path TEXT
        )
        """)
        c.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sender TEXT,
            channel TEXT,
            body TEXT,
            image_path TEXT,
            status TEXT
        )
        """)
        conn.commit()
        conn.close()


def log_detection_db(timestamp, parsed_results, image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for det in parsed_results:
        name = det.get('name') or 'unknown'
        conf = float(det.get('conf', 0.0))
        c.execute("INSERT INTO detections (timestamp, class_name, confidence, image_path) VALUES (?, ?, ?, ?)",
                  (timestamp, name, conf, image_path))
    conn.commit()
    conn.close()


def log_detection_csv(timestamp, classes, image_path):
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['timestamp', 'classes', 'image_path'])
        w.writerow([timestamp, ';'.join([f"{c}:{conf:.2f}" for c, conf in classes]), image_path])


def _get_bool_from_request(req, *names):
    for n in names:
        v = req.form.get(n)
        if v is not None:
            return v in ('1', 'true', 'True', 'on')
    return False


def _get_str_from_request(req, *names):
    for n in names:
        v = req.form.get(n)
        if v:
            return v
    return None


@app.route('/')
def index():
    twilio_from = os.getenv('TWILIO_WHATSAPP_FROM', os.getenv('TWILIO_FROM', 'whatsapp:+14155238886'))
    alert_to = os.getenv('ALERT_TO_WHATSAPP', '')
    return render_template('index.html', twilio_from=twilio_from, alert_to=alert_to)


@app.route('/results/<path:filename>')
def show_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    return jsonify(process_image_request(request))


@app.route('/detect', methods=['GET', 'POST'])
def detect_page():
    if request.method == 'GET':
        twilio_from = os.getenv('TWILIO_WHATSAPP_FROM', os.getenv('TWILIO_FROM', 'whatsapp:+14155238886'))
        alert_to = os.getenv('ALERT_TO_WHATSAPP', '')
        return render_template('index.html', twilio_from=twilio_from, alert_to=alert_to)
    result = process_image_request(request)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify(result)
    return render_template('result.html', result_url=result.get('result_url'), detections=result.get('detections', []),
                           alert_message=(result.get('alert_result') or {}).get('error_message'),
                           alert_channel='whatsapp' if result.get('alert_sent') else None,
                           result_file=os.path.basename(result.get('result_url') or ''))


def process_image_request(req):
    """
    Handle upload/image_url -> run YOLO -> save annotated image -> log -> optionally send alerts.
    Returns a dict suitable for jsonify().
    """
    # input: file upload or image_url
    image_url = req.form.get('image_url') or req.form.get('url')
    file_obj = None
    if 'image' in req.files and req.files['image'].filename:
        file_obj = req.files['image']
    if not file_obj and not image_url:
        return {'error': 'No image provided'}

    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    if file_obj:
        orig = secure_filename(file_obj.filename)
        filename = f"{ts}_{orig}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file_obj.save(save_path)
        except Exception as e:
            logger.exception("Failed saving uploaded file")
            return {'error': f'Failed to save uploaded file: {e}'}
    else:
        # fetch remote url
        try:
            import requests
            resp = requests.get(image_url, timeout=10)
            if resp.status_code != 200:
                return {'error': f'Failed to fetch image URL: HTTP {resp.status_code}'}
            from urllib.parse import urlparse
            path = urlparse(image_url).path
            base = os.path.basename(path) or 'remote.jpg'
            orig = secure_filename(base)
            filename = f"{ts}_{orig}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(save_path, 'wb') as f:
                f.write(resp.content)
        except Exception as e:
            logger.exception("Failed fetching remote URL")
            return {'error': f'Failed to fetch image URL: {e}'}

    # optional resize for inference
    try:
        from PIL import Image
        img_size = req.form.get('img_size') or req.form.get('imagesize') or str(DETECT_IMGSZ)
        try:
            img_size = int(img_size)
        except Exception:
            img_size = DETECT_IMGSZ
        img = Image.open(save_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.width > img_size or img.height > img_size:
            img.thumbnail((img_size, img_size), Image.LANCZOS)
            img.save(save_path, format='JPEG', quality=85)
    except Exception:
        # continue if pillow not available
        pass

    # run inference
    try:
        results = run_inference_on_image(MODEL, save_path, conf=DETECT_CONF, imgsz=DETECT_IMGSZ)
        parsed = results.get('parsed', []) if isinstance(results, dict) else []
        annotated = results.get('annotated') if isinstance(results, dict) else None
        if annotated is None:
            try:
                import cv2, numpy as np
                img_cv = cv2.imread(save_path)
                annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) if img_cv is not None else None
            except Exception:
                annotated = None
    except Exception as e:
        logger.exception("Inference failed")
        return {'error': f'Inference failed: {e}'}

    # save annotated image to results dir
    out_name = f"annot_{filename}.jpg"
    out_path = os.path.join(app.config['RESULTS_FOLDER'], out_name)
    try:
        if annotated is not None:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(annotated)
            img.save(out_path, format='JPEG', quality=85)
        else:
            import shutil
            shutil.copy(save_path, out_path)
    except Exception:
        try:
            import shutil
            shutil.copy(save_path, out_path)
        except Exception:
            out_path = save_path

    timestamp = datetime.utcnow().isoformat()
    detected = [(p.get('name', 'unknown'), p.get('conf', 0.0)) for p in parsed]
    try:
        init_db()
        log_detection_csv(timestamp, detected, out_path)
        log_detection_db(timestamp, parsed, out_path)
    except Exception:
        logger.exception("Logging failed")

    # optional CLIP logic (if available) - keep original behavior if present
    try:
        if is_clip_enabled() and parsed:
            names = [p.get('name') for p in parsed]
            top = clip_rerank(save_path, names, top_k=1)
            if top:
                parsed[0]['name'] = top[0][0]
    except Exception:
        pass

    result_url = url_for('show_result', filename=out_name, _external=True)

    # Build alert message AFTER inference
    body = "No animals detected"
    if parsed:
        primary = parsed[0].get('name') or 'animal'
        conf = parsed[0].get('conf', 0.0)
        body = f"Animal detected: {primary} (confidence={conf:.2f})"

    # parse flags from request (checkbox names must match client)
    send_whatsapp_flag = _get_bool_from_request(req, 'auto_whatsapp', 'auto_send_whatsapp', 'auto_send')
    send_telegram_flag = _get_bool_from_request(req, 'auto_telegram', 'auto_send_telegram')

    alert_result = None
    telegram_result = None
    alert_sent = False

    # WhatsApp alert (uploads only)
    if send_whatsapp_flag:
        try:
            whatsapp_result = alerts.send_whatsapp_alert(
                body,
                to_whatsapp=wa_to,
                from_whatsapp=wa_from,
                media_url=result_url
            )
            alert_sent = alert_sent or bool(whatsapp_result.get("ok"))
        except Exception as e:
            logger.exception("WhatsApp send failed")
            whatsapp_result = {"ok": False, "error_message": str(e)}

    # Telegram alert (uploads only, always attempt if enabled)
    telegram_result = None
    if str(os.getenv("SEND_TELEGRAM_ALERTS", "0")).lower() in ("1", "true", "yes"):
        try:
            telegram_result = send_telegram_alert(
                body, image_path=out_path
            )
            alert_sent = alert_sent or bool(telegram_result.get("ok"))
        except Exception as e:
            logger.exception("Telegram send failed")
            telegram_result = {"ok": False, "error": str(e)}

    # WhatsApp error handling for Twilio codes
    whatsapp_error_msg = None
    if whatsapp_result and not whatsapp_result.get("ok"):
        code = whatsapp_result.get("twilio_code")
        msg = whatsapp_result.get("error") or whatsapp_result.get("error_message")
        if code in (63038, 21910):
            whatsapp_error_msg = f"Twilio error {code}: {msg}"
        else:
            whatsapp_error_msg = msg

    # Cooldown logic (if present in your code, keep as-is)
    # If cooldown blocks, set alert_result accordingly

    alert_result = {
        "whatsapp": {
            "ok": whatsapp_result.get("ok") if whatsapp_result else False,
            "sid": whatsapp_result.get("sid") if whatsapp_result else None,
            "error_message": whatsapp_error_msg,
        },
        "telegram": telegram_result if telegram_result else {"ok": False, "error": "Not attempted"},
    }
    alert_sent = alert_result["whatsapp"]["ok"] or alert_result["telegram"].get("ok", False)

    return {
        'result_url': result_url,
        'detections': [{'name': p.get('name'), 'conf': p.get('conf', 0.0)} for p in parsed],
        'timestamp': timestamp,
        'alert_sent': alert_sent,
        'alert_result': alert_result,
        'telegram_result': telegram_result
    }
# ... later in process_image_request(), after building result_url and parsed etc.

    # NOTE: Twilio/WhatsApp + Telegram sending (upload flow only)
    auto_send = _get_bool_from_request(req, 'auto_whatsapp', 'auto_send_whatsapp', 'auto_send')
    send_telegram_flag = _get_bool_from_request(req, 'send_telegram', 'send_telegram_alert', 'send_telegram')
    wa_number = _get_str_from_request(req, 'whatsapp_phone', 'whatsapp_number', 'to') or os.getenv('ALERT_TO_WHATSAPP')
    wa_from = _get_str_from_request(req, 'whatsapp_from', 'from') or os.getenv('TWILIO_WHATSAPP_FROM', os.getenv('TWILIO_FROM'))
    telegram_to = os.getenv('TELEGRAM_CHAT_ID')  # we don't pass this directly (telegram helper reads env)

    alert_sent = False
    alert_result = None
    telegram_result = None

    # build a human friendly message
    if parsed:
        primary = parsed[0].get('name') or 'animal'
        primary_display = 'animal' if not primary or primary.lower() in ('unknown', 'object') else primary
        body = f"Animal detected: {primary_display} (confidence={parsed[0].get('conf', 0.0):.2f})\nImage: {result_url}"
    else:
        body = f"Detection completed. Image: {result_url}"

    # WHATSAPP via Twilio (only if auto_send requested)
    if auto_send and parsed:
        try:
            # server-side cooldown check (existing code re-used)
            allow_send = True
            retry_after = 0
            if os.path.exists(LAST_ALERT_FILE):
                try:
                    with open(LAST_ALERT_FILE, 'r') as f:
                        content = f.read().strip()
                        last_ts = float(content) if content else 0.0
                    diff = time.time() - last_ts
                    if diff < ALERT_COOLDOWN_SECONDS:
                        allow_send = False
                        retry_after = int(ALERT_COOLDOWN_SECONDS - diff)
                except Exception:
                    allow_send = True

            if not allow_send:
                alert_result = {'ok': False, 'sid': None, 'error_message': 'Server cooldown active', 'retry_after': retry_after}
                alert_sent = False
            else:
                # make sure we have a destination
                if not wa_number:
                    alert_result = {'ok': False, 'error_message': 'No WhatsApp destination (wa_number missing)'}
                else:
                    alert_result = alerts.send_whatsapp_alert(body, wa_number, wa_from)
                    alert_sent = bool(alert_result.get('ok'))
                    if alert_sent:
                        try:
                            with open(LAST_ALERT_FILE, 'w') as f:
                                f.write(str(time.time()))
                        except Exception:
                            pass
        except Exception as e:
            logger.exception("Error while attempting to send WhatsApp alert")
            alert_result = {'ok': False, 'error_message': str(e)}
            alert_sent = False

    # TELEGRAM (if requested)
    if send_telegram_flag and parsed:
        try:
            # send text and optionally photo
            # send_telegram_alert returns {'ok': True} or {'ok': False, 'error': '...'}
            telegram_result = send_telegram_alert(body, image_path=out_path)
        except Exception as e:
            logger.exception("Telegram send exception")
            telegram_result = {'ok': False, 'error': str(e)}

    # include results in response for frontend notifications
    return {
        'result_url': result_url,
        'detections': [{'name': p.get('name'), 'conf': p.get('conf', 0.0)} for p in parsed],
        'timestamp': timestamp,
        'alert_sent': alert_sent,
        'alert_result': alert_result,
        'telegram_result': telegram_result
    }


# --- Webcam-only routes (NO WhatsApp sending) ---

@app.route('/api/webcam_detect', methods=['POST'])
def api_webcam_detect():
    from utils.yolo_helper import run_yolo_detection
    img = request.files.get('image')
    img_size = int(request.form.get('img_size', 640))
    if not img:
        return jsonify({'error': 'No image'}), 400
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_path = f'static/webcam_alerts/temp_{ts}.jpg'
    img.save(temp_path)
    dets, annotated_path = run_yolo_detection(temp_path, img_size=img_size, save_dir='static/webcam_alerts')
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return jsonify({
        'detections': dets,
        'annotated_path': '/' + annotated_path.replace('\\', '/')
    })


@app.route('/save_webcam_snapshot', methods=['POST'])
def save_webcam_snapshot():
    data = request.get_json()
    img_path = data.get('img_path')
    ts = data.get('ts')
    return jsonify({'ok': True})


@app.route('/generate_webcam_report', methods=['POST'])
def generate_webcam_report():
    alert_dir = 'static/webcam_alerts'
    entries = []
    for fname in sorted(os.listdir(alert_dir)):
        if fname.startswith('annot_') and fname.endswith('.jpg'):
            parts = fname.split('_', 2)
            ts = parts[1] if len(parts) >= 2 else fname
            entries.append(ts)
    total_frames = len(entries)
    report_lines = [
        f"Webcam Monitoring Report",
        f"Total frames processed: {total_frames}",
        f"Timestamps:",
    ]
    for ts in entries:
        report_lines.append(f"  - {ts}")
    report_txt = '\n'.join(report_lines)
    report_dir = 'static/reports'
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"webcam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_txt)
    return jsonify({'report_path': '/' + report_path.replace('\\', '/')})


@app.route('/api/stats')
def api_stats():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date(timestamp), class_name, COUNT(*) FROM detections GROUP BY date(timestamp), class_name ORDER BY date(timestamp)")
    rows = c.fetchall()
    conn.close()
    data = {}
    dates = set()
    for day, cls, cnt in rows:
        dates.add(day)
        data.setdefault(cls, {})[day] = cnt
    dates = sorted(list(dates))
    series = {cls: [data.get(cls, {}).get(d, 0) for d in dates] for cls in data}
    return jsonify({'dates': dates, 'counts': series})


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    # return JSON for API requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.path.startswith('/api/'):
        return jsonify({'error': str(e), 'traceback': tb}), 500
    return f"<h1>Internal Server Error</h1><pre>{tb}</pre>", 500


# --- quick test endpoints for alerts (call in browser while server running) ---
@app.route('/tools/test_twilio', methods=['GET'])
def tools_test_twilio():
    """Return result of test_twilio_credentials and a sample WhatsApp send (no send when ?send=1 not present)."""
    try:
        res = alerts.test_twilio_credentials()
        send = request.args.get('send', '0') == '1'
        sample = None
        if send and res.get('ok'):
            # will attempt a real send; requires ALERT_TO_WHATSAPP in .env or ?to param
            to = request.args.get('to') or os.getenv('ALERT_TO_WHATSAPP')
            from_ = request.args.get('from') or os.getenv('TWILIO_WHATSAPP_FROM')
            sample = alerts.send_whatsapp_alert("Test message from Animal Detector", to_whatsapp=to, from_whatsapp=from_, media_url=None)
        return jsonify({'test': res, 'sample_send': sample})
    except Exception as e:
        current_app.logger.exception("tools/test_twilio failed")
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/tools/test_telegram', methods=['GET'])
def tools_test_telegram():
    """Quick telegram test; optional ?send=1 to attempt real send."""
    try:
        send = request.args.get('send', '0') == '1'
        sample = None
        if send:
            from utils.telegram_alert import send_telegram_alert
            sample = send_telegram_alert("Test message from Animal Detector (no image)")
        return jsonify({'sample_send': sample})
    except Exception as e:
        current_app.logger.exception("tools/test_telegram failed")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/tools/telegram_info', methods=['GET'])
def tools_telegram_info():
    """
    Returns getMe payload so you can verify the bot token.
    """
    res = telegram_get_me()
    return jsonify(res)

@app.route('/tools/telegram_updates', methods=['GET'])
def tools_telegram_updates():
    """
    Returns recent updates (messages) so you can find chat_id.
    Optional query params: offset, limit
    """
    try:
        offset = request.args.get('offset', None)
        limit = int(request.args.get('limit', '100') or 100)
        offset_val = int(offset) if offset else None
        res = telegram_get_updates(offset=offset_val, limit=limit)
        return jsonify(res)
    except Exception as e:
        current_app.logger.exception("tools/telegram_updates failed")
        return jsonify({'ok': False, 'error': str(e)}), 500

# small helper route to try a sample send (reads TELEGRAM_CHAT_ID or ?chat param)
@app.route('/tools/telegram_send_test', methods=['GET'])
def tools_telegram_send_test():
    chat = request.args.get('chat') or os.getenv('TELEGRAM_CHAT_ID')
    msg = request.args.get('msg') or "Test message from Animal Detector"
    if not chat:
        return jsonify({'ok': False, 'error': 'chat id not provided; send a message to bot and call /tools/telegram_updates to get chat id'}), 400
    # temporarily set environment for send_telegram_alert to use
    os.environ['TELEGRAM_CHAT_ID'] = str(chat)
    res = send_telegram_alert(msg)
    return jsonify(res)


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
