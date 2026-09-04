"""
app.py — MetraGuard MVP web application.

Run with:  python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import time

from ocr_engine import extract_text
from rule_engine import run_compliance_check

app = Flask(__name__)
app.secret_key = "metraguard-dev-secret"  # fine for local demo use only

UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("label_image")
    if not file or file.filename == "":
        flash("Please choose an image of the product label first.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG, PNG, or WEBP image.")
        return redirect(url_for("index"))

    image_bytes = file.read()

    # Save a copy so the evaluation report can show the original evidence image
    fname = f"{int(time.time())}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, fname)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    try:
        ocr_text, ocr_confidence = extract_text(image_bytes)
    except Exception as e:
        flash(f"Could not process image: {e}")
        return redirect(url_for("index"))

    report = run_compliance_check(ocr_text)

    return render_template(
        "result.html",
        report=report,
        ocr_text=ocr_text,
        ocr_confidence=ocr_confidence,
        image_path=f"uploads/{fname}",
    )


if __name__ == "__main__":
    print("MetraGuard MVP running -> http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
