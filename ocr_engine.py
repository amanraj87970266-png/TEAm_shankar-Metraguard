"""
ocr_engine.py — MetraGuard's AI Perception Layer (MVP version).

Pipeline: Image Preprocessing -> OCR/Text Detection -> (raw text out)
In the full pitch, this layer also does computer-vision layout analysis and
product/package classification. For the MVP we keep it to robust OCR, which
is the part judges will actually watch you demo.

Core principle honoured here: this file only EXTRACTS evidence (text).
It never decides compliance — that's rule_engine.py's job.

ROBUSTNESS NOTE: a single fixed preprocessing recipe (e.g. one adaptive
threshold) works great on a clean, flat, well-lit synthetic label, but real
phone photos vary a lot — glare, curved packaging, tilt, uneven lighting,
low resolution. So this module runs several preprocessing strategies and
several Tesseract page-segmentation modes, then automatically keeps whichever
combination gives the highest OCR confidence. This is still fully
deterministic and explainable (no ML judgment call here) — it's just "try a
few honest approaches, keep the one Tesseract itself is most confident about."
"""

import os
import io
import platform
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

# On Windows, the Tesseract installer often does NOT add tesseract.exe to PATH,
# which makes pytesseract fail with "TesseractNotFoundError" even though it's
# installed correctly. This auto-detects the default install location so the
# app works out of the box after a standard Windows install.
if platform.system() == "Windows":
    _default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_win_path):
        pytesseract.pytesseract.tesseract_cmd = _default_win_path


def _load_and_upscale(image_bytes: bytes) -> np.ndarray:
    # Load via Pillow first and correct EXIF orientation. Phone cameras save
    # portrait photos as landscape pixel data plus a rotation flag in EXIF —
    # OpenCV's decoder ignores that flag, which silently feeds Tesseract a
    # sideways image and produces near-blank or garbled OCR. This is the
    # single most common real-world failure mode for phone-photographed labels.
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)  # rotates pixels to match intended orientation
    pil_img = pil_img.convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to a consistent working resolution regardless of source camera
    # resolution — very large modern phone photos (12MP+) and very small ones
    # both hurt a fixed-blocksize adaptive threshold; a consistent target size
    # keeps preprocessing behaving predictably either way.
    h, w = gray.shape
    target = 1800
    if max(h, w) != target:
        scale = target / max(h, w)
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)

    return gray


def _variant_adaptive(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def _variant_otsu(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def _variant_sharpened_gray(gray: np.ndarray) -> np.ndarray:
    # No thresholding — sometimes plain contrast-enhanced grayscale beats binarization
    # on glossy/curved packaging where thresholding creates broken text.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(contrast, -1, kernel)


def _ocr_confidence(image: np.ndarray, config: str) -> tuple[str, float]:
    text = pytesseract.image_to_string(image, config=config)
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text.strip(), mean_conf


def extract_text(image_bytes: bytes) -> tuple[str, float]:
    """Returns (extracted_text, mean_confidence_0_to_100).

    Tries multiple preprocessing variants x page-segmentation modes and keeps
    whichever result Tesseract itself was most confident about.
    """
    gray = _load_and_upscale(image_bytes)

    variants = {
        "adaptive": _variant_adaptive(gray),
        "otsu": _variant_otsu(gray),
        "sharpened_gray": _variant_sharpened_gray(gray),
    }

    # PSM 6 = assume a single uniform block of text (good default for labels)
    # PSM 11 = sparse text, find as much as possible in any order (good for scattered label text)
    psm_modes = ["6", "11"]

    best_text, best_conf = "", -1.0
    for variant_img in variants.values():
        for psm in psm_modes:
            config = f"--oem 3 --psm {psm}"
            try:
                text, conf = _ocr_confidence(variant_img, config)
            except pytesseract.TesseractError:
                continue
            if conf > best_conf or (conf == best_conf and len(text) > len(best_text)):
                best_text, best_conf = text, conf

    return best_text, round(best_conf, 1)
