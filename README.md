# MetraGuard — MVP

**SIH26034** — Software System to check compliance of Packaged Commodities under
Legal Metrology (Packaged Commodities) Rules, 2011, by scanning products, images and labels.
Team Shankar.

This is a **working, fully offline MVP**. No cloud API keys, no internet connection
required at runtime — .

---

## 1. What this MVP actually does

You upload a photo of a product label. The system:

1. **Preprocesses** the image (grayscale, denoise, upscale, adaptive threshold) — `ocr_engine.py`
2. **Extracts text** with Tesseract OCR — `ocr_engine.py`
3. **Checks the extracted text** against the mandatory declarations required by
   Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011 — `rule_engine.py`
4. **Renders a report**: overall verdict (Compliant / Review Required / Potential
   Non-Compliance), a score, and — critically — the *exact text snippet* used as
   evidence for every decision.

This directly implements the "AI extracts evidence; rules determine compliance"
principle from your pitch deck, and it's a real, runnable slice of the 5-stage
pipeline you designed (Detect & extract → Classify → Apply rule engine → Flag → Report).

## 2. What's intentionally simplified for the 1-day MVP

Be upfront about this with judges — it shows maturity, not weakness:

| Full vision (deck) | MVP (today) | Why |
|---|---|---|
| CV/layout region detection | Whole-image OCR | Faster to build reliably in one day |
| Product/package classifier | Not implemented (flagged REVIEW) | Needs a trained model / taxonomy — Phase 2 |
| Multi-image package reconstruction | Single image per scan | Time constraint |
| PostgreSQL history & search | In-memory, single scan | No persistence layer needed to prove the concept |
| Multilingual OCR | English/Latin script (Tesseract `eng`) | Can add `hin` language pack in minutes if needed |

## 3. Setup (do this first, before the event if possible)

```bash
# 1. Install Tesseract OCR (the system binary, not just the Python wrapper)
#    Ubuntu/Debian:
sudo apt-get install tesseract-ocr
#    macOS:
brew install tesseract
#    Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki

# 2. Install Python dependencies
cd metraguard
pip install -r requirements.txt

# 3. Generate sample label images (for demoing without a real product)
python samples/generate_samples.py

# 4. Run the app
python app.py
```

Open **http://localhost:5000** in your browser. Upload `samples/sample_compliant.png`
or `samples/sample_noncompliant.png` to see it work immediately, or take a photo of
any real packaged product on your desk.

## 4. Files

```
metraguard/
├── app.py              Flask web app — routes, upload handling
├── ocr_engine.py        AI Perception Layer — image preprocessing + Tesseract OCR
├── rule_engine.py        Versioned Legal Rule Engine — the compliance logic
├── requirements.txt
├── templates/
│   ├── index.html       Upload page
│   └── result.html       Compliance report page
├── static/style.css      Styling
└── samples/
    └── generate_samples.py   Creates test label images
```

