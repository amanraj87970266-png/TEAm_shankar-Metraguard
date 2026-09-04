"""
rule_engine.py — MetraGuard's versioned legal rule engine.

This module encodes the mandatory declarations required on the label of a
pre-packaged commodity under the Legal Metrology Act, 2009 and the Legal
Metrology (Packaged Commodities) Rules, 2011 (Rule 6 in particular).

DESIGN PRINCIPLE (matches the pitch): the AI/OCR layer only *extracts text*.
This module is a plain, deterministic, versioned rule engine — no ML here.
That's what makes the compliance verdict explainable and auditable, and it's
also exactly what a judge will probe you on, so know this file well.

Each rule returns one of three states:
  PASS              -> declaration found with reasonable confidence
  REVIEW_REQUIRED   -> partial / low-confidence match, needs a human look
  NON_COMPLIANT     -> declaration not found at all
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

RULESET_VERSION = "LMPCR-2011-v1.0"  # "versioned rule repository" from the pitch


@dataclass
class FieldResult:
    field_id: str
    field_name: str
    legal_basis: str
    status: str  # PASS | REVIEW_REQUIRED | NON_COMPLIANT
    evidence: Optional[str] = None
    confidence: float = 0.0
    notes: str = ""


@dataclass
class ComplianceReport:
    overall_status: str  # COMPLIANT | REVIEW_REQUIRED | POTENTIAL_NON_COMPLIANCE
    score: float
    ruleset_version: str
    fields: List[FieldResult] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Individual field checks. Each is intentionally simple + explainable:
# regex/keyword matching over OCR text. This is the "deterministic legal
# rule engine" — swap in more patterns over time without touching the AI layer.
# ---------------------------------------------------------------------------

def _find(pattern, text, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return m.group(0).strip() if m else None


def check_manufacturer(text: str) -> FieldResult:
    pattern = r"(manufactured by|mfd by|mfg by|packed by|marketed by|packer)[:\-]?\s*[^\n]{5,80}"
    evidence = _find(pattern, text)
    if evidence:
        return FieldResult("manufacturer", "Name & Address of Manufacturer/Packer/Importer",
                            "Rule 6(1)(a)", "PASS", evidence, 0.9)
    # weaker fallback: any "Pvt Ltd" / "Ltd" / "Industries" style company name
    fallback = _find(r"[A-Z][A-Za-z&.,\s]{3,40}(Pvt\.?\s?Ltd\.?|Ltd\.?|Industries|Foods|Enterprises)", text)
    if fallback:
        return FieldResult("manufacturer", "Name & Address of Manufacturer/Packer/Importer",
                            "Rule 6(1)(a)", "REVIEW_REQUIRED", fallback, 0.4,
                            "Company-like name found but not explicitly labelled 'Manufactured/Packed by'.")
    return FieldResult("manufacturer", "Name & Address of Manufacturer/Packer/Importer",
                        "Rule 6(1)(a)", "NON_COMPLIANT", None, 0.0)


def check_common_name(text: str) -> FieldResult:
    # Heuristic: hard to verify generically without a product taxonomy.
    # We flag REVIEW_REQUIRED by default — a human/jury will understand this
    # is a known limitation to be solved with a product-category classifier (Phase 2).
    return FieldResult("common_name", "Common / Generic Name of Commodity",
                        "Rule 6(1)(b)", "REVIEW_REQUIRED", None, 0.3,
                        "Requires product classification model (planned Phase 2) to verify reliably.")


def check_net_quantity(text: str) -> FieldResult:
    pattern = r"\b(\d{1,4}(\.\d{1,3})?)\s?(g|gm|gram|grams|kg|ml|mL|litre|liter|l|L|N|pcs|pieces)\b"
    evidence = _find(pattern, text)
    if evidence:
        return FieldResult("net_quantity", "Net Quantity (Standard Unit)",
                            "Rule 6(1)(c) / Rule 8", "PASS", evidence, 0.85)
    return FieldResult("net_quantity", "Net Quantity (Standard Unit)",
                        "Rule 6(1)(c) / Rule 8", "NON_COMPLIANT", None, 0.0)


def check_mrp(text: str) -> FieldResult:
    pattern = r"(m\.?\s?r\.?\s?p\.?|max(imum)?\s+retail\s+price)[^\d₹]{0,15}(₹|rs\.?|inr)?\s?\d{1,3}(,\d{3})*(\.\d{1,2})?"
    evidence = _find(pattern, text)
    if evidence:
        return FieldResult("mrp", "Maximum Retail Price (incl. all taxes)",
                            "Rule 6(1)(e)", "PASS", evidence, 0.9)
    price_only = _find(r"(₹|rs\.?|inr)\s?\d{1,3}(,\d{3})*(\.\d{1,2})?", text)
    if price_only:
        return FieldResult("mrp", "Maximum Retail Price (incl. all taxes)",
                            "Rule 6(1)(e)", "REVIEW_REQUIRED", price_only, 0.4,
                            "A price was found but not explicitly labelled MRP.")
    return FieldResult("mrp", "Maximum Retail Price (incl. all taxes)",
                        "Rule 6(1)(e)", "NON_COMPLIANT", None, 0.0)


def check_mfg_date(text: str) -> FieldResult:
    pattern = (r"(mfg\.?\s?date|manufactur(ed|ing)\s+date|date\s+of\s+mfg|pkd\s+date|packed\s+on)"
               r"[:\-]?\s*(\d{1,2}[\/\-.])?(\d{1,2}|[A-Za-z]{3,9})[\/\-.\s]\d{2,4}")
    evidence = _find(pattern, text)
    if evidence:
        return FieldResult("mfg_date", "Month & Year of Manufacture/Pack/Import",
                            "Rule 6(1)(d)", "PASS", evidence, 0.85)
    date_only = _find(r"\b(0?[1-9]|1[0-2])[\/\-.](\d{2,4})\b", text)
    if date_only:
        return FieldResult("mfg_date", "Month & Year of Manufacture/Pack/Import",
                            "Rule 6(1)(d)", "REVIEW_REQUIRED", date_only, 0.4,
                            "A date-like pattern was found but not explicitly labelled.")
    return FieldResult("mfg_date", "Month & Year of Manufacture/Pack/Import",
                        "Rule 6(1)(d)", "NON_COMPLIANT", None, 0.0)


def check_consumer_care(text: str) -> FieldResult:
    email = _find(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = _find(r"(customer\s?care|consumer\s?care|helpline|toll\s?free)[^\d]{0,20}[\d][\d\-\s]{5,15}\d", text)
    if phone:
        return FieldResult("consumer_care", "Consumer Care / Complaint Contact",
                            "Rule 6(1)(f)", "PASS", phone, 0.85)
    if email:
        return FieldResult("consumer_care", "Consumer Care / Complaint Contact",
                            "Rule 6(1)(f)", "REVIEW_REQUIRED", email, 0.5,
                            "Email found; explicit 'customer care' label not confirmed.")
    return FieldResult("consumer_care", "Consumer Care / Complaint Contact",
                        "Rule 6(1)(f)", "NON_COMPLIANT", None, 0.0)


def check_country_of_origin(text: str) -> FieldResult:
    pattern = r"(country\s+of\s+origin|made\s+in)[:\-]?\s*[A-Za-z\s]{3,20}"
    evidence = _find(pattern, text)
    if evidence:
        return FieldResult("country_of_origin", "Country of Origin (if imported)",
                            "Rule 6(1)(h) / 2017 amendment", "PASS", evidence, 0.8,
                            "Not applicable for domestically manufactured goods.")
    return FieldResult("country_of_origin", "Country of Origin (if imported)",
                        "Rule 6(1)(h) / 2017 amendment", "REVIEW_REQUIRED", None, 0.2,
                        "Not found — PASS by exemption if product is domestic; flagged for human check.")


CHECKS = [
    check_manufacturer,
    check_common_name,
    check_net_quantity,
    check_mrp,
    check_mfg_date,
    check_consumer_care,
    check_country_of_origin,
]

# Fields whose absence is a hard compliance failure (core Rule 6 declarations).
# country_of_origin and common_name are excluded from the hard-fail set because
# they need extra context (import status / product taxonomy) to judge fairly —
# this mirrors the "Applicability layer" mentioned in the pitch.
HARD_REQUIRED = {"manufacturer", "net_quantity", "mrp", "mfg_date", "consumer_care"}


def run_compliance_check(ocr_text: str) -> ComplianceReport:
    results = [check(ocr_text) for check in CHECKS]

    non_compliant_hard = [r for r in results if r.field_id in HARD_REQUIRED and r.status == "NON_COMPLIANT"]
    review_hard = [r for r in results if r.field_id in HARD_REQUIRED and r.status == "REVIEW_REQUIRED"]

    if non_compliant_hard:
        overall = "POTENTIAL_NON_COMPLIANCE"
    elif review_hard or any(r.status == "REVIEW_REQUIRED" for r in results):
        overall = "REVIEW_REQUIRED"
    else:
        overall = "COMPLIANT"

    weight = {"PASS": 1.0, "REVIEW_REQUIRED": 0.5, "NON_COMPLIANT": 0.0}
    score = sum(weight[r.status] for r in results) / len(results) * 100

    return ComplianceReport(
        overall_status=overall,
        score=round(score, 1),
        ruleset_version=RULESET_VERSION,
        fields=results,
        raw_text=ocr_text,
    )
