"""
Core pipeline: J2K -> PNG conversion, Gemma vision OCR, structured field
extraction, and dictionary + LLM post-correction (spell correction).

Design notes
------------
* These are 19th/20th-c. handwritten French military registers. Small vision
  models CONFABULATE on hard handwriting, so every prompt is hardened against
  invention and the UI keeps a human in the loop (image beside transcription).
* OCR is a two-stage process: (1) verbatim vision transcription, (2) text->fields
  extraction. Keeping them separate lets the UI "showcase" the raw OCR and makes
  the field JSON far more reliable than asking one call to do both.
* Post-correction is also two-layer: closed-set fuzzy snapping (months, colours,
  departements) + an LLM spelling pass for names/places, always returning a diff.
"""
import io, os, json, base64, re, unicodedata
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2
import ollama
from rapidfuzz import process, fuzz

import fields as F
import dictionaries as D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
# Legacy single-folder pointer (kept for reference / status display)
SRC_IMAGES = PROJECT / "d_63392.IDX.001_I4417928 (1)" / "d_63392.IDX.001_I4417928" / "Images"
OUT = ROOT / "output"
PNG_DIR = OUT / "png"
RESULTS = OUT / "results.json"
for d in (OUT, PNG_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_VISION_MODEL = "qwen2.5vl:32b"  # primary vision reader (transcribe + vision-extract)
DEFAULT_TEXT_MODEL = "qwen2.5:7b"       # fast + good French for extraction/correction
QWEN_VISION_MODEL = "qwen2.5vl:7b"      # independent (smaller) vision reader for final verification —
                                         # now genuinely independent of the 32b primary, so it no longer self-skips
OCR_MAX_SIDE = 2200                    # downscale long side for the vision model

# ---------------------------------------------------------------------------
# Results persistence (simple JSON store keyed by image stem)
# ---------------------------------------------------------------------------
def load_results() -> dict:
    if RESULTS.exists():
        try:
            return json.loads(RESULTS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_results(data: dict):
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Image discovery + conversion  (scans EVERY */Images/ folder under the project)
# ---------------------------------------------------------------------------
_INDEX: dict[str, Path] = {}   # image name -> source image path
_SRC_DIR: Path | None = None   # user-supplied images directory (overrides auto-scan)
IMG_EXTS = (".j2k", ".jp2", ".jp2", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Row-crops produced by split_into_rows() (see below) aren't real files in any
# source folder, so plain directory scanning can never find them again after
# a restart — this small registry persists {name: png_path} so discover_images
# can merge them back in every time.
DERIVED_REGISTRY = OUT / "derived_images.json"

def _load_derived() -> dict:
    if DERIVED_REGISTRY.exists():
        try:
            return json.loads(DERIVED_REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_derived(d: dict):
    DERIVED_REGISTRY.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def set_source_dir(path: str) -> dict:
    """Point discovery at a specific directory the user passes in."""
    global _SRC_DIR
    p = Path(path.strip().strip('"').strip("'"))
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    _SRC_DIR = p
    idx = discover_images(refresh=True)
    return {"dir": str(p), "count": len(idx)}

def discover_images(refresh: bool = False) -> dict[str, Path]:
    """Find all supported images. If the user set a source dir, scan that
    (recursively); otherwise auto-scan any 'Images' folder in the project tree."""
    global _INDEX
    if _INDEX and not refresh:
        return _INDEX
    idx: dict[str, Path] = {}
    if _SRC_DIR is not None:
        for p in _SRC_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                idx[p.name] = p
    else:
        for p in PROJECT.rglob("*.j2k"):
            if "Images" in p.parts:
                idx[p.name] = p
        if not idx:                                  # fall back to any .j2k
            for p in PROJECT.rglob("*.j2k"):
                idx[p.name] = p
    # Row-splits, if any exist for pages in the CURRENT scan only — the
    # derived registry is a single global file shared across every folder
    # the app has ever been pointed at (e.g. a bulk dataset-prep run over a
    # different, much larger collection), so without this filter every
    # row-crop ever created bleeds into whatever folder is loaded next.
    current_stems = {Path(n).stem for n in idx}
    for name, p in _load_derived().items():
        m = _ROW_SPLIT_RE.match(name)
        parent_stem = m.group("parent") if m else Path(name).stem
        if parent_stem not in current_stems:
            continue
        pp = Path(p)
        if pp.exists():
            idx[name] = pp
    _INDEX = dict(sorted(idx.items()))
    return _INDEX

def list_source_images():
    return list(discover_images().keys())

def source_folders() -> list[str]:
    return sorted({str(p.parent) for p in discover_images().values()})

def png_path(name: str) -> Path:
    return PNG_DIR / (Path(name).stem + ".png")

def convert_one(name: str, force: bool = False) -> dict:
    """J2K -> lossless PNG. Returns {name, png, width, height, kb}."""
    src = discover_images().get(name)
    if src is None:
        raise FileNotFoundError(f"unknown image {name}")
    dst = png_path(name)
    if dst.exists() and not force:
        with Image.open(dst) as im:
            w, h = im.size
        return {"name": name, "png": dst.name, "width": w, "height": h,
                "kb": round(dst.stat().st_size / 1024, 1), "cached": True}
    with Image.open(src) as im:
        im.load()
        if im.mode not in ("L", "RGB", "I;16", "I"):
            im = im.convert("L")
        im.save(dst, "PNG", optimize=True)
        w, h = im.size
    return {"name": name, "png": dst.name, "width": w, "height": h,
            "kb": round(dst.stat().st_size / 1024, 1), "cached": False}

# ---------------------------------------------------------------------------
# Stage — Deskew (minAreaRect on the ink mass + Hough line transform on rule
# edges; the two are cross-checked and combined so a single noisy estimator
# can't rotate the page in the wrong direction).
# ---------------------------------------------------------------------------
def deskew_path(name: str) -> Path:
    return PNG_DIR / f"{Path(name).stem}_deskewed.png"

def _minarearect_angle(mask: np.ndarray) -> float | None:
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 50:
        return None
    angle = cv2.minAreaRect(coords)[-1]
    return -(90 + angle) if angle < -45 else -angle

def _hough_angle(gray: np.ndarray) -> float | None:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=200)
    if lines is None:
        return None
    angles = []
    for l in lines[:500]:
        rho, theta = l[0]
        deg = (theta * 180 / np.pi) - 90
        if -45 <= deg <= 45:
            angles.append(deg)
    return float(np.median(angles)) if angles else None

def deskew_image(name: str, force: bool = False) -> dict:
    """Detect + correct page skew. Returns {angle, rotated, method, file}."""
    dst = deskew_path(name)
    if dst.exists() and not force:
        return {"angle": 0.0, "rotated": None, "method": "cached", "file": dst.name}
    src = png_path(name)
    if not src.exists():
        convert_one(name)
    gray = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    a_rect = _minarearect_angle(mask)
    a_hough = _hough_angle(gray)
    candidates = [a for a in (a_rect, a_hough) if a is not None]
    if not candidates:
        angle, method = 0.0, "none"
    elif a_rect is not None and a_hough is not None and abs(a_rect - a_hough) < 5:
        angle, method = (a_rect + a_hough) / 2, "minAreaRect+hough"
    else:
        angle, method = candidates[0], ("minAreaRect" if a_rect is not None else "hough")
    angle = round(float(angle), 3)
    if abs(angle) < 0.1:
        ok, buf = cv2.imencode(".png", gray)
        Path(dst).write_bytes(buf.tobytes())
        return {"angle": angle, "rotated": False, "method": method, "file": dst.name}
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode(".png", rotated)
    Path(dst).write_bytes(buf.tobytes())
    return {"angle": angle, "rotated": True, "method": method, "file": dst.name}

# ---------------------------------------------------------------------------
# Stage — Contrast normalization: CLAHE (clip 2.5, 16x16 tiles) + light
# non-local-means denoise. Runs on the deskewed page.
# ---------------------------------------------------------------------------
def enhance_path(name: str) -> Path:
    return PNG_DIR / f"{Path(name).stem}_preprocessed.png"

def enhance_contrast(name: str, force: bool = False) -> dict:
    """CLAHE contrast normalization + denoise. Returns {file}."""
    dst = enhance_path(name)
    if dst.exists() and not force:
        return {"file": dst.name, "cached": True}
    src = deskew_path(name)
    if not src.exists():
        deskew_image(name)
    gray = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    eq = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(eq, h=7)
    ok, buf = cv2.imencode(".png", denoised)
    Path(dst).write_bytes(buf.tobytes())
    return {"file": dst.name, "cached": False}

# ---------------------------------------------------------------------------
# Stage — Row-splitting for multi-person ledger pages ("registre matricule"
# style scans: several people's records stacked in horizontal bands spanning
# a double-page spread, as opposed to the one-person-per-page "fiche
# individuelle" format the rest of the pipeline assumes). Detects the
# horizontal rule lines that separate each person's row and crops each band
# into its own image, which then goes through the exact same OCR/extraction
# pipeline as any normal single-person image.
# ---------------------------------------------------------------------------
def _row_band_positions(mask: np.ndarray) -> list[int]:
    """Return y-positions of horizontal rule lines inside an already-cropped
    (content-only, no black scan border) binary ink mask."""
    ch, cw = mask.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(cw * 0.15), 1))
    lines_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horiz_kernel)
    row_sums = lines_mask.sum(axis=1) / 255
    if row_sums.max() == 0:
        return []
    thresh = row_sums.max() * 0.4
    candidate_rows = np.where(row_sums > thresh)[0]
    if len(candidate_rows) == 0:
        return []
    groups, cur = [], [int(candidate_rows[0])]
    for r in candidate_rows[1:]:
        r = int(r)
        # real row dividers are never closer than ~100px apart on this form;
        # a second peak within ~35px of the first is the same physical rule
        # line detected twice (e.g. a slightly uneven/thick print), not a
        # second one — merge distance wider than the observed ~25-26px
        # near-duplicate offset seen on real pages
        if r - cur[-1] <= 35:
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    positions = [int(np.mean(g)) for g in groups]
    edge_margin = ch * 0.015
    return [p for p in positions if edge_margin < p < ch - edge_margin]

def detect_row_bands(name: str) -> list[tuple[int, int]]:
    """Detect each person's row as a (top, bottom) y-range in the plain
    converted image's coordinate space. Deliberately uses the RAW converted
    PNG rather than the deskewed one — cv2.warpAffine's interpolation subtly
    blurs thin printed rule lines even at a near-zero rotation angle, which
    was enough to drop them below the line-detection threshold on 2 of 5
    real test pages. These ledger pages run close enough to level that
    deskewing buys nothing here anyway. Handles a rule line the scan/print
    missed by filling in evenly-spaced synthetic dividers across any gap
    that's a multiple of the median row height."""
    src = png_path(name)
    if not src.exists():
        convert_one(name)
    img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    row_means = img.mean(axis=1); col_means = img.mean(axis=0)
    content_rows = np.where(row_means > 100)[0]
    content_cols = np.where(col_means > 100)[0]
    if len(content_rows) == 0 or len(content_cols) == 0:
        return [(0, h)]
    top, bottom = int(content_rows.min()), int(content_rows.max())
    left, right = int(content_cols.min()), int(content_cols.max())
    crop = img[top:bottom, left:right]
    ch, cw = crop.shape
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    positions = _row_band_positions(mask)
    bands = _bands_from_positions(positions, top)

    # This printed ledger form is physically identical page to page, so a
    # real row is always close to FORM_ROW_HEIGHT tall. When detection finds
    # too few/too many lines (a faint unprinted rule, a spurious duplicate
    # a few px off a real one) the result is implausible — a giant leftover
    # band or tiny slivers — and no amount of median-based gap-filling can
    # recover from bad input. Fall back to the form's known fixed geometry.
    heights = [b - a for a, b in bands]
    plausible = bool(heights) and all(
        FORM_ROW_HEIGHT * 0.6 <= h <= FORM_ROW_HEIGHT * 1.4 for h in heights)
    if not plausible:
        start = top + FORM_HEADER_OFFSET
        bands = [(start + i * FORM_ROW_HEIGHT, start + (i + 1) * FORM_ROW_HEIGHT)
                 for i in range(FORM_ROWS_PER_PAGE)]
    return bands

# Empirical geometry of this specific printed "registre matricule" form,
# measured from pages where line-detection worked cleanly — used as a
# fallback when detection fails on a given page (see plausibility check
# above). FORM_HEADER_OFFSET is relative to the per-page content-top (NOT an
# absolute y-position — content-top itself varies ~100-130px page to page
# depending on scan cropping, so it must be added back in by the caller).
# If a different ledger form is ever processed, these numbers won't fit it
# and detection will just fall back to a single full-page band.
FORM_HEADER_OFFSET = 390
FORM_ROW_HEIGHT = 790
FORM_ROWS_PER_PAGE = 4

def _bands_from_positions(positions: list[int], top: int) -> list[tuple[int, int]]:
    """Turn detected line y-positions (crop-local) into (top,bottom) bands
    (full-image coords), dropping the header band and gap-filling any
    divider the scan/print missed."""
    if len(positions) < 2:
        return []
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    normal_gaps = [g for g in gaps if g > 400] or gaps
    # a per-page median row-height is unreliable from very few samples (e.g.
    # exactly 2 gaps, where one is a multiple of the true row height skews
    # the "median" to their average) — anchor on the form's known row height
    # instead unless we have enough real samples to trust a local estimate
    med = float(np.median(normal_gaps)) if len(normal_gaps) >= 3 else float(FORM_ROW_HEIGHT)
    start_idx = 1 if gaps and gaps[0] < med * 0.6 else 0
    lines = positions[start_idx:]
    if len(lines) < 2:
        return []
    filled = [lines[0]]
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        n = max(round((b - a) / med), 1)
        for k in range(1, n + 1):
            filled.append(a + round((b - a) * k / n))
    return [(top + filled[i], top + filled[i + 1]) for i in range(len(filled) - 1)]

def split_into_rows(name: str, pad: int = 8, force: bool = False) -> list[dict]:
    """Crop each detected person-row (full width, both pages) into its own
    image and register it as a normal processable image. Each row-crop still
    goes through its own deskew/enhance once treated as an image in its own
    right — only the DETECTION here uses the raw (undeskewed) source, per
    detect_row_bands()'s docstring. Returns [{name, band}] for each row."""
    src = png_path(name)
    if not src.exists():
        convert_one(name)
    img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    bands = detect_row_bands(name)
    stem = Path(name).stem
    derived = _load_derived()
    out = []
    for i, (t, b) in enumerate(bands, start=1):
        t2, b2 = max(0, t - pad), min(h, b + pad)
        row_name = f"{stem}_row{i}.png"
        row_path = PNG_DIR / row_name
        if force or not row_path.exists():
            ok, buf = cv2.imencode(".png", img[t2:b2, :])
            row_path.write_bytes(buf.tobytes())
        derived[row_name] = str(row_path)
        out.append({"name": row_name, "band": [t2, b2]})
    _save_derived(derived)
    discover_images(refresh=True)
    return out

def _ocr_image_bytes(name: str) -> bytes:
    """Return the deskewed + CLAHE-enhanced, downscaled PNG (bytes) tuned for
    the OCR model. Falls back to a plain autocontrast pass if preprocessing
    hasn't run yet for some reason."""
    src = enhance_path(name)
    if not src.exists():
        try:
            deskew_image(name)
            enhance_contrast(name)
        except Exception:
            src = png_path(name)
            if not src.exists():
                convert_one(name)
    im = Image.open(src).convert("L")
    im = ImageOps.autocontrast(im, cutoff=1)
    w, h = im.size
    scale = OCR_MAX_SIDE / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Stage 1 — Vision OCR (verbatim transcription)
# ---------------------------------------------------------------------------
OCR_PROMPT = """You are a meticulous OCR/HTR engine for historical FRENCH military
conscription registers (1800s-1900s, printed forms with handwritten entries).

Transcribe EVERY piece of visible text on this page, VERBATIM.

STRICT RULES:
- Transcribe ONLY what is physically written. Never guess, invent, translate,
  paraphrase, or add explanatory text.
- Preserve the original French spelling, accents and capitalisation exactly.
- Keep the page's line-by-line layout. One text line per output line.
- If a word or number is unclear/illegible, write [?] in its place. Do not fill
  it in with a plausible guess.
- If the page is blank, a cover page, a target/calibration sheet, or an index,
  say exactly what it is in one short line and nothing more.
- Do NOT output any preamble, commentary, headings you invented, or closing notes.

Output the raw transcription now."""

def _clean_ocr_text(text: str) -> str:
    """Strip sentinels and kill runaway repetition loops that small vision models
    fall into (e.g. hundreds of '[?] le [?]' lines)."""
    text = re.sub(r"</?start_of_image>|</?end_of_image>", "", text)
    text = re.sub(r"^```.*?$", "", text, flags=re.MULTILINE)

    # 1) collapse identical / near-identical consecutive lines (keep at most 1)
    out, prev_key, run = [], None, 0
    for ln in text.splitlines():
        key = re.sub(r"\s+", " ", ln.strip().lower())
        # a "line" that is only illegible/placeholder filler
        filler = bool(re.fullmatch(r"(?:\[\?\]|le|la|au|à|du|de|et|,|\.|…|-|\s)*", key)) and "[?]" in key
        if key and key == prev_key:
            run += 1
            if run >= 1:            # drop the 2nd+ identical line
                continue
        else:
            prev_key, run = key, 0
        if filler and out and "[?]" in re.sub(r"\s+", " ", out[-1].lower()):
            continue                # drop consecutive filler lines
        out.append(ln)
    text = "\n".join(out)

    # 2) collapse inline "[?] le [?] le [?] …" style loops within a line
    text = re.sub(r"(?:\[\?\]\s*(?:le|la|au|à|du|de|et)?\s*){3,}", "[?] … ", text, flags=re.I)
    text = re.sub(r"(?:\[\?\]\s*){3,}", "[?] … ", text)
    # 3) if the SAME short phrase repeats 4+ times anywhere, keep 2 and cut
    text = re.sub(r"(\b[^\n]{2,40}?\b)(?:[\s,;·]*\1){3,}", r"\1 \1 …", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def run_ocr(name: str, model: str = DEFAULT_VISION_MODEL) -> dict:
    data = _ocr_image_bytes(name)
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": OCR_PROMPT, "images": [data]}],
        # repeat_penalty + capped output stop the model looping on '[?] le [?]'
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 1800,
                 "repeat_penalty": 1.35, "repeat_last_n": 320},
    )
    return {"raw_text": _clean_ocr_text(r["message"]["content"]), "ocr_model": model}

# ---------------------------------------------------------------------------
# Stage 2 — Structured field extraction (text -> template fields JSON)
# ---------------------------------------------------------------------------
def _field_spec_block() -> str:
    lines = []
    for label, key, vocab in F.MILITARY_FIELDS:
        hint = f"  ({vocab})" if vocab else ""
        lines.append(f'- "{label}"{hint}')
    return "\n".join(lines)

EXTRACT_PROMPT = """You extract structured fields from the OCR transcription of a
FRENCH military register record, following the Ancestry keying spec.

Return ONLY a JSON object whose keys are EXACTLY the field labels below. For every
field, output the value found in the transcription, or "" (empty string) if it is
not present. NEVER invent a value that is not supported by the transcription.

Keying rules:
- Months: use the French month name/abbreviation as written (e.g. janvier, août, 8bre) — do NOT translate to an English code.
- Hair/Eye colour: output the CI code (hair: BR BK LT FR RD CH GY; eye: BL BR GR GB GY BG H CH).
- Height like "1 m. 66 cent." -> "1m 66cm".
- Deceased Father/Mother: "Y" if 'feu'/'feue'/'defunt' precedes the name, else "".
- Enlistment Departement: leave "" if the value is Aude, L'Aude, or Tarn.
- Regiment: format as "{{n}}e Régt" (or "1er Régt" for the 1st) — omit any branch/arm descriptor (e.g. "d'Infanterie", "de Dragons"), that belongs in Branch/Unit instead.
- Event Type: "Military" or "Coverpage".
- Names keep their original French spelling.
- Printed ordinal/item markers ("1°", "2°", "3°"...) are the FORM'S OWN
  layout numbering (each printed line restarts its own count), not data —
  never copy one of these into Prefix or Suffix.
- Each field holds ONLY the value for its own printed label. In a dense
  multi-column table, do not concatenate text from a neighbouring column or
  the next printed label into one field (e.g. a unit symbol like "m" or
  "mètre" belongs to Height, not to Domicile).
- Given Name (Prénoms) can be a compound of multiple words (e.g. "Jean
  Baptiste", "François Antoine") — keep the whole printed phrase together
  as Given Name. The family name (Nom de famille) is the separate word
  printed above/beside it, usually on its own line.

Fields:
{fields}

Transcription:
\"\"\"
{transcription}
\"\"\"

Return only the JSON object, no commentary."""

def _empty_fields() -> dict:
    return {label: "" for label in F.FIELD_LABELS}

def _key_variants(label: str):
    """Forms a model might emit for a field label (exact, lower, snake_case)."""
    low = label.lower().strip()
    return {label, low, low.replace(" ", "_"), low.replace(" ", ""),
            low.replace("-", " "), low.replace(" ", "_").replace("-", "_")}

# precompute: normalized emitted-key -> canonical label
_KEY_LOOKUP = {}
for _lb in F.FIELD_LABELS:
    for _v in _key_variants(_lb):
        _KEY_LOOKUP[_v] = _lb

def _clean_val(v) -> str:
    """Strip illegible markers and OCR noise from an extracted field value."""
    s = ("" if v is None else str(v)).strip()
    # drop values that are only illegible markers / placeholders
    if re.fullmatch(r"[\[\]\?\.…\s\-–—…]*", s):
        return ""
    if s.lower() in ("n/a", "none", "null", "inconnu", "unknown", "illisible", "[illegible]"):
        return ""
    # remove embedded illegible tokens but keep the rest
    s = re.sub(r"\[\?\]|…", "", s).strip(" -–—,;")
    return s.strip()

def _remap_keys(parsed: dict) -> dict:
    """Map whatever keys the LLM emitted back to canonical field labels."""
    out = _empty_fields()
    for k, v in (parsed or {}).items():
        lb = _KEY_LOOKUP.get(str(k), _KEY_LOOKUP.get(str(k).lower().strip()))
        if lb:
            out[lb] = _clean_val(v)
    return out

def _parse_json_obj(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        # tolerate trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

def extract_fields(raw_text: str, model: str = DEFAULT_TEXT_MODEL) -> dict:
    if not raw_text.strip():
        return _empty_fields()
    prompt = EXTRACT_PROMPT.format(fields=_field_spec_block(), transcription=raw_text[:6000])
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "num_ctx": 8192},
    )
    parsed = _parse_json_obj(r["message"]["content"])
    return _remap_keys(parsed)

# ---------------------------------------------------------------------------
# Stage 2b — VISION structured extraction (layout-aware, w/ evidence+confidence)
# The model reads fields directly off the image, so it uses spatial layout
# (column headers, row alignment) that a plain transcription loses. Returns a
# per-field cell {value, confidence, evidence}. This is the accuracy backbone.
# ---------------------------------------------------------------------------
VISION_EXTRACT_PROMPT = """You are an expert indexer reading ONE image of a FRENCH
military conscription register (1800s-1900s printed form + handwriting).

Extract the fields listed below by looking at the image directly, using the printed
column headers and labels to locate each value.

Return ONLY a JSON object. Each key is a field label. Each value is an object:
  {{"value": <the value or "">, "confidence": "HIGH"|"MEDIUM"|"LOW", "evidence": <the exact words/label on the page that justify it, or "">}}

HARD RULES (accuracy over completeness — a blank is ALWAYS better than a guess):
- Output a value ONLY if you can actually see it on the page. If absent, value "".
- NEVER guess, complete, or infer a name/number/date to fill a blank or an
  unreadable spot. If you are not sure, leave value "" — do not invent.
- Copy numbers DIGIT FOR DIGIT exactly as written. Never add, drop, or infer a
  digit. If part of a number is unreadable, leave the whole field "".
- Do not carry a value from one field into another just because it "fits".
- confidence: HIGH = printed/clearly legible; MEDIUM = readable handwriting;
  LOW = hard to read / uncertain.
- evidence MUST be the exact words that literally appear on the page next to the
  value (usually the printed column label). If you cannot quote real on-page text
  as evidence, set the value to "". Never fabricate evidence.

Keying rules:
- Months: use the French month name/abbreviation as written (e.g. janvier, août, 8bre) — do NOT translate to an English code.
- Hair code: BR BK LT FR RD CH GY ; Eye code: BL BR GR GB GY BG H CH.
- Height "1 m. 66 cent." -> "1m 66cm".
- Deceased Father/Mother = "Y" only if 'feu'/'feue'/'defunt(e)' precedes the name.
- Enlistment Departement: "" if it is Aude, L'Aude or Tarn.
- Regiment: format as "{{n}}e Régt" (or "1er Régt" for the 1st) — omit any branch/arm descriptor (e.g. "d'Infanterie", "de Dragons"), that belongs in Branch/Unit instead.
- Event Type: "Military" or "Coverpage".
- Printed ordinal/item markers ("1°", "2°", "3°"...) are the FORM'S OWN
  layout numbering (each printed line restarts its own count), not data —
  never copy one of these into Prefix or Suffix.
- Each field holds ONLY the value for its own printed label. In a dense
  multi-column table, do not merge text from a neighbouring column or the
  next printed label into one field (e.g. a unit symbol like "m" or
  "mètre" belongs to Height, not to Domicile).
- Given Name (Prénoms) can be a compound of multiple words (e.g. "Jean
  Baptiste", "François Antoine") — keep the whole printed phrase together
  as Given Name. The family name (Nom de famille) is the separate word
  printed above/beside it, usually on its own line.

Fields:
{fields}

Return only the JSON object."""

def _remap_meta(parsed: dict) -> dict:
    """Normalise a {label: {value,confidence,evidence}} (or {label: value}) blob."""
    out = {lb: {"value": "", "confidence": "", "evidence": ""} for lb in F.FIELD_LABELS}
    for k, cell in (parsed or {}).items():
        lb = _KEY_LOOKUP.get(str(k), _KEY_LOOKUP.get(str(k).lower().strip()))
        if not lb:
            continue
        if isinstance(cell, dict):
            val = _clean_val(cell.get("value", ""))
            conf = str(cell.get("confidence", "")).upper().strip()[:6]
            ev = str(cell.get("evidence", "") or "")[:140]
        else:
            val = _clean_val(cell); conf = ""; ev = ""
        if val and conf not in ("HIGH", "MEDIUM", "LOW"):
            conf = "MEDIUM"
        out[lb] = {"value": val, "confidence": conf, "evidence": ev}
    return out

def vision_extract(name: str, model: str = DEFAULT_VISION_MODEL) -> dict:
    """Extract fields straight from the image. Returns {label:{value,confidence,evidence}}."""
    data = _ocr_image_bytes(name)
    prompt = VISION_EXTRACT_PROMPT.format(fields=_field_spec_block())
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [data]}],
        format="json",
        options={"temperature": 0, "num_ctx": 8192},
    )
    return _remap_meta(_parse_json_obj(r["message"]["content"]))

# ---------------------------------------------------------------------------
# Stage 3 — Consensus reconciliation of the two independent readers
# vision-extract (image)  vs  text-extract (transcription)
#   agree            -> HIGH confidence
#   only one present -> MEDIUM (or the vision confidence)
#   both differ      -> LOW, keep the vision reading, record the alternative
# ---------------------------------------------------------------------------
def reconcile(vision_meta: dict, text_fields: dict) -> tuple[dict, dict]:
    """Return (values{label:str}, meta{label:{confidence,evidence,source,alt}})."""
    values, meta = {}, {}
    for lb in F.FIELD_LABELS:
        vm = vision_meta.get(lb, {"value": "", "confidence": "", "evidence": ""})
        vv, vconf, vev = vm["value"], vm["confidence"] or "", vm["evidence"]
        tv = (text_fields.get(lb, "") or "").strip()
        nv, nt = _norm(vv), _norm(tv)
        alt = ""
        if vv and tv and nv == nt:
            val, conf, src = vv, "HIGH", "vision+text"
        elif vv and tv and nv != nt:
            val, conf, src = vv, "LOW", "conflict"; alt = tv
        elif vv and not tv:
            val, conf, src = vv, (vconf or "MEDIUM"), "vision"
        elif tv and not vv:
            val, conf, src = tv, "MEDIUM", "text"
        else:
            val, conf, src = "", "", ""
        values[lb] = val
        meta[lb] = {"confidence": conf, "evidence": vev, "source": src, "alt": alt}
    return values, meta

def confidence_stats(meta: dict, values: dict) -> dict:
    filled = [lb for lb in F.FIELD_LABELS if values.get(lb, "").strip()]
    def c(level): return sum(1 for lb in filled if meta.get(lb, {}).get("confidence") == level)
    return {"filled": len(filled), "high": c("HIGH"), "medium": c("MEDIUM"),
            "low": c("LOW"),
            "conflicts": sum(1 for lb in filled if meta.get(lb, {}).get("source") == "conflict"),
            "flagged": sum(1 for lb in filled if meta.get(lb, {}).get("flags"))}

# ---------------------------------------------------------------------------
# Deterministic doctype classification from the page's structural headers.
# These fiche-matricule section titles are printed (reliable) even when the
# handwritten names are misread, so they beat the model's Event Type guess.
# ---------------------------------------------------------------------------
_MIL_MARKERS = [
    "signalement", "detail des services", "mutations diverses", "conseil de revision",
    "numero matricule", "etat civil", "corps d affectation", "decision du conseil",
    "degre d instruction", "inscrit sous le", "classe dans", "antecedents judiciaires",
    "blessures", "campagnes", "renseignements physionomiques", "localites successives",
]
_COVER_MARKERS = [
    "subdivision de", "repertoire", "table alphabetique", "bureau de recrutement",
    "registre matricule des", "classe de mobilisation de", "liste generale", "volume", "tome",
]

_ROW_SPLIT_RE = re.compile(r"^(?P<parent>.+)_row\d+\.png$")

def is_row_split(name: str) -> bool:
    """True if `name` is a row-crop produced by split_into_rows(). Every such
    row is, by construction, one person's entry in a multi-person ledger
    page — there's no "coverpage" concept at the row level, and the
    structural section-header keywords classify_doctype() looks for live in
    the page's shared header band, which row-splitting deliberately excludes
    from every row. So neither the structural classifier nor (it turns out)
    the model's own per-row guess is reliable here — different rows of the
    exact same page were observed classifying into 3 different values,
    including one outside the 2-value vocabulary entirely. Event Type is
    forced to "Military" for these instead of guessed."""
    return bool(_ROW_SPLIT_RE.match(name))

def classify_doctype(transcript: str):
    """Return (doctype, confidence) from structural markers, or ('', '') if unclear."""
    t = _norm(transcript or "")
    mil = sum(1 for m in _MIL_MARKERS if m in t)
    cov = sum(1 for c in _COVER_MARKERS if c in t)
    if mil >= 2 and mil >= cov:
        return "Military", "HIGH"
    if mil == 1 and cov == 0:
        return "Military", "MEDIUM"
    if cov >= 1 and mil == 0:
        return "Coverpage", "MEDIUM"
    return "", ""

# ---------------------------------------------------------------------------
# Stage 6 — GROUNDING VERIFICATION (anti-hallucination)
# Cross-checks every extracted value against the INDEPENDENT transcription and
# against plausibility rules. A hallucinated value has no support in the
# transcript, so it is caught deterministically: flagged ⚠, confidence forced to
# LOW, and (in strict mode) blanked out entirely.
# ---------------------------------------------------------------------------
_NUM_FIELDS = {"Birth Day", "Birth Year", "Discharge Day", "Discharge Year",
               "Death Day", "Death Year", "Classe Year", "Entry Number"}
_DAY_FIELDS = {"Birth Day", "Discharge Day", "Death Day"}
_YEAR_FIELDS = {"Birth Year", "Discharge Year", "Death Year", "Classe Year"}
# values that are a TRANSFORMATION of page text (skip literal-support check)
_DERIVED_VOCAB = {"month", "hair", "eye", "yn", "event"}

def _transcript_index(text: str):
    norm = _norm(text)
    tokens = set(re.findall(r"[a-z]{2,}", norm))
    digits = set(re.findall(r"\d+", text))
    return norm, tokens, digits

def _value_supported(value: str, lb: str, vocab, tnorm, ttokens, tdigits) -> bool:
    """True if `value` is actually backed by something in the transcription."""
    if lb in _NUM_FIELDS:
        dv = re.sub(r"\D", "", value)
        if not dv:
            return True
        alld = re.sub(r"\D", "", tnorm)
        if dv in alld:
            return True
        return any(dv == d or (len(dv) >= 3 and fuzz.ratio(dv, d) >= 85) for d in tdigits)
    if vocab in _DERIVED_VOCAB:
        return True                      # coded/derived — cannot literal-match
    toks = sorted([t for t in re.findall(r"[a-z]+", _norm(value)) if len(t) >= 4],
                  key=len, reverse=True)
    if not toks:
        return True                      # too short to judge (initials, etc.)
    if not ttokens:
        return False
    best = process.extractOne(toks[0], ttokens, scorer=fuzz.ratio)
    return bool(best and best[1] >= 82)

def _plausible(value: str, lb: str):
    """Flag both out-of-range AND malformed values — a value that isn't even
    in the expected clean format (e.g. "19, 1905" for a year, "1er" for a
    day) is at least as suspect as one that's cleanly formatted but out of
    range, so it must never silently skip the check."""
    flags = []
    v = value.strip()
    if lb in _DAY_FIELDS and v:
        if not v.isdigit():
            flags.append("day-format")
        elif not (1 <= int(v) <= 31):
            flags.append("impossible-day")
    if lb in _YEAR_FIELDS and v:
        if not re.fullmatch(r"\d{4}", v):
            flags.append("year-format")
        elif not (1750 <= int(v) <= 1960):
            flags.append("year-out-of-range")
    if lb == "Height" and v:
        m = re.fullmatch(r"(\d)m (\d{1,2})cm", v)
        if not m:
            flags.append("height-format")
        elif int(m.group(1)) not in (1, 2):
            flags.append("height-implausible")
    return flags

def verify_grounding(values: dict, meta: dict, transcription: str,
                     strict: bool = False) -> dict:
    """Mutates meta in place: adds 'flags', downgrades suspect fields to LOW.
    In strict mode, blanks values whose text is unsupported by the transcript.
    Returns {dropped:[...], flagged:[...]}."""
    tnorm, ttokens, tdigits = _transcript_index(transcription or "")
    dropped, flagged = [], []
    for lb in F.FIELD_LABELS:
        v = (values.get(lb, "") or "").strip()
        m = meta.setdefault(lb, {"confidence": "", "evidence": "", "source": "", "alt": ""})
        m.setdefault("flags", [])
        if not v:
            continue
        vocab = F.LABEL_TO_VOCAB.get(lb)
        # 1) is the value corroborated by the INDEPENDENT transcription?
        #    (the core anti-hallucination signal — a value the transcript never
        #     saw is likely the vision model filling a gap it could not read)
        if not _value_supported(v, lb, vocab, tnorm, ttokens, tdigits):
            m["flags"].append("uncorroborated")
        # 2) plausibility / range checks
        m["flags"].extend(_plausible(v, lb))

        if m["flags"]:
            flagged.append(lb)
            if m.get("confidence") in ("HIGH", "MEDIUM"):
                m["confidence"] = "LOW"          # suspect -> force review
            if strict and "uncorroborated" in m["flags"]:
                dropped.append({"field": lb, "value": v, "reason": "not corroborated by transcript"})
                values[lb] = ""
    # 4) cross-field temporal sanity (birth before discharge/death/classe)
    def yr(lb):
        s = re.sub(r"\D", "", values.get(lb, ""))
        return int(s) if re.fullmatch(r"\d{4}", s) else None
    by = yr("Birth Year")
    if by:
        for lb in ("Discharge Year", "Death Year", "Classe Year"):
            y = yr(lb)
            if y and y < by:
                meta[lb].setdefault("flags", []).append("before-birth")
                if meta[lb].get("confidence") != "LOW":
                    meta[lb]["confidence"] = "LOW"
                if lb not in flagged:
                    flagged.append(lb)
    return {"dropped": dropped, "flagged": flagged}

# ---------------------------------------------------------------------------
# Stage — Qwen2.5-VL final verification pass. An INDEPENDENT vision reader
# (different model family from the Gemma readers above) re-reads the
# preprocessed page. It is used two ways:
#   1. Its own transcript is kept as a genuine third "final output" the UI
#      can show beside the Gemma transcript.
#   2. Any field the Gemma consensus left at LOW confidence (a conflict, or
#      an unreadable spot) is re-checked against Qwen's independent read —
#      if Qwen corroborates the kept value or its recorded alternative, the
#      field is upgraded to MEDIUM and the "uncorroborated" flag is cleared.
# This never invents a value that Gemma didn't already produce — Qwen can
# only confirm or leave a LOW field as-is, never introduce a new answer.
# ---------------------------------------------------------------------------
def qwen_finalize(name: str, values: dict, meta: dict, model: str = QWEN_VISION_MODEL) -> dict:
    try:
        qocr = run_ocr(name, model=model)
        qmeta = vision_extract(name, model=model)
    except Exception as e:
        return {"raw_text": "", "fields": {}, "upgraded": [], "error": str(e), "model": model}
    upgraded = []
    for lb in F.FIELD_LABELS:
        m = meta.get(lb)
        if not m or m.get("confidence") != "LOW":
            continue
        qv = (qmeta.get(lb, {}) or {}).get("value", "")
        if not qv:
            continue
        cur, alt = values.get(lb, ""), m.get("alt", "")
        if _norm(qv) == _norm(cur) or (alt and _norm(qv) == _norm(alt)):
            m["confidence"] = "MEDIUM"
            m["source"] = (m.get("source") or "") + "+qwen"
            m["flags"] = [f for f in m.get("flags", []) if f != "uncorroborated"]
            upgraded.append(lb)
    return {"raw_text": qocr["raw_text"], "fields": qmeta, "upgraded": upgraded,
            "model": model}

# ---------------------------------------------------------------------------
# Post-correction — layer 1: closed-set fuzzy snapping
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def _snap_vocab(value: str, vocab: str):
    """Return (snapped_value, changed_bool, note) using the closed set for `vocab`."""
    if not value:
        return value, False, ""
    spec = F.VOCAB_SETS.get(vocab)
    if spec is None:
        return value, False, ""

    if isinstance(spec, dict):
        # dict: canonical -> [variants]. Match against variants + canonical.
        candidates = {}
        for canon, variants in spec.items():
            candidates[_norm(canon)] = canon
            for v in variants:
                candidates[_norm(v)] = canon
        key = _norm(value)
        if key in candidates:
            snapped = candidates[key]
            return snapped, snapped != value, ("snap:" + vocab if snapped != value else "")
        best = process.extractOne(key, list(candidates.keys()), scorer=fuzz.WRatio)
        if best and best[1] >= 82:
            snapped = candidates[best[0]]
            if snapped != value:
                return snapped, True, f"snap:{vocab}({best[1]:.0f})"
        return value, False, ""
    else:
        # list of canonicals (states, cities, prefix, suffix)
        cand = {_norm(c): c for c in spec}
        key = _norm(value)
        if key in cand:
            snapped = cand[key]
            return snapped, snapped != value, ("case:" + vocab if snapped != value else "")
        best = process.extractOne(key, list(cand.keys()), scorer=fuzz.WRatio)
        # cities are open-ended -> only snap on very high confidence
        thresh = 90 if vocab == "city" else 85
        if best and best[1] >= thresh:
            snapped = cand[best[0]]
            if snapped != value:
                return snapped, True, f"snap:{vocab}({best[1]:.0f})"
        return value, False, ""

# ---------------------------------------------------------------------------
# Post-correction — layer 2: LLM spelling pass for free-text names/places
# ---------------------------------------------------------------------------
LLM_CORRECT_PROMPT = """You are correcting OCR spelling in fields read from a FRENCH
military register (Aude / Gironde region). Fix obvious OCR mistakes (merged/split
letters, wrong accents) so each value is a correct French name or place.

For each field you are given the read value and, for place fields, a short list of
KNOWN nearby place names as candidates. If the read value is clearly a garbled form
of one candidate, snap to that candidate; otherwise keep the read value.

RULES:
- Prefer the original. Only change it when the correction is clearly right.
- NEVER invent or fill an empty value. If input value is "", output "".
- Do NOT snap to a candidate unless it is obviously the same place misspelt.
- Keep proper French capitalisation and accents.

Return ONLY a JSON object mapping each field label to its corrected string value.

Input:
{payload}
"""

# free-text fields worth an LLM spelling pass (names + places)
LLM_FIELDS = [
    "Given Name", "Surname", "Father Given Name", "Father Surname",
    "Mother Given Name", "Mother Maiden Name", "Mother Surname",
    "Domicile", "Occupation", "Regiment", "Unit", "Branch", "Compagnie",
    "Battalion", "Rank",
    "Birth Commune", "Residence Commune", "Death Commune", "Enlistment City",
]

# Field label -> key in the external "Dictionary Updated" authoritative CI
# keying dictionaries (dictionaries.py). Hair/Eye/Month are deliberately left
# out: fields.py's small VOCAB_SETS already maps those straight to CI codes,
# and the external sheets for those two are noisy (mixed French/English
# keying-legend glosses) — better used only where a spelling-only snap makes
# sense (a bare canonical-value list), not a variant->code lookup.
FIELD_TO_DICTKEY = {
    "Given Name": "givenname", "Father Given Name": "givenname", "Mother Given Name": "givenname",
    "Surname": "surname", "Father Surname": "surname", "Mother Surname": "surname",
    "Mother Maiden Name": "surname",
    "Birth Commune": "city", "Birth Canton": "city", "Residence Commune": "city",
    "Residence Canton": "city", "Enlistment City": "city", "Death Commune": "city", "Domicile": "city",
    "Birth Departement": "state", "Enlistment Departement": "state",
    "Prefix": "prefix", "Father Prefix": "prefix", "Mother Prefix": "prefix",
    "Suffix": "suffix", "Father Suffix": "suffix", "Mother Suffix": "suffix",
    "Rank": "rank", "Regiment": "regiment", "Unit": "unit", "Branch": "branch",
    "Compagnie": "company", "Battalion": "battalion", "Occupation": "occupation",
}

def _place_candidates(value: str, vocab: str, k: int = 4):
    """Top-k known place names near `value` (fallback grounding for the LLM
    when the external dictionary directory isn't available)."""
    spec = F.VOCAB_SETS.get(vocab)
    pool = list(spec) if isinstance(spec, (list, tuple)) else list((spec or {}).keys())
    if not value or not pool:
        return []
    hits = process.extract(_norm(value), {c: _norm(c) for c in pool},
                           scorer=fuzz.WRatio, limit=k)
    return [h[2] for h in hits if h[1] >= 60]

def _dict_candidates(value: str, label: str, k: int = 4):
    """Grounding candidates for the LLM pass: prefer the big authoritative
    dictionary for this field, fall back to the small project vocab lists."""
    dictkey = FIELD_TO_DICTKEY.get(label)
    if dictkey and D.available():
        cands = D.top_candidates(value, dictkey, k=k)
        if cands:
            return cands
    vocab = F.LABEL_TO_VOCAB.get(label)
    return _place_candidates(value, vocab, k=k) if vocab else []

# Dictionaries that are a strict, tiny closed enum (not an open-ended name/
# place list) — canonicalizing against these carries no hallucination risk
# even for a HIGH-confidence field, because there's no "rare-but-correct"
# entry that could exist outside such a small enum. Exempt from the
# confidence gate below.
_CLOSED_ENUM_DICTS = {"branch"}

def dict_correct(fields_in: dict, meta: dict | None = None) -> tuple[dict, list]:
    """Snap free-text fields (names, places, ranks, units, occupations...)
    against the authoritative external CI keying dictionaries. Deterministic
    and conservative — only snaps on a high-confidence fuzzy match, and only
    ever replaces a value with a real dictionary entry, never invents one.

    Anti-hallucination: when `meta` is given, a field the two independent
    vision readers already AGREED on (HIGH confidence) is left untouched for
    the OPEN-ENDED dictionaries (names, places, ranks...) — that's exactly
    the kind of place a rare-but-correct real name/place could get silently
    overwritten by a similar-looking dictionary entry. Tiny closed enums
    (see _CLOSED_ENUM_DICTS) are exempt from this gate since there's no such
    risk."""
    corrected = dict(fields_in)
    diffs = []
    if not D.available():
        return corrected, diffs
    for label, dictkey in FIELD_TO_DICTKEY.items():
        if meta and dictkey not in _CLOSED_ENUM_DICTS and meta.get(label, {}).get("confidence") == "HIGH":
            continue
        v = (corrected.get(label, "") or "").strip()
        if not v:
            continue
        if dictkey in _CLOSED_ENUM_DICTS:
            match, score = D.token_contains_match(v, dictkey)
        else:
            match, score = D.best_match(v, dictkey)
        if match and match != v:
            diffs.append({"field": label, "original": v, "corrected": match,
                          "note": f"dict:{dictkey}({score:.0f})"})
            corrected[label] = match
    return corrected, diffs

def _normalize_height(v: str) -> str:
    """'1 m. 66 cent.' / '1m66' / '1 metre 66' -> '1m 66cm'."""
    if not v:
        return v
    m = re.search(r"(\d)\s*m[\. ètre]*\s*(\d{1,3})\s*(?:c|cent|cm|mm)?", v, re.I)
    if m:
        return f"{m.group(1)}m {int(m.group(2))}cm"
    return v

def _normalize_day(v: str) -> str:
    """French ordinal day notation ('1er', '1ere', 'premier') -> plain '1'.
    Only the 1st is ever written as an ordinal in these registers; every
    other day is already a bare number. Leaves anything else untouched —
    a value that still doesn't parse as a number gets caught by
    _plausible()'s day-format check instead of being guessed at here."""
    if not v:
        return v
    if re.fullmatch(r"1\s*(?:er|ere|ère|re)\.?", v.strip(), re.I) or v.strip().lower() == "premier":
        return "1"
    return v

def _normalize_regiment(v: str) -> str:
    """CI convention: '{n}e Régt' ('1er Régt' for the 1st) — drop any branch/
    arm descriptor the page may also carry (e.g. "d'Infanterie", "de
    Dragons"), that belongs in Branch/Unit, not Regiment. Safety-net for
    when the model didn't already follow the prompt's keying rule; leaves
    the value untouched if no regiment number can be found at all."""
    if not v:
        return v
    m = re.search(r"(\d+)\s*(?:er|ère|re|e)?\b", v, re.I)
    if not m:
        return v
    num = m.group(1)
    suffix = "er" if num == "1" else "e"
    return f"{num}{suffix} Régt"

# On ledger pages, the printed item number that starts each line ("1°",
# "2°", "3° fils de...") is the FORM'S OWN layout numbering, not a real
# Prefix/Suffix value — the model still leaks it through sometimes despite
# the prompt instruction (e.g. "3e fils de", even with a stray HTML <sup>
# tag from trying to render the printed superscript). No legitimate
# Prefix/Suffix value looks like this, so it's safe to strip deterministically.
_ORDINAL_MARKER_RE = re.compile(
    r"^(?:\s*\d+\s*(?:er|ère|e|°|<sup>e</sup>)\.?\s*)+(?:fils|fille)?\s*(?:de|d['’])?", re.I)

def _strip_ordinal_marker(v: str) -> str:
    return "" if v and _ORDINAL_MARKER_RE.match(v) else v

_PREFIX_SUFFIX_FIELDS = ("Prefix", "Father Prefix", "Mother Prefix",
                         "Suffix", "Father Suffix", "Mother Suffix")

def correct_fields(raw_fields: dict, use_llm: bool = True,
                   model: str = DEFAULT_TEXT_MODEL, meta: dict | None = None) -> dict:
    """Return {corrected: {...}, diffs: [{field, original, corrected, note}]}.

    `meta` (per-field {confidence, ...}), when supplied, gates every
    correction layer below away from HIGH-confidence fields — see
    dict_correct()'s docstring for why."""
    corrected = dict(raw_fields)
    diffs = []

    # Layer 0: deterministic keying-rule normalisers
    hv = _normalize_height(corrected.get("Height", ""))
    if hv != corrected.get("Height", ""):
        diffs.append({"field": "Height", "original": corrected.get("Height", ""),
                      "corrected": hv, "note": "format:height"})
        corrected["Height"] = hv
    for dlabel in ("Birth Day", "Discharge Day", "Death Day"):
        dv = _normalize_day(corrected.get(dlabel, ""))
        if dv != corrected.get(dlabel, ""):
            diffs.append({"field": dlabel, "original": corrected.get(dlabel, ""),
                          "corrected": dv, "note": "format:day"})
            corrected[dlabel] = dv
    for pslabel in _PREFIX_SUFFIX_FIELDS:
        pv = _strip_ordinal_marker(corrected.get(pslabel, ""))
        if pv != corrected.get(pslabel, ""):
            diffs.append({"field": pslabel, "original": corrected.get(pslabel, ""),
                          "corrected": pv, "note": "format:ordinal-marker"})
            corrected[pslabel] = pv
    # Enlistment Departement: blank if it is the expected default (Aude/L'Aude/Tarn)
    ed = corrected.get("Enlistment Departement", "")
    if _norm(ed) in ("aude", "l'aude", "laude", "tarn"):
        if ed:
            diffs.append({"field": "Enlistment Departement", "original": ed,
                          "corrected": "", "note": "rule:default-dept"})
        corrected["Enlistment Departement"] = ""

    # Layer 1a: authoritative external dictionary spelling-snap (names,
    # places, ranks, units, occupations...) — see dictionaries.py
    dc_corrected, dc_diffs = dict_correct(corrected, meta=meta)
    corrected = dc_corrected
    diffs.extend(dc_diffs)

    # Layer 1b: closed-set snapping for every vocab-controlled field. Fields
    # already handled by the big dictionary above are skipped here to avoid
    # double-processing — this layer now only does the remaining work: the
    # hair/eye/month variant->CI-code mapping, plus city/state/prefix/suffix
    # as a fallback when the external dictionary directory isn't available.
    for label in F.FIELD_LABELS:
        vocab = F.LABEL_TO_VOCAB.get(label)
        if not vocab:
            continue
        if label in FIELD_TO_DICTKEY and D.available():
            continue
        newv, changed, note = _snap_vocab(corrected.get(label, ""), vocab)
        if changed:
            diffs.append({"field": label, "original": raw_fields.get(label, ""),
                          "corrected": newv, "note": note})
            corrected[label] = newv

    # Layer 2: grounded LLM spelling pass on free-text names/places. Skips
    # HIGH-confidence fields too — an LLM asked to "fix spelling" can still
    # over-eagerly rewrite a value that's already correct, and a plausible
    # rewrite can slip past the grounding check if it still partially
    # matches the transcript, so the safest guard is to never hand the LLM
    # a field both vision readers already agreed on.
    if use_llm:
        payload = {}
        for lb in LLM_FIELDS:
            if meta and meta.get(lb, {}).get("confidence") == "HIGH":
                continue
            v = corrected.get(lb, "").strip()
            if not v:
                continue
            cell = {"value": v}
            cands = _dict_candidates(v, lb)
            if cands:
                cell["candidates"] = cands
            payload[lb] = cell
        if payload:
            try:
                r = ollama.chat(
                    model=model,
                    messages=[{"role": "user",
                               "content": LLM_CORRECT_PROMPT.format(
                                   payload=json.dumps(payload, ensure_ascii=False))}],
                    format="json",
                    options={"temperature": 0, "num_ctx": 4096},
                )
                fixed = _parse_json_obj(r["message"]["content"])
                for lb, val in fixed.items():
                    lb2 = _KEY_LOOKUP.get(str(lb), _KEY_LOOKUP.get(str(lb).lower().strip(), lb))
                    if lb2 in corrected and isinstance(val, str):
                        val = _clean_val(val)
                        if val and val != corrected[lb2]:
                            diffs.append({"field": lb2, "original": corrected[lb2],
                                          "corrected": val, "note": "llm"})
                            corrected[lb2] = val
            except Exception as e:
                diffs.append({"field": "_error", "original": "", "corrected": "",
                              "note": f"llm-correction-failed: {e}"})

    # Regiment format runs LAST, after the dictionary snap and LLM pass —
    # both of those can reintroduce the dictionary's full-word "Régiment"
    # spelling, so the "{n}e Régt" keying format has to be enforced as the
    # final step rather than in Layer 0, or it just gets undone downstream.
    rv = _normalize_regiment(corrected.get("Regiment", ""))
    if rv != corrected.get("Regiment", ""):
        diffs.append({"field": "Regiment", "original": corrected.get("Regiment", ""),
                      "corrected": rv, "note": "format:regiment"})
        corrected["Regiment"] = rv

    return {"corrected": corrected, "diffs": diffs}

# ---------------------------------------------------------------------------
# Full per-image pipeline — multi-stage, consensus-based, confidence-scored.
# Stages (each reported via `progress(stage, pct)` for the live UI):
#   convert -> transcribe -> vision-extract -> text-extract -> reconcile
#   -> correct -> done
# mode: "accurate" runs both readers + consensus; "fast" skips the 2nd vision call.
# ---------------------------------------------------------------------------
# Two phases, run by two separate buttons:
STAGES_OCR = ["Convert", "Deskew", "Enhance", "Transcribe", "Vision extract",
              "Text extract", "Reconcile", "Done"]
STAGES_POST = ["Correct", "Doctype", "Verify", "Qwen Verify", "Done"]
STAGES = STAGES_OCR + STAGES_POST[:-1]   # full-pipeline label list

# ---------------------------------------------------------------------------
# PHASE 1 — OCR / extraction only. Produces the raw template values (no
# correction, no anti-hallucination). This is what fills the OCR Excel.
# ---------------------------------------------------------------------------
def process_ocr(name: str, vision_model: str = DEFAULT_VISION_MODEL,
                text_model: str = DEFAULT_TEXT_MODEL, mode: str = "accurate",
                progress=None) -> dict:
    def emit(stage, pct):
        if progress:
            try: progress(stage, pct)
            except Exception: pass

    emit("Convert", 3); convert_one(name)
    emit("Deskew", 8); deskew = deskew_image(name)
    emit("Enhance", 14); enhance = enhance_contrast(name)
    emit("Transcribe", 22)
    ocr = run_ocr(name, model=vision_model)

    if mode == "accurate":
        emit("Vision extract", 45)
        vmeta = vision_extract(name, model=vision_model)
    else:
        vmeta = {lb: {"value": "", "confidence": "", "evidence": ""} for lb in F.FIELD_LABELS}

    emit("Text extract", 70)
    tfields = extract_fields(ocr["raw_text"], model=text_model)

    emit("Reconcile", 85)
    if mode == "accurate":
        values, meta = reconcile(vmeta, tfields)
    else:
        values = tfields
        meta = {lb: {"confidence": ("MEDIUM" if tfields.get(lb, "").strip() else ""),
                     "evidence": "", "source": "text", "alt": ""} for lb in F.FIELD_LABELS}

    # doctype from structure is part of raw OCR (reliable, deterministic)
    if is_row_split(name):
        values["Event Type"] = "Military"
        meta.setdefault("Event Type", {}).update(
            {"confidence": "HIGH", "source": "row-split", "alt": "", "evidence": "one row of a multi-person ledger page"})
    else:
        dt, dconf = classify_doctype(ocr["raw_text"])
        if dt:
            values["Event Type"] = dt
            meta.setdefault("Event Type", {}).update(
                {"confidence": dconf, "source": "structure", "alt": "", "evidence": "page section headers"})

    result = {
        "name": name, "png": png_path(name).name,
        "src": str(discover_images().get(name, "")),
        "deskew": deskew, "enhance": enhance,
        "raw_text": ocr["raw_text"], "ocr_model": ocr["ocr_model"],
        "text_model": text_model, "mode": mode,
        "phase": "ocr",
        "fields_raw": dict(values),
        "fields_corrected": dict(values),   # == raw until post-processing runs
        "meta": meta, "diffs": [], "audit": {}, "stats": confidence_stats(meta, values),
        "postprocessed": False,
    }
    store = load_results(); store[name] = result; save_results(store)
    emit("Done", 100)
    return result

# ---------------------------------------------------------------------------
# PHASE 2 — Post-processing on an already-OCR'd record: grounded spelling
# correction + anti-hallucination grounding verification. Run by button 2.
# ---------------------------------------------------------------------------
def postprocess(name: str, text_model: str = DEFAULT_TEXT_MODEL,
                use_llm_correction: bool = True, strict: bool = False,
                use_qwen: bool = True, progress=None) -> dict:
    def emit(stage, pct):
        if progress:
            try: progress(stage, pct)
            except Exception: pass

    store = load_results()
    rec = store.get(name)
    if not rec:
        raise ValueError(f"{name} has not been OCR'd yet")
    values = dict(rec.get("fields_raw", {}))
    meta = rec.get("meta", {})

    emit("Correct", 20)
    corr = correct_fields(values, use_llm=use_llm_correction, model=text_model, meta=meta)

    emit("Doctype", 40)
    if is_row_split(name):
        prev = corr["corrected"].get("Event Type", "")
        if prev != "Military":
            corr["diffs"].append({"field": "Event Type", "original": prev,
                                  "corrected": "Military", "note": "row-split:forced"})
        corr["corrected"]["Event Type"] = "Military"
        meta.setdefault("Event Type", {}).update(
            {"confidence": "HIGH", "source": "row-split", "alt": "", "evidence": "one row of a multi-person ledger page"})
    else:
        dt, dconf = classify_doctype(rec.get("raw_text", ""))
        if dt:
            prev = corr["corrected"].get("Event Type", "")
            if prev != dt:
                corr["diffs"].append({"field": "Event Type", "original": prev,
                                      "corrected": dt, "note": "doctype:markers"})
            corr["corrected"]["Event Type"] = dt
            meta.setdefault("Event Type", {}).update(
                {"confidence": dconf, "source": "structure", "alt": "", "evidence": "page section headers"})

    emit("Verify", 60)
    audit = verify_grounding(corr["corrected"], meta, rec.get("raw_text", ""), strict=strict)

    qwen = rec.get("qwen", {"raw_text": "", "fields": {}, "upgraded": []})
    # Skip the "independent" Qwen re-read when Qwen2.5-VL was ALSO the
    # primary vision reader for this image — re-running the identical
    # deterministic (temperature=0) model against itself can't corroborate
    # anything a second opinion wouldn't already have caught.
    if use_qwen and rec.get("ocr_model") == QWEN_VISION_MODEL:
        emit("Qwen Verify", 80)
        qwen = {"raw_text": "", "fields": {}, "upgraded": [], "model": QWEN_VISION_MODEL,
                "skipped": "primary vision reader was already Qwen2.5-VL"}
    elif use_qwen:
        emit("Qwen Verify", 80)
        qwen = qwen_finalize(name, corr["corrected"], meta)

    rec.update({
        "phase": "postprocessed", "postprocessed": True, "strict": strict,
        "fields_corrected": corr["corrected"], "meta": meta,
        "diffs": corr["diffs"], "audit": audit, "qwen": qwen,
        "stats": confidence_stats(meta, corr["corrected"]),
    })
    store[name] = rec; save_results(store)
    emit("Done", 100)
    return rec

# ---------------------------------------------------------------------------
# Convenience: full pipeline (OCR + post-process) for the single-image button.
# ---------------------------------------------------------------------------
def process_image(name: str, vision_model: str = DEFAULT_VISION_MODEL,
                  text_model: str = DEFAULT_TEXT_MODEL, use_llm_correction: bool = True,
                  mode: str = "accurate", strict: bool = False, use_qwen: bool = True,
                  progress=None) -> dict:
    process_ocr(name, vision_model=vision_model, text_model=text_model, mode=mode, progress=progress)
    return postprocess(name, text_model=text_model, use_llm_correction=use_llm_correction,
                       strict=strict, use_qwen=use_qwen, progress=progress)
