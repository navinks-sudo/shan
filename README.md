# Military Registers — ACTIGEN OCR Workbench

Local web app to convert the Aude/Gironde J2K register scans to PNG, run **Gemma (Ollama)**
vision OCR, extract fields into the **CI `Military` template**, apply **grounded post-correction /
spell-fix**, and export to Excel — with **consensus-based confidence scoring**, a **one-click
Process-All batch**, and a **live progress dashboard**.

## What makes the extraction accurate (accuracy design)

Each image runs a multi-stage, dual-reader pipeline:

1. **Transcribe** — vision OCR, verbatim (anti-hallucination prompt; illegible → `[?]`).
2. **Vision extract** — a *second* vision pass reads fields **straight off the image**, using
   the printed column headers/labels, and returns each field with an **evidence snippet** and a
   **HIGH/MEDIUM/LOW confidence**. (Layout-aware — catches what a flat transcript loses.)
3. **Text extract** — an independent reader pulls the same fields from the transcription.
4. **Reconcile (consensus)** — the two readers are compared per field:
   - **agree → HIGH** · **only one has it → MEDIUM** · **disagree → LOW** (kept + the
     alternative shown for review). Conflicts are counted so you review the risky fields first.
5. **Correct** — deterministic keying rules (height format, default-département blank) +
   closed-set fuzzy snapping (months, colours, départements) + a **grounded LLM spelling pass**
   that is given known Aude/Gironde place candidates so it snaps to real communes instead of
   inventing. Every change is shown as a diff.

`Accurate` mode runs all of the above; `Fast` mode skips the 2nd vision pass (single reader).

The UI colours every field by confidence (green/amber/red) with its evidence, so a human
verifies the risky values — AI output stays a draft.

### Anti-hallucination (grounding verification)

Even 12b confabulates on hard handwriting — on some pages it invents an entire alternate
person. A dedicated **Verify** stage catches this deterministically:

- **Corroboration check** — every extracted name/place/number is looked up in the INDEPENDENT
  transcription. A value the transcript never saw is flagged **⚠ "not in transcript"**
  (`uncorroborated`) and forced to LOW confidence. (Real example: page 0018 the model read
  `Boutler / Bélesta / Ariège`; the transcript said `Orias / Bayeux / Calvados` — all the
  fabricated fields were flagged, the real ones passed.)
- **Plausibility checks** — impossible day (>31), year out of 1750–1960, implausible height.
- **Cross-field sanity** — discharge/death/classe year before birth year → flagged.
- **Strict anti-hallucination toggle** — when on, any `uncorroborated` value is **blanked**
  instead of kept, so the template only contains transcript-backed values.

The scoreboard shows a **⚠ FLAG** count; flagged fields get a red border and a reason chip.
Review those against the image (and the Raw OCR tab) — that's where hallucinations concentrate.

## Run

1. Make sure **Ollama is running** and has the models: `gemma3:27b` (default) — plus optionally
   `gemma3:12b` / `gemma3:4b` — and `qwen2.5:7b`.
   Pull if needed: `ollama pull gemma3:27b` and `ollama pull qwen2.5:7b`.
2. Double-click **`run.bat`** (or run `python -m uvicorn app:app --port 8000` in this folder).
3. Open **http://127.0.0.1:8000** in a browser.

### The two-button workflow

1. **Point it at your images.** Paste a folder path into the **📁 Images folder** bar and click
   **Load folder** (leave blank to auto-scan the project's `Images/` folders). Supports
   `.j2k .jp2 .png .jpg .tif …`, scanned recursively.
2. **① Process OCR → Excel** — OCRs every image (default **gemma3:27b**) into the CI template.
   Live dashboard shows progress. When done, **⬇ Excel** downloads the raw-OCR template.
3. **② Post-process** — runs spelling correction + anti-hallucination grounding on the OCR'd
   records. Tick **Strict anti-hallucination** first to blank any value not backed by the
   transcript. **⬇ Excel** again for the cleaned template.

**Process scope** (All / Unprocessed only / Military only / Coverpage only) applies to both
buttons. Both phases are independent — you can re-run post-processing with different settings
without re-doing the slow OCR.

## Workflow in the UI

* **⚡ Process All Folders** — scans **every `*/Images/` folder** under the project and runs the
  whole batch. A live dashboard shows a progress ring, current file + stage, **ETA**, the 7
  pipeline stages lighting up, and running HIGH/MED/LOW tallies. **Stop** halts after the
  current image. (Pick model + `Accurate`/`Fast` first; `gemma3:4b` is best for a fast full run.)
* **Per image:** click one in the list, press **▶ Run OCR**. Fields appear colour-coded by
  confidence with evidence + any conflict alternative; the **Raw OCR** tab shows the transcript.
* Edit any field inline → **↻ Re-fix** (re-runs correction) → **💾 Save**.
* **⬇ Excel** downloads `Military_OCR_Output.xlsx` — one row per image, 45 CI `Military` field
  columns, plus a `Field Map` sheet (label → Ancestry field name).

Models: `gemma3:12b` default; `27b` = most accurate/slowest; `4b` = fastest (best for full-batch
runs, weakest on hard handwriting — the confidence flags make its errors visible).

## How accuracy / anti-hallucination is handled

* These are 1800s–1900s **handwritten** French registers — small vision models confabulate.
  Every prompt forbids guessing; illegible text becomes `[?]` and is stripped from fields.
* Non-record images (targets, covers, blank pages) yield an empty record — nothing invented.
* The image sits beside the OCR so a human verifies every value (AI output is a draft).
* Bigger model = better: use `gemma3:27b` for the final pass on hard pages.

## Post-correction (spell correction)

Two layers, both producing a visible diff:
1. **Closed-set fuzzy snapping** (`rapidfuzz`) for controlled vocab: months→3-letter codes,
   hair/eye colours→CI codes, départements, prefixes/suffixes, Y/N flags. See `fields.py`.
2. **LLM spelling pass** (`qwen2.5:7b`) on free-text names/places — fixes OCR letter errors
   while refusing to invent or change already-plausible values.

To improve place-name correction, extend the `CITIES` / `STATES` lists in `fields.py` with the
project's official Aude/Gironde commune dictionary.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend + routes |
| `pipeline.py` | J2K→PNG, vision OCR, field extraction, post-correction |
| `fields.py` | 45-field `Military` schema + controlled vocabularies |
| `export_xlsx.py` | Excel export in the CI template |
| `static/index.html` | 3-pane UI |
| `output/png/` | converted PNGs |
| `output/results.json` | saved OCR + fields per image |
| `output/Military_OCR_Output.xlsx` | exported workbook |

Source images: `../d_63392.IDX.001_I4417928 (1)/d_63392.IDX.001_I4417928/Images/*.j2k`
