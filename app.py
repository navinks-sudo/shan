"""
FastAPI backend for the Aude/Gironde Military Registers OCR workbench.

Endpoints
  GET  /                       -> UI (static/index.html)
  GET  /api/images             -> list images + per-image status
  GET  /api/models             -> available ollama models
  POST /api/convert            -> convert all J2K -> PNG
  GET  /png/{stem}.png         -> serve converted PNG
  GET  /api/result/{name}      -> stored result for one image
  POST /api/ocr                -> {name, vision_model, text_model, llm_correct} full pipeline
  POST /api/reocr              -> re-run OCR only (keeps model choice)
  POST /api/correct            -> re-run post-correction on stored/edited fields
  POST /api/save               -> persist user-edited corrected fields
  GET  /api/export.xlsx        -> download Excel in CI 'Military' template
"""
import json, re, time, threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama

import pipeline as P
import export_xlsx
import fields as F
import dictionaries as D

app = FastAPI(title="Military Registers OCR Workbench")
STATIC = Path(__file__).resolve().parent / "static"


@app.on_event("startup")
def _warm_dictionaries():
    """Pre-parse + cache the external CI keying dictionaries (the ~1.74M-row
    surname list alone takes ~20s from raw .xlsx) in a background thread so
    the first real post-processing request isn't the one paying that cost."""
    threading.Thread(target=D.warm_all, daemon=True).start()


@app.get("/api/dictionaries")
def api_dictionaries():
    return {"available": D.available(), "dir": str(D.DICT_DIR), "counts": D.stats()}

# ---------------------------------------------------------------------------
# Batch job state (single in-memory job; polled by the UI for live progress)
# ---------------------------------------------------------------------------
JOB = {
    "running": False, "stop": False,
    "total": 0, "done": 0, "pct": 0.0,
    "current": None, "stage": None, "stage_pct": 0,
    "started": None, "finished": None, "eta": None,
    "items": [],           # completed: {name, ok, event, stats, error}
    "totals": {"filled": 0, "high": 0, "medium": 0, "low": 0, "conflicts": 0},
    "params": {},
}
_LOCK = threading.Lock()

def _run_batch(names, vmodel, tmodel, llm, mode, only_new, strict=False, task="full", use_qwen=True):
    store = P.load_results()
    todo = [n for n in names if not (only_new and n in store)]
    stages = {"ocr": P.STAGES_OCR, "postprocess": P.STAGES_POST}.get(task, P.STAGES + ["Done"])
    with _LOCK:
        JOB.update(running=True, stop=False, total=len(todo), done=0, pct=0.0,
                   items=[], started=time.time(), finished=None, eta=None, task=task, stages=stages,
                   totals={"filled": 0, "high": 0, "medium": 0, "low": 0, "conflicts": 0, "flagged": 0},
                   params={"vision": vmodel, "text": tmodel, "mode": mode, "llm": llm, "strict": strict,
                           "task": task, "qwen": use_qwen})
    for i, name in enumerate(todo):
        if JOB["stop"]:
            break
        def cb(stage, pct, _i=i, _n=name):
            elapsed = time.time() - JOB["started"]
            frac = (_i + pct / 100.0) / max(1, len(todo))
            eta = (elapsed / frac - elapsed) if frac > 0.02 else None
            with _LOCK:
                JOB.update(current=_n, stage=stage, stage_pct=pct,
                           pct=round(frac * 100, 1), eta=round(eta) if eta else None)
        try:
            if task == "ocr":
                rec = P.process_ocr(name, vision_model=vmodel, text_model=tmodel, mode=mode, progress=cb)
            elif task == "postprocess":
                rec = P.postprocess(name, text_model=tmodel, use_llm_correction=llm,
                                    strict=strict, use_qwen=use_qwen, progress=cb)
            else:
                rec = P.process_image(name, vision_model=vmodel, text_model=tmodel,
                                      use_llm_correction=llm, mode=mode, strict=strict,
                                      use_qwen=use_qwen, progress=cb)
            st = rec.get("stats", {})
            with _LOCK:
                for k in JOB["totals"]:
                    JOB["totals"][k] += st.get(k, 0)
                JOB["items"].append({"name": name, "ok": True,
                                     "event": rec["fields_corrected"].get("Event Type", ""),
                                     "stats": st})
        except Exception as e:
            with _LOCK:
                JOB["items"].append({"name": name, "ok": False, "error": str(e)})
        with _LOCK:
            JOB["done"] = i + 1
    with _LOCK:
        JOB.update(running=False, current=None, stage="Complete",
                   stage_pct=100, pct=100.0, finished=time.time(), eta=0)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/fields")
def api_fields():
    return {"labels": F.FIELD_LABELS,
            "map": [{"label": l, "key": k, "vocab": v} for l, k, v in F.MILITARY_FIELDS]}


@app.get("/api/models")
def api_models():
    try:
        data = ollama.list()
        names = [m.get("model") or m.get("name") for m in data.get("models", [])]
        names = [n for n in names if n]
    except Exception:
        names = []
    vision = [n for n in names if n.startswith("gemma3") or "vl" in n.lower()] or names
    return {"all": names, "vision": vision,
            "default_vision": P.DEFAULT_VISION_MODEL, "default_text": P.DEFAULT_TEXT_MODEL,
            "default_qwen_vision": P.QWEN_VISION_MODEL,
            "qwen_available": any(n.split(":")[0] == P.QWEN_VISION_MODEL.split(":")[0] for n in names)}


_ROW_RE = re.compile(r"^(?P<parent>.+)_row(?P<n>\d+)\.png$")

@app.get("/api/images")
def api_images():
    results = P.load_results()
    names = P.list_source_images()
    # how many rows each parent was split into, so a row can show "2 of 4"
    row_counts = {}
    for n in names:
        m = _ROW_RE.match(n)
        if m:
            row_counts[m.group("parent")] = row_counts.get(m.group("parent"), 0) + 1

    out = []
    for name in names:
        pngp = P.png_path(name)
        rec = results.get(name)
        m = _ROW_RE.match(name)
        row_info = None
        if m:
            row_info = {"parent": m.group("parent"), "index": int(m.group("n")),
                        "total": row_counts.get(m.group("parent"), 1)}
        # row-crop filenames never carry the parent's original extension
        # (.j2k/.png/...), so compare stems, not raw names
        has_splits = Path(name).stem in row_counts
        out.append({
            "name": name,
            "converted": pngp.exists(),
            "processed": rec is not None,
            "event_type": (rec or {}).get("fields_corrected", {}).get("Event Type", "") if rec else "",
            "n_diffs": len((rec or {}).get("diffs", [])) if rec else 0,
            "stats": (rec or {}).get("stats", {}) if rec else {},
            "row_split": row_info,
            "has_row_splits": has_splits,
        })
    folders = P.source_folders()
    return {"images": out, "count": len(out),
            "folders": folders, "exists": len(folders) > 0,
            "src_dir": str(P._SRC_DIR) if P._SRC_DIR else "(auto: project Images folders)"}


@app.post("/api/convert")
def api_convert():
    converted = [P.convert_one(n) for n in P.list_source_images()]
    return {"converted": len(converted), "items": converted}


@app.get("/png/{stem}.png")
def api_png(stem: str):
    p = P.PNG_DIR / f"{stem}.png"
    if not p.exists():
        # try to convert from source
        src = P.SRC_IMAGES / f"{stem}.j2k"
        if src.exists():
            P.convert_one(src.name)
    if not p.exists():
        raise HTTPException(404, "PNG not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/result/{name}")
def api_result(name: str):
    rec = P.load_results().get(name)
    if not rec:
        raise HTTPException(404, "not processed yet")
    return rec


class SplitRowsReq(BaseModel):
    name: str
    force: bool = False


@app.post("/api/split-rows")
def api_split_rows(req: SplitRowsReq):
    if req.name not in P.list_source_images():
        raise HTTPException(404, "unknown image")
    try:
        rows = P.split_into_rows(req.name, force=req.force)
    except Exception as e:
        raise HTTPException(500, f"row split failed: {e}")
    return {"parent": req.name, "rows": rows}


class OcrReq(BaseModel):
    name: str
    vision_model: str | None = None
    text_model: str | None = None
    llm_correct: bool = True
    mode: str = "accurate"
    strict: bool = False
    use_qwen: bool = True


@app.post("/api/ocr")
def api_ocr(req: OcrReq):
    if req.name not in P.list_source_images():
        raise HTTPException(404, "unknown image")
    try:
        return P.process_image(
            req.name,
            vision_model=req.vision_model or P.DEFAULT_VISION_MODEL,
            text_model=req.text_model or P.DEFAULT_TEXT_MODEL,
            use_llm_correction=req.llm_correct,
            mode=req.mode,
            strict=req.strict,
            use_qwen=req.use_qwen,
        )
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")


# ---------------------------------------------------------------------------
# Batch processing (single "Process All" job) + live progress
# ---------------------------------------------------------------------------
class BatchReq(BaseModel):
    vision_model: str | None = None
    text_model: str | None = None
    llm_correct: bool = True
    mode: str = "accurate"
    strict: bool = False
    only_new: bool = False          # skip already-processed images
    names: list[str] | None = None  # explicit subset of images to process
    doctype: str | None = None      # or filter existing results by Event Type (e.g. "Military")
    task: str = "full"              # "ocr" | "postprocess" | "full"
    use_qwen: bool = True           # run the Qwen2.5-VL final verification pass


@app.post("/api/batch/start")
def api_batch_start(req: BatchReq):
    if JOB["running"]:
        raise HTTPException(409, "a batch is already running")
    all_names = P.list_source_images()
    store = P.load_results()
    if req.names:
        names = [n for n in all_names if n in set(req.names)]
    elif req.doctype:
        names = [n for n in all_names
                 if store.get(n, {}).get("fields_corrected", {}).get("Event Type") == req.doctype]
    else:
        names = all_names
    # post-processing only applies to images that have already been OCR'd
    if req.task == "postprocess":
        names = [n for n in names if n in store]

    # "Unprocessed only" (only_new) is task-aware, so a shutdown mid-run resumes
    # cleanly instead of redoing work:
    #   ocr/full     -> skip images that have no result yet
    #   postprocess  -> skip images already post-processed (clean resume)
    if req.only_new:
        if req.task == "postprocess":
            names = [n for n in names if not store.get(n, {}).get("postprocessed")]
        else:
            names = [n for n in names if n not in store]

    if not names:
        raise HTTPException(404, "no matching images (for post-processing, run OCR first)")
    # only_new already applied above -> pass False so _run_batch does not re-filter
    t = threading.Thread(target=_run_batch, args=(
        names, req.vision_model or P.DEFAULT_VISION_MODEL,
        req.text_model or P.DEFAULT_TEXT_MODEL, req.llm_correct, req.mode,
        False, req.strict, req.task, req.use_qwen),
        daemon=True)
    t.start()
    return {"started": True, "count": len(names), "task": req.task}


class DirReq(BaseModel):
    path: str


@app.post("/api/set-dir")
def api_set_dir(req: DirReq):
    try:
        info = P.set_source_dir(req.path)
    except Exception as e:
        raise HTTPException(400, str(e))
    return info


@app.post("/api/batch/stop")
def api_batch_stop():
    JOB["stop"] = True
    return {"stopping": True}


@app.get("/api/batch/status")
def api_batch_status():
    with _LOCK:
        j = dict(JOB)
    j.setdefault("stages", P.STAGES)
    return j


class CorrectReq(BaseModel):
    name: str
    fields: dict
    llm_correct: bool = True
    text_model: str | None = None


@app.post("/api/correct")
def api_correct(req: CorrectReq):
    corr = P.correct_fields(req.fields, use_llm=req.llm_correct,
                            model=req.text_model or P.DEFAULT_TEXT_MODEL)
    store = P.load_results()
    if req.name in store:
        store[req.name]["fields_corrected"] = corr["corrected"]
        store[req.name]["diffs"] = corr["diffs"]
        P.save_results(store)
    return corr


class SaveReq(BaseModel):
    name: str
    fields_corrected: dict


@app.post("/api/save")
def api_save(req: SaveReq):
    store = P.load_results()
    if req.name not in store:
        raise HTTPException(404, "not processed yet")
    store[req.name]["fields_corrected"] = req.fields_corrected
    P.save_results(store)
    return {"ok": True}


@app.get("/api/export.xlsx")
def api_export(corrected: bool = True):
    out = P.OUT / "Military_OCR_Output.xlsx"
    export_xlsx.build_workbook(str(out), use_corrected=corrected)
    return FileResponse(out, filename="Military_OCR_Output.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
