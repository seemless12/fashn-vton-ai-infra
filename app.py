"""
app.py
------
FastAPI server for FASHN VTON v1.5 with adaptive masked/maskless routing.

Run directly:
    uvicorn app:app --host 0.0.0.0 --port 8000

Or via run_colab.py, which also spins up a Cloudflare Tunnel so you get a
public URL to hit from anywhere (browser, Postman, another notebook, a
frontend, etc).
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import json
import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from engine_optimized import TryOnRequest, VTONEngine
import license_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vton_server")


class _QuietPollingFilter(logging.Filter):
    """
    The client-side poll loop hits GET /api/try-on/status/{job_id} every
    few seconds while a job runs -- useful for the client, pure noise in
    the server log (it can bury the one line you actually care about:
    the Sampling progress bar). This drops only the *successful* status
    polls; 404s, other endpoints, and everything else still logs normally.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("/api/try-on/status/" in msg and '" 200' in msg)


logging.getLogger("uvicorn.access").addFilter(_QuietPollingFilter())

WEIGHTS_DIR = os.environ.get("FASHN_WEIGHTS_DIR", "./weights")

app = FastAPI(title="FASHN VTON API (Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_engine: Optional[VTONEngine] = None

# ----------------------------------------------------------------------
# Async job queue
# ----------------------------------------------------------------------
# Inference can easily take longer than a tunnel/proxy's request timeout
# (Cloudflare quick tunnels, in particular, are not built for long-held
# connections). Instead of blocking the HTTP request for the whole
# generation, /api/try-on enqueues a job and returns immediately; the
# caller polls /api/try-on/status/{job_id} until the image is ready.
#
# max_workers=1 serializes generations -- appropriate for a single GPU.
# Bump this only if you know your GPU/VRAM can handle concurrent runs.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict = {}
_jobs_lock = threading.Lock()

JOB_TTL_SECONDS = 30 * 60  # sweep out stale completed/failed jobs after this

# Redis Configuration (Optional, for decoupled worker setup)
REDIS_MODE = os.environ.get("REDIS_MODE", "0") == "1"
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
QUEUE_NAME = "fashn_jobs_queue"

redis_client = None
if REDIS_MODE:
    import redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    logger.info("Running in REDIS_MODE. AI Engine will NOT be loaded here.")


def _sweep_stale_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        stale = [
            jid
            for jid, job in _jobs.items()
            if job.get("status") in ("completed", "failed")
            and job.get("finished_at", 0) < cutoff
        ]
        for jid in stale:
            _jobs.pop(jid, None)


@app.on_event("startup")
def load_model():
    if REDIS_MODE:
        return
        
    global _engine
    logger.info("Loading VTON engine (this can take a minute on first boot)...")
    _engine = VTONEngine(weights_dir=WEIGHTS_DIR)
    logger.info("Model loaded, server ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _engine is not None}


@app.get("/api/credits/check")
def check_credits(device_id: str, license_key: Optional[str] = None):
    """
    Returns remaining quota and plan info for the given device or license key.
    """
    return license_manager.check_credits(device_id, license_key)


@app.post("/api/license/activate")
async def activate_license(
    device_id: str = Form(...),
    license_key: str = Form(...),
):
    """
    Validates and activates a PKR 500 license key for the user.
    """
    status = license_manager.check_credits(device_id, license_key)
    if not status.get("valid"):
        raise HTTPException(status_code=400, detail=status.get("error", "Invalid or expired license key."))
    return status


@app.post("/tryon")
async def tryon(
    # NOTE: this endpoint blocks for the full duration of generation. If
    # you're calling through a tunnel/proxy with a request timeout shorter
    # than your inference time, use /api/try-on (submit) + GET
    # /api/try-on/status/{job_id} (poll) instead -- see README.md.

    person_image: UploadFile = File(..., description="Photo of the person"),
    garment_image: UploadFile = File(..., description="Garment photo (model-worn or flat-lay)"),
    category: Optional[str] = Form(
        None, description='"tops" | "bottoms" | "one-pieces" — omit to auto-detect'
    ),
    garment_photo_type: str = Form("model", description='"model" or "flat-lay"'),
    mode: str = Form(
        "auto",
        description='"auto" (recommended) | "masked" | "maskless" | "both" (returns whichever scores better)',
    ),
    num_samples: int = Form(1),
    steps: int = Form(15, ge=10, le=50, description="10=fast, 15=balanced, 25=high, 30=ultra"),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    autocrop: bool = Form(True, description="Auto tight-crop flat-lay garment photos"),
    refine: bool = Form(False, description="Optional free local detail-refine pass on the garment region (adds real time -- see README)"),
    refine_strength: float = Form(0.25, description="Denoising strength for the refine pass (low = subtle, preserves shape)"),
    refine_guidance_scale: float = Form(3.0, description="Refine pass CFG scale -- keep low (~3.0) to avoid over-sharpening/oversaturation"),
    refine_steps: int = Form(25, description="Inference steps for the refine pass"),
    return_debug: bool = Form(False, description="Include routing decisions in headers"),
):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly.")

    try:
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_pil = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded images: {e}")

    req = TryOnRequest(
        person_image=person_pil,
        garment_image=garment_pil,
        category=category,
        garment_photo_type=garment_photo_type,
        mode=mode,
        num_samples=num_samples,
        num_timesteps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        autocrop=autocrop,
        refine=refine,
        refine_strength=refine_strength,
        refine_guidance_scale=refine_guidance_scale,
        refine_steps=refine_steps,
    )

    try:
        result = _engine.run(req)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    buf.seek(0)

    headers = {}
    if return_debug:
        headers.update(
            {
                "X-Category-Used": result.category_used,
                "X-Segmentation-Free": str(result.segmentation_free_used),
                "X-Extent-Ratio": f"{result.extent_ratio:.3f}",
                "X-Dominant-Class": result.dominant_class,
                "X-Candidates-Tried": str(result.candidates_tried),
                "X-Refined": str(result.refined),
            }
        )

    return StreamingResponse(buf, media_type="image/png", headers=headers)


def _run_job(job_id: str, req: TryOnRequest):
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["started_at"] = time.time()

    try:
        result = _engine.run(req)
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "status": "completed",
                    "image_bytes": buf.getvalue(),
                    "finished_at": time.time(),
                    "meta": {
                        "category_used": result.category_used,
                        "segmentation_free_used": result.segmentation_free_used,
                        "extent_ratio": result.extent_ratio,
                        "dominant_class": result.dominant_class,
                        "candidates_tried": result.candidates_tried,
                        "refined": result.refined,
                        "steps": req.num_timesteps,
                        "generation_time": round(time.time() - _jobs[job_id]["started_at"], 2),
                    },
                }
            )
        logger.info("Job %s completed", job_id)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(
                {"status": "failed", "error": str(e), "finished_at": time.time()}
            )


@app.post("/api/try-on")
async def submit_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: Optional[str] = Form(
        None, description='Free-form or canonical, e.g. "t-shirt", "kurta", "tops"'
    ),
    garment_photo_type: str = Form("model"),
    mode: str = Form("auto"),
    num_samples: int = Form(1),
    steps: int = Form(15, ge=10, le=50),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    autocrop: bool = Form(True),
    refine: bool = Form(False, description="Optional free local detail-refine pass (adds real time)"),
    refine_strength: float = Form(0.25),
    refine_guidance_scale: float = Form(3.0),
    refine_steps: int = Form(25),
    device_id: str = Form("default_dev"),
    license_key: Optional[str] = Form(None),
):
    """
    Enqueue a try-on job and return immediately with a job_id + poll_endpoint.
    This is the recommended entry point for anything behind a tunnel/proxy
    with request timeouts shorter than generation time -- pair it with
    GET /api/try-on/status/{job_id}.
    """
    # Enforce subscription / trial credit quota
    quota = license_manager.check_credits(device_id, license_key)
    if not quota.get("valid") or quota.get("credits_remaining", 0) <= 0:
        raise HTTPException(
            status_code=402,
            detail="Credit limit reached. Please upgrade to the PKR 500 pack (100 try-ons) to continue."
        )

    # Deduct 1 credit before queueing
    if not license_manager.deduct_credit(device_id, license_key):
        raise HTTPException(
            status_code=402,
            detail="Credit deduction failed. Quota may be exhausted."
        )

    if not REDIS_MODE and _engine is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly.")

    if not REDIS_MODE:
        _sweep_stale_jobs()

    try:
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded images: {e}")

    job_id = uuid.uuid4().hex

    if REDIS_MODE:
        # Push raw bytes as base64 to Redis, defer Image.open to worker
        payload = {
            "person_image_b64": base64.b64encode(person_bytes).decode("utf-8"),
            "garment_image_b64": base64.b64encode(garment_bytes).decode("utf-8"),
            "category": category,
            "garment_photo_type": garment_photo_type,
            "mode": mode,
            "num_samples": num_samples,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "autocrop": autocrop,
            "refine": refine,
            "refine_strength": refine_strength,
            "refine_guidance_scale": refine_guidance_scale,
            "refine_steps": refine_steps,
        }
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "pending",
            "created_at": str(time.time())
        })
        redis_client.rpush(QUEUE_NAME, json.dumps({
            "job_id": job_id,
            "payload": json.dumps(payload)
        }))
    else:
        try:
            person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
            garment_pil = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode images: {e}")

        req = TryOnRequest(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type=garment_photo_type,
            mode=mode,
            num_samples=1,
            num_timesteps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            autocrop=autocrop,
            refine=refine,
            refine_strength=refine_strength,
            refine_guidance_scale=refine_guidance_scale,
            refine_steps=refine_steps,
        )

        with _jobs_lock:
            _jobs[job_id] = {"status": "pending", "created_at": time.time()}
        _executor.submit(_run_job, job_id, req)

    return {
        "job_id": job_id,
        "status": "pending",
        "poll_endpoint": f"/api/try-on/status/{job_id}",
    }


@app.get("/api/try-on/status/{job_id}")
def poll_tryon(job_id: str):
    """
    Poll a job's status.
    """
    if REDIS_MODE:
        job_data = redis_client.hgetall(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found.")
            
        status = job_data.get(b"status", b"").decode("utf-8")
        if status == "completed":
            image_b64 = job_data.get(b"image_b64", b"").decode("utf-8")
            image_bytes = base64.b64decode(image_b64)
            meta_str = job_data.get(b"meta", b"{}").decode("utf-8")
            meta = json.loads(meta_str)
            
            headers = {
                "X-Category-Used": str(meta.get("category_used", "")),
                "X-Segmentation-Free": str(meta.get("segmentation_free_used", "")),
                "X-Extent-Ratio": f"{meta.get('extent_ratio', 0):.3f}",
                "X-Dominant-Class": str(meta.get("dominant_class", "")),
                "X-Candidates-Tried": str(meta.get("candidates_tried", "")),
                "X-Refined": str(meta.get("refined", "")),
            }
            
            # Save image to static folder
            os.makedirs("static", exist_ok=True)
            image_path = f"static/{job_id}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            redis_client.delete(f"job:{job_id}")
            return JSONResponse({
                "status": "completed",
                "steps": meta.get("steps", 15),
                "generation_time": meta.get("generation_time", 0.0),
                "image_url": f"/static/{job_id}.png"
            })
            
        if status == "failed":
            return JSONResponse({"status": "failed", "error": job_data.get(b"error", b"").decode("utf-8")})
            
        return JSONResponse({"status": status})
    else:
        with _jobs_lock:
            job = _jobs.get(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found (expired or invalid job_id).")

        status = job["status"]

        if status == "completed":
            image_bytes = job["image_bytes"]
            meta = job.get("meta", {})
            headers = {
                "X-Category-Used": str(meta.get("category_used", "")),
                "X-Segmentation-Free": str(meta.get("segmentation_free_used", "")),
                "X-Extent-Ratio": f"{meta.get('extent_ratio', 0):.3f}",
                "X-Dominant-Class": str(meta.get("dominant_class", "")),
                "X-Candidates-Tried": str(meta.get("candidates_tried", "")),
                "X-Refined": str(meta.get("refined", "")),
            }
            with _jobs_lock:
                _jobs.pop(job_id, None)  # one-shot fetch, free the memory
                
            # Save image to static folder
            os.makedirs("static", exist_ok=True)
            image_path = f"static/{job_id}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            return JSONResponse({
                "status": "completed",
                "steps": meta.get("steps", 15),
                "generation_time": meta.get("generation_time", 0.0),
                "image_url": f"/static/{job_id}.png"
            })

        if status == "failed":
            return JSONResponse({"status": "failed", "error": job.get("error", "unknown error")})

        return JSONResponse({"status": status})


@app.post("/tryon/debug")
async def tryon_debug(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: Optional[str] = Form(None),
    garment_photo_type: str = Form("model"),
    mode: str = Form("auto"),
    num_samples: int = Form(1),
    steps: int = Form(15, ge=10, le=50),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    autocrop: bool = Form(True),
    refine: bool = Form(False),
    refine_strength: float = Form(0.25),
    refine_guidance_scale: float = Form(3.0),
    refine_steps: int = Form(25),
):
    """Same as /tryon but returns JSON with the routing decision + a base64 image,
    handy while you're tuning LOOSE_EXTENT_THRESHOLD in engine.py."""
    import base64

    if _engine is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly.")

    person_pil = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
    garment_pil = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")

    req = TryOnRequest(
        person_image=person_pil,
        garment_image=garment_pil,
        category=category,
        garment_photo_type=garment_photo_type,
        mode=mode,
        num_samples=num_samples,
        num_timesteps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        autocrop=autocrop,
        refine=refine,
        refine_strength=refine_strength,
        refine_guidance_scale=refine_guidance_scale,
        refine_steps=refine_steps,
    )
    result = _engine.run(req)

    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse(
        {
            "category_used": result.category_used,
            "segmentation_free_used": result.segmentation_free_used,
            "extent_ratio": result.extent_ratio,
            "dominant_class": result.dominant_class,
            "candidates_tried": result.candidates_tried,
            "sharpness_score": result.score,
            "image_base64": b64,
        }
    )
