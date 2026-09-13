"""
gateway.py
----------
Shopping Buddy AI - Decoupled Smart Gateway (Always Online 24/7).
Zero GPU Footprint ($0 to $4/mo).

Responsibilities:
  1. Always-on health check and instant credit validation.
  2. Accepts virtual try-on requests from the Chrome Extension.
  3. Checks AWS EC2 GPU instance state (i-0c8c4e055ad9259da):
     - If stopped, wakes it up via boto3.
     - Tracks real-time boot status for the extension ("waking_gpu").
  4. Forwards inference payload to the GPU worker once ready.
  5. Relays the generated try-on image back to the client.
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.parse
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import aws_gpu_manager
import license_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smart_gateway")

app = FastAPI(title="Shopping Buddy Smart Gateway (Decoupled)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dir for serving generated images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory job registry
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()


def _multipart_post(url: str, fields: dict, files: dict, timeout: int = 120) -> dict:
    """Helper to post multipart/form-data using standard urllib (no external deps needed)."""
    boundary = "----ShoppingBuddyBoundary" + uuid.uuid4().hex
    body = bytearray()

    # Form fields
    for key, val in fields.items():
        if val is None:
            continue
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{val}\r\n".encode("utf-8"))

    # File fields
    for field_name, (filename, file_bytes, mime) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "ShoppingBuddyGateway/1.0",
        },
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def _process_tryon_job(
    job_id: str,
    person_bytes: bytes,
    garment_bytes: bytes,
    category: str,
    garment_photo_type: str,
    mode: str,
    steps: int,
    guidance_scale: float,
    device_id: str,
):
    """Background task to wake GPU, dispatch inference, and store result."""
    try:
        async with _jobs_lock:
            _jobs[job_id]["stage"] = "waking_gpu"
            _jobs[job_id]["message"] = "Checking dedicated AI GPU state..."

        def on_wake_progress(msg: str):
            if job_id in _jobs:
                _jobs[job_id]["stage"] = "waking_gpu"
                _jobs[job_id]["message"] = msg

        # Step 1: Ensure GPU is running & healthy
        gpu_ip = await asyncio.to_thread(aws_gpu_manager.ensure_gpu_ready, on_wake_progress)

        async with _jobs_lock:
            _jobs[job_id]["stage"] = "processing"
            _jobs[job_id]["message"] = "Fitting garment in virtual fitting room..."
            _jobs[job_id]["gpu_ip"] = gpu_ip

        # Step 2: Dispatch job to GPU server
        gpu_url = f"http://{gpu_ip}:{aws_gpu_manager.APP_PORT}/api/try-on"
        fields = {
            "category": category,
            "garment_photo_type": garment_photo_type,
            "mode": mode,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "device_id": device_id,
        }
        files = {
            "person_image": ("person.jpg", person_bytes, "image/jpeg"),
            "garment_image": ("garment.jpg", garment_bytes, "image/jpeg"),
        }

        logger.info(f"Forwarding job {job_id} to GPU worker: {gpu_url}")
        res = await asyncio.to_thread(_multipart_post, gpu_url, fields, files, 180)
        gpu_job_id = res.get("job_id")
        poll_endpoint = res.get("poll_endpoint", f"/api/try-on/status/{gpu_job_id}")

        # Step 3: Poll GPU worker until completion
        poll_url = f"http://{gpu_ip}:{aws_gpu_manager.APP_PORT}{poll_endpoint}"
        poll_start = time.time()
        completed_data = None

        while time.time() - poll_start < 180:
            await asyncio.sleep(1.0)
            try:
                poll_req = urllib.request.Request(poll_url, headers={"User-Agent": "ShoppingBuddyGateway/1.0"})
                with urllib.request.urlopen(poll_req, timeout=5) as poll_resp:
                    status_json = json.loads(poll_resp.read().decode("utf-8"))
                    if status_json.get("status") == "completed":
                        completed_data = status_json
                        break
                    elif status_json.get("status") in ("failed", "error"):
                        raise RuntimeError(status_json.get("error", "GPU generation failed."))
            except urllib.error.URLError:
                continue

        if not completed_data:
            raise TimeoutError("Inference timed out on GPU worker.")

        # Step 4: Finalize and save result
        image_b64 = completed_data.get("image", "")
        if not image_b64 and completed_data.get("image_url"):
            # Fetch image bytes from GPU server static URL
            img_fetch_url = f"http://{gpu_ip}:{aws_gpu_manager.APP_PORT}{completed_data['image_url']}"
            with urllib.request.urlopen(img_fetch_url, timeout=10) as img_resp:
                raw_bytes = img_resp.read()
                image_b64 = f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

        async with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["stage"] = "completed"
            _jobs[job_id]["image"] = image_b64
            _jobs[job_id]["steps"] = steps
            _jobs[job_id]["finished_at"] = time.time()
            _jobs[job_id]["meta"] = completed_data.get("meta", {})

        logger.info(f"Job {job_id} successfully completed on GPU worker {gpu_ip}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed during GPU orchestration: {e}")
        async with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


# -------------------------------------------------------------------
# GATEWAY PUBLIC API ENDPOINTS
# -------------------------------------------------------------------

@app.get("/health")
def health():
    """Always-online gateway health check with live GPU status."""
    gpu_info = aws_gpu_manager.get_gpu_state()
    return {
        "status": "ok",
        "gateway": True,
        "gpu_instance_id": aws_gpu_manager.INSTANCE_ID,
        "gpu_state": gpu_info.get("state"),
        "gpu_ip": gpu_info.get("public_ip"),
    }


@app.get("/api/credits/check")
def check_credits(device_id: str, license_key: Optional[str] = None):
    """Instant local credit and plan validation without waking GPU."""
    return license_manager.check_credits(device_id, license_key)


@app.post("/api/license/activate")
async def activate_license(
    device_id: str = Form(...),
    license_key: str = Form(...)
):
    """Instant license activation without waking GPU."""
    return license_manager.activate_license(license_key, device_id)


@app.post("/api/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form(""),
    garment_photo_type: str = Form("model"),
    mode: str = Form("auto"),
    steps: int = Form(15, ge=10, le=50),
    guidance_scale: float = Form(1.5, ge=1.0, le=5.0),
    device_id: str = Form("dev_guest"),
    license_key: Optional[str] = Form(None),
):
    """
    Primary try-on submission endpoint.
    Validates quota, deducts credit, and orchestrates GPU wake-up in background.
    """
    credit_info = license_manager.check_credits(device_id, license_key)
    if not credit_info.get("has_credits", False):
        raise HTTPException(
            status_code=402,
            detail="Credit limit reached. Please upgrade to the PKR 500 pack (100 try-ons) to continue."
        )

    if not license_manager.deduct_credit(device_id, license_key):
        raise HTTPException(
            status_code=402,
            detail="Credit deduction failed. Quota may be exhausted."
        )

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    job_id = uuid.uuid4().hex
    async with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "stage": "waking_gpu",
            "message": "Initiating AI pipeline...",
            "created_at": time.time(),
        }

    # Dispatch to GPU background task
    asyncio.create_task(
        _process_tryon_job(
            job_id=job_id,
            person_bytes=person_bytes,
            garment_bytes=garment_bytes,
            category=category,
            garment_photo_type=garment_photo_type,
            mode=mode,
            steps=steps,
            guidance_scale=guidance_scale,
            device_id=device_id,
        )
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "poll_endpoint": f"/api/try-on/status/{job_id}",
    }


@app.get("/api/try-on/status/{job_id}")
async def poll_tryon(job_id: str):
    """Poll endpoint providing stage-aware feedback to the Chrome Extension."""
    async with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    status = job.get("status")
    if status == "completed":
        return JSONResponse({
            "status": "completed",
            "image": job.get("image"),
            "steps": job.get("steps", 15),
            "meta": job.get("meta", {}),
        })
    elif status == "failed":
        return JSONResponse({
            "status": "failed",
            "error": job.get("error", "Try-on generation failed.")
        }, status_code=500)

    # Still pending/waking/processing
    return JSONResponse({
        "status": "pending",
        "stage": job.get("stage", "processing"),
        "message": job.get("message", "Styling your garment..."),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
