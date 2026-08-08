import os
import json
import time
import base64
import logging
from io import BytesIO
from PIL import Image
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpu_worker")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
QUEUE_NAME = "fashn_jobs_queue"

logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client.ping()
    logger.info("Connected to Redis successfully.")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

# Lazy load the engine
engine = None

def get_engine():
    global engine
    if engine is None:
        logger.info("Loading VTON Engine...")
        from engine import VTONEngine
        engine = VTONEngine(weights_dir=os.environ.get("FASHN_WEIGHTS_DIR", "./weights"))
        logger.info("VTON Engine loaded.")
    return engine

def process_job(job_id: str, payload_str: str):
    logger.info(f"Processing job: {job_id}")
    
    # 1. Update status to processing
    redis_client.hset(f"job:{job_id}", "status", "processing")
    redis_client.hset(f"job:{job_id}", "started_at", str(time.time()))

    try:
        # 2. Parse payload
        payload = json.loads(payload_str)
        person_bytes = base64.b64decode(payload["person_image_b64"])
        garment_bytes = base64.b64decode(payload["garment_image_b64"])
        
        # 3. Decode images (Moved from FastAPI to GPU worker)
        person_pil = Image.open(BytesIO(person_bytes)).convert("RGB")
        garment_pil = Image.open(BytesIO(garment_bytes)).convert("RGB")

        # 4. Construct Request
        from engine import TryOnRequest
        req = TryOnRequest(
            person_image=person_pil,
            garment_image=garment_pil,
            category=payload.get("category"),
            garment_photo_type=payload.get("garment_photo_type", "model"),
            mode=payload.get("mode", "auto"),
            num_samples=payload.get("num_samples", 1),
            steps=payload.get("steps", 15),
            guidance_scale=payload.get("guidance_scale", 1.5),
            seed=payload.get("seed", 42),
            autocrop=payload.get("autocrop", True),
            refine=payload.get("refine", False),
            refine_strength=payload.get("refine_strength", 0.25),
            refine_guidance_scale=payload.get("refine_guidance_scale", 3.0),
            refine_steps=payload.get("refine_steps", 25)
        )

        # 5. Run Engine
        vton_engine = get_engine()
        result = vton_engine.run(req)

        # 6. Encode Result
        buf = BytesIO()
        result.image.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 7. Store Result back in Redis
        meta = {
            "category_used": result.category_used,
            "segmentation_free_used": result.segmentation_free_used,
            "extent_ratio": result.extent_ratio,
            "dominant_class": result.dominant_class,
            "candidates_tried": result.candidates_tried,
            "refined": result.refined,
        }
        
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "completed",
            "image_b64": result_b64,
            "meta": json.dumps(meta),
            "finished_at": str(time.time())
        })
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Job {job_id} failed.")
        redis_client.hset(f"job:{job_id}", mapping={
            "status": "failed",
            "error": str(e),
            "finished_at": str(time.time())
        })

def main():
    logger.info("Worker started. Listening for jobs...")
    while True:
        try:
            # Blocking pop from queue (timeout 5s)
            result = redis_client.blpop(QUEUE_NAME, timeout=5)
            if result:
                _, job_data = result
                job_dict = json.loads(job_data)
                process_job(job_dict["job_id"], job_dict["payload"])
        except Exception as e:
            logger.error(f"Error popping from queue: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
