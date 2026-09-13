"""
aws_gpu_manager.py
------------------
Programmatic AWS EC2 Lifecycle Manager for Shopping Buddy AI.
Controls the GPU instance (g5.xlarge: i-0c8c4e055ad9259da) in ap-south-1.
Supports:
  - Checking instance state and public IP
  - Waking the instance on-demand if stopped
  - Health probing until the FastAPI AI engine is ready to accept requests
  - Stopping the instance to eliminate compute billing
"""

import logging
import os
import sys
import time
from typing import Dict, Optional, Callable
import urllib.request
import urllib.error
import json

import boto3
import botocore.exceptions

logger = logging.getLogger("aws_gpu_manager")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
INSTANCE_ID = os.environ.get("FASHN_INSTANCE_ID", "i-0c8c4e055ad9259da")
APP_PORT = int(os.environ.get("FASHN_APP_PORT", 8000))

# Maximum wait time for EC2 to transition to running (in seconds)
MAX_BOOT_WAIT_SECONDS = 120
# Maximum wait time for FastAPI /health to respond ok once booted
MAX_HEALTH_WAIT_SECONDS = 60


def get_ec2_client():
    return boto3.client("ec2", region_name=AWS_REGION)


def get_gpu_state() -> Dict[str, Optional[str]]:
    """
    Returns current instance state and public IP.
    Example return: {"state": "stopped", "public_ip": None}
    """
    client = get_ec2_client()
    try:
        res = client.describe_instances(InstanceIds=[INSTANCE_ID])
        reservations = res.get("Reservations", [])
        if not reservations or not reservations[0].get("Instances"):
            return {"state": "unknown", "public_ip": None}
            
        instance = reservations[0]["Instances"][0]
        state = instance["State"]["Name"]
        public_ip = instance.get("PublicIpAddress", None)
        return {"state": state, "public_ip": public_ip}
    except Exception as e:
        logger.error(f"Failed to query EC2 state for {INSTANCE_ID}: {e}")
        return {"state": "error", "error": str(e), "public_ip": None}


def is_service_healthy(public_ip: str, timeout: int = 3) -> bool:
    """
    Pings the /health endpoint of the GPU instance.
    """
    if not public_ip:
        return False
    url = f"http://{public_ip}:{APP_PORT}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ShoppingBuddyGateway/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                return data.get("status") == "ok" or data.get("model_loaded") is True
    except Exception:
        return False
    return False


def ensure_gpu_ready(status_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Ensures that the GPU instance is running and its AI engine is healthy.
    If the instance is stopped, it starts it up, waits for a public IP,
    and verifies that /health is responding.
    Returns the reachable public IP of the GPU server.
    """
    client = get_ec2_client()
    info = get_gpu_state()
    state = info.get("state")
    public_ip = info.get("public_ip")

    logger.info(f"Current GPU instance state: {state} (IP: {public_ip})")

    if state == "running" and public_ip:
        if is_service_healthy(public_ip, timeout=4):
            logger.info(f"GPU server is already online and healthy at http://{public_ip}:{APP_PORT}")
            return public_ip
        else:
            logger.info("Instance is running but service is not yet responding. Waiting for health probe...")

    if state in ("stopped", "stopping"):
        if state == "stopping":
            if status_callback:
                status_callback("Waiting for previous instance shutdown to complete...")
            logger.info("Waiting for instance to finish stopping...")
            waiter = client.get_waiter("instance_stopped")
            waiter.wait(InstanceIds=[INSTANCE_ID])

        started = False
        attempt = 1
        max_attempts = 15  # Up to 7.5 minutes of auto-polling every 30s
        while not started and attempt <= max_attempts:
            try:
                if status_callback:
                    if attempt == 1:
                        status_callback("Waking up dedicated AI GPU instance...")
                    else:
                        status_callback(f"AWS GPU busy (Out of Capacity). Auto-retrying every 30s to onboard slot (Attempt {attempt}/{max_attempts})...")
                logger.info(f"Starting EC2 instance {INSTANCE_ID} (Attempt {attempt}/{max_attempts})...")
                client.start_instances(InstanceIds=[INSTANCE_ID])
                started = True
                logger.info("Successfully requested EC2 start_instances!")
            except botocore.exceptions.ClientError as e:
                err_code = e.response.get("Error", {}).get("Code", "")
                err_msg = e.response.get("Error", {}).get("Message", "")
                if "capacity" in err_code.lower() or "capacity" in err_msg.lower() or err_code in ("InsufficientInstanceCapacity", "RequestLimitExceeded"):
                    logger.warning(f"AWS GPU Capacity Unavailable ({err_code}: {err_msg}). Retrying in 30 seconds (Attempt {attempt}/{max_attempts})...")
                    if status_callback:
                        status_callback(f"AWS GPU cluster is at capacity. Auto-polling in 30s to secure slot (Attempt {attempt}/{max_attempts})...")
                    attempt += 1
                    time.sleep(30)
                else:
                    logger.error(f"Non-capacity EC2 ClientError: {err_code} - {err_msg}")
                    raise e
            except Exception as e:
                if "capacity" in str(e).lower():
                    logger.warning(f"Capacity error: {e}. Retrying in 30s...")
                    attempt += 1
                    time.sleep(30)
                else:
                    raise e

        if not started:
            raise TimeoutError(f"Could not secure AWS GPU capacity after {max_attempts} attempts.")

    # Wait for instance to reach running state
    start_time = time.time()
    while time.time() - start_time < MAX_BOOT_WAIT_SECONDS:
        info = get_gpu_state()
        state = info.get("state")
        public_ip = info.get("public_ip")
        
        if state == "running" and public_ip:
            logger.info(f"Instance reached running state with Public IP: {public_ip}")
            break
            
        elapsed = int(time.time() - start_time)
        msg = f"Waking up dedicated AI GPU... ({elapsed}s elapsed)"
        if status_callback:
            status_callback(msg)
        time.sleep(3)
    else:
        raise TimeoutError(f"Instance {INSTANCE_ID} failed to reach 'running' state within {MAX_BOOT_WAIT_SECONDS}s.")

    # Wait for FastAPI service to respond to /health
    if status_callback:
        status_callback("AI GPU booted. Initializing neural network weights...")
    health_start = time.time()
    while time.time() - health_start < MAX_HEALTH_WAIT_SECONDS:
        if is_service_healthy(public_ip, timeout=3):
            logger.info(f"GPU server {public_ip}:{APP_PORT} is fully healthy and ready!")
            if status_callback:
                status_callback("AI Engine ready!")
            return public_ip
        time.sleep(3)
        
    # If health probe timed out but IP is active, return IP and let request retry
    logger.warning("Health probe timed out, but instance is running. Returning Public IP.")
    return public_ip


def stop_gpu() -> bool:
    """
    Shuts down the GPU instance to drop compute billing to $0.00/hr.
    """
    client = get_ec2_client()
    try:
        logger.info(f"Stopping instance {INSTANCE_ID} to save GPU compute costs...")
        client.stop_instances(InstanceIds=[INSTANCE_ID])
        return True
    except Exception as e:
        logger.error(f"Failed to stop instance {INSTANCE_ID}: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_gpu()
    elif len(sys.argv) > 1 and sys.argv[1] == "start":
        ip = ensure_gpu_ready(status_callback=lambda msg: print(f"Progress: {msg}"))
        print(f"Server is ready at: http://{ip}:{APP_PORT}")
    else:
        st = get_gpu_state()
        print(f"GPU Instance ({INSTANCE_ID}) State: {st['state']} | Public IP: {st.get('public_ip')}")
