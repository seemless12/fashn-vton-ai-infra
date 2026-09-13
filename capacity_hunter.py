"""
capacity_hunter.py
------------------
Continuous 30-Second AWS GPU Capacity Acquirer ("Capacity Hunter").
Addresses the 'InsufficientInstanceCapacity' error common with AWS GPU instances (g5.xlarge).

Monitors instance i-0c8c4e055ad9259da. If it is stopped or AWS reports out of capacity,
it auto-retries every 30 seconds until a physical GPU slot opens up and the instance
is successfully onboarded.

Usage:
  python capacity_hunter.py
  python capacity_hunter.py --interval 30
"""

import argparse
import logging
import os
import sys
import time

import boto3
import botocore.exceptions

import aws_gpu_manager

logger = logging.getLogger("capacity_hunter")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

INSTANCE_ID = aws_gpu_manager.INSTANCE_ID
REGION = aws_gpu_manager.AWS_REGION


def hunt_capacity(retry_interval: int = 30):
    client = boto3.client("ec2", region_name=REGION)
    logger.info("=" * 65)
    logger.info(f" Starting Shopping Buddy GPU Capacity Hunter for {INSTANCE_ID}")
    logger.info(f" Region: {REGION} | Check Interval: {retry_interval} seconds")
    logger.info("=" * 65)

    attempt = 1
    while True:
        info = aws_gpu_manager.get_gpu_state()
        state = info.get("state")
        public_ip = info.get("public_ip")

        logger.info(f"[Attempt {attempt}] Current Instance State: {state.upper()}")

        if state == "running" and public_ip:
            logger.info("=" * 65)
            logger.info(f" SUCCESS! GPU Instance is RUNNING with Public IP: {public_ip}")
            logger.info(f" API Endpoint: http://{public_ip}:{aws_gpu_manager.APP_PORT}")
            logger.info("=" * 65)
            
            logger.info("Checking AI service health...")
            if aws_gpu_manager.is_service_healthy(public_ip, timeout=5):
                logger.info("AI Service is HEALTHY and ready to process virtual try-on requests!")
            else:
                logger.info("AI Service is still booting models into VRAM. Will be ready in ~15-30 seconds.")
            return public_ip

        if state == "stopping":
            logger.info("Instance is currently stopping. Waiting for shutdown before onboarding...")
            time.sleep(10)
            continue

        # Try to start instance
        try:
            logger.info(f"Requesting AWS to start instance {INSTANCE_ID}...")
            client.start_instances(InstanceIds=[INSTANCE_ID])
            logger.info("AWS accepted start request! Capacity granted.")
            logger.info("Waiting for instance to reach RUNNING state and assign Public IP...")
            
            # Wait for running state
            for _ in range(40):
                time.sleep(3)
                curr = aws_gpu_manager.get_gpu_state()
                if curr.get("state") == "running" and curr.get("public_ip"):
                    logger.info("=" * 65)
                    logger.info(f" ONBOARDING COMPLETE! Public IP: {curr['public_ip']}")
                    logger.info("=" * 65)
                    return curr["public_ip"]

        except botocore.exceptions.ClientError as e:
            err_code = e.response.get("Error", {}).get("Code", "")
            err_msg = e.response.get("Error", {}).get("Message", "")
            if "capacity" in err_code.lower() or "capacity" in err_msg.lower() or err_code in ("InsufficientInstanceCapacity", "RequestLimitExceeded"):
                logger.warning(
                    f"AWS Data Center OUT OF CAPACITY ({err_code})!\n"
                    f"  Details: {err_msg}\n"
                    f"  Next onboarding attempt in {retry_interval} seconds..."
                )
            else:
                logger.error(f"AWS Error: {err_code} - {err_msg}")
        except Exception as ex:
            logger.error(f"Unexpected error: {ex}")

        attempt += 1
        time.sleep(retry_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous 30s AWS GPU Capacity Hunter")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    args = parser.parse_args()

    hunt_capacity(retry_interval=args.interval)
