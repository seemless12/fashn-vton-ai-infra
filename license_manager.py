"""
license_manager.py
------------------
SQLite-based license & credit tracking for Shopping Buddy AI.
Supports:
- Device-level free trial quotas (default: 3 generations)
- Paid activation license keys (PKR 500 for 100 generations)
- Secure credit deduction on GPU generation start
"""

import sqlite3
import uuid
import time
import os
from pathlib import Path

DB_PATH = Path(os.environ.get("LICENSE_DB_PATH", "licenses.db"))

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables if they do not exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                key TEXT PRIMARY KEY,
                credits_total INTEGER NOT NULL,
                credits_remaining INTEGER NOT NULL,
                tier TEXT NOT NULL DEFAULT 'starter',
                created_at REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS device_trials (
                device_id TEXT PRIMARY KEY,
                free_credits INTEGER NOT NULL DEFAULT 3,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()

# Ensure DB tables exist on load
init_db()

def create_license_key(credits: int = 100, tier: str = "starter") -> str:
    """
    Generate a formatted activation key:
    e.g. SB-500-A4F1-9E2B (PKR 500 offer)
    """
    raw = uuid.uuid4().hex[:8].upper()
    prefix = "SB-500" if tier == "starter" else f"SB-{tier.upper()[:3]}"
    key = f"{prefix}-{raw[:4]}-{raw[4:]}"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO licenses (key, credits_total, credits_remaining, tier, created_at, is_active) VALUES (?, ?, ?, ?, ?, 1)",
            (key, credits, credits, tier, time.time())
        )
        conn.commit()
    return key

def check_credits(device_id: str, license_key: str = None) -> dict:
    """
    Check current remaining credits.
    Priority: Paid License Key > Device Free Trial.
    """
    with get_db() as conn:
        # 1. If user provided a license key, validate and return key status
        if license_key and license_key.strip():
            clean_key = license_key.strip().upper()
            cur = conn.execute("SELECT * FROM licenses WHERE key = ? AND is_active = 1", (clean_key,))
            row = cur.fetchone()
            if row:
                return {
                    "valid": True,
                    "type": "paid",
                    "tier": row["tier"],
                    "credits_remaining": row["credits_remaining"],
                    "credits_total": row["credits_total"],
                    "offer_name": "Pro Pack (PKR 500)"
                }
            return {
                "valid": False,
                "error": "Invalid or expired license key",
                "credits_remaining": 0,
                "credits_total": 0
            }

        # 2. Fall back to device trial
        cur = conn.execute("SELECT * FROM device_trials WHERE device_id = ?", (device_id,))
        row = cur.fetchone()
        if not row:
            # First time user on this device gets 3 free try-ons
            conn.execute(
                "INSERT INTO device_trials (device_id, free_credits, created_at) VALUES (?, 3, ?)",
                (device_id, time.time())
            )
            conn.commit()
            return {
                "valid": True,
                "type": "trial",
                "tier": "free_trial",
                "credits_remaining": 3,
                "credits_total": 3,
                "offer_name": "Free Trial (3 Free Try-Ons)"
            }

        return {
            "valid": True,
            "type": "trial",
            "tier": "free_trial",
            "credits_remaining": row["free_credits"],
            "credits_total": 3,
            "offer_name": "Free Trial (3 Free Try-Ons)"
        }

def deduct_credit(device_id: str, license_key: str = None) -> bool:
    """
    Deducts 1 credit from either the paid key or device trial.
    Returns True if deduction succeeded, False if quota exhausted.
    """
    with get_db() as conn:
        if license_key and license_key.strip():
            clean_key = license_key.strip().upper()
            cur = conn.execute("SELECT credits_remaining FROM licenses WHERE key = ? AND is_active = 1", (clean_key,))
            row = cur.fetchone()
            if row and row["credits_remaining"] > 0:
                conn.execute(
                    "UPDATE licenses SET credits_remaining = credits_remaining - 1 WHERE key = ?",
                    (clean_key,)
                )
                conn.commit()
                return True
            return False

        # Deduct from device trial
        cur = conn.execute("SELECT free_credits FROM device_trials WHERE device_id = ?", (device_id,))
        row = cur.fetchone()
        if row and row["free_credits"] > 0:
            conn.execute(
                "UPDATE device_trials SET free_credits = free_credits - 1 WHERE device_id = ?",
                (device_id,)
            )
            conn.commit()
            return True
        return False
