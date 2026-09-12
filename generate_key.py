"""
generate_key.py
---------------
CLI utility to generate PKR 500 activation keys for Shopping Buddy customers.

Usage:
    python generate_key.py [credits] [tier] [count]

Example:
    python generate_key.py 100 starter 1
    # Output:
    # Created 1 license key(s) (PKR 500 Offer - 100 Credits):
    #   - SB-500-A4F1-9E2B
"""

import sys
from license_manager import create_license_key

def main():
    credits = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tier = sys.argv[2] if len(sys.argv) > 2 else "starter"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"==================================================")
    print(f"Shopping Buddy — License Generator")
    print(f"Offer: PKR 500 Pack ({credits} generations)")
    print(f"==================================================")

    for i in range(count):
        key = create_license_key(credits=credits, tier=tier)
        print(f"[{i+1}/{count}] Key: {key}")

if __name__ == "__main__":
    main()
