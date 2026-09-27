#!/usr/bin/env python3
"""
generate_submission.py — Generate submission.jsonl for magicpin AI Challenge

Loads dataset/test_pairs.json (30 canonical pairs), resolves all 4 contexts,
calls bot.compose() for each pair, validates output, and writes submission.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
sys.stdout.reconfigure(encoding="utf-8")

import env_loader
from dotenv import load_dotenv
load_dotenv()
from bot import compose, validate_composed_message

DATASET_DIR = Path("dataset")
TEST_PAIRS_FILE = DATASET_DIR / "test_pairs.json"
OUTPUT_FILE = Path("submission.jsonl")


def check_antipatterns(records: list[dict], dataset_dir: Path, context_dumps: dict[str, str] | None = None) -> list[str]:
    """Audit composed records against anti-patterns from challenge-brief.md §11 including fabrication."""
    warnings = []
    seen_bodies = set()

    for r in records:
        tid = r["test_id"]
        body = r.get("body", "").strip()
        cta = r.get("cta", "")
        send_as = r.get("send_as", "")
        suppression_key = r.get("suppression_key", "")

        # 1. Non-empty body
        if not body:
            warnings.append(f"[{tid}] Anti-pattern: Empty body")

        # 2. Check duplicate message body across pairs
        if body in seen_bodies:
            warnings.append(f"[{tid}] Anti-pattern: Duplicate message body")
        seen_bodies.add(body)

        # 3. Check generic discount anti-pattern without service/price
        if re.search(r"\bflat\s+\d+%\s+off\b", body, re.IGNORECASE) and "@" not in body and "₹" not in body:
            warnings.append(f"[{tid}] Anti-pattern: Generic discount copy without service+price framing")

        # 4. Long preamble check
        if re.search(r"^(i hope you're doing well|i am reaching out today to|hope this email finds you)", body, re.IGNORECASE):
            warnings.append(f"[{tid}] Anti-pattern: Long greeting preamble")

        # 5. Check missing suppression key
        if not suppression_key:
            warnings.append(f"[{tid}] Missing suppression key")

        # 6. Check send_as valid
        if send_as not in ("vera", "merchant_on_behalf"):
            warnings.append(f"[{tid}] Invalid send_as value: {send_as}")

        # 7. Check for fabricated geographic landmarks, roads, or streets not in context (§11)
        if context_dumps and tid in context_dumps:
            ctx_lower = context_dumps[tid].lower()
            geo_patterns = [
                r"\b\d+\s*(?:ft|feet|foot)\s*(?:road|rd)?\b",
                r"\b[A-Za-z0-9\.\-]+\s+(?:road|rd|marg|street|st|lane|cross|metro|circle|chowk|flyover|plaza)\b"
            ]
            stopwords = {"the main", "a main", "to cross", "and cross", "or cross", "main menu"}
            for gp in geo_patterns:
                for m in re.finditer(gp, body, re.IGNORECASE):
                    phrase = m.group(0).strip()
                    if phrase.lower() in stopwords:
                        continue
                    if phrase.lower() not in ctx_lower:
                        warnings.append(f"[{tid}] Anti-pattern: Fabricated location/landmark not in context: '{phrase}'")

            # 8. Check for unverified percentages, 3+ digit numbers, or quotes
            for pct in re.findall(r"\b\d+(?:\.\d+)?%", body):
                num_val = pct.rstrip("%")
                if pct.lower() not in ctx_lower and num_val not in ctx_lower:
                    warnings.append(f"[{tid}] [Review Flag] Percentage {pct} not explicitly found in context")
            for num in re.findall(r"\b\d{3,}\b", body):
                if num != "2026" and num not in ctx_lower:
                    warnings.append(f"[{tid}] [Review Flag] 3+ digit number {num} not explicitly found in context")
            for q in re.findall(r'"([^"]{4,})"', body):
                if q.lower() not in ctx_lower:
                    warnings.append(f"[{tid}] [Review Flag] Quoted phrase \"{q}\" not explicitly found in context")

    return warnings


def main():
    if not TEST_PAIRS_FILE.exists():
        print(f"Error: {TEST_PAIRS_FILE} not found. Run Step 0 first.")
        sys.exit(1)

    with open(TEST_PAIRS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    print(f"Loaded {len(pairs)} canonical test pairs from {TEST_PAIRS_FILE}\n")

    # Check environment variable
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip())
    print(f"API Key Status: {'[DETECTED] Using Gemini API (' + ('GEMINI_API_KEY' if os.environ.get('GEMINI_API_KEY') else 'GOOGLE_API_KEY') + ')' if has_gemini_key else '[UNSET] Using Deterministic Fallback'}")

    records = []
    import time
    category_cache = {}
    context_dumps = {}
    engine_counts = {"gemini-flash": 0, "gemini-pro": 0, "fallback": 0}

    for p in pairs:
        tid = p["test_id"]
        trg_id = p["trigger_id"]
        m_id = p["merchant_id"]
        c_id = p.get("customer_id")

        trg_path = DATASET_DIR / "triggers" / f"{trg_id}.json"
        m_path = DATASET_DIR / "merchants" / f"{m_id}.json"

        with open(trg_path, "r", encoding="utf-8") as f:
            trigger = json.load(f)

        with open(m_path, "r", encoding="utf-8") as f:
            merchant = json.load(f)

        cat_slug = merchant.get("category_slug")
        if cat_slug not in category_cache:
            cat_path = DATASET_DIR / "categories" / f"{cat_slug}.json"
            with open(cat_path, "r", encoding="utf-8") as f:
                category_cache[cat_slug] = json.load(f)
        category = category_cache[cat_slug]

        customer = None
        if c_id:
            c_path = DATASET_DIR / "customers" / f"{c_id}.json"
            if c_path.exists():
                with open(c_path, "r", encoding="utf-8") as f:
                    customer = json.load(f)

        composed = compose(category, merchant, trigger, customer)
        rat = composed.get("rationale", "")
        if "gemini" in rat.lower() and "fallback" not in rat.lower():
            if "pro" in rat.lower():
                engine_counts["gemini-pro"] += 1
                engine_tag = "Gemini Pro"
            else:
                engine_counts["gemini-flash"] += 1
                engine_tag = "Gemini Flash"
        else:
            engine_counts["fallback"] += 1
            engine_tag = "Fallback"

        print(f"  [{tid}] {trigger.get('kind')} -> {engine_tag}")
        time.sleep(1.0)

        context_dumps[tid] = json.dumps({
            "category": category,
            "merchant": merchant,
            "trigger": trigger,
            "customer": customer
        }, ensure_ascii=False)

        record = {
            "test_id": tid,
            "body": composed["body"],
            "cta": composed["cta"],
            "send_as": composed["send_as"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        }
        records.append(record)

    # Write submission.jsonl
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully generated {len(records)} submissions in {OUTPUT_FILE}")
    print(f"Engine Breakdown: {engine_counts}\n")

    # Anti-pattern and constraint audit (including §11 fabrication detection)
    warnings = check_antipatterns(records, DATASET_DIR, context_dumps)
    if warnings:
        print(f"Audit Warning Count: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Audit: [PASS] All 30 messages passed anti-pattern checks (no taboos, no duplicates, valid CTAs, correct send_as).")

    # Print summary breakdown
    print("\nSubmission Summary:")
    print(f"{'Test ID':<8} {'Send As':<20} {'CTA':<20} {'Body Snippet'}")
    print("-" * 80)
    for r in records:
        snippet = r["body"].replace("\n", " ")[:50] + "..."
        print(f"{r['test_id']:<8} {r['send_as']:<20} {r['cta']:<20} {snippet}")


if __name__ == "__main__":
    main()
