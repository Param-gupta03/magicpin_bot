"""
bot_server.py — FastAPI HTTP Server for magicpin AI Challenge ("Vera")
=====================================================================

Implements the 5 challenge endpoints exactly per challenge-testing-brief.md §2:
- POST /v1/context   (Context push)
- POST /v1/tick      (Proactive initiation)
- POST /v1/reply     (Multi-turn reply handling)
- GET  /v1/healthz   (Liveness probe)
- GET  /v1/metadata  (Candidate / bot metadata)
- POST /v1/teardown  (Privacy wipe at test end per §11)

Backed by compose() in bot.py and respond() in conversation_handlers.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("vera.server")

from dotenv import load_dotenv
load_dotenv()
import env_loader
from bot import compose
from conversation_handlers import respond

# =============================================================================
# METADATA CONFIGURATION — Edit these constants as needed
# =============================================================================
TEAM_NAME = "Vera Champion"
TEAM_MEMBERS = ["magicpin Candidate"]
MODEL = "gemini-flash-latest"
APPROACH = "4-Context dynamic composer with Gemini few-shot guidance, deterministic safety fallback, and intent-aware conversation engine"
CONTACT_EMAIL = "candidate@example.com"
VERSION = "1.0.0"
SUBMITTED_AT = "2026-09-27T12:00:00Z"

# =============================================================================
# In-Memory State
# =============================================================================
START_TIME = time.time()

# Context store keyed by (scope, context_id)
# Value: {"version": int, "payload": dict, "delivered_at": str}
CONTEXT_STORE: Dict[Tuple[str, str], Dict[str, Any]] = {}

# Conversation store keyed by conversation_id
# Value: {"turns": [...], "auto_reply_count": int, "merchant_id": str, "customer_id": str, "trigger_id": str}
CONVERSATION_STORE: Dict[str, Dict[str, Any]] = {}
MERCHANT_AUTO_REPLY_COUNT: Dict[str, int] = {}

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

app = FastAPI(title="magicpin Vera Assistant Server", version=VERSION)


# =============================================================================
# Endpoint 1: POST /v1/context
# =============================================================================
@app.post("/v1/context")
async def push_context(request: Request):
    """
    Receive context pushes.
    - Idempotent by (scope, context_id, version).
    - Higher version replaces prior version atomically.
    - 409 if version <= current_version.
    - 400 on malformed input or invalid scope.
    """
    try:
        data = await request.json()
        if not data or not isinstance(data, dict):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"accepted": False, "reason": "empty_payload", "details": "Payload cannot be empty"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_json", "details": str(e)},
        )

    scope = data.get("scope")
    context_id = data.get("context_id")
    version = data.get("version")
    payload = data.get("payload")
    delivered_at = data.get("delivered_at") or (datetime.now(timezone.utc).isoformat() + "Z")

    if scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Scope must be one of {sorted(VALID_SCOPES)}"},
        )

    if not context_id or version is None or payload is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_payload", "details": "Missing context_id, version, or payload"},
        )

    key = (scope, str(context_id))
    existing = CONTEXT_STORE.get(key)

    if existing is not None:
        current_version = existing["version"]
        if version <= current_version:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": current_version,
                },
            )

    # Store atomically
    stored_at = datetime.now(timezone.utc).isoformat() + "Z"
    CONTEXT_STORE[key] = {
        "version": version,
        "payload": payload,
        "delivered_at": delivered_at,
        "stored_at": stored_at,
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{context_id}_v{version}",
        "stored_at": stored_at,
    }


# =============================================================================
# Endpoint 2: POST /v1/tick
# =============================================================================
@app.post("/v1/tick")
async def on_tick(request: Request):
    """
    Periodic wake-up for proactive initiation.
    Resolves available triggers against in-memory contexts, composes messages,
    and returns actions. Returns {"actions": []} if nothing to send.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    available_triggers = data.get("available_triggers", [])
    if not isinstance(available_triggers, list):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_payload", "details": "available_triggers must be a list"},
        )
    actions: List[Dict[str, Any]] = []
    seen_merchant_convs = set()

    logger.info("[/v1/tick] Received %d triggers: %s", len(available_triggers), available_triggers)

    for trg_id in available_triggers:
        trg_key = ("trigger", str(trg_id))
        trg_entry = CONTEXT_STORE.get(trg_key)
        if not trg_entry:
            logger.warning("[/v1/tick] Trigger %s not found in CONTEXT_STORE", trg_id)
            continue

        trigger_doc = trg_entry["payload"]
        merchant_id = trigger_doc.get("merchant_id")
        if not merchant_id:
            logger.warning("[/v1/tick] Trigger %s has no merchant_id", trg_id)
            continue

        m_key = ("merchant", str(merchant_id))
        m_entry = CONTEXT_STORE.get(m_key)
        if not m_entry:
            logger.warning("[/v1/tick] Merchant %s not found in CONTEXT_STORE", merchant_id)
            continue

        merchant_doc = m_entry["payload"]
        cat_slug = merchant_doc.get("category_slug")
        if not cat_slug:
            logger.warning("[/v1/tick] Merchant %s has no category_slug", merchant_id)
            continue

        cat_key = ("category", str(cat_slug))
        cat_entry = CONTEXT_STORE.get(cat_key)
        if not cat_entry:
            logger.warning("[/v1/tick] Category %s not found in CONTEXT_STORE", cat_slug)
            continue

        category_doc = cat_entry["payload"]

        # Resolve customer if applicable
        customer_doc = None
        customer_id = trigger_doc.get("customer_id")
        if customer_id:
            c_key = ("customer", str(customer_id))
            c_entry = CONTEXT_STORE.get(c_key)
            if not c_entry:
                logger.warning("[/v1/tick] Customer %s context not found for trigger %s; skipping", customer_id, trg_id)
                continue
            customer_doc = c_entry["payload"]

        conversation_id = f"conv_{merchant_id}_{trg_id}"
        pair_key = (merchant_id, conversation_id)
        if pair_key in seen_merchant_convs:
            continue
        seen_merchant_convs.add(pair_key)

        try:
            composed = await asyncio.to_thread(compose, category_doc, merchant_doc, trigger_doc, customer_doc)
            body = composed["body"]
            tmpl_name = composed.get("template_name", f"vera_{trigger_doc.get('kind', 'generic')}_v1")
            tmpl_params = composed.get("template_params", [
                merchant_doc.get("identity", {}).get("owner_first_name") or merchant_doc.get("identity", {}).get("name", "Merchant"),
                category_doc.get("slug", "")
            ])
            action_item = {
                "conversation_id": conversation_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed["send_as"],
                "trigger_id": trg_id,
                "template_name": tmpl_name,
                "template_params": tmpl_params,
                "body": body,
                "cta": composed["cta"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }
            actions.append(action_item)
            logger.debug("[/v1/tick] Action created for %s -> %s", trg_id, conversation_id)

            # Initialize conversation state
            CONVERSATION_STORE[conversation_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id": trg_id,
                "category_slug": cat_slug,
                "auto_reply_count": 0,
                "turns": [
                    {
                        "turn_number": 1,
                        "from_role": composed["send_as"],
                        "message": body,
                        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    }
                ],
            }
        except Exception as e:
            logger.error("[/v1/tick] Compose exception for %s: %s", trg_id, e)
            continue

    return {"actions": actions}


# =============================================================================
# Endpoint 3: POST /v1/reply
# =============================================================================
@app.post("/v1/reply")
async def on_reply(request: Request):
    """
    Handle replies from merchants or customers in an ongoing conversation.
    Returns:
        { "action": "send", "body": ..., "cta": ..., "rationale": ... }
        { "action": "wait", "wait_seconds": ..., "rationale": ... }
        { "action": "end", "rationale": ... }
    """
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "invalid_json", "details": str(e)},
        )

    conv_id = data.get("conversation_id", "")
    merchant_id = data.get("merchant_id", "")
    from_role = data.get("from_role", "merchant")
    if "message" not in data or data.get("message") is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "missing_message", "details": "The 'message' field is required"},
        )
    message = data.get("message", "")
    turn_number = data.get("turn_number", 2)
    received_at = data.get("received_at", datetime.now(timezone.utc).isoformat() + "Z")

    state = CONVERSATION_STORE.setdefault(
        conv_id,
        {
            "merchant_id": merchant_id,
            "customer_id": data.get("customer_id"),
            "trigger_id": None,
            "category_slug": None,
            "auto_reply_count": 0,
            "turns": [],
        },
    )

    # Retrieve context docs for intelligent conversation answering
    m_entry = CONTEXT_STORE.get(("merchant", str(merchant_id)))
    merchant_doc = m_entry["payload"] if m_entry else None

    cat_slug = state.get("category_slug")
    if not cat_slug and merchant_doc:
        cat_slug = merchant_doc.get("category_slug")
    cat_entry = CONTEXT_STORE.get(("category", str(cat_slug))) if cat_slug else None
    category_doc = cat_entry["payload"] if cat_entry else None

    trg_entry = CONTEXT_STORE.get(("trigger", str(state.get("trigger_id")))) if state.get("trigger_id") else None
    trigger_doc = trg_entry["payload"] if trg_entry else None

    # Produce the next conversational move
    # Track auto replies per merchant across conversation IDs (e.g. for simulator runs)
    m_auto_count = MERCHANT_AUTO_REPLY_COUNT.get(str(merchant_id), 0)
    state["auto_reply_count"] = max(state.get("auto_reply_count", 0), m_auto_count)
    if turn_number >= 3:
        state["auto_reply_count"] = max(state["auto_reply_count"], 1)

    response_action = respond(
        conversation_state=state,
        latest_message=message,
        from_role=from_role,
        merchant_context=merchant_doc,
        category_context=category_doc,
        trigger_context=trigger_doc,
    )

    # Track auto-reply frequency
    if response_action.get("action") == "wait" and "auto-reply" in response_action.get("rationale", "").lower():
        state["auto_reply_count"] = state.get("auto_reply_count", 0) + 1
        MERCHANT_AUTO_REPLY_COUNT[str(merchant_id)] = m_auto_count + 1
    elif response_action.get("action") == "end":
        MERCHANT_AUTO_REPLY_COUNT[str(merchant_id)] = 0

    # Persist the merchant's incoming turn
    state["turns"].append({
        "turn_number": turn_number,
        "from_role": from_role,
        "message": message,
        "received_at": received_at,
    })

    # Persist the bot's outgoing turn
    state["turns"].append({
        "turn_number": turn_number + 1,
        "from_role": "vera",
        "action": response_action.get("action"),
        "body": response_action.get("body"),
        "rationale": response_action.get("rationale"),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    })

    return response_action


# =============================================================================
# Endpoint 4: GET /v1/healthz
# =============================================================================
@app.get("/v1/healthz")
async def healthz():
    """
    Liveness and context readiness probe.
    """
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in CONTEXT_STORE.keys():
        if scope in counts:
            counts[scope] += 1

    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


# =============================================================================
# Endpoint 5: GET /v1/metadata
# =============================================================================
@app.get("/v1/metadata")
async def metadata():
    """
    Candidate and model metadata.
    """
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": MODEL,
        "approach": APPROACH,
        "contact_email": CONTACT_EMAIL,
        "version": VERSION,
        "submitted_at": SUBMITTED_AT,
    }


# =============================================================================
# Endpoint 6 (Optional): POST /v1/teardown
# =============================================================================
@app.post("/v1/teardown")
async def teardown():
    """
    Wipe all in-memory context and conversation data for privacy compliance (§11).
    """
    CONTEXT_STORE.clear()
    CONVERSATION_STORE.clear()
    MERCHANT_AUTO_REPLY_COUNT.clear()
    return {"accepted": True, "status": "wiped"}
