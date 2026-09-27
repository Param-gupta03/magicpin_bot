"""
conversation_handlers.py — Multi-turn conversation logic for Vera
magicpin AI Challenge

Implements:
    respond(conversation_state: dict, latest_message: str, from_role: str,
            merchant_context: dict | None, category_context: dict | None,
            trigger_context: dict | None) -> dict
"""

from __future__ import annotations

import re
from typing import List, Optional

# Canned auto-reply patterns commonly seen in WhatsApp Business
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"thanks for reaching out",
    r"our team will respond shortly",
    r"automated assistant",
    r"automated message",
    r"automatic reply",
    r"we will get back to you",
    r"currently unavailable",
    r"hamari team tak pahuncha deti hoon",
    r"aapki jaankari ke liye bahut-bahut shukriya",
    r"i am an automated assistant",
]

# Explicit opt-out / stop patterns
OPTOUT_PATTERNS = [
    r"\bstop\b",
    r"\bunsubscribe\b",
    r"\bnot interested\b",
    r"\bstop messaging\b",
    r"\bdon't message\b",
    r"\bdont message\b",
    r"\bleave me alone\b",
    r"\bmat bhejo\b",
    r"\bband karo\b",
    r"\buseless\b",
    r"\bwhy are you bothering\b",
]

# Defer / busy patterns
DEFER_PATTERNS = [
    r"\bbusy right now\b",
    r"\bbusy hoon\b",
    r"\bcheck back later\b",
    r"\bcall later\b",
    r"\bnot now\b",
    r"\blater please\b",
    r"\bkal baat\b",
    r"\bthodi der baad\b",
    r"\bafter some time\b",
]

# Explicit affirmative / intent handoff patterns
AFFIRMATIVE_PATTERNS = [
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\bsure\b",
    r"\bok\b",
    r"\bokay\b",
    r"\bgo ahead\b",
    r"\blet's do it\b",
    r"\blets do it\b",
    r"\bhaan\b",
    r"\btheek hai\b",
    r"\bsend\b",
    r"\bdraft\b",
    r"\bproceed\b",
    r"\bbilkul\b",
    r"\bkardo\b",
    r"\bconfirm\b",
    r"^\s*1\s*$",
    r"^\s*2\s*$",
]

# Out-of-scope curveball patterns
OUT_OF_SCOPE_PATTERNS = [
    (r"\bgst\b|\btax\b|\bfiling\b|\bca\b", "GST and tax filing", "your CA/tax accountant"),
    (r"\bloan\b|\bfinancing\b|\bbank credit\b", "business financing and loans", "your bank"),
    (r"\bfssai\b|\blicense\b|\bhygiene rating cert\b", "food licensing compliance", "the municipal/FSSAI office"),
]


def is_auto_reply(latest_message: str, prior_messages: List[str]) -> bool:
    """Detect canned auto-replies via repetition or common template phrasing."""
    lower_msg = latest_message.strip().lower()

    # 1. Regex match for canned auto-replies
    for pat in AUTO_REPLY_PATTERNS:
        if re.search(pat, lower_msg):
            return True

    # 2. Verbatim repetition: repeated 3+ times across prior merchant messages
    if lower_msg and prior_messages:
        repeat_count = sum(1 for m in prior_messages if m.strip().lower() == lower_msg)
        if repeat_count >= 2:  # current message makes it 3
            return True

    return False


def is_affirmative(message: str) -> bool:
    """Detect affirmative intent transitions."""
    lower = message.strip().lower()
    return any(re.search(pat, lower) for pat in AFFIRMATIVE_PATTERNS)


def is_opt_out(message: str) -> bool:
    """Detect explicit opt-out or hostility."""
    lower = message.strip().lower()
    return any(re.search(pat, lower) for pat in OPTOUT_PATTERNS)


def is_deferral(message: str) -> bool:
    """Detect request for later callback/wait."""
    lower = message.strip().lower()
    return any(re.search(pat, lower) for pat in DEFER_PATTERNS)


def respond(
    conversation_state: dict,
    latest_message: str,
    from_role: str = "merchant",
    merchant_context: Optional[dict] = None,
    category_context: Optional[dict] = None,
    trigger_context: Optional[dict] = None
) -> dict:
    """
    Produce the next response for an ongoing conversation.
    Returns:
        { "action": "send", "body": ..., "cta": ..., "rationale": ... }
        { "action": "wait", "wait_seconds": ..., "rationale": ... }
        { "action": "end", "rationale": ... }
    """
    turns = conversation_state.get("turns", [])
    prior_merchant_msgs = [
        t.get("message", "") for t in turns 
        if t.get("from_role") == "merchant"
    ]
    auto_reply_count = conversation_state.get("auto_reply_count", 0)

    # 1. Check for Auto-reply
    if is_auto_reply(latest_message, prior_merchant_msgs):
        if auto_reply_count >= 1:
            # Second auto-reply -> end conversation gracefully
            return {
                "action": "end",
                "rationale": "Second consecutive auto-reply received; gracefully exiting conversation to avoid message loops."
            }
        else:
            # First auto-reply -> back off
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Detected merchant auto-reply (canned phrasing). Backing off 4 hours to wait for human owner."
            }

    # 2. Check for Opt-Out / Hostility
    if is_opt_out(latest_message):
        return {
            "action": "end",
            "rationale": "Merchant explicitly requested opt-out or expressed disinterest; closing conversation cleanly."
        }

    # 3. Check for Deferral
    if is_deferral(latest_message):
        return {
            "action": "wait",
            "wait_seconds": 3600,
            "rationale": "Merchant asked to reconnect later; backing off 1 hour."
        }

    # 4. Check for Out-of-Scope Curveballs
    lower_msg = latest_message.strip().lower()
    for pat, topic, handler in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, lower_msg):
            return {
                "action": "send",
                "body": (
                    f"I'll have to leave {topic} to {handler} — that's outside what Vera can help with directly. "
                    f"Coming back to our active campaign — want me to send over the promotion draft to review? Reply YES."
                ),
                "cta": "binary_yes_stop",
                "rationale": f"Out-of-scope inquiry regarding {topic} politely redirected back to marketing workflow."
            }

    # 5. Check for Affirmative / Intent Transition (Action Mode)
    if is_affirmative(latest_message):
        # Determine topic and owner/biz details
        owner = ""
        biz_name = "your business"
        locality = "your area"
        if merchant_context:
            identity = merchant_context.get("identity", {})
            owner = identity.get("owner_first_name", "")
            biz_name = identity.get("name", biz_name)
            locality = identity.get("locality", locality)

        trg_kind = trigger_context.get("kind", "") if trigger_context else ""

        if "research" in trg_kind or "cde" in trg_kind:
            return {
                "action": "send",
                "body": (
                    f"Sending the clinical summary and abstract now (2 pages). "
                    f"I've also pre-filled the patient WhatsApp note ready to broadcast to your roster. "
                    f"Want me to send the broadcast draft to your number right now? Reply YES."
                ),
                "cta": "binary_yes_stop",
                "rationale": "Merchant gave affirmative confirmation; transitioned immediately into fulfillment and delivery."
            }
        elif "thali" in trg_kind or "planning" in trg_kind:
            return {
                "action": "send",
                "body": (
                    f"Great! I am drafting the 1-page package flyer and WhatsApp announcement for {biz_name} in {locality} now. "
                    f"It will be ready in 90 seconds. Would you like me to push the introductory announcement to Google Business as well? Reply YES."
                ),
                "cta": "binary_yes_stop",
                "rationale": "Immediate action transition honoring merchant planning commitment with concrete 90-second timeline."
            }
        else:
            return {
                "action": "send",
                "body": (
                    f"Perfect! I am generating the ready-to-share campaign draft for {biz_name} right now. "
                    f"Takes under 2 minutes. Want me to send the final preview before publishing? Reply YES."
                ),
                "cta": "binary_yes_stop",
                "rationale": "Affirmative intent detected; immediately executing campaign drafting without redundant qualification."
            }

    # 6. Default engaged conversational response (with turn limit guard)
    if len(prior_merchant_msgs) >= 2:
        return {
            "action": "wait",
            "wait_seconds": 86400,
            "rationale": "Conversation reached turn limit without resolution; backing off 24h to prevent messaging fatigue."
        }

    return {
        "action": "send",
        "body": (
            f"Understood! We can customize the timing and service details to suit your schedule. "
            f"Would you like me to send the draft copy over for a quick review? Reply YES."
        ),
        "cta": "binary_yes_stop",
        "rationale": "Engaged merchant follow-up keeping friction low with a binary yes/no next step."
    }
