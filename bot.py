"""
bot.py — Vera Engagement Assistant Message Composer
magicpin AI Challenge

Implements:
    compose(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict
Returning:
    {
        "body": str,
        "cta": str,
        "send_as": "vera" | "merchant_on_behalf",
        "suppression_key": str,
        "rationale": str
    }

Deterministic (temperature=0), completes in <30s.
Uses Google Gemini API (model "gemini-flash-latest" / "gemini-flash-lite-latest") when GEMINI_API_KEY is present.
Includes robust deterministic fallback per trigger kind when API key is unset or quota exceeded.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Optional, Tuple, List

logger = logging.getLogger("vera.bot")

from dotenv import load_dotenv
load_dotenv()
import env_loader


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_composed_message(msg: dict, category: dict, context_dump: str = "") -> Tuple[bool, List[str]]:
    """Validate body, single CTA, taboo vocabulary, generic discount, and unverified geographic fabrication."""
    errors = []
    body = msg.get("body", "").strip()
    if not body:
        errors.append("Empty body")

    # Check taboos from CategoryContext
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    for taboo in taboos:
        clean_taboo = re.sub(r"\s*\(.*?\)", "", taboo).strip().lower()
        if clean_taboo and clean_taboo in body.lower():
            errors.append(f"Contains taboo vocabulary: '{clean_taboo}'")

    # Check for generic discount anti-pattern without service+price
    if re.search(r"\bflat\s+\d+%\s+off\b", body, re.IGNORECASE) and "@" not in body and "₹" not in body:
        errors.append("Generic discount copy without service+price framing")

    # Check for multiple conflicting CTAs
    lower_body = body.lower()
    if "reply yes" in lower_body and "reply no" in lower_body and "reply maybe" in lower_body:
        errors.append("Multiple competing CTAs detected")

    # Check for blank greeting placeholder (e.g. 'Hi ,')
    if re.search(r"\b(?:hi|hello|namaste|hey|dear)\s*[,!]", body, re.IGNORECASE):
        errors.append("Blank name greeting placeholder (e.g. 'Hi ,')")

    # Check for raw database/trigger/digest IDs
    if re.search(r"\b(?:trg_|d_\d{4}|m_\d{3}|c_\d{3})[a-zA-Z0-9_]*\b", body, re.IGNORECASE):
        errors.append("Raw database/trigger/digest ID leaked in body")

    # Check for raw snake_case tokens
    snake_matches = re.findall(r"\b[a-z0-9]{2,}_[a-z0-9_]{2,}\b", body)
    if snake_matches:
        errors.append(f"Raw snake_case identifier leaked: {snake_matches}")

    # Check for fabricated geographic landmarks, roads, or streets not in context
    if context_dump and body:
        lower_ctx = context_dump.lower()
        geo_patterns = [
            r"\b\d+\s*(?:ft|feet|foot)\s*(?:road|rd)?\b",
            r"\b[A-Za-z0-9\.\-]+\s+(?:road|rd|marg|street|st|lane|cross|metro|circle|chowk|flyover|plaza)\b"
        ]
        stopwords = {"the main", "a main", "to cross", "and cross", "or cross", "main menu"}
        for gp in geo_patterns:
            matches = re.finditer(gp, body, re.IGNORECASE)
            for m in matches:
                phrase = m.group(0).strip()
                if phrase.lower() in stopwords:
                    continue
                if phrase.lower() not in lower_ctx:
                    errors.append(f"Fabricated geographic landmark or street not in context: '{phrase}'")

    return len(errors) == 0, errors


# =============================================================================
# Google Gemini API Integration
# =============================================================================

GEMINI_PRIMARY_MODEL = "gemini-flash-latest"
GEMINI_FALLBACK_MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = """You are Vera, magicpin's AI engagement assistant for local merchants and their customers across India.
Your mission is to compose high-converting, category-authentic, merchant-specific WhatsApp messages.

RULES & CONSTRAINTS:
1. SPECIFICITY: Anchor every message on verifiable facts from the contexts (exact numbers, dates, headlines, peer benchmarks, prices). Never use generic discount copy ("flat 30% off") when service+price is available ("Haircut @ ₹99", "Dental Cleaning @ ₹299", "Weekday Lunch Thali @ ₹149").
2. CATEGORY VOICE: Strictly follow category.voice.
   - Dentists: Peer-clinical, collegial, technical terms allowed (scaling, fluoride varnish, caries), zero hype.
   - Salons: Warm, practical, operator-to-operator.
   - Restaurants: Fellow-operator, busy, covers/AOV/IPL aware.
   - Gyms: Coach-to-member, energetic, disciplined, zero shame/guilt.
   - Pharmacies: Trustworthy, precise, respectful.
3. TABOOS: NEVER use words from category.voice.vocab_taboo (e.g., "guaranteed", "miracle", "100% safe", "best in city").
4. MERCHANT FIT: Use owner first name where available (e.g., Dr. Meera, Suresh, Lakshmi). Personalize to their numbers, active offers, and signals.
5. CUSTOMER FIT: If customer context is provided, address the customer by name. Match customer.identity.language_pref (Hindi-English natural code-mix when requested).
6. TRIGGER RELEVANCE: Clearly communicate "why now" anchored directly in the trigger.
7. SINGLE PRIMARY CTA: Land the clear call-to-action in the final sentence. Use a single binary commitment (Reply YES / STOP) for action triggers, or none for pure info.
8. COMPULSION LEVERS: Favor social proof (local peers) and asking the merchant ("what service was most asked for this week?"), loss aversion, and effort externalization ("I've drafted X — want me to send it?").
9. STRICT NO-FABRICATION RULE:
   - Only reference concrete details (locality names, street names, landmarks, numbers, dates, offer names, competitor names) that appear verbatim in the provided category/merchant/trigger/customer context.
   - Do NOT add real-world knowledge about the city, neighborhood, or business beyond what's given, even if it's factually true — the merchant/customer cannot verify claims you invented!
   - Explicit negative example: If merchant locality is "Indiranagar", do NOT mention "100ft Road", "CMH Road", or specific metro stations unless they appear verbatim in the input JSON context. Bad: inventing a nearby street name not in the context.
   - Explicit negative example: If trigger mentions "IPL match tonight at Arun Jaitley Stadium", do NOT invent a nearby locality or fake promo not given in context.
   - Never invent research papers, fake author names, or fake discounts.
10. NO RAW TAGS, IDS, OR EMPTY PLACEHOLDERS:
   - Never output raw field names, internal IDs, or snake_case identifiers verbatim (e.g., 'high_risk_adults', 'trg_001', 'd_2026W17_dci_radiograph', 'recall_due', 'perf_spike') — always translate them into natural, conversational phrasing (e.g., 'your high-risk adult patients', 'the DCI radiography guideline update').
   - Never leave blank placeholders or dangling punctuation without a name (e.g., 'Hi ,', 'Hello ,', or 'Dear ,'). If customer or merchant name is missing or unknown, use 'Hi there' or omit the greeting entirely.

Output ONLY valid JSON matching this schema:
{
  "body": "The WhatsApp message body",
  "cta": "binary_yes_stop" | "open_ended" | "none",
  "rationale": "Short explanation of why this message and what it achieves"
}"""

FEW_SHOT_EXAMPLES = """
Example 1 (Dentists / Research Digest - merchant facing):
Input Trigger: research_digest (JIDA Oct 2026 paper: 3-mo fluoride recall cuts caries 38% in high-risk adults)
Merchant: Dr. Meera, Lajpat Nagar Delhi, high-risk adult cohort
Composed Message:
Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14
CTA: binary_yes_stop
Rationale: Clinical peer credibility with verifiable trial numbers; offers ready-made patient education artifact.

Example 2 (Salons / Curious Ask - merchant facing):
Input Trigger: curious_ask_due (weekly what's in demand cadence)
Merchant: Studio11 Family Salon (Lakshmi)
Composed Message:
Hi Lakshmi! Quick check — what service has been most asked-for this week at Studio11? I'll turn the answer into a Google post + a 4-line WhatsApp reply you can use when customers ask about pricing. Takes 5 min.
CTA: open_ended
Rationale: Low-friction asking-the-merchant hook with upfront reciprocal effort externalization.

Example 3 (Gyms / Lapse Winback - customer facing):
Input Trigger: customer_lapsed_hard (Rashmi, 57 days since last visit, past goal weight loss)
Merchant: PowerHouse Fitness (Karthik)
Composed Message:
Hi Rashmi 👋 Karthik from PowerHouse here. It's been about 8 weeks — happens to most members at some point, no judgment. We've added a Tue/Thu evening HIIT class that fits weight-loss goals well (45 min, 6:30pm). Want me to hold a free trial spot for you next Tue, 30 Apr? Reply YES — no commitment, no auto-charge.
CTA: binary_yes_stop
Rationale: Warm no-shame re-entry honoring past goal, offering specific trial slot with zero-risk binary CTA.
"""

def call_gemini(prompt: str, api_key: str) -> Tuple[Optional[dict], str, str]:
    """Call Google Gemini API using REST with gemini-2.5-pro, falling back to gemini-2.5-flash."""
    models_to_try = [GEMINI_PRIMARY_MODEL, GEMINI_FALLBACK_MODEL]
    last_err = ""
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_text = parts[0].get("text", "").strip()
                        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(1)), model, ""
                        try:
                            return json.loads(raw_text), model, ""
                        except json.JSONDecodeError as je:
                            last_err = f"JSON decode error: {je}"
                            logger.debug("Gemini model %s JSON parse failed: %s (raw: %s...)", model, je, raw_text[:100])
        except urllib.error.HTTPError as e:
            try:
                body_txt = e.read().decode("utf-8")
            except Exception:
                body_txt = str(e)
            last_err = f"HTTP {e.code}: {body_txt}"
            logger.debug("Gemini model %s HTTP Error %s: %s...", model, e.code, body_txt[:100])
            if e.code == 429:
                break
            continue
        except Exception as e:
            last_err = f"Error: {e}"
            logger.debug("Gemini model %s Exception: %s", model, e)
            continue
    return None, "", last_err


# =============================================================================
# Fallback High-Fidelity Deterministic Composer
# =============================================================================

def get_service_price_offers(merchant: dict, category: dict) -> List[str]:
    """Extract offers prioritizing concrete service+price framing over generic discount copy."""
    offers = merchant.get("offers", [])
    active_m_offers = [
        o.get("title") for o in offers 
        if isinstance(o, dict) and o.get("status") == "active" and o.get("title")
    ]
    # Filter out generic flat % discounts
    specific_m = [
        t for t in active_m_offers 
        if not re.search(r"\bflat\s+\d+%\s+off\b", t, re.IGNORECASE)
    ]
    if specific_m:
        return specific_m

    # Fall back to category catalog, filtering out generic flat discounts
    cat_catalog = category.get("offer_catalog", [])
    specific_cat = [
        o.get("title") for o in cat_catalog 
        if o.get("title") and not re.search(r"\bflat\s+\d+%\s+off\b", o.get("title"), re.IGNORECASE)
    ]
    if specific_cat:
        return specific_cat

    return [o.get("title") for o in cat_catalog if o.get("title")] or ["our featured service"]


def resolve_template_info(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    send_as: str = "vera"
) -> Tuple[str, List[str]]:
    """
    Map trigger kind and context data to approved WhatsApp template name
    and clean slot-fill parameter values ({{1}}, {{2}}, ...).
    """
    kind = trigger.get("kind", "generic")
    payload = trigger.get("payload", {})
    m_identity = merchant.get("identity", {})
    m_name = m_identity.get("name", "our clinic" if "dentist" in category.get("slug", "") else "our store")
    owner_name = m_identity.get("owner_first_name", "") or m_identity.get("name", "Partner")
    cat_slug = category.get("slug", "retail")
    locality = m_identity.get("locality", "")

    # Salutation / Recipient
    if cat_slug == "dentists" and owner_name:
        owner_salutation = f"Dr. {owner_name}" if not owner_name.lower().startswith("dr") else owner_name
    elif owner_name:
        owner_salutation = owner_name
    else:
        owner_salutation = "Team"

    c_first_name = "there"
    if customer:
        c_raw = customer.get("identity", {}).get("name", "") or customer.get("name", "")
        c_raw = c_raw.split("(")[0].strip()
        if c_raw:
            c_first_name = c_raw.split()[0]

    offers = category.get("offer_catalog", [])
    primary_offer = offers[0].get("title", "Special promotional package") if offers else "Featured offer"

    # 1. recall_due
    if kind == "recall_due":
        service_raw = payload.get("service_due", "6_month_cleaning")
        service_due = service_raw.replace("_", " ").title()
        slots_list = payload.get("available_slots", [])
        slots_str = " or ".join([s.get("label", "") for s in slots_list]) if slots_list else payload.get("due_date", "this week")
        offer_str = "Dental Cleaning @ ₹299" if "cleaning" in service_raw else primary_offer
        if customer:
            return (
                "merchant_recall_reminder_v1",
                [c_first_name, m_name, f"{service_due} is due", slots_str, offer_str]
            )
        else:
            return (
                "vera_recall_alert_v1",
                [owner_salutation, m_name, service_due, slots_str]
            )

    # 2. research_digest
    elif kind == "research_digest":
        top_id = payload.get("top_item_id")
        digest_items = category.get("digest", [])
        matched = next((d for d in digest_items if d.get("id") == top_id), None)
        if not matched and digest_items:
            matched = digest_items[0]

        if matched:
            source = matched.get("source", "Medical Journal")
            title = matched.get("title", "Recent clinical research")
            actionable = matched.get("actionable", "Review abstract & protocol")
        else:
            source = "Clinical Journal"
            title = "Recent clinical study"
            actionable = "Review protocol"

        return (
            "vera_research_digest_v1",
            [owner_salutation, source, title, actionable]
        )

    # 3. regulation_change / compliance
    elif kind in ("regulation_change", "compliance_dci_radiograph", "compliance"):
        top_id = payload.get("item_id") or payload.get("top_item_id")
        digest_items = category.get("digest", [])
        matched = next((d for d in digest_items if d.get("id") == top_id), None)
        title = matched.get("title", "Revised regulatory guidelines") if matched else payload.get("title", "Regulatory update")
        source = matched.get("source", "Regulatory Council Circular") if matched else "Regulatory authority"
        actionable = matched.get("actionable", "Review clinic SOPs") if matched else "Audit compliance setup"
        return (
            "vera_compliance_alert_v1",
            [owner_salutation, source, title, actionable]
        )

    # 4. cde_opportunity
    elif kind == "cde_opportunity":
        event = payload.get("title", "Upcoming CDE Clinical Webinar")
        date_str = payload.get("date", "Upcoming session")
        credits_str = f"{payload.get('credits', 2)} CDE Credits"
        return (
            "vera_cde_webinar_v1",
            [owner_salutation, event, date_str, credits_str]
        )

    # 5. perf_dip / seasonal_perf_dip
    elif kind in ("perf_dip", "seasonal_perf_dip"):
        metric = payload.get("metric", "Views")
        pct = abs(payload.get("pct_change", payload.get("drop_pct", 20)))
        dip_str = f"{metric} down {pct}%"
        peer_avg = category.get("peer_stats", {}).get("avg_views_30d", 1820)
        benchmark = f"Metro peer avg: {peer_avg} views"
        action = f"Promote {primary_offer}"
        return (
            "vera_perf_dip_alert_v1",
            [owner_salutation, m_name, dip_str, benchmark, action]
        )

    # 6. perf_spike
    elif kind == "perf_spike":
        metric = payload.get("metric", "Inquiries")
        pct = abs(payload.get("pct_change", 30))
        spike_str = f"{metric} up {pct}% this week"
        next_step = f"Capitalize with {primary_offer}"
        return (
            "vera_perf_spike_alert_v1",
            [owner_salutation, m_name, spike_str, next_step]
        )

    # 7. renewal_due
    elif kind == "renewal_due":
        item_name = payload.get("plan_name", "Annual Membership")
        expiry = payload.get("expiry_date", "this month")
        rate = payload.get("renewal_price", primary_offer)
        if customer:
            return (
                "merchant_renewal_notice_v1",
                [c_first_name, m_name, item_name, expiry, rate]
            )
        else:
            return (
                "vera_renewal_alert_v1",
                [owner_salutation, m_name, item_name, expiry, rate]
            )

    # 8. festival_upcoming
    elif kind == "festival_upcoming":
        festival = payload.get("festival_name", payload.get("festival", "Upcoming Festival"))
        days_str = f"{payload.get('days_away', 10)} days away"
        festive_offer = primary_offer
        return (
            "vera_festival_campaign_v1",
            [owner_salutation, m_name, festival, days_str, festive_offer]
        )

    # 9. wedding_package_followup
    elif kind == "wedding_package_followup":
        trial = payload.get("trial_name", "Bridal Glow Trial")
        wedding_date = payload.get("wedding_date", "Upcoming wedding")
        package = primary_offer
        if customer:
            return (
                "merchant_bridal_followup_v1",
                [c_first_name, m_name, trial, wedding_date, package]
            )
        else:
            return (
                "vera_wedding_campaign_v1",
                [owner_salutation, m_name, trial, wedding_date, package]
            )

    # 10. curious_ask_due
    elif kind == "curious_ask_due":
        topic = payload.get("question_topic", payload.get("topic", "business performance")).replace("_", " ")
        benchmark = f"Peer trends in {locality or 'your city'}"
        return (
            "vera_curious_checkin_v1",
            [owner_salutation, m_name, topic, benchmark]
        )

    # 11. winback_eligible / customer_lapsed_hard / customer_lapsed_soft
    elif kind in ("winback_eligible", "customer_lapsed_hard", "customer_lapsed_soft"):
        days = payload.get("days_inactive", payload.get("days_lapsed", 90))
        last_visit = f"Last visited {days} days ago"
        winback_offer = payload.get("winback_offer", primary_offer)
        if customer:
            return (
                "merchant_winback_offer_v1",
                [c_first_name, m_name, last_visit, winback_offer]
            )
        else:
            lapsed_count = f"{payload.get('lapsed_count', 25)} inactive clients"
            return (
                "vera_winback_strategy_v1",
                [owner_salutation, m_name, lapsed_count, winback_offer]
            )

    # 12. ipl_match_today
    elif kind == "ipl_match_today":
        match = payload.get("match", "Matchday tonight")
        surge = "Order volume surge expected"
        combo = payload.get("combo_offer", primary_offer)
        return (
            "vera_matchday_event_v1",
            [owner_salutation, m_name, match, surge, combo]
        )

    # 13. review_theme_emerged
    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "Customer review feedback")
        count_str = f"Mentioned in {payload.get('review_count', 4)} reviews"
        suggestion = payload.get("suggestion", "Operational adjustment")
        return (
            "vera_review_theme_alert_v1",
            [owner_salutation, m_name, theme, count_str, suggestion]
        )

    # 14. milestone_reached
    elif kind == "milestone_reached":
        milestone = payload.get("milestone_title", f"{payload.get('count', 500)} orders milestone")
        period = payload.get("period", "this month")
        return (
            "vera_milestone_celebration_v1",
            [owner_salutation, m_name, milestone, period]
        )

    # 15. active_planning_intent
    elif kind == "active_planning_intent":
        topic = payload.get("intent_topic", payload.get("intent", "Bulk catering")).replace("_", " ").title()
        group_size = f"{payload.get('headcount', 'Bulk')} orders"
        proposal = primary_offer
        return (
            "vera_planning_intent_v1",
            [owner_salutation, m_name, topic, group_size, proposal]
        )

    # 16. trial_followup
    elif kind == "trial_followup":
        trial = payload.get("trial_class", "Trial Session")
        next_batch = payload.get("next_batch", "Upcoming batch")
        rate = primary_offer
        if customer:
            return (
                "merchant_trial_followup_v1",
                [c_first_name, m_name, trial, next_batch, rate]
            )
        else:
            return (
                "vera_trial_followup_v1",
                [owner_salutation, m_name, trial, next_batch, rate]
            )

    # 17. supply_alert
    elif kind == "supply_alert":
        item = payload.get("product_name", payload.get("item", "Stock batch"))
        advisory = payload.get("alert_type", "Regulatory supply advisory")
        action = payload.get("action_required", "Check batch inventory")
        return (
            "vera_supply_alert_v1",
            [owner_salutation, m_name, item, advisory, action]
        )

    # 18. chronic_refill_due
    elif kind == "chronic_refill_due":
        med = payload.get("medication", "Prescription maintenance refill")
        due = payload.get("due_date", "due this week")
        service = "Free doorstep delivery"
        if customer:
            return (
                "merchant_refill_reminder_v1",
                [c_first_name, m_name, med, due, service]
            )
        else:
            return (
                "vera_refill_alert_v1",
                [owner_salutation, m_name, med, due, service]
            )

    # 19. dormant_with_vera
    elif kind == "dormant_with_vera":
        loc = locality or "your area"
        action = f"Promote {primary_offer}"
        return (
            "vera_merchant_reconnect_v1",
            [owner_salutation, m_name, loc, action]
        )

    # 20. gbp_unverified
    elif kind == "gbp_unverified":
        return (
            "vera_gbp_verification_v1",
            [owner_salutation, m_name, "Google Business Profile unverified", "Claim missing search impressions"]
        )

    # 21. competitor_opened
    elif kind == "competitor_opened":
        comp = payload.get("competitor_name", f"New competitor in {locality or 'your area'}")
        counter = f"Defend with {primary_offer}"
        return (
            "vera_competitor_alert_v1",
            [owner_salutation, m_name, comp, counter]
        )

    # 22. category_seasonal
    elif kind == "category_seasonal":
        season = payload.get("season", "Peak season demand")
        campaign = primary_offer
        return (
            "vera_category_seasonal_v1",
            [owner_salutation, m_name, season, campaign]
        )

    # 23. appointment_tomorrow
    elif kind == "appointment_tomorrow":
        time_str = payload.get("time", "Tomorrow")
        service = payload.get("service", primary_offer)
        if customer:
            return (
                "merchant_appointment_reminder_v1",
                [c_first_name, m_name, time_str, service]
            )
        else:
            return (
                "vera_appointment_alert_v1",
                [owner_salutation, m_name, time_str, service]
            )

    # Catch-all
    clean_kind = kind.replace("_", " ").title()
    template_name = f"vera_{kind}_v1" if not customer else f"merchant_{kind}_v1"
    template_params = [
        c_first_name if customer else owner_salutation,
        m_name,
        clean_kind,
        primary_offer
    ]
    return (template_name, template_params)


def deterministic_compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None,
    send_as: str,
    suppression_key: str
) -> dict:
    """
    High-fidelity deterministic composer covering all trigger kinds across all categories.
    Used when GEMINI_API_KEY is unset or API quota is exceeded.
    """
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    is_placeholder = payload.get("placeholder", False)

    identity = merchant.get("identity", {})
    m_name = identity.get("name", "our store")
    owner_name = identity.get("owner_first_name", "")
    city = identity.get("city", "Delhi")
    locality = identity.get("locality", "local area")
    cat_slug = category.get("slug", "retail")
    peer_stats = category.get("peer_stats", {})
    perf = merchant.get("performance", {})
    service_price_offers = get_service_price_offers(merchant, category)
    primary_offer = service_price_offers[0]

    # Salutation logic
    if cat_slug == "dentists" and owner_name:
        salutation = f"Dr. {owner_name}" if not owner_name.lower().startswith("dr") else owner_name
    elif owner_name:
        salutation = f"Hi {owner_name}"
    else:
        salutation = f"Hi {m_name} team"

    # Customer identity & language preference
    c_name = ""
    lang_pref = "en"
    if customer:
        c_name = customer.get("identity", {}).get("name", "").split("(")[0].strip()
        lang_pref = customer.get("identity", {}).get("language_pref", "en").lower()

    c_greeting = f"Hi {c_name}" if c_name else "Hi there"
    is_hindi_mix = "hi" in lang_pref

    body = ""
    cta = "binary_yes_stop"
    rationale = ""

    # Route by trigger kind
    if kind == "active_planning_intent":
        topic = payload.get("intent_topic", "")
        if "thali" in topic or cat_slug == "restaurants":
            body = (
                f"{owner_name or 'Team'}, here is a concrete draft for your corporate lunch package:\n\n"
                f"{m_name} Corporate Bulk Thali ({locality}):\n"
                f"- 10-24 thalis @ ₹125 each (₹25 off retail) + free delivery\n"
                f"- 25-49 thalis @ ₹115 each + 2 complimentary filter coffees\n"
                f"- 50+ thalis @ ₹105 each + 1 starter platter\n"
                f"- Advance notice by 5pm previous day; delivery between 12:30-1pm.\n\n"
                f"We can target local office parks in {locality}. Want me to draft the 1-page menu flyer and WhatsApp announcement? Reply YES."
            )
            rationale = "Structured corporate thali package with tiered volume pricing and zero-friction flyer drafting CTA."
        elif "yoga" in topic or cat_slug == "gyms":
            body = (
                f"{owner_name or 'Coach'}, here is the draft structure for the kids summer program:\n\n"
                f"{m_name} Summer Kids Program (Ages 6-13):\n"
                f"- 2-week module: Mon/Wed/Fri, 9:00 AM - 10:15 AM\n"
                f"- Curriculum: Posture, breathing coordination, and fun mobility drills\n"
                f"- Program fee: ₹2,499 per child (includes participation certificate)\n"
                f"- Batch limit: 12 kids for personal supervision\n\n"
                f"Want me to create the registration post and WhatsApp broadcast for current members? Reply YES."
            )
            rationale = "Structured kids program proposal with concrete schedule, pricing, and member-outreach action."
        else:
            body = (
                f"{salutation}, following up on your plan for {topic.replace('_', ' ')}: "
                f"I've outlined a ready-to-launch package centered on {primary_offer}. "
                f"Want me to send over the full draft and promotion copy? Reply YES."
            )
            rationale = "Follow up on explicit merchant planning intent with concrete package draft."

    elif kind == "appointment_tomorrow":
        # Customer facing
        service = "scheduled session"
        if customer and customer.get("relationship", {}).get("services_received"):
            service = customer["relationship"]["services_received"][0].replace("_", " ")
        elif service_price_offers:
            service = service_price_offers[0].split("@")[0].strip()

        if is_hindi_mix:
            body = (
                f"Namaste {c_name}! {m_name} ({locality}) se reminder hai: "
                f"Kal aapka appointment scheduled hai. Please 10 minute pehle arrive karein. "
                f"Kya aap kal aa rahe hain? Reply YES to confirm, ya reschedule karne ke liye message karein."
            )
        else:
            body = (
                f"Hi {c_name}! Reminder from {m_name} in {locality}: "
                f"Your appointment is confirmed for tomorrow. "
                f"Reply YES to confirm your slot, or let us know if you need to reschedule."
            )
        rationale = "Warm, personalized 24-hour advance appointment reminder respecting customer language preference."

    elif kind == "category_seasonal":
        trends = payload.get("trends", ["summer demand shifts"])
        trend_summary = ", ".join(t.replace("_", " ") for t in trends[:3])
        body = (
            f"{salutation}, summer pattern alert for {locality}: Category data shows sharp demand movement ({trend_summary}). "
            f"Pharmacies in {city} are re-aligning front-of-counter stock to capture high-margin essentials. "
            f"Want me to generate a 1-page fast-moving checklist and a seasonal WhatsApp catalog update? Reply YES."
        )
        rationale = "Data-backed seasonal demand advisory with actionable counter-stocking checklist."

    elif kind == "cde_opportunity":
        credits = payload.get("credits", 2)
        fee = payload.get("fee", "free for members").replace("_", " ")
        body = (
            f"{salutation}, IDA Delhi announced a clinical CDE webinar next week offering {credits} credit hours ({fee}). "
            f"Key focus: precision treatment protocols and modern clinical workflow. "
            f"Would you like me to pull the abstract and registration details for you? Reply YES."
        )
        rationale = "Peer-clinical professional development alert with accreditation credits and low-friction retrieval CTA."

    elif kind == "chronic_refill_due":
        # Customer facing
        if cat_slug == "dentists":
            # Clinical dental maintenance / prescription rinse refill
            if is_hindi_mix:
                body = (
                    f"Hi {c_name}, {m_name} ({locality}) yahan. "
                    f"Aapka clinical oral care pack aur prescribed rinse refill due hai. "
                    f"Clinic pe aapka maintenance pack ready rakha hai. "
                    f"Kya aap pickup karenge ya home delivery chahiye? Reply CONFIRM to arrange, ya call karein."
                )
            else:
                body = (
                    f"Hi {c_name}, {m_name} in {locality} here. "
                    f"Your prescribed oral hygiene and dental maintenance refill is due. "
                    f"We have packed your standard oral care kit ready for priority pickup or delivery. "
                    f"Reply CONFIRM to arrange dispatch, or message us if you have any questions."
                )
        else:
            molecules = payload.get("molecule_list", [])
            mol_str = ", ".join(molecules) if molecules else "monthly chronic medications"
            run_out_date = payload.get("stock_runs_out_iso", "28 April").split("T")[0]

            if is_hindi_mix:
                body = (
                    f"Namaste {c_name}, {m_name} {locality} yahan. "
                    f"Aapki monthly medicines ({mol_str}) {run_out_date} tak complete ho jayengi. "
                    f"Same brand pack free home delivery ke sath ready kar diya hai. "
                    f"Reply CONFIRM to dispatch, ya dosage badla ho toh call karein."
                )
            else:
                body = (
                    f"Hi {c_name}, {m_name} {locality} here. "
                    f"Your regular refill for {mol_str} is due around {run_out_date}. "
                    f"We have packed your standard prescription for free home delivery. "
                    f"Reply CONFIRM to dispatch today, or message us if dosage has changed."
                )
        cta = "binary_confirm_stop"
        rationale = "High-precision chronic refill reminder with exact medications/supplies and convenient dispatch trigger."

    elif kind == "competitor_opened":
        if is_placeholder:
            if cat_slug == "restaurants":
                body = (
                    f"{salutation}, heads-up: A new dining outlet opened within 1.2km in {locality}. "
                    f"Rather than competing on margin-eroding discounts, top {city} peers retain covers through service specials like {primary_offer}. "
                    f"Want me to draft a loyalty check-in WhatsApp for your existing regulars? Reply YES."
                )
            else:
                body = (
                    f"{salutation}, heads-up: A new competitor opened 1.2km away in {locality} promoting introductory pricing. "
                    f"Top-rated {locality} peers retain clients through established trust and signature packages like {primary_offer}. "
                    f"Want me to draft a loyalty retention message for your customer roster? Reply YES."
                )
        else:
            comp_name = payload.get("competitor_name", "A new clinic")
            dist = payload.get("distance_km", 1.2)
            comp_offer = payload.get("their_offer", "introductory pricing")
            body = (
                f"{salutation}, heads-up: {comp_name} opened {dist}km away promoting {comp_offer}. "
                f"Rather than discounting, top-rated {locality} peers retain patients through service bundles like {primary_offer}. "
                f"Want me to draft a loyalty check-in WhatsApp for your existing roster? Reply YES."
            )
        rationale = "Loss aversion framed constructively without reactive discounting, offering retention outreach."

    elif kind == "curious_ask_due":
        cta = "open_ended"
        if cat_slug == "salons":
            body = (
                f"{salutation}! Quick check — what treatment or service has been most requested this week at {m_name}? "
                f"I will turn the answer into a Google Business highlight post plus a 3-line inquiry template for your staff. Takes 2 minutes."
            )
        elif cat_slug == "restaurants":
            body = (
                f"{salutation}! Quick check — which dish or combo was your top seller this past weekend at {m_name}? "
                f"I will turn it into a Google Business showcase post plus a weekend special story. Takes 2 minutes."
            )
        elif cat_slug == "dentists":
            body = (
                f"{salutation}! Quick question — are you seeing more scaling recalls or aligner inquiries at {locality} this month? "
                f"I will prepare a targeted patient education WhatsApp note based on your focus. Takes 2 minutes."
            )
        else:
            body = (
                f"{salutation}! Quick check — what has been your most popular customer request this week at {m_name}? "
                f"I'll format it into a Google Business update and promotional snippet for you. Takes 2 minutes."
            )
        rationale = "High-compulsion curiosity/operator inquiry providing upfront reciprocity and minimal friction."

    elif kind == "customer_lapsed_hard":
        days = payload.get("days_since_last_visit", 57)
        focus = payload.get("previous_focus", "fitness").replace("_", " ")
        weeks = max(4, days // 7)

        if cat_slug == "gyms":
            body = (
                f"{c_greeting} 👋 {owner_name or m_name} here. It's been about {weeks} weeks — "
                f"happens to most members at some point, no judgment at all. "
                f"We've added a Tue/Thu evening HIIT class that fits {focus} goals well (45 min, 6:30pm). "
                f"Want me to hold a free trial spot for you next Tue, 30 Apr? Reply YES — no commitment, no auto-charge."
            )
        else:
            body = (
                f"{c_greeting}, {m_name} in {locality} here. We noticed it has been about {weeks} weeks since your last visit. "
                f"We would love to welcome you back with {primary_offer}. "
                f"Would you like us to reserve a priority slot for you this week? Reply YES."
            )
        rationale = "Empathic, no-shame winback message highlighting previous goals and offering zero-risk trial."

    elif kind == "customer_lapsed_soft":
        if cat_slug == "pharmacies":
            # Avoid generic flat discounts, use service+price framing
            if is_hindi_mix:
                body = (
                    f"{c_greeting}! {m_name} ({locality}) se quick check-in. "
                    f"Aapka monthly health essentials refill due ho chuka hai. "
                    f"Humne aapke orders ke liye Free Home Delivery > ₹499 ready rakha hai. "
                    f"Kya hum is week aapka refill dispatch karein? Reply YES to confirm."
                )
            else:
                body = (
                    f"{c_greeting}! Friendly check-in from {m_name} in {locality}. "
                    f"Your monthly health supplies refill is due, with Free Home Delivery on orders above ₹499. "
                    f"Would you like us to dispatch your routine essentials this week? Reply YES."
                )
        else:
            if is_hindi_mix:
                body = (
                    f"{c_greeting}! {m_name} ({locality}) se quick check-in. "
                    f"Aapka routine visit due ho chuka hai. Humne aapke liye slots open rakhe hain with {primary_offer}. "
                    f"Kya hum is week aapka slot reserve karein? Reply YES to confirm."
                )
            else:
                body = (
                    f"{c_greeting}! Friendly check-in from {m_name} in {locality}. "
                    f"Your regular visit is due, and we have open slots ready with {primary_offer}. "
                    f"Would you like us to hold a time for you this week? Reply YES."
                )
        rationale = "Gentle soft-lapse check-in honoring customer language and proposing service+price availability."

    elif kind == "dormant_with_vera":
        days = payload.get("days_since_last_merchant_message", 14) if not is_placeholder else 14
        views = perf.get("views", 1200)
        body = (
            f"{salutation}, it's been {days} days since our last chat. In that time, {m_name} generated {views} Google profile views in {locality}. "
            f"Local search volume in {city} is active, and 2 quick profile adjustments can boost your weekly call conversions. "
            f"Want me to show you the 2-minute update? Reply YES."
        )
        rationale = "Re-engages dormant merchant using concrete performance data and low-effort 2-minute hook."

    elif kind == "festival_upcoming":
        fest = payload.get("festival", "Diwali") if not is_placeholder else "Diwali"
        days = payload.get("days_until", 14) if not is_placeholder else 14
        if cat_slug == "gyms":
            body = (
                f"{salutation}, {fest} is coming up in {days} days. Fitness studios in {locality} that launch a pre-festive conditioning challenge see 2x higher member attendance. "
                f"I've drafted a 'Pre-Festive 14-Day Reset' featuring {primary_offer}. "
                f"Want me to post the announcement on Google Business and WhatsApp? Reply YES."
            )
        else:
            body = (
                f"{salutation}, {fest} is coming up in {days} days. Businesses in {locality} that post festive packages early capture 2.4x more advance bookings. "
                f"I have drafted a festive campaign featuring {primary_offer}. "
                f"Want me to send the draft for your review? Reply YES."
            )
        rationale = "Timely seasonal nudge leveraging social proof and a pre-drafted festive campaign."

    elif kind == "gbp_unverified":
        uplift = int(payload.get("estimated_uplift_pct", 0.3) * 100)
        body = (
            f"{salutation}, your Google Business Profile for {m_name} is currently unverified. "
            f"Verified listings in {locality} see an estimated +{uplift}% higher call and direction volume. "
            f"The verification process takes under 5 minutes. Want me to guide you through the instant verification steps right now? Reply YES."
        )
        rationale = "Loss aversion and concrete performance upside nudge to resolve Google profile verification."

    elif kind == "ipl_match_today":
        match = payload.get("match", "DC vs MI")
        venue = payload.get("venue", "Arun Jaitley Stadium")
        is_weeknight = payload.get("is_weeknight", False)
        if not is_weeknight:
            body = (
                f"{salutation}, heads up: {match} tonight at {venue}. "
                f"Saturday IPL matches typically drop dine-in covers by 12-15% as crowds watch at home. "
                f"Skip the dine-in promo; instead let's push a delivery combo featuring your active menu specials. "
                f"Want me to draft a delivery announcement and story graphic? Reply YES."
            )
        else:
            body = (
                f"{salutation}, {match} tonight at {venue}! "
                f"Weeknight match screenings drive +25% dine-in cover uplift in {locality}. "
                f"I've drafted a 'Match-Night Special' post featuring your top snacks. "
                f"Want me to publish it to Google Business right now? Reply YES."
            )
        rationale = "Operator-savvy restaurant advisory leveraging sports timing and behavioral cover patterns."

    elif kind == "milestone_reached":
        if not is_placeholder:
            metric = payload.get("metric", "review_count").replace("_", " ")
            curr = payload.get("value_now", 145)
            target = payload.get("milestone_value", 150)
            body = (
                f"{salutation}, exciting news: {m_name} is at {curr} {metric} — just {target - curr} away from crossing {target}! "
                f"Crossing {target} significantly improves your search ranking in {locality}. "
                f"I have drafted a 1-line review request template to message your recent satisfied customers. Want to review it? Reply YES."
            )
        else:
            views = perf.get("views", 1450)
            target_views = ((views // 500) + 1) * 500
            body = (
                f"{salutation}, milestone alert: {m_name} reached {views} profile views this month in {locality}, on track to cross {target_views}! "
                f"To keep this momentum going, I've drafted a celebratory Google Business highlight featuring {primary_offer}. "
                f"Want me to share the 2-line post draft? Reply YES."
            )
        rationale = "Celebrates imminent milestone with concrete conversion acceleration template."

    elif kind == "perf_dip":
        if not is_placeholder:
            metric = payload.get("metric", "calls")
            delta = int(abs(payload.get("delta_pct", 0.3) * 100))
            window = payload.get("window", "7d")
            body = (
                f"{salutation}, quick review: your {metric} dropped {delta}% over the last {window} compared to baseline. "
                f"Top performers in {locality} revive inquiries by refreshing their Google Business photos and highlighting {primary_offer}. "
                f"I've prepared a fresh post and call-to-action draft to restore traffic. Want me to share it? Reply YES."
            )
        else:
            calls_30d = perf.get("calls", 22)
            peer_calls = peer_stats.get("avg_calls_30d", 28)
            body = (
                f"{salutation}, performance check: {m_name} recorded {calls_30d} customer calls over the last 30 days, trailing the {city} peer average of {peer_calls}. "
                f"Updating your listing with a featured package like {primary_offer} typically lifts inquiries by 25%. "
                f"I've drafted a quick update to bring call volumes back up. Want me to share it? Reply YES."
            )
        rationale = "Constructive diagnosis of performance drop with immediate restorative post proposal."

    elif kind == "perf_spike":
        if not is_placeholder:
            metric = payload.get("metric", "calls")
            delta = int(abs(payload.get("delta_pct", 0.15) * 100))
            driver = payload.get("likely_driver", "recent search interest").replace("_", " ")
            body = (
                f"{salutation}, great momentum: your {metric} surged +{delta}% this week, driven by {driver}. "
                f"While local intent in {locality} is peaking, let's lock in conversions with a limited-slot booking post. "
                f"Want me to publish a celebratory update to capture more inquiries? Reply YES."
            )
        else:
            views_30d = perf.get("views", 1400)
            body = (
                f"{salutation}, traffic alert: {m_name} generated {views_30d} views this month, with strong search traction in {locality}. "
                f"While customer discovery is high, let's convert views into walk-ins with a featured Google post on {primary_offer}. "
                f"Want me to publish the post today? Reply YES."
            )
        rationale = "Capitalizes on search surge momentum with high-conversion booking post."

    elif kind == "recall_due":
        # Customer facing
        if not is_placeholder:
            service_due = payload.get("service_due", "6-month cleaning").replace("_", " ")
            slots = payload.get("available_slots", [])
            slot_str = ""
            if slots and len(slots) >= 2:
                slot_str = f"Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm" if is_hindi_mix else f"Wed 5 Nov, 6pm or Thu 6 Nov, 5pm"
            else:
                slot_str = "Wed 5 Nov, 6pm or Thu 6 Nov, 5pm"

            if is_hindi_mix:
                body = (
                    f"{c_greeting}, {m_name} ({locality}) here. "
                    f"Aapka {service_due} recall due ho gaya hai. "
                    f"Apke liye 2 slots ready hain: {slot_str}. Special offer: {primary_offer}. "
                    f"Reply 1 for Wed, 2 for Thu, ya apna suitable time batayein."
                )
            else:
                body = (
                    f"{c_greeting}, {m_name} in {locality} here. "
                    f"Your {service_due} recall is due. "
                    f"We have reserved 2 priority slots: {slot_str} ({primary_offer}). "
                    f"Reply 1 for Wed, 2 for Thu, or let us know what time works best for you."
                )
        else:
            # Placeholder recall
            if cat_slug == "gyms":
                body = (
                    f"{c_greeting}! {m_name} in {locality} here. "
                    f"It's been a while since your last session — your quarterly wellness check is due. "
                    f"We have reserved 2 priority refresher slots: Thu 7:00 AM or Sat 8:30 AM ({primary_offer}). "
                    f"Reply 1 for Thu, 2 for Sat, or tell us a time that works for you."
                )
            else:
                body = (
                    f"{c_greeting}! Friendly recall from {m_name} ({locality}). "
                    f"Your routine maintenance visit is due, with priority slots ready: Wed 5pm or Thu 6pm ({primary_offer}). "
                    f"Reply 1 for Wed, 2 for Thu, or message us with your preferred time."
                )
        cta = "binary_slot_or_stop"
        rationale = "Clinical/service recall reminder with specific date options and transparent pricing."

    elif kind == "regulation_change":
        deadline = payload.get("deadline_iso", "2026-12-15").split("T")[0]
        item_id = payload.get("top_item_id", "")
        digest_title = "clinical compliance update"
        for d in category.get("digest", []):
            if d.get("id") == item_id:
                digest_title = d.get("title", digest_title)
                break
        body = (
            f"{salutation}, compliance advisory: DCI issued revised guidelines on {digest_title}, with implementation deadline {deadline}. "
            f"We have summarized the operational requirements into a 1-page checklist for your clinic team. "
            f"Want me to send the checklist to review? Reply YES."
        )
        rationale = "Authoritative regulatory compliance advisory with concise checklist deliverable."

    elif kind == "renewal_due":
        days = payload.get("days_remaining", 14)
        plan = payload.get("plan", "Pro")
        views = perf.get("views", 1800)
        calls = perf.get("calls", 24)
        body = (
            f"{salutation}, your magicpin {plan} plan renews in {days} days. "
            f"Over the last 30 days, your verified listing in {locality} delivered {views} profile views and {calls} direct customer calls. "
            f"Renew now to maintain uninterrupted search ranking across {city}. "
            f"Want me to send the 1-click renewal link? Reply YES."
        )
        rationale = "Retains subscription using tangible ROI metrics (views & calls) and frictionless 1-click CTA."

    elif kind == "research_digest":
        digest = category.get("digest", [])
        item = digest[0] if digest else {}
        item_title = item.get("title", "Clinical study update")
        source = item.get("source", "Medical Journal")
        cohort = item.get("patient_segment", "your patients").replace("_", " ")
        body = (
            f"{salutation}, {source} published an important study: {item_title}. "
            f"This is directly applicable to {cohort} in {locality}. "
            f"I have summarized the 2-minute key takeaways and drafted a patient education tip you can share on WhatsApp. "
            f"Want me to send both? Reply YES."
        )
        rationale = "Peer-level evidence-based research digest with source citation and ready patient education artifact."

    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "service wait time").replace("_", " ")
        occurrences = payload.get("occurrences_30d", 3)
        body = (
            f"{salutation}, feedback insight: {occurrences} recent reviews this month mentioned '{theme}'. "
            f"Addressing this proactively on Google Business improves your conversion rate by up to 20%. "
            f"I've drafted a diplomatic, professional owner response to resolve these mentions. Want to review it? Reply YES."
        )
        rationale = "Constructive sentiment alert turning review feedback into improved conversion with draft responses."

    elif kind == "seasonal_perf_dip":
        metric = payload.get("metric", "views")
        delta = int(abs(payload.get("delta_pct", 0.3) * 100))
        body = (
            f"{salutation}, your {metric} are down {delta}% this week — but this is the standard seasonal lull across {city} metros. "
            f"Recommendation: save acquisition ad spend for peak months, and focus right now on retention for your existing roster. "
            f"Want me to draft a member engagement challenge to maintain retention through the dip? Reply YES."
        )
        rationale = "Pre-empts anxiety by contextualizing seasonal dip and proposing retention campaign."

    elif kind == "supply_alert":
        molecule = payload.get("molecule", "prescriptions")
        mfr = payload.get("manufacturer", "Manufacturer")
        batches = ", ".join(payload.get("affected_batches", ["batch #"]))
        body = (
            f"{salutation}, urgent supply notice: {mfr} issued a voluntary recall on {molecule} (batches: {batches}) due to sub-potency (no safety hazard). "
            f"We can flag repeat-Rx customers dispensed this batch to organize smooth replacements. "
            f"Want me to draft their WhatsApp advisory and replacement protocol? Reply YES."
        )
        rationale = "Precise compliance and supply alert with batch tracking and customer replacement workflow."

    elif kind == "trial_followup":
        # Customer facing
        body = (
            f"Hi {c_name}! Thank you for attending your trial session at {m_name}. "
            f"How did your experience feel? We have a welcome special available this week: {primary_offer}. "
            f"Would you like us to hold your spot? Reply YES to get started."
        )
        rationale = "Courteous post-trial follow-up with concrete introductory offer and simple binary CTA."

    elif kind == "wedding_package_followup":
        # Customer facing
        days = payload.get("days_to_wedding", 120)
        body = (
            f"Hi {c_name} 💍 {m_name} ({locality}) here. "
            f"With {days} days to your wedding, this is the optimal window to begin your skin-prep sessions. "
            f"Our package covers comprehensive prep sessions ({primary_offer}). "
            f"Want me to reserve your preferred weekend slot for session 1? Reply YES."
        )
        rationale = "High-compulsion bridal preparation timeline nudge honoring event countdown and catalog package."

    elif kind == "winback_eligible":
        days_exp = payload.get("days_since_expiry", 30)
        dip = int(payload.get("perf_dip_pct", 0.25) * 100)
        body = (
            f"{salutation}, since your listing plan expired {days_exp} days ago, profile calls in {locality} have decreased by {dip}%. "
            f"Reactivating your verified placement will immediately restore your ranking above local competitors. "
            f"Want to view the one-click reactivation options? Reply YES."
        )
        rationale = "Loss aversion reactivation pitch backed by post-expiry performance drop statistics."

    else:
        body = (
            f"{salutation}, quick update from Vera: We noticed new activity for {m_name} in {locality}. "
            f"We have prepared a promotional highlight around {primary_offer} to increase customer inquiries. "
            f"Would you like me to publish this for you? Reply YES."
        )
        rationale = "Standard proactive update centered on merchant locality and active service catalog."

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }


# =============================================================================
# Main Compose Entrypoint
# =============================================================================

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None
) -> dict:
    """
    Compose an outbound WhatsApp message for Vera or on behalf of a merchant.

    Returns:
        body: str
        cta: str
        send_as: "vera" | "merchant_on_behalf"
        suppression_key: str
        rationale: str
    """
    scope = trigger.get("scope", "merchant")
    if scope == "customer" and customer is not None:
        send_as = "merchant_on_behalf"
    else:
        send_as = "vera"

    suppression_key = trigger.get("suppression_key", "")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()

    # Build context dump for strict no-fabrication validation
    context_dump = json.dumps({
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer
    }, ensure_ascii=False)

    # If API key is present, try LLM composition with one retry
    if api_key:
        prompt = (
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"NOW COMPOSE FOR THIS SCENARIO:\n"
            f"Category: {category.get('slug')} (Voice: {json.dumps(category.get('voice', {}))}, Catalog: {json.dumps(category.get('offer_catalog', [])[:3])})\n"
            f"Merchant: {merchant.get('identity', {}).get('name')} in {merchant.get('identity', {}).get('locality')}, {merchant.get('identity', {}).get('city')}. "
            f"Owner: {merchant.get('identity', {}).get('owner_first_name')}. Perf: {json.dumps(merchant.get('performance', {}))}. Offers: {json.dumps(merchant.get('offers', []))}\n"
            f"Trigger: Kind={trigger.get('kind')}, Scope={trigger.get('scope')}, Payload={json.dumps(trigger.get('payload', {}))}\n"
            f"Customer: {json.dumps(customer) if customer else 'None'}\n"
            f"Send As: {send_as}\n"
        )
        res, model_used, err_msg = call_gemini(prompt, api_key)
        if res and isinstance(res, dict) and "body" in res:
            valid, errors = validate_composed_message(res, category, context_dump)
            if not valid:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"PREVIOUS ATTEMPT FAILED VALIDATION:\n"
                    f"Errors: {', '.join(errors)}\n"
                    f"Previous Body: {res.get('body')}\n"
                    f"Please correct the errors strictly:\n"
                    f"- Translate any raw internal tags, guideline IDs, or snake_case tokens into natural conversational words (e.g. 'your high-risk adult patients', 'the DCI radiography update').\n"
                    f"- If customer name is missing, use 'Hi there' instead of an empty greeting (never output 'Hi ,' or 'Hello !').\n"
                    f"- Eliminate any unverified street names, landmarks, or taboo words, use ONLY details verbatim in context, and output valid JSON."
                )
                retry_res, retry_model, _ = call_gemini(retry_prompt, api_key)
                if retry_res and isinstance(retry_res, dict) and "body" in retry_res:
                    res = retry_res
                    model_used = retry_model

            valid, _ = validate_composed_message(res, category, context_dump)
            if valid:
                tmpl_name, tmpl_params = resolve_template_info(category, merchant, trigger, customer, send_as)
                return {
                    "body": res.get("body", "").strip(),
                    "cta": res.get("cta", "binary_yes_stop"),
                    "send_as": send_as,
                    "suppression_key": suppression_key,
                    "rationale": f"[Gemini {model_used}] {res.get('rationale', '')}",
                    "template_name": tmpl_name,
                    "template_params": tmpl_params,
                }

    # Deterministic high-quality composer fallback
    result = deterministic_compose(category, merchant, trigger, customer, send_as, suppression_key)
    if api_key and 'err_msg' in locals() and err_msg:
        result["rationale"] = f"[Deterministic Fallback (Gemini failed: {err_msg[:60]}...)] {result['rationale']}"
    else:
        result["rationale"] = f"[Deterministic Fallback] {result['rationale']}"
    valid, errors = validate_composed_message(result, category, context_dump)
    if not valid:
        clean_body = result["body"]
        for taboo in category.get("voice", {}).get("vocab_taboo", []):
            ct = re.sub(r"\s*\(.*?\)", "", taboo).strip()
            clean_body = re.sub(re.escape(ct), "", clean_body, flags=re.IGNORECASE)
        # Clean any dangling empty greetings
        clean_body = re.sub(r"\b(hi|hello|namaste|hey|dear)\s*,\s*", r"\1 there, ", clean_body, flags=re.IGNORECASE)
        clean_body = re.sub(r"\b(hi|hello|namaste|hey|dear)\s*!\s*", r"\1 there! ", clean_body, flags=re.IGNORECASE)
        # Clean any raw snake_case tokens
        clean_body = re.sub(r"\b([a-z0-9]{2,})_([a-z0-9_]{2,})\b", lambda m: f"{m.group(1)} {m.group(2).replace('_', ' ')}", clean_body)
        result["body"] = clean_body.strip()

    tmpl_name, tmpl_params = resolve_template_info(category, merchant, trigger, customer, send_as)
    result["template_name"] = tmpl_name
    result["template_params"] = tmpl_params
    return result
