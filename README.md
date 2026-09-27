# magicpin AI Challenge — Vera Assistant Rebuild

## 1. Approach & Architecture

Vera is rebuilt around the **4-Context Framework** (`CategoryContext`, `MerchantContext`, `TriggerContext`, `CustomerContext`), implementing a production-grade WhatsApp engagement engine in `bot.py` with multi-turn support in `conversation_handlers.py`:

1. **Context Resolution & Scope Routing**: Outbound messages dynamically assign `send_as: "vera"` for merchant operations or `"merchant_on_behalf"` when addressing a customer. Every action preserves `suppression_key` for deduplication.
2. **LLM Composition with Few-Shot Grounding**: When configured, `compose()` calls Google Gemini (`gemini-flash-latest`, temperature=0.0) with domain-authentic few-shot exemplars across all 5 verticals (dentists, gyms, pharmacies, restaurants, salons). Enforces single binary commitment CTAs (Reply YES / STOP) or 2-slot selections in the final sentence, honoring natural Hindi-English code-mix.
3. **Multi-Layer Guardrails & Anti-Fabrication**:
   - **Taboo Scrubbing**: Enforces `CategoryContext.voice.vocab_taboo` (`"guaranteed"`, `"miracle"`, `"100% safe"`).
   - **No-Fabrication Engine**: Rejects unstated geographic landmarks, street names, fabricated statistics, or unverified quotes not traceable to context.
   - **Tag & Greeting Sanitization**: Automatically translates internal snake_case identifiers into natural phrasing and eliminates blank name placeholders.
4. **Deterministic Multi-Vertical Fallback**: If API quotas are exceeded (HTTP 429) or offline evaluation is conducted, the engine seamlessly falls back to a deterministic composer covering all 26 trigger families with verified service+price anchors (`Dental Cleaning @ ₹299`, `Haircut @ ₹99`).

## 2. Key Tradeoffs Made

| Decision | Tradeoff | Rationale |
|---|---|---|
| **Deterministic Fallback vs. Stubs** | High implementation surface across 26 trigger kinds | Guarantees testability, offline evaluation, and zero disruption during API rate limits. |
| **Strict Binary CTAs vs. Multi-choice** | Limits branching conversation paths | Binary commitments (YES/STOP) maximize reply conversion and eliminate decision fatigue on WhatsApp. |
| **Unit Economics over Percentage Discounts** | Requires catalog filtering and price matching | Local customers convert significantly higher on explicit service+price tags than generic "Flat 20% off". |
| **Post-Composition Regex Validation** | Minor latency overhead on failed assertions | Hard guarantees that regulatory taboos and hallucinated streets never reach end users. |

## 3. What Additional Context Would Have Helped Most

1. **Unit Economics & Margin Data**: Service-level gross margins to prioritize high-margin treatments (e.g. aligners/implants vs cleanings) over raw booking volume.
2. **Hourly Capacity & Staff Utilization**: Salon chair occupancy and restaurant cover heatmaps to target off-peak lull slots (e.g. Tuesday 2–4 PM).
3. **Historical WhatsApp Drop-off Triggers**: Exact churn thresholds on message character count and frequency by category.
4. **Merchant Language Preference**: Specifying whether the merchant owner prefers pure Hindi, English, or vernacular code-mix.
