#!/usr/bin/env python3
"""
eval.py — LLM-as-judge evaluation for the voice shopping bot /process endpoint.

Usage (from backend/):
    ANTHROPIC_API_KEY=sk-ant-... python3 eval.py

Runs 15 curated test cases against the live FastAPI server, scores each
response with Claude as judge, and writes a full report to eval_report.json.
"""

import json
import re
import sys
from datetime import datetime

import anthropic
import httpx

SERVER_URL  = "http://localhost:8000"
JUDGE_MODEL = "claude-sonnet-4-5"
PASS_THRESHOLD = 7          # judge score >= this counts as a pass

# ── Eval dataset ───────────────────────────────────────────────────────────
# 15 samples across 6 scenario types:
#   A  add single item           B  add multiple items
#   C  get offers                D  greeting / other
#   E  Hindi / Hinglish input    F  quantity edge cases

EVAL_DATASET = [
    # ── A: single item ─────────────────────────────────────────────────────
    {
        "id": 1, "type": "A",
        "input": "I need 10 kg sugar",
        "expected_intent": "add_to_cart",
        "notes": "baseline single-item English",
    },
    {
        "id": 2, "type": "A",
        "input": "give me 2 litres of mustard oil",
        "expected_intent": "add_to_cart",
        "notes": "liquid unit (litre)",
    },

    # ── B: multiple items ───────────────────────────────────────────────────
    {
        "id": 3, "type": "B",
        "input": "I want to order 2 dozen eggs and 5 litres milk",
        "expected_intent": "add_to_cart",
        "notes": "multiple items, dozen + litre units",
    },
    {
        "id": 4, "type": "B",
        "input": "add 500 gram sugar and 1 litre sunflower oil to my order",
        "expected_intent": "add_to_cart",
        "notes": "gram unit (not kg), two items",
    },
    {
        "id": 5, "type": "B",
        "input": "1 packet salt, 3 packets biscuits, and 2 boxes matchsticks",
        "expected_intent": "add_to_cart",
        "notes": "packet + box units, three items",
    },

    # ── C: get offers ───────────────────────────────────────────────────────
    {
        "id": 6, "type": "C",
        "input": "do you have any offers today?",
        "expected_intent": "get_offers",
        "notes": "direct offer query",
    },
    {
        "id": 7, "type": "C",
        "input": "what deals are available this week?",
        "expected_intent": "get_offers",
        "notes": "offer query with time context",
    },
    {
        "id": 8, "type": "C",
        "input": "show me your best discounts",
        "expected_intent": "get_offers",
        "notes": "discount synonym",
    },

    # ── D: greeting / other ─────────────────────────────────────────────────
    {
        "id": 9, "type": "D",
        "input": "hello, good morning",
        "expected_intent": "other",
        "notes": "greeting with no shopping intent",
    },
    {
        "id": 10, "type": "D",
        "input": "thank you, that's all for now",
        "expected_intent": "other",
        "notes": "closing statement",
    },

    # ── E: Hindi / Hinglish ─────────────────────────────────────────────────
    {
        "id": 11, "type": "E",
        "input": "mujhe 5 kg atta chahiye",
        "expected_intent": "add_to_cart",
        "notes": "pure Hindi: 'I need 5 kg flour'",
    },
    {
        "id": 12, "type": "E",
        "input": "kya koi discount hai aaj?",
        "expected_intent": "get_offers",
        "notes": "Hindi: 'Is there any discount today?'",
    },
    {
        "id": 13, "type": "E",
        "input": "bhai 3 bag gehun ka atta dena",
        "expected_intent": "add_to_cart",
        "notes": "informal Hindi: 'Bro give me 3 bags of wheat flour'",
    },

    # ── F: quantity edge cases ──────────────────────────────────────────────
    {
        "id": 14, "type": "F",
        "input": "I want 0.5 kg ghee and 2.5 kg maida",
        "expected_intent": "add_to_cart",
        "notes": "decimal quantities",
    },
    {
        "id": 15, "type": "F",
        "input": "mujhe ek dozen kela aur ek dozen anda chahiye",
        "expected_intent": "add_to_cart",
        "notes": "Hindi + English mix: 'I need one dozen bananas and one dozen eggs'",
    },
]

# ── Judge prompt ───────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an evaluator for a voice shopping assistant used by retail shop owners in India.
The assistant handles three intents: add_to_cart, get_offers, other.

You will receive a JSON object with:
- input          : the user's original speech
- notes          : context about the test case
- expected_intent: the correct intent
- actual_intent  : what the system detected
- actual_confidence: the system's confidence
- actual_reply   : the system's response to the user

Score 0–10 using these criteria:
  10   Intent correct AND reply is accurate, complete, and natural
  8–9  Intent correct, reply is good but slightly verbose or imperfect
  6–7  Intent correct, reply has a noticeable issue (missing item, wrong qty)
  4–5  Intent wrong but reply is still somewhat useful or coherent
  2–3  Intent wrong AND reply is generic, unhelpful, or missing key info
  0–1  Completely wrong intent AND misleading or empty reply

Special rules:
- Hindi/Hinglish inputs: reward correct intent detection even if reply is English
- Decimal quantities (0.5 kg, 2.5 kg): reward if captured correctly
- "other" intent: a reply like "I'm not sure how to help" is perfectly correct
- Mixed-intent inputs: be lenient; either reasonable intent is acceptable

Respond ONLY with valid JSON (no markdown fences). Example:
{"score": 8, "reason": "Intent correct. Reply confirmed items were added."}"""

# ── Helpers ────────────────────────────────────────────────────────────────

def _strip_json(text: str) -> dict:
    """Parse JSON from Claude's reply, tolerating markdown code fences."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text.strip())


def check_server() -> None:
    """
    Verify the FastAPI server is reachable before spending API budget on eval.
    Exits with a helpful message if not.
    """
    try:
        r = httpx.get(f"{SERVER_URL}/docs", timeout=4.0)
        r.raise_for_status()
    except httpx.ConnectError:
        print(f"\n❌  Cannot connect to {SERVER_URL}")
        print("    The FastAPI server is not running. Start it with:\n")
        print("        cd backend")
        print("        ANTHROPIC_API_KEY=sk-ant-... venv/bin/uvicorn main:app --reload\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n⚠️  Server health check failed: {e}")
        sys.exit(1)


def call_process(text: str) -> dict:
    """POST /process and return the parsed response dict."""
    r = httpx.post(
        f"{SERVER_URL}/process",
        json={"text": text},
        timeout=90.0,   # /process makes 1–2 Claude calls internally
    )
    r.raise_for_status()
    return r.json()


def judge_response(client: anthropic.Anthropic, sample: dict, result: dict) -> dict:
    """
    Ask Claude to score the bot's response.
    Returns {"score": int, "reason": str}.
    """
    payload = {
        "input":              sample["input"],
        "notes":              sample.get("notes", ""),
        "expected_intent":    sample["expected_intent"],
        "actual_intent":      result.get("intent", ""),
        "actual_confidence":  result.get("confidence", ""),
        "actual_reply":       result.get("reply", ""),
    }
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=250,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
    )
    return _strip_json(response.content[0].text)

# ── Main eval loop ─────────────────────────────────────────────────────────

def run_eval() -> dict:
    check_server()
    client = anthropic.Anthropic()

    results      = []
    failures     = []
    total_score  = 0
    intent_hits  = 0
    judge_passes = 0

    n = len(EVAL_DATASET)

    print("\n🧪  Voice Shopping Bot — LLM-as-Judge Eval")
    print(f"    {n} cases  ·  server: {SERVER_URL}  ·  judge: {JUDGE_MODEL}\n")
    print(f"{'#':>2}  {'Input':<48}  {'Exp':>10}  {'Got':>10}  Score  ✓")
    print("─" * 86)

    for sample in EVAL_DATASET:
        sid  = sample["id"]
        text = sample["input"]
        exp  = sample["expected_intent"]

        # ── Step 1: call the bot ──────────────────────────────────────────
        try:
            result = call_process(text)
        except httpx.HTTPStatusError as e:
            result = {"intent": "error", "confidence": "low",
                      "reply": f"HTTP {e.response.status_code}"}
        except Exception as e:
            result = {"intent": "error", "confidence": "low", "reply": str(e)}

        actual_intent = result.get("intent", "error")
        intent_match  = (actual_intent == exp)
        if intent_match:
            intent_hits += 1

        # ── Step 2: LLM judge ─────────────────────────────────────────────
        try:
            verdict = judge_response(client, sample, result)
            score   = max(0, min(10, int(verdict.get("score", 0))))
            reason  = verdict.get("reason", "")
        except Exception as e:
            score  = 0
            reason = f"Judge error: {e}"

        total_score += score
        if score >= PASS_THRESHOLD:
            judge_passes += 1

        # ── Collect row ───────────────────────────────────────────────────
        row = {
            "id":                sid,
            "type":              sample.get("type", ""),
            "input":             text,
            "notes":             sample.get("notes", ""),
            "expected_intent":   exp,
            "actual_intent":     actual_intent,
            "actual_confidence": result.get("confidence", ""),
            "actual_reply":      result.get("reply", ""),
            "intent_match":      intent_match,
            "judge_score":       score,
            "judge_reason":      reason,
        }
        results.append(row)
        if score < PASS_THRESHOLD or not intent_match:
            failures.append(row)

        # ── Live progress line ────────────────────────────────────────────
        mark = "✓" if (intent_match and score >= PASS_THRESHOLD) else "✗"
        print(f"{sid:>2}  {text[:48]:<48}  {exp:>10}  {actual_intent:>10}  {score:>3}/10  {mark}")

    # ── Summary ───────────────────────────────────────────────────────────
    intent_acc_pct  = round(intent_hits  / n * 100, 1)
    judge_pass_pct  = round(judge_passes / n * 100, 1)
    avg_score       = round(total_score  / n, 2)

    print("─" * 86)
    print("\n📊  Summary")
    print(f"    Intent accuracy  : {intent_hits}/{n}  ({intent_acc_pct}%)")
    print(f"    Avg judge score  : {avg_score} / 10")
    print(f"    Judge pass rate  : {judge_passes}/{n}  ({judge_pass_pct}%)  "
          f"[threshold ≥ {PASS_THRESHOLD}]")

    if failures:
        print(f"\n⚠️   Failures ({len(failures)} case{'s' if len(failures) != 1 else ''}):\n")
        for f in failures:
            match_tag = "intent-ok" if f["intent_match"] else f"intent-wrong (got {f['actual_intent']})"
            print(f"  #{f['id']} [{f['judge_score']}/10]  [{f['type']}]  {f['input']!r}")
            print(f"        {match_tag}")
            print(f"        {f['judge_reason']}\n")
    else:
        print("\n✅  All cases passed!")

    return {
        "summary": {
            "total":                n,
            "intent_correct":       intent_hits,
            "intent_accuracy_pct":  intent_acc_pct,
            "avg_judge_score":      avg_score,
            "judge_pass_count":     judge_passes,
            "judge_pass_rate_pct":  judge_pass_pct,
            "judge_pass_threshold": PASS_THRESHOLD,
            "timestamp":            datetime.utcnow().isoformat() + "Z",
            "server_url":           SERVER_URL,
            "judge_model":          JUDGE_MODEL,
        },
        "results":  results,
        "failures": failures,
    }


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_eval()

    report_path = "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n💾  Report saved → {report_path}\n")
