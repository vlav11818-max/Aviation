"""Batch-translate the Russian narrative fields in
``resources/aviation/incidents.yaml`` to English.

Rule-based parsing leaves some fields half-translated (mixed
Cyrillic/Latin). Run this script when you have a real LLM key
configured (any provider LiteLLM supports); it walks every incident
whose ``translation_status`` is ``"partial"`` and rewrites the
narrative fields in place, then flips the status to ``"clean"``.

Usage:

    OPENAI_API_KEY=... python scripts/translate_incidents.py
    OPENAI_API_KEY=... python scripts/translate_incidents.py --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm import call_llm  # noqa: E402


NARRATIVE_FIELDS = ("failure_type", "outcome", "casualties", "twist_potential")
LIST_NARRATIVE_FIELDS = ("dramatic_details",)


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[а-яА-ЯёЁ]", text or ""))


async def translate_incident(inc: dict, model: str) -> dict:
    """Translate the mixed-language narrative fields on one incident."""
    payload = {
        k: inc.get(k, "") for k in NARRATIVE_FIELDS
    }
    payload["dramatic_details"] = inc.get("dramatic_details", []) or []
    prompt = (
        "You are translating an aviation-incident record from mixed "
        "Russian/English into clean English. Preserve every aviation "
        "term (ATC phraseology, ICAO codes, aircraft designators, unit "
        "abbreviations). Preserve proper names. Keep the same terse "
        "reference tone — this is a data record, not prose. Do NOT add "
        "or invent details. Return ONLY the JSON object below with each "
        "field translated into English.\n\n"
        f"INCIDENT NAME: {inc.get('name', '')}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return this exact JSON schema (values in English):\n"
        "{\n"
        '  "failure_type": "...",\n'
        '  "outcome": "...",\n'
        '  "casualties": "...",\n'
        '  "twist_potential": "...",\n'
        '  "dramatic_details": ["...", "..."]\n'
        "}"
    )
    resp = await call_llm(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2500,
        json_mode=True,
    )
    try:
        # Extract the JSON payload (be defensive).
        m = re.search(r"\{.*\}", resp.text, re.S)
        data = json.loads(m.group(0) if m else resp.text)
    except Exception as exc:
        print(f"  ! parse failed for {inc.get('name')}: {exc}", file=sys.stderr)
        return inc
    for f in NARRATIVE_FIELDS:
        if f in data and isinstance(data[f], str):
            inc[f] = data[f].strip()
    if "dramatic_details" in data and isinstance(data["dramatic_details"], list):
        inc["dramatic_details"] = [str(x).strip() for x in data["dramatic_details"] if str(x).strip()]
    inc["translation_status"] = "clean" if not _still_needs(inc) else "partial"
    return inc


def _still_needs(inc: dict) -> bool:
    for f in NARRATIVE_FIELDS:
        if _has_cyrillic(inc.get(f) or ""):
            return True
    for d in inc.get("dramatic_details", []) or []:
        if _has_cyrillic(d):
            return True
    return False


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini",
                    help="LiteLLM model id (needs the matching env-var key)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Translate at most this many partials (0 = all)")
    ap.add_argument("--path", default="resources/aviation/incidents.yaml")
    args = ap.parse_args()

    path = Path(args.path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    incidents = data.get("incidents") or []
    partials = [i for i in incidents if i.get("translation_status") == "partial"]
    if not partials:
        print("Nothing to translate — all incidents are marked 'clean'.")
        return
    to_do = partials if args.limit == 0 else partials[: args.limit]
    print(f"Translating {len(to_do)} incident(s) using {args.model}…")
    for i, inc in enumerate(to_do, start=1):
        print(f"  [{i:>3}/{len(to_do)}] {inc.get('name', '')[:80]}")
        try:
            await translate_incident(inc, args.model)
        except Exception as exc:
            print(f"    ! {type(exc).__name__}: {exc}")
    path.write_text(
        yaml.safe_dump({"incidents": incidents}, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )
    remaining = sum(1 for i in incidents if i.get("translation_status") == "partial")
    print(f"Done. {remaining} incident(s) still marked 'partial'.")


if __name__ == "__main__":
    asyncio.run(main())
