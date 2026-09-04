"""Seed-data loaders for the aviation factory.

Reads ``resources/aviation/*.yaml`` on first use and caches the result.
Everything here is read-only reference data: an incident catalog, a
character-name pool, and a fictional-airline pool.

Small filter helpers (:func:`filter_incidents`, :func:`sample_name`)
let the Planner and Streamlit UI pick a seed that matches the requested
axis values without shipping the whole file into the LLM prompt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml


RESOURCE_DIR = Path("resources/aviation")


@dataclass
class IncidentSeed:
    name: str
    section: str = ""
    date: str = ""
    aircraft: str = ""
    location: str = ""
    failure_type: str = ""
    outcome: str = ""
    casualties: str = ""
    dramatic_details: list[str] = field(default_factory=list)
    sub_genre_primary: str = ""
    sub_genre_secondary: str = ""
    causation_type: str = ""
    aircraft_type: str = ""
    setting_layer2: str = ""
    setting_geographic: str = ""
    monetization_risk: str = "LOW"
    twist_potential: str = ""
    sources: list[str] = field(default_factory=list)
    translation_status: str = "clean"

    def summary_for_prompt(self) -> str:
        """Return a compact block for injection into the planner prompt."""
        lines = [
            f"SEED: {self.name}",
            f"Date: {self.date}",
            f"Aircraft: {self.aircraft}",
            f"Location: {self.location}",
            f"Failure: {self.failure_type}",
            f"Outcome: {self.outcome}",
            f"Casualties: {self.casualties}",
            f"Causation: {self.causation_type}",
            f"Monetization risk: {self.monetization_risk}",
        ]
        if self.dramatic_details:
            lines.append("Dramatic details (pick at least three to preserve):")
            for d in self.dramatic_details:
                lines.append(f"  - {d}")
        if self.twist_potential:
            lines.append(f"Twist potential: {self.twist_potential}")
        if self.sources:
            lines.append("Sources:")
            for s in self.sources:
                lines.append(f"  - {s}")
        return "\n".join(lines)


@dataclass
class NamePoolEntry:
    first_name: str
    last_name: str
    ethnicity: str = ""
    role_hint: str = ""


@dataclass
class AirlinePoolEntry:
    name: str
    region: str = ""
    carrier_type: str = ""


# ── loaders ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_incidents() -> list[IncidentSeed]:
    path = RESOURCE_DIR / "incidents.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[IncidentSeed] = []
    for row in data.get("incidents") or []:
        if not isinstance(row, dict):
            continue
        try:
            seed = IncidentSeed(**{k: v for k, v in row.items() if k in IncidentSeed.__annotations__})
        except TypeError:
            # Skip malformed rows rather than crashing the whole factory.
            continue
        # Missing / blank risk defaults to LOW (the parser leaves it "" when
        # the source cell was empty).
        if not seed.monetization_risk.strip():
            seed.monetization_risk = "LOW"
        out.append(seed)
    return out


@lru_cache(maxsize=1)
def load_names() -> list[NamePoolEntry]:
    path = RESOURCE_DIR / "character_names.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [NamePoolEntry(**r) for r in (data.get("names") or []) if isinstance(r, dict)]


@lru_cache(maxsize=1)
def load_airlines() -> list[AirlinePoolEntry]:
    path = RESOURCE_DIR / "fictional_airlines.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [AirlinePoolEntry(**r) for r in (data.get("airlines") or []) if isinstance(r, dict)]


def reload_all() -> None:
    """Force a re-read of every cached resource file (used by tests)."""
    load_incidents.cache_clear()
    load_names.cache_clear()
    load_airlines.cache_clear()


# ── filters ─────────────────────────────────────────────────────────


def filter_incidents(
    *,
    sub_genre: str | None = None,
    aircraft_type: str | None = None,
    max_risk: str | None = None,  # "LOW" / "MED" / "HIGH"
    excluded_names: Iterable[str] = (),
) -> list[IncidentSeed]:
    """Return incidents matching every provided filter.

    ``max_risk``: `LOW` keeps only LOW; `MED` keeps LOW+MED; `HIGH` keeps all.
    ``excluded_names``: incident.name entries already used (case-insensitive).
    """
    risk_order = {"LOW": 0, "MED": 1, "HIGH": 2}
    threshold = risk_order.get((max_risk or "HIGH").upper(), 2)
    excluded = {n.strip().lower() for n in excluded_names}
    out: list[IncidentSeed] = []
    for inc in load_incidents():
        if sub_genre and inc.sub_genre_primary.strip().lower() != sub_genre.strip().lower() \
                and inc.sub_genre_secondary.strip().lower() != sub_genre.strip().lower():
            continue
        if aircraft_type and inc.aircraft_type.strip().lower() != aircraft_type.strip().lower():
            continue
        if risk_order.get(inc.monetization_risk.upper(), 0) > threshold:
            continue
        if inc.name.strip().lower() in excluded:
            continue
        out.append(inc)
    return out


def sample_names(
    n: int,
    *,
    excluded_first_names: Iterable[str] = (),
    ethnicity_bias: str | None = None,
    rng: random.Random | None = None,
) -> list[NamePoolEntry]:
    """Return ``n`` distinct names not in ``excluded_first_names``.

    Prefers ``ethnicity_bias`` when possible; falls back to any match
    when the bias pool is exhausted.
    """
    rng = rng or random.Random()
    pool = list(load_names())
    excluded = {x.strip().lower() for x in excluded_first_names}
    pool = [p for p in pool if p.first_name.lower() not in excluded]
    if ethnicity_bias:
        biased = [p for p in pool if ethnicity_bias.lower() in p.ethnicity.lower()]
        if len(biased) >= n:
            rng.shuffle(biased)
            return biased[:n]
    rng.shuffle(pool)
    return pool[:n]


def sample_airline(
    *,
    excluded_names: Iterable[str] = (),
    region_hint: str | None = None,
    rng: random.Random | None = None,
) -> AirlinePoolEntry | None:
    rng = rng or random.Random()
    pool = [a for a in load_airlines() if a.name.strip().lower() not in {x.strip().lower() for x in excluded_names}]
    if region_hint:
        biased = [a for a in pool if region_hint.lower() in a.region.lower()]
        if biased:
            return rng.choice(biased)
    if not pool:
        return None
    return rng.choice(pool)
