"""The eight rotation axes and their per-axis cooldowns.

Codified from the Layer 2 story-parameters brief. Each axis has:

* an enum-like list of canonical values,
* a cooldown window (number of consecutive stories where a value must
  not repeat),
* an ordered target quota per quarter (36-video window) — soft rule
  the Global History Manager reports but does not enforce.

The Planner picks one value per axis for a new story, subject to the
per-axis cooldown check in :mod:`core.history`. Values are stored in
the DB as the exact strings from this module so cooldown checks are
purely lexical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── ordered enums ──────────────────────────────────────────────────


class SubGenre(str, Enum):
    MIRACLE_LANDING = "Miracle emergency landing"
    ATC_HERO = "ATC hero"
    IN_FLIGHT_MEDICAL = "In-flight medical emergency"
    WEATHER = "Weather disaster & near-miss"
    MECHANICAL = "Mechanical failure recreation"
    CABIN_CRISIS = "Cabin crisis"
    GENERAL_AVIATION = "General aviation drama"
    SEARCH_AND_RESCUE = "Search & rescue operation"
    MILITARY = "Military aviation (declassified)"
    MYSTERY = "Aviation mystery"
    HUMAN_FACTOR_TRAGEDY = "Human-factor tragedy"


class AircraftClass(str, Enum):
    WIDE_BODY = "Wide-body commercial"
    NARROW_BODY = "Narrow-body commercial"
    REGIONAL_JET = "Regional jet"
    TURBOPROP = "Turboprop commercial"
    BUSINESS_JET = "Business jet"
    SMALL_PISTON = "Small piston single-engine"
    TWIN_PISTON = "Twin piston"
    CARGO = "Cargo aircraft"
    BUSH = "Bush plane"
    HELICOPTER = "Helicopter"
    MILITARY_TRANSPORT = "Military transport"
    VINTAGE = "Vintage / warbird"


class Setting(str, Enum):
    NORTH_ATLANTIC = "North Atlantic"
    TRANS_PACIFIC = "Trans-Pacific"
    ALASKA_BUSH = "Alaska bush"
    ANDES_CROSSING = "Andes crossing"
    SAHARA_DESERT = "Sahara / desert"
    AMAZON_OVERFLIGHT = "Amazon overflight"
    ARCTIC_CIRCLE = "Arctic circle"
    ISLANDS_OCEANIC = "Islands & oceanic"
    MOUNTAIN_AIRPORT = "Mountain airport"
    COASTAL_OCEAN = "Coastal / ocean approach"
    WINTER_STORM = "Winter storm zone"
    TROPICAL_STORM = "Tropical storm / typhoon"
    VOLCANIC = "Volcanic region"
    WARZONE = "Warzone corridor (historic)"


class IncidentType(str, Enum):
    SINGLE_ENGINE_FAILURE = "Single engine failure"
    DUAL_ENGINE_FAILURE = "Dual engine failure"
    UNCONTAINED_ENGINE_FAILURE = "Uncontained engine failure"
    FUEL_EXHAUSTION = "Fuel exhaustion"
    FUEL_CONTAMINATION = "Fuel contamination"
    FUEL_DUMPING = "Fuel dumping emergency"
    GEAR_MALFUNCTION = "Landing gear malfunction"
    HYDRAULIC_FAILURE = "Hydraulic system failure"
    AUTOPILOT_MALFUNCTION = "Autopilot malfunction / runaway trim"
    INSTRUMENT_FAILURE = "Instrument failure"
    CABIN_DEPRESSURIZATION = "Cabin depressurization"
    ELECTRICAL_FIRE = "Electrical fire"
    BIRD_STRIKE = "Bird strike"
    SEVERE_ICING = "Severe icing"
    LIGHTNING_STRIKE = "Lightning strike"
    WIND_SHEAR = "Wind shear on approach"
    MICROBURST = "Microburst"
    WAKE_TURBULENCE = "Wake turbulence encounter"
    VOLCANIC_ASH = "Volcanic ash encounter"
    GPS_JAMMING = "GPS jamming / spoofing"
    PILOT_INCAPACITATION = "Pilot incapacitation"
    RUNWAY_INCURSION = "Runway incursion"
    NEAR_COLLISION = "Near mid-air collision (TCAS event)"
    ATC_ERROR = "ATC error / miscommunication"
    LANGUAGE_BARRIER = "Language barrier miscommunication"
    CABIN_FIRE = "Cabin fire"
    MEDICAL_DIVERSION = "Passenger medical emergency requiring diversion"


class TwistType(str, Enum):
    OVERLOOKED_HERO = "The overlooked hero"
    RECENTLY_CERTIFIED_PART = "The failed part was certified yesterday"
    RESCUER_KNEW_PILOT = "The rescuer knew the pilot personally"
    HISTORICAL_PARALLEL = "Historical parallel"
    PASSENGER_SKILL = "The passenger's skill saved them"
    IGNORED_FORECAST = "Weather forecast was ignored warning"
    MANUFACTURER_KNEW = "The manufacturer knew"
    UNIT_CONFUSION = "Timezone / unit confusion"
    AUTOPILOT_HERO = "The autopilot did save the day"
    GHOST_RADIO = "Radio contact from the ghost"
    FAMILY_ON_COMM = "The pilot's family was on ATC comm"
    UNLIKELY_ALTERNATE = "The alternate airport that shouldn't have worked"
    FOUND_DECADES_LATER = "The plane was found decades later"
    PREVIOUS_CRASH_EXPLAINS = "A previous crash's black box explains this one"
    WHISTLEBLOWER_FIRED = "The whistleblower who was fired"


class Resolution(str, Enum):
    MIRACLE_RUNWAY = "Miracle runway landing"
    DITCHING = "Ditching (water landing) — all survive"
    BELLY_LANDING = "Belly landing"
    OFF_AIRPORT_LANDING = "Landing on highway / field / beach"
    DIVERSION = "Emergency diversion"
    ATC_TALK_DOWN = "ATC talks down non-pilot passenger"
    AUTOLAND = "Autoland saves the day"
    RESCUE_EXTRACTION = "Rescue helicopter extraction"
    WILDERNESS_SURVIVAL = "Wilderness survival then rescue"
    GHOST_PLANE = "Ghost plane recovered"
    POSTHUMOUS_HERO = "Investigation reveals hero action posthumously"
    TRAGIC_BUT_LESSONS = "Tragic loss, lessons saved future flights"


class EmotionalBeat(str, Enum):
    REDEMPTION = "Redemption"
    SELFLESS_SACRIFICE = "Selfless sacrifice"
    UNLIKELY_HERO = "Unlikely hero recognized"
    BUREAUCRATIC_INJUSTICE = "Bureaucratic injustice overcome"
    IMPOSSIBLE_ODDS = "Impossible odds beaten"
    WIDOW_CLOSURE = "Widow gets closure"
    ROOKIE_RESPECT = "Rookie earns respect"
    LEGENDARY_RETIREMENT = "Retirement flight becomes legendary"


class HookPattern(str, Enum):
    A_SPECIFIC_DETAIL = "A — Specific seat/detail + shock"
    B_TIME_COMPRESSION = "B — Time compression"
    C_DIRECT_QUOTE = "C — Direct quote first"
    D_LOCATION_STATUS = "D — Location + status"
    E_NUMBER_MYSTERY = "E — The number nobody knew"
    F_IMPOSSIBLE_TASK = "F — Impossible task"
    G_PASSENGER_QUESTION = "G — The passenger question"


class NarrativeStructureV2(str, Enum):
    """Six documented structures (replaces the old 5-element enum).

    Kept as a separate enum so the old ``models.aviation_bible.NarrativeStructure``
    still works for saved JSON blobs.
    """

    THREE_ACT = "three_act_classic"
    KISHOTENKETSU = "kishotenketsu"
    IN_MEDIA_RES = "in_media_res_flashback"
    RASHOMON = "rashomon_multi_pov"
    INVESTIGATION = "investigation_first"
    BRAID = "documentary_braid"


# The Master Overview's list of 35 protagonist archetypes.
PROTAGONIST_ARCHETYPES: list[str] = [
    "Rookie First Officer in first commercial flight",
    "Female Captain in a doubting environment",
    "Retired Captain on last flight before retirement",
    "Senior Captain 60+ with the trick they don't teach",
    "Off-duty pilot passenger",
    "Test pilot on an experimental airframe",
    "Alaskan / Canadian-North bush pilot",
    "Cargo pilot on a solo night sector",
    "Coast Guard / MedEvac helicopter pilot",
    "Flight attendant with a military past",
    "Purser taking command",
    "Cabin crew trained in emergency ops",
    "Junior ATC controller on a night shift",
    "Veteran ATC controller with 30 years experience",
    "Approach controller at a mountain airport",
    "Flight dispatcher who spotted a flaw in the plan",
    "Ground crew mechanic on pre-flight inspection",
    "Airport fire chief coordinating crash response",
    "Meteorologist who warned but was not heard",
    "Passenger — off-duty doctor / nurse",
    "Passenger — engineer / mechanic",
    "Passenger — ex-military",
    "Passenger — student pilot",
    "Passenger — child with unusual awareness",
    "Passenger — elderly war veteran",
    "Passenger — expectant mother",
    "Passenger — celebrity travelling incognito",
    "Reporter aboard by chance, documenting events",
    "Rescue swimmer from a helicopter",
    "Search & Rescue commander",
    "NTSB investigator unravelling causes post-factum",
    "Coast Guard captain hearing mayday",
    "Mountain rescue team reaching the crash site",
    "Ranger in the wilds, first to find wreckage",
    "Amateur radio operator who caught the last transmission",
]


# Sub-genre quarterly target quotas per the master brief (12 videos / month × 3 months).
SUBGENRE_QUARTERLY_QUOTAS: dict[SubGenre, int] = {
    SubGenre.MIRACLE_LANDING: 9,
    SubGenre.HUMAN_FACTOR_TRAGEDY: 3,
    SubGenre.MECHANICAL: 6,
    SubGenre.WEATHER: 6,
    SubGenre.CABIN_CRISIS: 3,
    SubGenre.SEARCH_AND_RESCUE: 3,
    SubGenre.GENERAL_AVIATION: 3,
    SubGenre.ATC_HERO: 3,
    SubGenre.IN_FLIGHT_MEDICAL: 3,
    SubGenre.MILITARY: 1,
    SubGenre.MYSTERY: 1,
}


# Narrative-structure quarterly targets.
STRUCTURE_QUARTERLY_QUOTAS: dict[NarrativeStructureV2, int] = {
    NarrativeStructureV2.THREE_ACT: 11,
    NarrativeStructureV2.INVESTIGATION: 7,
    NarrativeStructureV2.IN_MEDIA_RES: 5,
    NarrativeStructureV2.BRAID: 5,
    NarrativeStructureV2.KISHOTENKETSU: 4,
    NarrativeStructureV2.RASHOMON: 4,
}


@dataclass(frozen=True)
class AxisSpec:
    """One tracked rotation axis."""

    name: str
    cooldown: int             # story-history window in which the value cannot repeat
    values: list[str]         # canonical value strings (for validation)


AXES: dict[str, AxisSpec] = {
    "sub_genre": AxisSpec("sub_genre", cooldown=2, values=[e.value for e in SubGenre]),
    "aircraft_type": AxisSpec("aircraft_type", cooldown=5, values=[e.value for e in AircraftClass]),
    "setting": AxisSpec("setting", cooldown=4, values=[e.value for e in Setting]),
    "protagonist_archetype": AxisSpec("protagonist_archetype", cooldown=6, values=PROTAGONIST_ARCHETYPES),
    "inciting_incident": AxisSpec("inciting_incident", cooldown=8, values=[e.value for e in IncidentType]),
    "twist_type": AxisSpec("twist_type", cooldown=10, values=[e.value for e in TwistType]),
    "resolution": AxisSpec("resolution", cooldown=6, values=[e.value for e in Resolution]),
    "hook_pattern": AxisSpec("hook_pattern", cooldown=3, values=[e.value for e in HookPattern]),
    "narrative_structure": AxisSpec("narrative_structure", cooldown=2, values=[e.value for e in NarrativeStructureV2]),
    "narrator_voice": AxisSpec("narrator_voice", cooldown=4, values=[]),  # free text
    "character_first_name": AxisSpec("character_first_name", cooldown=15, values=[]),
    "fictional_airline": AxisSpec("fictional_airline", cooldown=8, values=[]),
}

# Hook pattern has an additional constraint: max 3 uses in the last 10 stories.
HOOK_MAX_USES_WINDOW = 10
HOOK_MAX_USES = 3


class MonetizationRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MED"
    HIGH = "HIGH"


class CausationType(str, Enum):
    MECHANICAL = "Mechanical"
    STRUCTURAL = "Structural"
    WEATHER = "Weather"
    HUMAN_FACTOR = "Human-Factor"
    MAINTENANCE = "Maintenance"
    FUEL = "Fuel"
    FIRE = "Fire"
    SECURITY = "Security"
    BIRD_STRIKE = "Bird-Strike"
    UNKNOWN = "Unknown"
