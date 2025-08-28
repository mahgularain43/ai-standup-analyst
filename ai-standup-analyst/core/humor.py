import random
from typing import List, Dict, Any

PERSONA_PRESETS = {
    "Data Analyst": {
        "description": "Professional with a hint of dry humor",
        "templates": {
            "trend_up":  ["{metric} up {pct:.1f}% — Excel finally smiled."],
            "trend_down":["{metric} down {pct:.1f}% — the graph looks sad, not me."],
            "anomaly":   ["Outlier in {metric}: {value:.2f} at row {idx} (z={z:.1f})."],
            "segment":   ["{best} owns {metric} ({best_val:.1f}); {worst} brings snacks ({worst_val:.1f})."],
            "generic":   ["Numbers don’t lie, they just roast quietly."],
        },
    },
    "Stand-Up Comic": {
        "description": "Full comedy mode — roast everything",
        "templates": {
            "trend_up":  [
                "{metric} shot up {pct:.1f}% — finally something going up besides rent!",
                "Huge glow-up: {metric} +{pct:.1f}%. Even my GPA is jealous."
            ],
            "trend_down":[
                "{metric} dropped {pct:.1f}% — like my screen time after a power cut.",
                "{metric} slid {pct:.1f}% — who put the banana peel on the chart?"
            ],
            "anomaly":   [
                "{metric} hit {value:.2f} (z={z:.1f}) — that data point pre-gamed.",
                "We found the party crasher in {metric}: {value:.2f} (z={z:.1f})."
            ],
            "segment":   [
                "{best} dominates {metric} ({best_val:.1f}). {worst}? Power nap at {worst_val:.1f}.",
                "Leader: {best} in {metric} ({best_val:.1f}). {worst} said ‘I’ll be there in 5’ at {worst_val:.1f}."
            ],
            "generic":   ["Tip your data scientist."],
        },
    },
    "Executive Coach": {
        "description": "Motivational with gentle humor",
        "templates": {
            "trend_up":  ["{metric} grew {pct:.1f}% — momentum looks great!"],
            "trend_down":["{metric} dipped {pct:.1f}% — a setup for a comeback."],
            "anomaly":   ["Notable spike in {metric}: {value:.2f} (z={z:.1f}). What drove it?"],
            "segment":   ["{best} leads {metric} ({best_val:.1f}). Let’s share best practices."],
            "generic":   ["Action item: keep winning charts coming."],
        },
    },
    "Roast Master": {
        "description": "Spicy, savage, still office-safe",
        "templates": {
            "trend_up":  ["{metric} up {pct:.1f}% — who finally read the onboarding doc?"],
            "trend_down":["{metric} down {pct:.1f}% — like my hopes on a Monday."],
            "anomaly":   ["{metric} at {value:.2f} (z={z:.1f}) — sir, this is a Wendy’s."],
            "segment":   ["{best} carries {metric} ({best_val:.1f}); {worst} contributes vibes ({worst_val:.1f})."],
            "generic":   ["The data did you dirty, not me."],
        },
    },
    "Dad Jokes": {
        "description": "Wholesome groaners",
        "templates": {
            "trend_up":  ["{metric} rose {pct:.1f}%. It’s outstanding — like a dad in a field."],
            "trend_down":["{metric} fell {pct:.1f}%. I wood joke, but that’s a lumber drop."],
            "anomaly":   ["Anomaly in {metric}: {value:.2f} (z={z:.1f}). Un-data-precedented!"],
            "segment":   ["{best} best at {metric} ({best_val:.1f}). The rest? Segment-tally behind."],
            "generic":   ["I had a joke about stats, but it’s not significant."],
        },
    },
    "Tech Support": {
        "description": "Calm, sarcastic, vaguely enterprise",
        "templates": {
            "trend_up":  ["{metric} +{pct:.1f}% — have you tried turning your KPIs off and on again?"],
            "trend_down":["{metric} −{pct:.1f}% — works on my dashboard."],
            "anomaly":   ["{metric} {value:.2f} (z={z:.1f}). Logging ticket: ‘gremlin in prod.’"],
            "segment":   ["{best} leads {metric} ({best_val:.1f}). {worst} is waiting on access."],
            "generic":   ["We can fix it. After lunch."],
        },
    },
}

def _apply_tone(joke: str, tone: int) -> str:
    # 1..10 scale; 1 = dry, 10 = chaotic
    if tone >= 8:
        joke = joke.replace("!", "!!!")
        if "up" in joke: joke = "🚀 " + joke
        if "down" in joke: joke = "💀 " + joke
    elif tone <= 3:
        joke = joke.replace("!!!", ".").replace("!", ".")
        joke = joke.replace("crashed", "declined").replace("shot up", "increased")
    return joke

def _render_from_insight(ins: Dict[str, Any], templates: Dict[str, list]) -> str:
    t = ins.get("type")

    if t == "trend":
        pct = abs(ins.get("value", 0.0))
        direction_word = ins.get("direction", "").lower()
        key = "trend_up" if direction_word.startswith("increas") else "trend_down"
        tmpl = random.choice(templates.get(key, templates["generic"]))
        return tmpl.format(metric=ins["metric"], pct=pct)

    if t == "anomaly":
        tmpl = random.choice(templates["anomaly"])
        return tmpl.format(
            metric=ins["metric"],
            value=ins.get("value", 0.0),
            z=ins.get("z_score", 0.0),
            idx=ins.get("index", 0)
        )

    if t == "segment":
        tmpl = random.choice(templates["segment"])
        return tmpl.format(
            metric=ins["metric"],
            best=ins.get("best_segment", "Best"),
            best_val=ins.get("best_value", 0.0),
            worst=ins.get("worst_segment", "Worst"),
            worst_val=ins.get("worst_value", 0.0)
        )

    return random.choice(templates["generic"])

def make_jokes(insights: List[Dict[str, Any]], persona: str = "Data Analyst", tone: int = 6) -> List[str]:
    p = PERSONA_PRESETS.get(persona, PERSONA_PRESETS["Data Analyst"])["templates"]
    jokes: List[str] = []
    for ins in insights:
        line = _render_from_insight(ins, p)
        jokes.append(_apply_tone(line, tone))
    return jokes
