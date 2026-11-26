"""Support tickets, generated: what customers write to a note-taking app.

An ordinary week is a mix of billing questions, how-to questions, bug
reports, account changes, sign-in trouble and the odd thank-you, each
written a little differently (greetings, sign-offs, wording). The
stories change one thing at a time, so their answers are known:

* ``app-update``  — the week after version 5 shipped: sign-in and sync
  complaints crowd in, messages grow longer and angrier and mention the
  update itself;
* ``chat-widget`` — the same topics, but most tickets now come from an
  in-app chat box: short, lower-case, few greetings, abbreviations;
* ``two-weeks``   — two ordinary weeks: nothing changed.

The texts are English on purpose: the text plugin's measures (stop
words, readability) and its embeddings are made for English.

    from ddoc.examples import support
    ref_dir, cur_dir = support.write_pair("app-update", out_dir)
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TICKETS = 200

_PLANS = ["Plus", "Pro", "Family", "Team"]
_DEVICES = ["iPhone", "Android phone", "iPad", "laptop", "work PC"]
_FEATURES = ["export a notebook to PDF", "share a note with my team", "add a reminder", "change the font size",
             "move notes between notebooks", "turn on two-step verification", "restore a deleted note"]

TOPICS: Dict[str, List[str]] = {
    "billing": ["I was charged twice for my {plan} subscription this month.",
                "Can I get a refund for the {plan} plan? I cancelled it last week.",
                "My card was declined when I tried to renew {plan}, but the bank says it is fine.",
                "Is there a student discount for the {plan} plan?"],
    "howto": ["How do I {feature}?", "Is it possible to {feature} on my {device}?",
              "I cannot find where to {feature} in the settings."],
    "bug": ["The app crashes when I open a long note on my {device}.",
            "Images in my notes do not load on the {device}.",
            "Search does not find notes that I know exist."],
    "account": ["Please change the email address on my account.",
                "How do I delete my account and all my data?",
                "I would like to merge two accounts into one."],
    "signin": ["I forgot my password and the reset email never arrives.",
               "I cannot sign in on my {device}.", "The app logged me out and will not let me back in."],
    "thanks": ["Just wanted to say the app is great, thank you!", "Thanks for fixing the sync problem so quickly."],
}

_AFTER_UPDATE: Dict[str, List[str]] = {
    "signin": ["Since the version 5 update I cannot sign in on my {device} at all.",
               "After updating to v5 the app keeps logging me out every few minutes.",
               "The new sign-in screen in version 5 rejects my password, which works on the website."],
    "sync": ["My notes stopped syncing between my {device} and my {device2} after the update.",
             "Since v5, edits I make on my {device} never show up anywhere else.",
             "The update duplicated half of my notebooks and now nothing syncs."],
}
_ANGRY = ["This is really frustrating.", "I have tried reinstalling twice already!!",
          "Please fix this as soon as possible, I use the app for work.", "Nothing has worked so far.",
          "I have been a paying customer for years and this is unacceptable."]
_HELLO = ["Hi,", "Hello,", "Hello team,", "Hi there,", "Good morning,", ""]
_BYE = ["Thanks.", "Thank you!", "Regards", "Best,", "Cheers", ""]

MIX_USUAL = {"billing": 0.28, "howto": 0.28, "bug": 0.14, "account": 0.12, "signin": 0.12, "thanks": 0.06}
MIX_UPDATE = {"signin": 0.34, "sync": 0.22, "billing": 0.14, "howto": 0.14, "bug": 0.10, "thanks": 0.06}

# story → (baseline style, current style)
STORIES: Dict[str, Tuple[str, str]] = {
    "app-update": ("usual", "update"),
    "chat-widget": ("usual", "chat"),
    "two-weeks": ("usual", "usual"),
}


def _fill(rng: random.Random, template: str) -> str:
    device, device2 = rng.sample(_DEVICES, 2)
    return template.format(plan=rng.choice(_PLANS), feature=rng.choice(_FEATURES), device=device, device2=device2)


def _chat(text: str, rng: random.Random) -> str:
    """The same request typed into a chat box."""
    short = text.split(". ")[0].rstrip(".!")
    for a, b in (("I cannot", "cant"), ("cannot", "cant"), ("do not", "dont"), ("does not", "doesnt"),
                 ("Please ", "pls "), ("please", "pls"), ("thank you", "thx"), ("Thank you", "thx")):
        short = short.replace(a, b)
    short = short[0].lower() + short[1:]
    return short + rng.choice(["", "", "?", " ??", " pls", " asap"])


def ticket(rng: random.Random, style: str) -> str:
    mix = MIX_UPDATE if style == "update" else MIX_USUAL
    topic = rng.choices(list(mix), weights=list(mix.values()))[0]
    # after the update most sign-in tickets are about it; sync trouble is new
    fresh = style == "update" and topic in _AFTER_UPDATE and (topic == "sync" or rng.random() < 0.85)
    pool = _AFTER_UPDATE[topic] if fresh else TOPICS[topic]
    body = _fill(rng, rng.choice(pool))
    if style == "chat":
        return _chat(body, rng)
    if style == "update" and topic in ("signin", "sync", "bug"):
        body += " " + " ".join(rng.sample(_ANGRY, rng.randint(1, 3)))
    elif rng.random() < 0.3:
        body += " " + _fill(rng, rng.choice(TOPICS[topic]))
    return " ".join(x for x in (rng.choice(_HELLO), body, rng.choice(_BYE)) if x)


def write_set(folder: Path, style: str, *, seed: int, n: int = TICKETS) -> Path:
    """``n`` tickets under ``folder/tickets/`` (``data.csv``: id, text) with
    the ``ddoc.yaml`` that tells ddoc the column holds text."""
    data = Path(folder) / "tickets"
    data.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with (data / "data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text"])
        for i in range(n):
            w.writerow([i, ticket(rng, style)])
    (data / "ddoc.yaml").write_text(
        "modality: text\ntext_column: text\nid_column: id\nlanguage: english\n", encoding="utf-8")
    return Path(folder)


def write_pair(story: str, out_dir: Path, *, seed: Optional[int] = None) -> Tuple[Path, Path]:
    """Write ``ref/`` and ``cur/`` of a story under ``out_dir``."""
    before, after = STORIES[story]
    seed = 21 if seed is None else seed
    out_dir = Path(out_dir)
    return (write_set(out_dir / "ref", before, seed=seed),
            write_set(out_dir / "cur", after, seed=seed + 1))
