#!/usr/bin/env python3
"""Build a listing of library events, to be searched alongside the books.

    python scripts/build_events.py

A reader asking "what can I do about my CV" wants the job club on Tuesday at
least as much as a book on interview technique, and a catalogue that only holds
books cannot tell them. The events are a second collection in the same index, so
one query reaches both and they compete for the same result slots.

EVERYTHING HERE IS INVENTED
---------------------------
There is no public feed of library events, and inventing a real service's
programme would be worse than useless. So Northfield Libraries is fictional and
so is every session below.

That has a consequence worth stating plainly rather than burying: an invented
corpus can demonstrate cross-collection search and **cannot measure it**. The
book records were moved to real Open Library data precisely because invented
titles are unusually descriptive - "The Battle of Hastings" gives the game away -
and anything written here has the same flattery built in. Whatever the events
retrieval looks like, it is a demonstration of the shape of the feature, not
evidence about its quality.

They are written to be awkward in the ways real listings are awkward: the title
often says nothing useful ("Tuesday Club", "Drop-in"), the useful words live in
the description, and several sessions are near-duplicates of each other at
different branches. That is the part worth testing against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "events_northfield.json"

BRANCHES = ["Northfield Central", "Ashcombe Branch", "Rowley Green",
            "Marsden Park", "Fenwick Road"]
PALETTE = ["#4A3B52", "#1F3A3D", "#3A5A40", "#33475B", "#5C3A3A"]

#: (title, audience, description)
#:
#: Titles are deliberately uneven. Some name the thing; several are the sort of
#: opaque house name a real programme is full of, where the only way to find the
#: session is to read what it actually is.
EVENTS: list[tuple[str, str, str]] = [
    ("Rhyme Time", "Babies and toddlers",
     "Songs, rhymes and action games for babies and toddlers with their "
     "grown-ups. Half an hour, no booking, and nobody minds if your child "
     "wanders off halfway through. Good for language before children start "
     "talking."),
    ("Story Time", "Under 5s",
     "Picture books read aloud for pre-school children, with a simple craft "
     "afterwards. Drop in."),
    ("Chatterbooks", "Children 8 to 12",
     "A reading group for children who already like books and want to argue "
     "about them. Each month the group picks the next title."),
    ("Summer Reading Challenge", "Children 4 to 11",
     "Read six library books over the summer holidays and collect stickers and "
     "a certificate. Sign up any time; there is no cost."),
    ("Homework Club", "Secondary school age",
     "A quiet table, a plug socket and someone to ask when you are stuck. "
     "Staffed after school on weekdays during term."),
    ("Code Club", "Children 9 to 13",
     "An introduction to programming using Scratch and micro:bit boards. No "
     "experience needed and laptops are provided. Places are limited."),
    ("Lego Club", "Children 5 to 11",
     "Building session with the library's Lego. Turn up and join in."),
    ("Tuesday Club", "Adults",
     "A weekly get-together over tea and biscuits for anyone who would "
     "otherwise be on their own all day. Conversation, occasionally a quiz, no "
     "pressure to speak."),
    ("Knit and Natter", "Adults",
     "Bring whatever you are making, or borrow needles and wool and be shown "
     "how to start. Knitting, crochet and mending all welcome."),
    ("Repair Cafe", "All ages",
     "Bring a broken lamp, a jammed zip, a toaster that has stopped, and a "
     "volunteer will try to fix it with you rather than for you. Small "
     "electricals, textiles and bicycles."),
    ("Seed Swap", "Adults and families",
     "Bring seeds you have saved and take home something different. Growers "
     "on hand for questions about small gardens, allotments and containers."),
    ("Gardening Talk: Growing in Small Spaces", "Adults",
     "A practical hour on vegetables in pots, window boxes and shared yards - "
     "what actually crops when you have no ground to dig."),
    ("Beekeeping for Beginners", "Adults",
     "An introduction to keeping a hive: the equipment, the time it takes "
     "through the year, what the law requires, and whether a small garden is "
     "big enough. Run with the county beekeepers association."),
    ("Job Club", "Adults",
     "Help with CVs, application forms and interview practice. Bring a draft "
     "or start from nothing. Computers available; no appointment needed."),
    ("Universal Credit Drop-in", "Adults",
     "Support with online claims, journal messages and evidence uploads, with "
     "someone sitting beside you rather than over the phone."),
    ("Money Advice Session", "Adults",
     "Free confidential appointments with a debt adviser: rent and council tax "
     "arrears, energy bills, buy-now-pay-later and priority debts. Booking "
     "required."),
    ("Digital Help Drop-in", "Adults",
     "One-to-one help with a phone, tablet or laptop. Email, video calls, "
     "banking apps, online forms, or simply making the text bigger. Bring your "
     "own device if you have one."),
    ("Introduction to Spreadsheets", "Adults",
     "A beginners' session on rows, columns and simple sums, aimed at people "
     "who need it for work and have never been shown."),
    ("Family History Workshop", "Adults",
     "Getting started with census returns, parish registers and the library's "
     "subscription genealogy databases. Bring what you already know about one "
     "person and work back."),
    ("Local History Talk: The Canal and the Mills", "Adults",
     "An illustrated talk on how the waterway shaped this area, with "
     "photographs from the local studies collection."),
    ("Local History Talk: Northfield in the War Years", "Adults",
     "Rationing, evacuation and the shelters, drawn from the oral history "
     "recordings held in the local studies collection."),
    ("Archive Open Afternoon", "Adults",
     "The local studies room open without an appointment, with staff to show "
     "you the maps, directories and newspaper microfilm."),
    ("Author Event: An Evening With a Crime Writer", "Adults",
     "A novelist talks about writing detective fiction, followed by questions "
     "and signing. Tickets free but must be reserved."),
    ("Poetry Evening", "Adults",
     "An open-mic night for local poets and anyone who would rather listen. "
     "Five minutes each, read your own or someone else's."),
    ("Reading Group", "Adults",
     "A monthly discussion of one novel, with copies reserved for members in "
     "advance. New members always welcome, including people who did not finish "
     "the book."),
    ("Large Print Reading Group", "Adults",
     "The same monthly discussion with all titles available in large print and "
     "audio."),
    ("Writing Group", "Adults",
     "A workshop for people writing fiction or memoir, with time to read work "
     "aloud and get responses from the room."),
    ("Bereavement Cafe", "Adults",
     "An informal space for anyone who has lost someone, at any point after "
     "the death. Nobody has to talk. Tea, quiet company and someone from the "
     "local bereavement service who can point you at further support if you "
     "want it."),
    ("Carers Group", "Adults",
     "For people looking after a partner, parent or child. A couple of hours "
     "away from it, with practical advice on respite and benefits."),
    ("Memory Cafe", "Adults",
     "For people living with dementia and those who care for them. Music, "
     "photographs and familiar objects, in a room set up to be easy to be in."),
    ("Walk and Talk", "Adults",
     "A gentle forty-minute walk from the library door and back, at the pace "
     "of the slowest person. Good for anyone getting moving again after "
     "illness or a long spell indoors."),
    ("Chair-based Exercise", "Older adults",
     "Strength and balance work done sitting down, led by a qualified "
     "instructor. Aimed at people who have had a fall or are worried about "
     "having one."),
    ("Mindfulness Hour", "Adults",
     "An introduction to breathing and attention exercises for stress, taught "
     "without jargon. No mats, no special clothes."),
    ("Baby Massage", "Parents of babies",
     "A four-week course on massage techniques that can help with colic, "
     "sleep and settling. Booking required."),
    ("Bump and Beyond", "Expectant and new parents",
     "Antenatal and postnatal drop-in with a health visitor: feeding, "
     "sleeping, weaning and how you are doing yourself."),
    ("Toddler Mealtimes Workshop", "Parents and carers",
     "A practical session on fussy eating - what is normal, what helps and "
     "what quietly makes it worse. Run with the community dietitian."),
    ("Sleep Workshop for Parents", "Parents and carers",
     "For families with a child who will not settle or wakes repeatedly. "
     "Realistic approaches rather than a single method."),
    ("English Conversation Club", "Adults",
     "Practice speaking English in a small friendly group. All levels, no "
     "grading, no charge."),
    ("Citizenship Test Practice", "Adults",
     "Study support for the Life in the UK test, with the official handbook "
     "and practice papers available to borrow."),
    ("Board Games Afternoon", "All ages",
     "The library's games out on the tables. Chess, draughts, and a shelf of "
     "modern games someone will teach you."),
    ("Film Afternoon", "Adults",
     "A screening with subtitles and the lights left partly on, followed by "
     "tea. Titles chosen by the audience."),
    ("Art Club", "Adults",
     "Drawing and painting with materials provided. No tutor and no standard "
     "to meet - people work on their own things alongside each other."),
    ("Photography Walk", "Adults and teenagers",
     "A guided walk taking pictures around the town, then back to look at what "
     "everyone got. Phone cameras entirely welcome."),
    ("Astronomy Evening", "All ages",
     "An introduction to the winter sky with the local astronomical society, "
     "including telescopes outside if it is clear."),
    ("Bike Maintenance Basics", "Teenagers and adults",
     "Punctures, brakes and gears, on your own bike. Tools provided."),
    ("Dog Training Talk", "Adults",
     "A trainer on the first weeks with a new puppy: toilet training, biting, "
     "sleeping and walking on a lead."),
    ("Volunteer Information Morning", "Adults",
     "What volunteering at the library involves - shelving, reading groups, "
     "digital help and the home delivery service."),
    ("Home Library Service Information", "Adults",
     "For anyone who cannot get to a library: how books, audiobooks and large "
     "print are brought to the door by volunteers."),
]

#: Phrases in a description that already settle whether booking is needed.
#: Deriving it rather than rolling for it stops the record contradicting itself -
#: the first draft printed "Booking: Book a place" above a description that says
#: "no booking", which is the kind of detail a librarian reads first.
_DROP_IN = ("drop in", "no booking", "turn up", "no appointment",
            "without an appointment", "sign up any time")
_MUST_BOOK = ("booking required", "places are limited", "must be reserved",
              "tickets free but must be reserved")

#: Some sessions repeat and some happen once. A real programme is a mix, and
#: the repeats are what create the near-duplicate records that make retrieval
#: interesting - several branches running "Rhyme Time" with near-identical text.
REPEATING = {
    "Rhyme Time", "Story Time", "Homework Club", "Knit and Natter",
    "Job Club", "Digital Help Drop-in", "Tuesday Club", "Reading Group",
    "English Conversation Club", "Chair-based Exercise", "Lego Club",
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
TIMES = ["10.00am", "10.30am", "11.00am", "1.30pm", "2.00pm", "4.00pm",
         "6.00pm", "7.00pm"]
START = date(2026, 10, 5)


def booking_for(description: str, digest: bytes) -> str:
    """What the description already says, or a coin toss when it says nothing."""
    lowered = description.lower()
    if any(phrase in lowered for phrase in _MUST_BOOK):
        return "Book a place"
    if any(phrase in lowered for phrase in _DROP_IN):
        return "Just turn up"
    return "Book a place" if digest[3] % 3 == 0 else "Just turn up"


def render_text(audience: str, description: str, when: str,
                branch: str, booking: str, cost: str) -> str:
    """The listing as the service stores it - and all that gets indexed.

    Same shape as a catalogue record: labelled lines, no marketing copy. The
    description carries the meaning, which is the point - the titles mostly do
    not.
    """
    return "\n".join([
        f"For: {audience}",
        f"When: {when}",
        f"Where: {branch}",
        f"Booking: {booking}   Cost: {cost}",
        f"About: {description}",
    ])


def build(target: int) -> list[dict]:
    records: list[dict] = []
    for index, (title, audience, description) in enumerate(EVENTS):
        repeats = 7 if title in REPEATING else 1
        for occurrence in range(repeats):
            if len(records) >= target:
                break
            slug = f"{title}-{occurrence}".lower().replace(" ", "-")
            digest = hashlib.sha256(slug.encode()).digest()

            branch = BRANCHES[(index + occurrence) % len(BRANCHES)]
            # A session called "Tuesday Club" that meets on Mondays is nobody's
            # idea of realistic mess - it is just a hash ignoring the title.
            named = next((d for d in WEEKDAYS if d.lower() in title.lower()), None)
            weekday = named or WEEKDAYS[digest[0] % len(WEEKDAYS)]
            time = TIMES[digest[1] % len(TIMES)]
            starts = START + timedelta(days=int(digest[2]) % 84)
            recurring = title in REPEATING
            when = (f"{weekday}s, {time}" if recurring
                    else f"{weekday} {starts.strftime('%-d %B %Y')}, {time}")

            booking = booking_for(description, digest)
            cost = "Free" if digest[4] % 8 else "£3"
            places = 0 if digest[5] % 9 == 0 else 2 + digest[6] % 18

            records.append({
                # --- corpus fields: what BetterSearch indexes ---------------
                "doc_id": f"ev-{digest.hex()[:10]}",
                "title": title,
                # Every occurrence of a weekly session is its own record, as it
                # is in a real programme - each has its own date, branch and
                # place count. They share a series id so a result list can show
                # the session once rather than seven times.
                "series_id": "s-" + hashlib.sha256(title.encode()).hexdigest()[:8],
                "text": render_text(audience, description, when, branch,
                                    booking, cost),
                "tags": ["event"],
                # --- display fields: the page reads these -------------------
                "record_type": "event",
                "audience": audience,
                "when": when,
                "recurring": recurring,
                "booking": booking,
                "cost": cost,
                "places": places,
                "blurb": description,
                "subjects": [],
                "location": branch,
                # `format` and `available` are read by the existing facets and
                # the existing card, so events fill them rather than growing a
                # parallel set of fields that mean the same thing.
                "format": "Event",
                "available": places,
                "copies": places,
                "fiction": False,
                "author": "",
                "author_dates": "",
                "year": str(starts.year),
                "publisher": "",
                "language": "English",
                "cover": PALETTE[digest[7] % len(PALETTE)],
            })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    records = build(args.target)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps({
        "name": "northfield-events",
        # Contributed to the page's example queries alongside the books', so
        # serving both collections offers a query for each.
        "presets": ["somewhere to go on a Tuesday afternoon"],
        "description": (
            "Events at a fictional library service. Every session, date, branch "
            "and place count here is invented - there is no public feed of "
            "library events - so this collection demonstrates searching books "
            "and events together and measures nothing about how well it works."
        ),
        "documents": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    words = sorted(len(r["text"].split()) + len(r["title"].split()) for r in records)
    print(f"Wrote {len(records)} events to {args.out} "
          f"({args.out.stat().st_size // 1024} KB)")
    print(f"  distinct sessions    {len(EVENTS)}")
    print(f"  repeating sessions   {len(REPEATING)} (7 occurrences each)")
    print(f"  words per record     min {words[0]}  median {words[len(words)//2]}"
          f"  max {words[-1]}")
    print(f"  needs booking        {sum(1 for r in records if r['booking'] == 'Book a place')}")
    print(f"  fully booked         {sum(1 for r in records if not r['places'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
