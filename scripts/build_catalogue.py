#!/usr/bin/env python3
"""Build the demonstration library catalogue.

Emits data/catalogue_library.json, which doubles as a BetterSearch corpus:
Document.from_dict reads only doc_id/title/text/url/tags and ignores the rest,
so the display fields ride along in the same file and

    bettersearch ingest --corpus data/catalogue_library.json

works with no conversion step.

The records are invented. A cataloguer judges the shape of a record and how
search behaves over it, not whether they recognise the titles - and inventing
avoids both misrepresenting real works and getting bibliographic facts subtly
wrong, which is the first thing a cataloguer would notice.

Three properties are deliberate, each taken from a real catalogue page:

* a surname collision, so one search returns three unrelated people
* degraded records - bracketed titles, unknown dates, missing authors
* a blurb on roughly one record in five, because that is what real catalogues
  look like: the inconsistency is itself the argument
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("data/catalogue_library.json")

# Cover colours, picked to sit together and stay legible with white text.
PALETTE = ["#2E4A62", "#1B4D3E", "#6B3A4B", "#4A4067", "#7A4A2B",
           "#2C5A5A", "#553F2E", "#3E4C5E"]

# Physical branches only. Electronic items are shelved nowhere and are handled
# separately below - a printed book must never report its location as "Online".
BRANCHES = ["Northfield Central", "Ashcombe Branch", "Rowley Green",
            "Marsden Park", "Fenwick Road"]

# title | author | dates | year | publisher | format | subjects | fiction | blurb
R = [
    # ---- spirituality / wellbeing -------------------------------------------
    ("Living Lightly", "Grewal, Prem", "1931-1990", "2011", "Wayfarer Press", "Book",
     ["Meditation", "Spiritual life", "Mind and body"], False,
     "A collection drawn from forty years of talks, arguing that ordinary attention is the whole of the practice."),
    ("The Art of Stillness", "Grewal, Prem", "1931-1990", "2008", "Wayfarer Press", "Book",
     ["Meditation", "Spiritual life"], False, None),
    ("One Breath at a Time", "Grewal, Prem", "1931-1990", "2014", "Wayfarer Press", "Book",
     ["Breathing exercises", "Meditation", "Stress management"], False, None),
    ("Nothing to Do", "Grewal, Prem", "1931-1990", "UUUU", "Wayfarer Press", "Book",
     ["Spiritual life"], False, None),
    ("The Quiet Room", "Halloran, Beatrice", "", "2019", "Kestrel & Vane", "Book",
     ["Contemplation", "Solitude", "Mind and body"], False, None),
    ("Sitting Still", "Okonjo, Adaeze", "", "2021", "Lantern House", "Book",
     ["Meditation", "Mindfulness"], False,
     "A working nurse's account of twenty minutes a day, and what changed."),
    ("Attention, and Other Small Miracles", "Verhoeven, Sanne", "", "2017", "Bellwether", "Book",
     ["Contemplation", "Psychology"], False, None),
    ("The Unhurried Life", "Bramwell, Douglas", "1944-2020", "2003", "Foxglove", "Large print",
     ["Simplicity", "Conduct of life"], False, None),
    ("Sixty Seconds", "Okonjo, Adaeze", "", "2024", "Lantern House", "eAudiobook",
     ["Meditation", "Stress management"], False, None),

    # ---- grief and loss ------------------------------------------------------
    ("After the Funeral", "Restrick, Marian", "", "2018", "Coldharbour", "Book",
     ["Bereavement", "Grief", "Consolation"], False,
     "Written in the year after her husband died, and deliberately offering no stages and no timetable."),
    ("Carrying On", "Restrick, Marian", "", "2022", "Coldharbour", "Book",
     ["Bereavement", "Widows"], False, None),
    ("The Empty Chair", "Kealey, Dermot", "", "2015", "Bellwether", "Book",
     ["Grief", "Family", "Loss"], False, None),
    ("What Nobody Tells You", "Fairhead, Judith", "", "2020", "Coldharbour", "Large print",
     ["Bereavement", "Consolation"], False, None),
    ("A Year of Thursdays", "Kealey, Dermot", "", "2023", "Bellwether", "eAudiobook",
     ["Grief", "Memoirs"], False, None),

    # ---- sleep, anxiety, health ---------------------------------------------
    ("Night Shift", "Corrigan, Estelle", "", "2019", "Meridian Health", "Book",
     ["Insomnia", "Sleep disorders", "Self-care"], False,
     "Why so many of us wake at three in the morning, and what the evidence actually supports."),
    ("The Tired Generation", "Corrigan, Estelle", "", "2023", "Meridian Health", "Book",
     ["Fatigue", "Sleep disorders", "Public health"], False, None),
    ("Wide Awake at Four", "Pelletier, Nina", "", "2016", "Kestrel & Vane", "Book",
     ["Insomnia", "Cognitive therapy"], False, None),
    ("The Racing Mind", "Ademola, Tunde", "", "2021", "Meridian Health", "Book",
     ["Anxiety", "Cognitive therapy", "Self-care"], False, None),
    ("Small Steps Out", "Ademola, Tunde", "", "2024", "Meridian Health", "Book",
     ["Anxiety", "Phobias", "Self-care"], False,
     "A graded programme for leaving the house again, written for people who have stopped."),
    ("Your Back, Explained", "Whittle, Gordon", "", "2014", "Meridian Health", "Book",
     ["Backache", "Physical therapy"], False, None),
    ("The Change", "Marsh-Ibori, Grace", "", "2022", "Lantern House", "Book",
     ["Menopause", "Women's health"], False, None),
    ("Breathe Out", "Pelletier, Nina", "", "2020", "Kestrel & Vane", "eAudiobook",
     ["Relaxation", "Stress management"], False, None),

    # ---- parenting and children's food ---------------------------------------
    ("The Fussy Years", "Delacroix-Bell, Hannah", "", "2021", "Rookery", "Book",
     ["Child rearing", "Nutrition", "Toddlers"], False,
     "Refuses the phrase 'picky eater' and treats mealtime standoffs as a normal developmental stage."),
    ("Small Plates", "Delacroix-Bell, Hannah", "", "2018", "Rookery", "Book",
     ["Cooking", "Infants - Nutrition", "Family"], False, None),
    ("Will They Ever Eat Greens?", "Nkemelu, Chidi", "", "2023", "Rookery", "Book",
     ["Child rearing", "Nutrition"], False, None),
    ("The Long Bedtime", "Delacroix-Bell, Hannah", "", "2019", "Rookery", "Book",
     ["Child rearing", "Sleep", "Toddlers"], False, None),
    ("Three and Furious", "Nkemelu, Chidi", "", "2020", "Rookery", "Large print",
     ["Child rearing", "Temper tantrums"], False, None),

    # ---- cosy / gentle crime --------------------------------------------------
    ("Murder at Marlowe Hall", "Grewal, Ravi", "", "2019", "Pennyfarthing", "Book",
     ["Detective and mystery stories", "England - Fiction"], True,
     "The first Hensleigh mystery. A country house, a missing will, and nothing more graphic than a cold teapot."),
    ("The Knitting Circle Murders", "Grewal, Ravi", "", "2021", "Pennyfarthing", "Book",
     ["Detective and mystery stories"], True, None),
    ("Death Takes the Bus", "Grewal, Ravi", "", "2023", "Pennyfarthing", "Book",
     ["Detective and mystery stories"], True, None),
    ("A Body in the Bandstand", "Cholmondeley, Pearl", "", "2020", "Pennyfarthing", "Large print",
     ["Detective and mystery stories", "Villages - Fiction"], True, None),
    ("The Vicar's Last Sermon", "Cholmondeley, Pearl", "", "2022", "Pennyfarthing", "Book",
     ["Detective and mystery stories", "Clergy - Fiction"], True, None),
    ("Tea, Cake and Arsenic", "Cholmondeley, Pearl", "", "2024", "Pennyfarthing", "eAudiobook",
     ["Detective and mystery stories"], True, None),

    # ---- harder crime ---------------------------------------------------------
    ("The Cutting Room", "Vasilenko, Irina", "", "2018", "Blackwater", "Book",
     ["Detective and mystery stories", "Thrillers"], True, None),
    ("Nine Grams", "Vasilenko, Irina", "", "2020", "Blackwater", "Book",
     ["Thrillers", "Organized crime - Fiction"], True, None),
    ("Cold Frame", "Obuya, Marcus", "", "2022", "Blackwater", "Book",
     ["Detective and mystery stories", "Police procedural"], True, None),

    # ---- literary fiction ------------------------------------------------------
    ("The Salt House", "Trevanion, Bryony", "", "2017", "Aldgate", "Book",
     ["Domestic fiction", "Wales - Fiction"], True,
     "Three sisters return to a house on the estuary to decide what to do with it, and with each other."),
    ("A Slight Delay", "Trevanion, Bryony", "", "2021", "Aldgate", "Book",
     ["Domestic fiction"], True, None),
    ("The Understudy", "Ferreira-Lynch, Dom", "", "2019", "Aldgate", "Book",
     ["Theatre - Fiction", "Psychological fiction"], True, None),
    ("Weather Permitting", "Ferreira-Lynch, Dom", "", "2023", "Aldgate", "Book",
     ["Domestic fiction", "Humorous fiction"], True, None),
    ("The Loan", "Sciacca, Emilia", "", "2016", "Aldgate", "Large print",
     ["Domestic fiction", "Debt - Fiction"], True, None),
    ("Hold the Line", "Obuya, Marcus", "", "2024", "Aldgate", "Book",
     ["Domestic fiction", "Work - Fiction"], True, None),

    # ---- cookery ---------------------------------------------------------------
    ("Tuesday Dinners", "Rasmussen, Kirsten", "", "2020", "Copper Pot", "Book",
     ["Cooking", "Quick and easy cooking"], False,
     "Forty meals that assume you got home at half past six and nobody has been to the shops."),
    ("One Pan, One Oven", "Rasmussen, Kirsten", "", "2022", "Copper Pot", "Book",
     ["Cooking", "Quick and easy cooking"], False, None),
    ("The Batch Book", "Ferreira-Lynch, Dom", "", "2021", "Copper Pot", "Book",
     ["Cooking", "Make-ahead cooking"], False, None),
    ("Feeding Six for Forty Pounds", "Marsh-Ibori, Grace", "", "2023", "Copper Pot", "Book",
     ["Cooking", "Low budget cooking", "Family"], False, None),
    ("Bread, Slowly", "Rasmussen, Kirsten", "", "2018", "Copper Pot", "Book",
     ["Baking", "Bread"], False, None),
    ("The Allotment Kitchen", "Whittle, Gordon", "", "2019", "Copper Pot", "Large print",
     ["Cooking", "Vegetables", "Gardening"], False, None),

    # ---- history ---------------------------------------------------------------
    ("The Year the Fields Changed Hands", "Pemberton-Hyde, Alistair", "", "2015", "Thornbury", "Book",
     ["Great Britain - History - Medieval period", "Land tenure"], False,
     "How a single generation redrew who owned England, told through four manors and their surviving rolls."),
    ("The Survey", "Pemberton-Hyde, Alistair", "", "2018", "Thornbury", "Book",
     ["Great Britain - History - Medieval period", "Land tenure"], False, None),
    ("Castles of Earth and Timber", "Pemberton-Hyde, Alistair", "", "2012", "Thornbury", "Book",
     ["Castles", "Military architecture", "Great Britain - History"], False, None),
    ("The Last Anglo-Saxon Court", "Weatherall, Susannah", "", "2020", "Thornbury", "Book",
     ["Great Britain - History", "Kings and rulers"], False, None),
    ("Embroidered Histories", "Weatherall, Susannah", "", "2022", "Thornbury", "Book",
     ["Textile fabrics", "Great Britain - History", "Art and history"], False, None),
    ("The Crossing", "Weatherall, Susannah", "", "2024", "Thornbury", "Book",
     ["Great Britain - History", "Naval history"], False, None),
    ("Roads Before Rails", "Bramwell, Douglas", "1944-2020", "2009", "Thornbury", "Book",
     ["Roads - History", "Transport", "Great Britain - History"], False, None),
    ("The Wall and the Frontier", "Bramwell, Douglas", "1944-2020", "2011", "Thornbury", "Large print",
     ["Romans", "Great Britain - History", "Fortification"], False, None),

    # ---- nature and science -----------------------------------------------------
    ("What the Hedgerow Knows", "Fairhead, Judith", "", "2021", "Kestrel & Vane", "Book",
     ["Natural history", "Hedges", "Countryside"], False, None),
    ("The Quiet Decline", "Okonjo, Adaeze", "", "2023", "Kestrel & Vane", "Book",
     ["Insects", "Biodiversity", "Environmental protection"], False,
     "What thirty years of one family's moth-trap records reveal about a landscape emptying out."),
    ("Reading the Sky", "Corrigan, Estelle", "", "2017", "Kestrel & Vane", "Book",
     ["Meteorology", "Weather forecasting"], False, None),
    ("Deep Water, Slow Time", "Vasilenko, Irina", "", "2022", "Kestrel & Vane", "Book",
     ["Oceanography", "Climate"], False, None),
    ("The Ice Record", "Weatherall, Susannah", "", "2019", "Kestrel & Vane", "Book",
     ["Climatology", "Glaciers", "Palaeoclimatology"], False, None),

    # ---- children's --------------------------------------------------------------
    ("The Wednesday Garden", "Sciacca, Emilia", "", "2018", "Rookery", "Book",
     ["Children's stories", "Gardens - Fiction", "Friendship - Fiction"], True,
     "A girl inherits a walled garden nobody has opened in forty years. For readers of eight and up."),
    ("The Boy Who Counted Rain", "Sciacca, Emilia", "", "2021", "Rookery", "Book",
     ["Children's stories", "Family - Fiction"], True, None),
    ("Marigold and the Long Walk Home", "Nkemelu, Chidi", "", "2022", "Rookery", "Book",
     ["Children's stories", "Adventure stories"], True, None),
    ("The Lighthouse at the End of Term", "Trevanion, Bryony", "", "2020", "Rookery", "eAudiobook",
     ["Children's stories", "Schools - Fiction"], True, None),
    ("Ten Things About Beetles", "Fairhead, Judith", "", "2023", "Rookery", "Book",
     ["Children's non-fiction", "Insects"], False, None),

    # ---- DVDs ----------------------------------------------------------------------
    ("The Salt House [DVD]", "Trevanion, Bryony", "", "2022", "Aldgate Screen", "DVD",
     ["Feature films", "Domestic fiction"], True, None),
    ("Marlowe Hall [DVD]", "Grewal, Ravi", "", "2023", "Pennyfarthing Screen", "DVD",
     ["Feature films", "Detective and mystery films"], True, None),
    ("A Year on the Estuary [DVD]", "Fairhead, Judith", "", "2021", "Kestrel Screen", "DVD",
     ["Documentary films", "Natural history"], False, None),

    # ---- degraded records, as found in every real catalogue -------------------------
    ("[Punjabi book]", "Grewal, Prem", "1931-1990", "UUUU", "", "Book",
     ["Spiritual life"], False, None),
    ("[Sound recording]", "", "", "UUUU", "", "eAudiobook", [], False, None),
    ("Untitled volume 2", "Bramwell, Douglas", "1944-2020", "19??", "Thornbury", "Book",
     ["Great Britain - History"], False, None),
    ("[Large print item]", "", "", "2004", "", "Large print", ["Fiction"], True, None),
    ("Selected works", "Grewal, Prem", "1931-1990", "1998", "Wayfarer Press", "Book",
     ["Spiritual life", "Collected works"], False, None),
]


def render_record(author, dates, year, publisher, subjects, blurb):
    """The record text as a catalogue actually stores it - and all that is indexed.

    The blurb is included when present because a real catalogue indexes its
    summary field too. Only about one record in five has one, which is the
    inconsistency the enrichment work exists to fix.
    """
    lines = []
    who = f"{author}, {dates}" if author and dates else author
    lines.append(f"Author: {who}" if who else "Author: [unknown]")
    pub = ", ".join(x for x in (publisher, year) if x)
    lines.append(f"Published: {pub}" if pub else "Published: [no date]")
    if subjects:
        lines.append("Subjects: " + "; ".join(subjects))
    if blurb:
        lines.append(f"Summary: {blurb}")
    return "\n".join(lines)


def build():
    records = []
    for i, (title, author, dates, year, publisher, fmt, subjects, fiction, blurb) in enumerate(R):
        doc_id = f"nl-{100 + i:05d}"
        digest = hashlib.sha256(doc_id.encode()).digest()
        available = 0 if digest[2] % 5 == 0 else 1 + digest[3] % 3
        copies = available + digest[4] % 3
        records.append({
            # --- corpus fields: what BetterSearch indexes -------------------
            "doc_id": doc_id,
            "title": title,
            "text": render_record(author, dates, year, publisher, subjects, blurb),
            "tags": [fmt.lower()],
            # --- display fields: ignored by the library, used by the page ---
            "author": author,
            "author_dates": dates,
            "year": year,
            "publisher": publisher,
            "format": fmt,
            "language": "English",
            "fiction": fiction,
            "subjects": subjects,
            "location": "Online" if fmt == "eAudiobook" else BRANCHES[digest[5] % len(BRANCHES)],
            "available": max(copies, 1) if fmt == "eAudiobook" else available,
            "copies": max(copies, 1),
            "blurb": blurb,
            "cover": PALETTE[digest[6] % len(PALETTE)],
        })

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "name": "northfield-libraries",
        "description": (
            "Demonstration catalogue for a fictional library service. Records are "
            "invented; availability figures are illustrative."
        ),
        "documents": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    words = [len(r["text"].split()) + len(r["title"].split()) for r in records]
    with_blurb = sum(1 for r in records if r["blurb"])
    print(f"Wrote {len(records)} records to {OUT}")
    print(f"Words per record: min {min(words)}  median {sorted(words)[len(words)//2]}  max {max(words)}")
    print(f"Records with a blurb: {with_blurb} ({with_blurb/len(records):.0%})")
    print(f"Formats: {sorted({r['format'] for r in records})}")


if __name__ == "__main__":
    build()
