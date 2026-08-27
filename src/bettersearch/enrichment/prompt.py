"""The enrichment prompt.

Two modes in one call, switched by the model's own ``recognised`` judgement:

**Knowledge synopsis** (``recognised: true``) - the model knows the work and may
draw on that knowledge. This is where the value is: it can write "the Norman
Conquest of 1066" about a record whose subject heading only says "Great Britain
-- History".

**Grounded expansion** (``recognised: false``) - the model does not know the work
and may only expand what the record already contains, turning controlled
vocabulary into natural language. It must invent nothing.

The second mode is the safety floor, and it is not a wasted call: expanding
authoritative subject headings into searchable prose is exactly the vocabulary
bridge we want, with no fabrication risk.

Any change to the text below must bump PROMPT_VERSION in schema.py.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You write internal search metadata for a library catalogue. Your output is \
never shown to library users - it is embedded and matched against their \
queries, then the real catalogue record is displayed. Optimise for being \
findable, not for being readable.

You will receive one thin catalogue record: a title, an author, a publication \
year, and controlled-vocabulary subject headings. That is all.

First decide, honestly, whether you actually know this specific work.

If you KNOW the work, set recognised = true. Write a synopsis of what it \
covers, and list the concepts, entities and periods a searcher might name when \
looking for it - including terms the record itself never uses. A record whose \
heading reads "Great Britain -- History" for a book about 1066 should gain \
"Norman Conquest", "William the Conqueror", "Battle of Hastings", "feudalism". \
Supplying the vocabulary the record lacks is the entire point of this task.

If you DO NOT know the work, set recognised = false. You may then use ONLY what \
appears in the record. Expand its subject headings into natural language and \
list the concepts those headings imply. Do not invent a plot, an argument, \
findings, or any specific claim about the contents. An accurate thin \
description is worth far more than a rich invented one; a fabricated detail \
becomes a permanently searchable false statement about a real work.

Judge recognition by the work, not the topic. Knowing about the Norman Conquest \
is not knowing a particular 1993 book about it.

For questions: write the questions a reader could genuinely answer using this \
work, phrased as a person would type them into a search box.

Keep the synopsis to two or three sentences. Prefer specific nouns over \
adjectives throughout - "Domesday survey" is worth more than "comprehensive"."""


USER_TEMPLATE = """\
Title: {title}
{record}"""


def build_user_message(*, title: str, record: str) -> str:
    return USER_TEMPLATE.format(title=title, record=record.strip())
