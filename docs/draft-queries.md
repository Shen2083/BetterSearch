# The 41 draft eval queries

These decide what the 4,000-record measurement actually proves, so they are
the one part of this I should not be marking myself. Edit freely.

## What I need from you

1. **Would a real person type this into a library catalogue?** If not, reword or cut it.
2. **Cut freely.** 30 good queries beat 41 mediocre ones. Nothing is wasted —
   relevance judgements are generated afterwards, so changing a query here costs nothing.
3. **Please keep several controls.** They are deliberately keyword-friendly and stop
   the set flattering semantic search by construction.
4. **Add your own.** Anything you have heard a real borrower ask.

⚠️ marks a query where I already saw a problem — those are the ones most worth your eye.

To edit: `data/eval_real_queries_draft.json`. To see what each one currently
retrieves from the real catalogue: `docs/query-review.md`.

---

## natural-language subject  (10)

**1. `how to keep bees in a small garden`**  

> Subject exists in the corpus; wording will not match headings.

**2. `learning to grow vegetables in pots`**  

> Container gardening, phrased as a beginner would.

**3. `books about the night sky for beginners`**  

> Astronomy without using the word.

**4. `understanding how computers actually work`**  

> Vague, non-technical phrasing.

**5. `starting a business with no money`**  

> Small business, framed by constraint.

**6. `what to do about money worries`**  

> Personal finance, emotional framing.

**7. `identifying garden birds`**  

> Should be reachable by both lanes.

**8. `learning to take better photographs`**  

> Photography, skill framing.

**9. `knitting patterns for beginners`**  

> Close to the heading - a keyword lane should do well.

**10. `woodworking with hand tools`**  

> Corpus uses 'woodwork'; tests morphology.


## subject, natural phrasing  (5)

**11. `second world war in the pacific`**  

> Corpus heading is 'World War, 1939-1945'. Classic vocabulary mismatch.

**12. `history of my local area`**  

> Local history; deliberately unanchored to a place.

**13. `poems about loss`**  

> Crosses poetry and bereavement.

**14. `graphic novels for teenagers`**  

> Two facets at once.

**15. `cycling long distances`**  

> Endurance cycling, phrased plainly.


## task  (8)

**16. `getting my toddler to eat vegetables`**  

> Parenting; no heading will contain this.

**17. `helping a child who will not sleep`**  

> Spans parenting and sleep.

**18. `coping after someone dies`**  ⚠️

> Bereavement in plain words.
>
> **Problem seen:** 115 grief-adjacent records exist, but they are Ubik, Mort, Death Note — fiction where death is a theme, not bereavement support.

**19. `preparing for a job interview`**  

> Job hunting.

**20. `what to expect when you are pregnant`**  

> Near a famous title - tests title bleed.

**21. `getting fit again after illness`**  

> Fitness with a qualifier.

**22. `learning to cook on a budget`**  

> Cookery plus constraint.

**23. `dog training for a new puppy`**  

> Should be easy for both lanes.


## mood  (7)

**24. `something gentle to read before bed`**  ⚠️

> No subject heading exists for this. Core product claim.
>
> **Problem seen:** Both lanes read it literally as books *about* bedtime, not a calming novel.

**25. `a funny book to cheer me up`**  

> Humour, phrased by effect not genre.

**26. `gripping but not too violent`**  ⚠️

> Two constraints, one negative. Negation is hard for both lanes.
>
> **Problem seen:** Both lanes fail. Negation defeats them. Keep as evidence of a limit, or cut.

**27. `something short I can finish in one sitting`**  ⚠️

> Length, which is not in the record at all. May be unanswerable - worth knowing.
>
> **Problem seen:** Keyword accidentally WINS (finds short-story collections); semantic returns Woodworking. Length is not in the record at all.

**28. `a book that will make me cry`**  

> Pure affect.

**29. `cosy mystery, nothing gruesome`**  

> The gentle-crime case that keyword got backwards on the small corpus.

**30. `an uplifting story after a hard year`**  ⚠️

> Affect plus life context.
>
> **Problem seen:** Semantic latches onto 'hard year' and returns grief titles — it inverts the intent.


## read-alike  (3)

**31. `books like Agatha Christie`**  

> Read-alike. Keyword will return Christie herself; semantic should find similar authors. The judge decides which is right.

**32. `if I liked Lord of the Rings what next`**  

> Conversational read-alike.

**33. `something similar to Sherlock Holmes`**  

> Detective read-alike.


## half-remembered  (3)

**34. `that book about a woman and bees in South Carolina`**  

> The Secret Life of Bees. Tests vague known-item.

**35. `the one about a boy wizard at school`**  

> May not be in the corpus; a legitimate zero-result case.

**36. `a novel where the beekeeper is the detective`**  

> Deliberately specific and possibly unanswerable.


## control (keyword should win)  (5)

**37. `Joy of Cooking`**  

> Exact title. If semantic loses here that is expected and fine.

**38. `Agatha Christie`**  

> Exact author.

**39. `The Secret Life of Bees`**  

> Exact title; pairs with the half-remembered version above.

**40. `Betty Crocker`**  

> Exact author, single work.

**41. `Sue Monk Kidd`**  

> Exact author.


---

**41 queries** · 5 flagged · 5 controls

Provenance recorded in the README: drafted by the system's author, reviewed and
edited by Shen. Not claimed as independently written.