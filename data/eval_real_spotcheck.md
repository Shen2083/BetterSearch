# Judge spot check

50 random judgements from `claude-haiku-4-5`, for you to sanity-check by eye.
The question is not whether you agree with every one - it is whether the
judge is reading **intent** or just matching words. If it is grading a
thriller as relevant to "something gentle to read before bed" because the
record says "night", the eval is not measuring what we think it is.

Measured self-consistency on a re-judged sample: **98%** identical.

**0** — _the one about a boy wizard at school_
```
The Thing About December
Author: Donal Ryan
Published: Transworld Publishers Limited, 2013
Subjects: Fathers and sons; Fiction; Bereavement; Land tenure; Fiction, general
```
judge: Contemporary Irish fiction about bereavement and land tenure, not fantasy or wizards.

**0** — _The Secret Life of Bees_
```
How to Keep Bees
Author: Miller, Arthur C. inspector of apiaries.
Published: CreateSpace Independent Publishing, 2014
Subjects: Beekeeping
```
judge: Practical beekeeping manual, not the Sue Monk Kidd novel requested.

**2** — _knitting patterns for beginners_
```
Winter essential knits
Author: Quail Studio
Published: Quail Publishing, 2015
Subjects: Knitting; Patterns; Knitting, patterns
```
judge: Knitting patterns book directly matches beginner-focused query intent.

**0** — _the one about a boy wizard at school_
```
The Wonderful Story of Henry Sugar and Six More
Author: Roald Dahl
Published: Alfred A. Knopf, 1977
Subjects: Juvenile books; Teenage fiction; Children's stories, English; Short stories; JUVENILE FICTION / Short Stories; JUVENILE FICTION / Fantasy & Magic; JUV
```
judge: Short story collection, not novel about boy wizard at school.

**2** — _the one about a boy wizard at school_
```
Carry On
Author: Rainbow Rowell
Published: Macmillan, 2015
Subjects: gay; fantasy; magical schools; Young adult fiction, fantasy, contemporary; Young adult fiction, romance, paranormal; Magic; JUVENILE FICTION / Love & Romance; JUVENILE FICTION / Fantasy & Mag
```
judge: Boy wizard at magical school - this directly matches the query intent.

**1** — _a novel where the beekeeper is the detective_
```
The keeper of the bees
Author: Gene Stratton-Porter
Published: Indiana UP, 1925
Subjects: Fiction; Bee culture; Beekeepers; Fiction, general; Indiana, fiction
```
judge: Features a beekeeper protagonist, but not a detective novel.

**0** — _Sue Monk Kidd_
```
Tree of Birds
Author: Susan Meddaugh
Published: Tandem Library, 1988
Subjects: Juvenile fiction; Fiction; Migration; Birds; School stories; Humorous stories; Schools; Humorous fiction
```
judge: Wrong author. Sue Monk Kidd writes adult fiction, not juvenile books.

**2** — _an uplifting story after a hard year_
```
Option B
Author: Sheryl Sandberg
Published: Knopf, 2017
Subjects: Relationships; Parenting; Grief; Resilience (Personality trait); Loss (Psychology); Bereavement; Family; BIOGRAPHY & AUTOBIOGRAPHY
```
judge: Resilience and bereavement recovery directly address uplifting after hardship.

**0** — _an uplifting story after a hard year_
```
The Lonely Voice
Author: Frank O'Connor
Published: Melville House Pub., 1963
Subjects: Short story; Short stories - literary criticism
```
judge: Literary criticism of short stories, not an uplifting narrative to read.

**0** — _woodworking with hand tools_
```
Wood machining processes
Author: Koch, Peter
Published: Ronald Press Co., 1964
Subjects: Woodwork; Woodworking machinery
```
judge: Reader wants hand tools; record focuses on machinery processes.

**0** — _a novel where the beekeeper is the detective_
```
Fables & Reflections
Author: Neil Gaiman
Published: Vertigo, 1993
Subjects: Literature; Comics & graphic novels, general; Fantasy; Graphic novel; Mythology; Fairy tales; Fables; Dreams
```
judge: Fables collection, not a novel with beekeeper detective plot.

**2** — _understanding how computers actually work_
```
Code
Author: Charles Petzold
Published: Microsoft Press, 1999
Subjects: Computer programming; Coding theory; Datenverarbeitung; Computers; Hardware; Coderingstheorie; Programmatuurtechniek; Einfu hrung
```
judge: Petzold's Code explains computer fundamentals from circuits to programming.

**1** — _history of my local area_
```
Writing local history
Author: David Dymond
Published: Bedford Square Press / NCVO, 1981
Subjects: Local History
```
judge: Guide to researching local history, not history of a specific area.

**1** — _second world war in the pacific_
```
World War II
Author: Grolier Educational
Published: Gill & MacMillan, 2006
Subjects: World War, 1939-1945; Encyclopedias; World War (1939-1945) fast (OCoLC)fst01180924
```
judge: General WWII coverage, but likely lacks Pacific theatre depth.

**2** — _knitting patterns for beginners_
```
Vogue Knitting
Author: Trisha Malcolm
Published: Sixth&Spring Books, 1999
Subjects: Patterns; Knitting; Knitting, patterns
```
judge: Directly addresses beginner knitting patterns with authority.

**1** — _an uplifting story after a hard year_
```
Chicken Soup for the Grieving Soul
Author: Jack Canfield
Published: HCI, 2003
Subjects: Self-Improvement; Bereavement; Nonfiction
```
judge: Uplifting stories present, but focused on grief rather than general renewal.

**0** — _gripping but not too violent_
```
Love your life not theirs
Author: Rachel Cruze
Published: Dreamscape Media, 2016
Subjects: Money; Personal Finance; Finance, personal
```
judge: Personal finance book, not fiction. Tone/violence level irrelevant.

**0** — _the one about a boy wizard at school_
```
Coach Hyatt is a riot!
Author: Dan Gutman
Published: HarperTrophy, 2008
Subjects: Fiction; Humorous stories; Football in fiction; Schools; Football; Coaches (Athletics); Schools in fiction; Juvenile Fiction
```
judge: Sports comedy about a football coach, not wizard school fiction.

**2** — _preparing for a job interview_
```
Job hunting for dummies
Author: Max Messmer
Published: Dialektika, 1995
Subjects: Miniature books; Job hunting; Specimens; Employment interviewing; Career changes
```
judge: Directly addresses job hunting and employment interviewing. Practical guide for job seekers.

**0** — _a book that will make me cry_
```
The Meltdown
Author: Jeff Kinney
Published: Baumhaus Verlag GmbH, 2007
Subjects: Humorous stories; Families; Winter; Diaries; JUVENILE FICTION / Humorous Stories; Juvenile fiction; Snow; Fiction
```
judge: Humorous juvenile fiction designed to make readers laugh, not cry.

**1** — _gripping but not too violent_
```
Calamity
Author: Brandon Sanderson
Published: Delacorte Press, 2016
Subjects: Science fiction; Guerrilla warfare; Juvenile fiction; Supervillains; JUVENILE FICTION / Boys & Men; Fiction; Children's fiction; Adventure and adventurers, fiction
```
judge: Science fiction adventure with action, but guerrilla warfare suggests violence level unclear.

**2** — _Agatha Christie_
```
The Pale Horse
Author: Agatha Christie
Published: Collins, 1952
Subjects: English Detective and mystery stories; Fiction; Mystery; Detective and mystery stories; English Novelists; Large type books; English literature; Fiction, crime
```
judge: Work by Agatha Christie, exactly what reader requested.

**2** — _woodworking with hand tools_
```
Traditional woodworking handtools
Author: Graham Blackburn
Published: Gramercy Books, 1998
Subjects: Woodworking tools; Handbooks, manuals; Woodwork; Tools; Travail du bois; Guides, manuels; Outillage
```
judge: Directly matches query: traditional woodworking with hand tools, practical handbook.

**2** — _starting a business with no money_
```
Starting a Business All-In-One for Dummies
Author: Consumer Dummies
Published: Wiley & Sons, Incorporated, John, 2015
Subjects: New business enterprises; Business planning; Small business, planning
```
judge: Directly addresses business startup basics. Practical guide for new entrepreneurs.

**1** — _poems about loss_
```
Poems
Author: Matthew Arnold
Published: The Macmillan Company, 1853
Subjects: Philosophy; Education; Poetry (poetic works by one author); English poetry; Modern Poetry; Arnold, matthew, 1822-1888; Literature, history and criticism; Poetry
```
judge: General poetry collection; loss not confirmed in sparse record metadata.

**2** — _something gentle to read before bed_
```
Sleep like a tiger
Author: Mary Logue
Published: HMH Books for Young Readers, 2012
Subjects: Animals; Fiction; Sleep; Bedtime; Sleep behavior; Sleep behavior in animals; Juvenile fiction; Picture books for children
```
judge: Gentle bedtime picture book about sleep. Exactly what they're seeking.

**0** — _Sue Monk Kidd_
```
"Surely You're Joking, Mr. Feynman"
Author: Richard Phillips Feynman
Published: W. W. Norton/Commonwealth Publishing, 1984
Subjects: Anecdotes; Anecdotes, facetiae, satire; Art; Biography; Humor; Music; Physicists; Professors
```
judge: Different author entirely. No connection to Sue Monk Kidd's works.

**0** — _if I liked Lord of the Rings what next_
```
The millionaire next door
Author: Thomas J. Stanley
Published: MJF Books, 1996
Subjects: Rich people; Millionaires; Richesse; Millionnaires; Saving and investment; Riches; Personal Banking; Economics
```
judge: Non-fiction finance book, unrelated to fantasy adventure fiction.

**0** — _the one about a boy wizard at school_
```
The Test
Author: Peggy Kern
Published: Scholastic Paperbacks, 2011
Subjects: Juvenile fiction; High school students; Teenage pregnancy; Fiction; Children's fiction; Pregnancy, fiction; Schools, fiction
```
judge: School setting present, but no wizard or magical elements. Different topic entirely.

**0** — _poems about loss_
```
Turn Up the Heat
Author: Philip L. Goglia
Published: Plume, 2002
Subjects: Nutrition; Physical fitness; Weight loss
```
judge: Book about weight loss and nutrition, not emotional loss or grief poetry.

**2** — _what to do about money worries_
```
Thou Shall Prosper
Author: Rabbi Daniel Lapin
Published: Wiley & Sons, Incorporated, John, 2002
Subjects: Wealth; Judaism; Personal Finance; Money; Finance, personal; Money, religious aspects
```
judge: Personal finance book directly addressing money concerns and wealth-building.

**1** — _history of my local area_
```
Index to American Genealogies: And to Genealogical Material Contained in All Works Such as Town ..
Author: Daniel Steele Durrie
Published: J. Munsell's sons, 1900
Subjects: Bibliography; Local History; Genealogy
```
judge: Genealogy index related to local history, but primarily a reference tool rather than history narrative.

**2** — _poems about loss_
```
Poems and Prose
Author: Edgar Allan Poe
Published: Alfred A. Knopf, 1995
Subjects: American fantasy poetry; Poetry (poetic works by one author); Laments; Narrative poetry; Poetry; Grief; American poetry; American Children's poetry
```
judge: Collection explicitly includes grief and laments; Poe renowned for elegiac poetry.

**0** — _a book that will make me cry_
```
Getting Through the Night
Author: Eugenia Price
Published: (Ulverscroft) [distributor], 1982
Subjects: Grief; Consolation; Christianity; Religious aspects of Bereavement; Religious aspects of Grief; Bereavement; Large type books
```
judge: Self-help for grief, not emotional fiction to move reader to tears.

**2** — _history of my local area_
```
Unraveling the past
Author: Maria Luisa T. Camagay
Published: Vibal Group, Inc., 2018
Subjects: Study and teaching; Sources; Local History; Education
```
judge: Local history book directly matches reader's query intent.

**2** — _how to keep bees in a small garden_
```
How to Keep Bees
Author: Miller, Arthur C. inspector of apiaries.
Published: CreateSpace Independent Publishing, 2014
Subjects: Beekeeping
```
judge: Directly addresses beekeeping fundamentals; matches query intent perfectly.

**1** — _Joy of Cooking_
```
The compleat I hate to cook book
Author: Peg Bracken
Published: Bantam Books, 1986
Subjects: Cookery; International Cookery; Cooking
```
judge: Cooking book but humorous approach, not the classic Joy of Cooking

**0** — _a funny book to cheer me up_
```
The pin-up
Author: Mark Gabor
Published: Bell, 1972
Subjects: Glamour photography; Graphic arts; Photography of women; Photography, history
```
judge: Photography history book, not humorous fiction or entertainment.

**1** — _something gentle to read before bed_
```
Chicken Little (Read-Aloud Storybook)
Author: RH Disney
Published: RH/Disney, 2005
Subjects: Children: Kindergarten; Children: Grades 1-2; Juvenile fiction
```
judge: Read-aloud storybook, but Chicken Little's panic plot may not be soothing.

**1** — _learning to grow vegetables in pots_
```
Square Foot Gardening
Author: Mel Bartholomew
Published: Rodale Books, 1981
Subjects: Square foot gardening; Vegetable gardening; Gardening
```
judge: Vegetable gardening book, but focuses on square foot method, not container/pot growing.

**2** — _an uplifting story after a hard year_
```
the sun and her flowers
Author: Rupi Kaur
Published: Kenwann, 2017
Subjects: Self-actualization (Psychology); Poetry; American poetry; New York Times bestseller; New York Times reviewed; Poetry (poetic works by one author); American poetry -- 21st century; Sel
```
judge: Poetry collection about self-actualization and personal growth. Uplifting tone matches intent.

**2** — _helping a child who will not sleep_
```
Gentle Sleep Book
Author: Sarah Ockwell-Smith
Published: Little, Brown Book Group Limited, 2015
Subjects: Infants; Preschool children; Toddlers; Children, sleep; Pediatrics; Sleep; Care
```
judge: Directly addresses children's sleep issues with parenting guidance approach.

**2** — _second world war in the pacific_
```
Unbroken
Author: Laura Hillenbrand
Published: April Yaynclk, 2010
Subjects: New York Times bestseller; Prisoners of war; Campaigns; American Aerial operations; World War, 1939-1945; Long-distance runners; United States; Biography
```
judge: Detailed WWII Pacific prisoner-of-war biography with aerial campaigns and combat.

**1** — _learning to grow vegetables in pots_
```
Question box
Author: United States. Department of Agriculture. Radio Service
Published: United States Department of, 1941
Subjects: Vegetables; Nutrition; Berries; Gardening; Victory gardens
```
judge: Gardening and vegetables present, but unclear if specifically about container/pot gardening.

**2** — _knitting patterns for beginners_
```
Knitting Now
Author: Gabi Tubbs
Published: Scribner, 1985
Subjects: Patterns; Knitting; Knitting, patterns
```
judge: Directly matches query for knitting patterns. Appears to be instructional resource.

**0** — _the one about a boy wizard at school_
```
Magical child
Author: Joseph Chilton Pearce
Published: Bantam Books, 1977
Subjects: Child psychology; Child rearing; Psychologie; Enfants
```
judge: Non-fiction psychology text, not fiction about wizard boy at school.

**0** — _getting fit again after illness_
```
Women & madness
Author: Phyllis Chesler
Published: PALGRAVE MACMILLAN, 1972
Subjects: Women; Sex role; Psychology; Mental health; Social aspects; Mental illness; Sociological aspects; Health and hygiene
```
judge: Mental illness text, not fitness/recovery guidance after illness.

**0** — _that book about a woman and bees in South Carolina_
```
The splendour of South Indian music
Author: P. T. Chelladurai
Published: Vaigarai Publishers, 1991
Subjects: Carnatic music; History and criticism; Music; Ragas
```
judge: About South Indian music history, not South Carolina or bees.

**0** — _Sue Monk Kidd_
```
Humble Pi
Author: Matt Parker
Published: Mundi, 2019
Subjects: Mathematics; Mathematics, popular works; Mathematics, miscellanea; Humor, general
```
judge: Reader asked for Sue Monk Kidd; this is a math humor book by different author.

**1** — _an uplifting story after a hard year_
```
You Can Heal Your Life
Author: Louise Hay
Published: Hay House Publishing, 2008
Subjects: Self-actualization (psychology); Mind and body; Self-care, health
```
judge: Self-help wellness book, not narrative fiction. Adjacent to emotional uplift intent.
