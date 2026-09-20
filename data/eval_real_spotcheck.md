# Judge spot check

50 random judgements from `claude-haiku-4-5`, for you to sanity-check by eye.
The question is not whether you agree with every one - it is whether the
judge is reading **intent** or just matching words. If it is grading a
thriller as relevant to "something gentle to read before bed" because the
record says "night", the eval is not measuring what we think it is.

Measured self-consistency on a re-judged sample: **94%** identical.

**1** — _books like Agatha Christie_
```
The Secret Bedroom
Author: Robert Lawrence Stine
Published: Gareth Stevens Pub, 1991
Subjects: Fiction; Horror tales; Revenge; Horror; collectionID:FearStreet; Children's fiction; Mystery and detective stories; Horror stories
```
judge: Mystery element present, but horror focus differs from Christie's detective style.

**2** — _what to do about money worries_
```
Focus on personal finance
Author: Jack R. Kapoor
Published: McGraw-Hill/Irwin, 2013
Subjects: Personal Finance; Investments
```
judge: Directly addresses personal finance and money management concerns.

**0** — _the one about a boy wizard at school_
```
The Great Automatic Grammatizator
Author: Roald Dahl
Published: Tandem Library, 1982
Subjects: Teen & Young Adult Fiction; Teen & Young Adult Mystery & Suspense; Juvenile audience; Fiction; Juvenile Fiction; Young Adult Fiction; Wit and humor; Short stories
```
judge: Roald Dahl story collection, not about a boy wizard at school.

**1** — _cycling long distances_
```
A Walk in the Woods
Author: Bill Bryson
Published: Goldmann, 1997
Subjects: Description and travel; Natural history; Travel; Journeys; Hiking; Nonfiction; Natural history, united states; Appalachian trail
```
judge: Long-distance travel/journey narrative, but hiking not cycling.

**0** — _learning to take better photographs_
```
Woodworking
Author: Nancy MacDonald
Published: CENGAGE Delmar Learning, 2008
Subjects: Carpentry; Woodwork
```
judge: Woodworking book has no connection to photography instruction.

**2** — _poems about loss_
```
Poems and Prose
Author: Edgar Allan Poe
Published: Alfred A. Knopf, 1995
Subjects: American fantasy poetry; Poetry (poetic works by one author); Laments; Narrative poetry; Poetry; Grief; American poetry; American Children's poetry
```
judge: Poe collection with grief and laments as subjects directly matches loss poetry query.

**2** — _books like Agatha Christie_
```
The Secret of Chimneys
Author: Agatha Christie
Published: Wydawnictwo Hachette, 1925
Subjects: Fiction; Mystery; Superintendent Battle (Fictitious character); English Detective and mystery stories; Murder; Investigation; Conspiracies; Manors
```
judge: Exact author match. Classic mystery novel by Agatha Christie herself.

**2** — _dog training for a new puppy_
```
The Dog's Mind
Author: Bruce Fogle
Published: Pelham Books/Stephen Greene Press, 1990
Subjects: Dogs; Behavior; Psychology; Training; Dogs, psychology
```
judge: Covers dog behavior, psychology and training - directly addresses puppy training needs.

**2** — _something similar to Sherlock Holmes_
```
Death in the Clouds
Author: Agatha Christie
Published: Ren min wen xue, 1935
Subjects: Detective and mystery stories; Fiction; Hercule Poirot (Fictitious character); Private investigators; Murder; English Detective and mystery stories; Mystery; Murder in ficti
```
judge: Classic detective mystery by renowned author similar to Holmes stories.

**1** — _cycling long distances_
```
Three Men on the Bummel
Author: Jerome Klapka Jérôme
Published: Sutton Publishing Ltd, 1900
Subjects: Cycling; Humor; British; Fiction
```
judge: Fiction about cycling, but comedic travel narrative, not instructional guide.

**1** — _understanding how computers actually work_
```
Advances in Computers, Volume 49 (Advances in Computers)
Author: Marvin V. Zelkowitz
Published: Elsevier Science & Technology, 1995
Subjects: Computers, periodicals; Electronic data processing; Informatique; Computers; Ordinateurs; Machine Theory; Reference; G
```
judge: Technical coverage of computers, but academic volume may not suit beginner understanding.

**2** — _a book that will make me cry_
```
Déjalos ir con amor
Author: Nancy O'Connor
Published: Trillas, 1990
Subjects: Bereavement; Psychological aspects; Grief
```
judge: Book about grief and bereavement directly addresses emotional need to process loss.

**1** — _a book that will make me cry_
```
Destroy Me
Author: Tahereh Mafi
Published: Farshore, 2012
Subjects: Dystopian; science fiction; young adult; romance; dystopian romance; science fiction romance; forbidden love; power dynamics
```
judge: Emotionally intense dystopian romance, may move reader but not guaranteed tearful.

**2** — _graphic novels for teenagers_
```
Jujutsu Kaisen, Vol. 7
Author: Gege Akutami
Published: Viz Media, 2020
Subjects: Comics & graphic novels, manga, horror; Comics & graphic novels, manga, media tie-in; Comics & graphic novels, manga, fantasy
```
judge: Manga graphic novel, appropriate for teenage audience, horror and fantasy genres.

**0** — _something similar to Sherlock Holmes_
```
Home woodworking
Author: F. E. Sherlock
Published: Newnes Technical, 1982
Subjects: Amateurs' manuals; Woodwork; Woodworking; Woodwork - General; Crafts / Hobbies
```
judge: Woodworking manual, not detective fiction. Author surname coincidence.

**0** — _if I liked Lord of the Rings what next_
```
One of us is next
Author: Karen M. McManus
Published: Alfaguara Infantil, 2020
Subjects: Young adult fiction, mysteries & detective stories; Young adult fiction, school & education, general; Young adult fiction, thrillers & suspense; Young adult fiction, socia
```
judge: Contemporary YA mystery thriller; completely unrelated to epic fantasy.

**1** — _Joy of Cooking_
```
Guide to good food
Author: Velda L. Largen
Published: Goodheart-Wilcox Co., 1979
Subjects: Food; Nutrition; International Cookery; Cookery, International; General; Education / Teaching; Education; EDU
```
judge: Similar cookbook but not the exact famous title reader sought.

**0** — _a novel where the beekeeper is the detective_
```
Hector Finds a Fortune (Ready-for-Chapters)
Author: Elizabeth Shreeve
Published: Fitzgerald Books, 2004
Subjects: Uncles; Insects; Juvenile fiction; Fiction; Friendship; Travel; Beekeepers; Beekeeping
```
judge: Juvenile adventure featuring beekeeping, not detective mystery novel.

**0** — _graphic novels for teenagers_
```
A New Beginning
Author: John Flanagan
Published: Puffin Books, 2013
Subjects: JUVENILE FICTION; Action & Adventure; Apprentices; Fantasy fiction; Fiction; Fantasy; JUVENILE FICTION / Action & Adventure / General; JUVENILE FICTION / Fantasy & Magic
```
judge: Text novel, not graphic novel. Wrong format entirely.

**0** — _a book that will make me cry_
```
Boys don't cry
Author: Malorie Blackman
Published: Random House Children's Books, 2010
Subjects: Teenage fathers; Fiction; Single parents
```
judge: Emotional manipulation title, but YA fiction about fatherhood unlikely to meet emotional need.

**2** — _a funny book to cheer me up_
```
No Brainer
Author: Jeff Kinney
Published: Bound to Stay Bound Books, 2023
Subjects: Children's fiction; Enjoying; funny; Silly; Juvenile fiction
```
judge: Funny children's book by popular humorous author, explicitly marked silly.

**0** — _dog training for a new puppy_
```
Anatomy of the dog
Author: Malcolm E. Miller
Published: W.B. Saunders, 1964
Subjects: Anatomy; Dogs; Veterinary anatomy; Dogs, anatomy; Chiens; Anatomie; Dogs, pictorial works; Anatomy & histology
```
judge: Book covers anatomy, not puppy training or behavior.

**2** — _helping a child who will not sleep_
```
Sleep Is for Everyone
Author: Paul Showers
Published: HarperCollins, 1974
Subjects: Sleep; Sleep, juvenile literature; Juvenile literature; Sleep -- Juvenile literature.; Children's fiction; Sleep, fiction
```
judge: Children's book about sleep, directly addresses the reader's need.

**2** — _something short I can finish in one sitting_
```
I Really Like Slop!
Author: Mo Willems
Published: Babaryba, 2015
Subjects: Pigs; Swine; Elephants; Juvenile fiction; Friendship; Food habits; Fiction; New York Times bestseller
```
judge: Short children's book easily readable in one sitting.

**1** — _a funny book to cheer me up_
```
Science of Happiness
Author: Stefan Klein
Published: Hachette Books, 2009
Subjects: Popular science; Psychology; Happiness; Pleasure; Smiling
```
judge: About happiness but likely serious science, not humorous entertainment.

**0** — _the one about a boy wizard at school_
```
The Last Straw
Author: Jeff Kinney
Published: Baumhaus Verlag GmbH, 2008
Subjects: Diaries; Middle schools; Juvenile fiction; Family life; Fiction; Schools; Families; Parent-Child Relations
```
judge: School setting but no wizard or fantasy elements; wrong series entirely.

**1** — _graphic novels for teenagers_
```
Watchmen
Author: Alan Moore
Published: Panini Verlags GmbH, 1986
Subjects: Watchmen (Comic strip); Graphic novels; Comic books, strips; New York Times bestseller; Comics & graphic novels, science fiction; Superheroes; Crimes against; Comics & graphic novels, s
```
judge: Graphic novel, but adult-oriented dark content unsuitable for most teenagers.

**2** — _identifying garden birds_
```
Bird (Eyewitness Guide)
Author: David Burnie
Published: Distributed by Random House, 1988
Subjects: Nature; Nonfiction; Aves; Juvenile literature; Birds; Literatura juvenil; Uccelli; Libri per ragazzi
```
judge: Comprehensive bird identification guide, appropriate scope and format.

**0** — _Sue Monk Kidd_
```
The Hostile Hospital
Author: Lemony Snicket
Published: HarperCollins Publishers New Zealand, 2001
Subjects: Juvenile fiction; Orphans; Medical fiction; Brothers and sisters; Children's fiction; Orphans, fiction; Brothers and sisters, fiction; Humorous stories
```
judge: Reader wants Sue Monk Kidd; this is Lemony Snicket, different author entirely.

**0** — _Joy of Cooking_
```
A new system of domestic cookery
Author: Maria Eliza Ketelby Rundell
Published: Robert M'Dermut, 1800
Subjects: American Cookery; Cookery; English Cookery; Cooking; American Cooking; English Cooking
```
judge: Different cookbook from different author and era. Not Joy of Cooking.

**1** — _learning to grow vegetables in pots_
```
Growing vegetables west of the Cascades
Author: Steve Solomon
Published: Sasquatch Books, 1989
Subjects: Vegetable gardening; Organic gardening; Gardening; Nonfiction
```
judge: Vegetable gardening guide, but focused on regional growing, not container gardening.

**0** — _a novel where the beekeeper is the detective_
```
Learning about bees from Mr. Krebs
Author: Alice K. Flanagan
Published: Children's Press, 1999
Subjects: Juvenile literature; Bee culture; Beekeepers; Honeybee; Occupations; Bees
```
judge: Nonfiction about beekeeping, not a detective novel.

**0** — _getting fit again after illness_
```
Getting a job you love during a tough economy
Author: Guy, Bill the jobs guy
Published: GlobaLeadershiPublishing, 2014
Subjects: Job hunting
```
judge: Job hunting guide, unrelated to fitness or health recovery.

**1** — _woodworking with hand tools_
```
Woodworking technology
Author: James J. Hammond
Published: Taplinger Pub Co, 1961
Subjects: Woodwork; Woodwork (Manual training); Textbooks
```
judge: Woodworking topic but unclear if hand tools focus or general technology.

**2** — _books like Agatha Christie_
```
Masterpieces of mystery and suspense
Author: Martin H. Greenberg
Published: Doubleday Book & Music Clubs, Inc., 1988
Subjects: American Detective and mystery stories; English Detective and mystery stories; Detective and mystery stories; Literature, collections
```
judge: Anthology of detective and mystery stories, exactly Christie's genre and style.

**0** — _The Secret Life of Bees_
```
Osobnosti apidológie
Author: Jozef Čižmárik
Published: Alexandra, 2003
Subjects: Biography; Beekeepers
```
judge: Book about beekeepers' biographies, not the novel 'The Secret Life of Bees'.

**1** — _history of my local area_
```
Index to American Genealogies: And to Genealogical Material Contained in All Works Such as Town ..
Author: Daniel Steele Durrie
Published: J. Munsell's sons, 1900
Subjects: Bibliography; Local History; Genealogy
```
judge: Genealogy resource with local history focus, but specialized index rather than narrative history.

**1** — _Joy of Cooking_
```
Frugal housewife
Author: Lydia Maria Child
Published: T. Tegg and Son, 1829
Subjects: Home economics; Cookery; American Cookery; Handbooks, manuals; Low budget cookery; American Cooking; Cooking; Low budget cooking
```
judge: Historic cookbook on budget cooking, not the Joy of Cooking specifically.

**0** — _The Secret Life of Bees_
```
Practical queen rearing
Author: Frank Chapman Pellett
Published: American Bee Journal
Subjects: Bee-keeping
```
judge: Non-fiction beekeeping manual, not the novel reader seeks.

**0** — _something short I can finish in one sitting_
```
Woodworking
Author: Moran, Bob
Published: Reader's Digest, 1996
Subjects: Woodwork; Woodworking tools; Technique
```
judge: Instructional woodworking book, not leisure reading for quick completion

**1** — _Joy of Cooking_
```
Theory & practice of good cooking
Author: James Beard
Published: Penguin, 1977
Subjects: Cooking; Cookery
```
judge: Classic cooking guide by renowned chef, but different book than Joy of Cooking

**0** — _how to keep bees in a small garden_
```
Small business management
Author: Justin G. Longenecker
Published: Cengage South-Western, 2000
Subjects: Management; Small business; Small business, management
```
judge: About small business management, not beekeeping or gardening.

**2** — _starting a business with no money_
```
Entrepreneurial finance
Author: Philip J. Adelman
Published: Prentice Hall, 1997
Subjects: Finance; Small business; Budgeting & financial management; Small businesses & self-employed; Small Business Finance; Business & Economics; Small Business/Entrepreneurshi
```
judge: Directly addresses entrepreneurial finance and small business startup needs.

**0** — _that book about a woman and bees in South Carolina_
```
Die Imker
Author: Roth, Gerhard
Published: S. Fischer, 2022
Subjects: Beekeepers; Fiction
```
judge: German book about beekeepers, not South Carolina woman and bees story.

**0** — _something gentle to read before bed_
```
Music notation
Author: Gardner Read
Published: Taplinger Pub. Co., 1964
Subjects: Musical notation; Musique; Notation
```
judge: Technical manual about music notation, not a gentle bedtime read.

**2** — _something gentle to read before bed_
```
A Gentle Reminder
Author: Bianca Sparacino
Published: Thought catalog, 2020
Subjects: Self-help; personal development; self-love; self-care; mindfulness; motivation; inspiration; personal growth
```
judge: Gentle self-help with mindfulness and self-care subjects perfectly suits bedtime reading.

**0** — _learning to grow vegetables in pots_
```
The Marijuana Growing Bible
Author: Marijuana
Published: Marijuana Cannabis Association, 2021
Subjects: Gardening
```
judge: Growing marijuana illegally/inappropriately, not vegetable gardening instruction.

**0** — _a book that will make me cry_
```
The Berenstain Bears Hug and Make Up (Berenstain Bears)
Author: Mike Berenstain
Published: HarperFestival, 2006
Subjects: Children: Kindergarten; Bears, fiction; Family life, fiction; Children's fiction; Automobile travel, fiction
```
judge: Children's picture book unlikely to evoke emotional crying in adult reader.

**2** — _what to do about money worries_
```
Creating money
Author: Sanaya Roman
Published: H J Kramer Inc., published in a joint, 1988
Subjects: Channeling (Spiritualism); Money; Success; Mind, body, spirit: disciplines & techniques; Personal Finance; Personal Guidance; Self-Help; New Age / Body, Mind &
```
judge: Self-help book directly addressing money concerns and personal finance guidance.

**1** — _an uplifting story after a hard year_
```
A beekeeper's year
Author: Janet Luke
Published: 2017
Subjects: Bee culture; Beekeepers
```
judge: Gentle nature topic, but unclear if narrative is uplifting or primarily instructional.
