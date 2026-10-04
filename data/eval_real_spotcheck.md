# Judge spot check

50 random judgements from `claude-haiku-4-5`, for you to sanity-check by eye.
The question is not whether you agree with every one - it is whether the
judge is reading **intent** or just matching words. If it is grading a
thriller as relevant to "something gentle to read before bed" because the
record says "night", the eval is not measuring what we think it is.

Measured self-consistency on a re-judged sample: **94%** identical.

**2** — _something similar to Sherlock Holmes_
```
Peril at End House
Author: Agatha Christie
Published: Center Point Large Print, 1932
Subjects: Captain Hastings (Fictitious character); English Detective and mystery stories; Fiction; Hercule Poirot (Fictitious character); Mystery; Private investigators; Detec
```
judge: Classic detective mystery similar to Holmes; features famous fictional detective solving cases.

**0** — _The Secret Life of Bees_
```
The Butterfly Garden
Author: Dot Hutchison
Published: Limos Yayinlari, 2016
Subjects: Kidnapping victims; Missing persons; Kidnapping; Investigation; Fiction; Missing persons, fiction; Fiction, suspense; Fiction, thrillers
```
judge: Different book. Similar nature theme, but wrong title and author entirely.

**0** — _that book about a woman and bees in South Carolina_
```
Merry Hall
Author: Nichols, Beverley
Published: Timber Press, 1951
Subjects: Gardening; Biography; Gardeners; Homes and haunts; English Authors
```
judge: English garden memoir, not about bees or South Carolina woman

**0** — _understanding how computers actually work_
```
Superintelligence
Author: Nick Bostrom
Published: TEELL, 2014
Subjects: Moral and ethical aspects; Cognitive science; Artificial intelligence; Social aspects; Philosophy; Computers and civilization; Intelligence artificielle; Philosophie
```
judge: About AI ethics and philosophy, not how computers function mechanically.

**0** — _Betty Crocker_
```
Practical cookery
Author: Victor Ceserani
Published: Hodder, 1962
Subjects: Cookery; Cookery, English; Cookery, International; English Cookery; International Cookery; Quantity cookery; For NVQ / SVQ 2; For SNVQ Intermediate
```
judge: Not Betty Crocker. Different author, different cookbook entirely.

**0** — _cosy mystery, nothing gruesome_
```
Recursion
Author: Blake Crouch
Published: Knopf Doubleday Publishing Group, 2019
Subjects: Fiction, suspense; Fiction, thrillers; Fiction, thrillers, general; New York Times bestseller; New York Times reviewed; Fiction, thrillers, suspense; Epidemics; Fiction
```
judge: Suspense thriller, not a cosy mystery. Subject matter appears dark/intense.

**0** — _cosy mystery, nothing gruesome_
```
The Raven Boys
Author: Maggie Stiefvater
Published: Scholastic Corperation, 2012
Subjects: Children's fiction; Magic, fiction; Extrasensory perception, fiction; Occultism, fiction; Occultism; Magic; Clairvoyance; Paranormal fiction
```
judge: Fantasy with paranormal elements, not a mystery. No detective plot.

**0** — _understanding how computers actually work_
```
Exploring Bioinformatics
Author: Caroline St. Clair
Published: Jones & Bartlett Learning, LLC, 2013
Subjects: Bioinformatics; Computational biology; Genetic Phenomena; Bio-informatique; Popular Science; Popular Science and Mathematics
```
judge: Bioinformatics is specialized application, not how computers fundamentally work.

**0** — _that book about a woman and bees in South Carolina_
```
The Lottery, or The Adventures of James Harris
Author: Shirley Jackson
Published: Brand: Modern Library, 1949
Subjects: Horror stories; Rites and ceremonies; Fiction; Manners and customs; Villages; Stoning; Lotteries; short story
```
judge: Horror stories about lotteries and stoning, not about bees or South Carolina

**0** — _Sue Monk Kidd_
```
The complete book of progressive knitting
Author: Ida Riley Duncan
Published: Liveright publishing corporation, 1940
Subjects: Knitting; knitting design
```
judge: Different author entirely. No connection to Sue Monk Kidd's works.

**0** — _Sue Monk Kidd_
```
Night School
Author: Richard Wiseman
Published: PAN, 2014
Subjects: Sleep; Popular works; Sleep, popular works; Popular science
```
judge: Different author entirely. Reader seeks Sue Monk Kidd's works.

**0** — _cosy mystery, nothing gruesome_
```
Carrion comfort
Author: Dan Simmons
Published: Headline Bk.Pub., 1989
Subjects: American Horror tales; Vampires; Fiction; Psychic ability; Horror tales, American; Fiction, fantasy, general; Vampires, fiction; Fiction, horror
```
judge: Horror novel with vampires; explicitly gruesome subject matter.

**0** — _gripping but not too violent_
```
Killing Floor
Author: Lee Child
Published: Albatros, 1997
Subjects: English language; Fiction; Private investigators; Murder; Drifters; Mystery; Government investigators; Fiction, suspense
```
judge: Killing Floor is violent action thriller, not suitable for reader's needs.

**1** — _something similar to Sherlock Holmes_
```
Destination Unknown
Author: Agatha Christie
Published: Collins, 1954
Subjects: English Detective and mystery stories; Mystery; Detective and mystery stories; Open Library Staff Picks; Fiction; Scientists; Conspiracy; English literature
```
judge: Mystery/detective fiction by classic author, but not Sherlock Holmes

**1** — _books about the night sky for beginners_
```
The planet factory
Author: Elizabeth Tasker
Published: Bloomsbury Publishing Plc, 2017
Subjects: Extrasolar planets; Astronomy; Planets; Popular works; Popular science
```
judge: Astronomy book but focuses on exoplanets, not introductory night sky observation.

**1** — _something short I can finish in one sitting_
```
Desperation
Author: Stephen King
Published: BCA, 1996
Subjects: Fiction; Travelers; Traffic police; American fiction; Horror tales; Horror; Suspense; Psychological fiction
```
judge: Stephen King novel is readable but 600+ pages, not short-sitting material.

**2** — _gripping but not too violent_
```
The Girl in Room 105
Author: Chetan Bhagat
Published: Amazon Publishing, 2018
Subjects: India, fiction; Fiction, suspense; Fiction, thrillers, general; Fiction, thrillers, suspense; Man-woman relationships; Fiction
```
judge: Suspense thriller with relational drama, likely gripping without extreme violence.

**1** — _Joy of Cooking_
```
The Way to Cook
Author: Julia Child
Published: Knopf, 1985
Subjects: International Cookery; Cookery; Cookbooks; Cooking; International cooking
```
judge: Different cookbook by same author; similar cooking focus but not the requested title.

**1** — _starting a business with no money_
```
A Summary Of Rich Dad Poor Dad
Author: Snowball Publishing
Published: WWW.Snowballpublishing.com, 2011
Subjects: Rich people; Personal Finance; Investments
```
judge: Personal finance book, but unclear if addresses zero-capital startup scenarios.

**0** — _second world war in the pacific_
```
Tai-Pan (Asian Saga
Author: James Clavell
Published: Macmillan Pub Co, 1966
Subjects: 19th century; British; Fiction; Historical Fiction; History; Fiction, historical, general; Hong kong (china), fiction; Fiction, general
```
judge: Historical fiction set in 19th century Hong Kong, not WWII Pacific history.

**2** — _something gentle to read before bed_
```
The Napping House (La Casa Adormecida)
Author: Audrey Wood
Published: Perfection Learning Prebound, 1984
Subjects: Fiction; Children's Books/Baby-Preschool; Fleas; Sleep; Fleas in fiction; Sleep in fiction; Spanish language materials; Juvenile fiction
```
judge: Perfect bedtime story: gentle children's book explicitly about sleep and napping.

**0** — _Betty Crocker_
```
The flavor bible
Author: Karen Page
Published: Little, Brown and Company, 2008
Subjects: Cookery; Cooking & Food; Gastronomy; Nonfiction; Cookbooks; Cooking
```
judge: Reader asked for Betty Crocker, got unrelated cookbook author.

**0** — _dog training for a new puppy_
```
Henry and Mudge
Author: Cynthia Rylant
Published: Houghton Mifflin, 1987
Subjects: Dogs in fiction; Juvenile fiction; Spanish language materials; Fiction; Dogs; Children's fiction; Adventure and adventurers, fiction; Mudge (fictitious character), fiction
```
judge: Fiction story about a boy and his dog, not puppy training guidance.

**2** — _poems about loss_
```
Ariel
Author: Sylvia Plath
Published: Suhrkamp Verlag KG, 1965
Subjects: American poetry; Poetry; Textual Criticism; open_syllabus_project; Poetry (poetic works by one author); American literature; Poésie américaine; General
```
judge: Sylvia Plath's Ariel is a major poetry collection addressing grief, mortality, and loss.

**0** — _what to do about money worries_
```
Managerial Accounting
Author: Jerry J. Weygandt
Published: Wiley, 1999
Subjects: Rechnungslegung; Comptabilité de gestion; Managerial accounting; Bilanz; Management accounting; Lotus 1-2-3 (computer program); Microsoft excel (computer program)
```
judge: Technical business textbook, not personal finance guidance for money worries.

**0** — _Sue Monk Kidd_
```
Foundations of Child and Youth Care
Author: Carol Stuart
Published: Kendall Hunt Publishing, 2009
Subjects: Child care; Youth, health and hygiene; Child rearing; Parenting; Family
```
judge: Reader asked for Sue Monk Kidd; this is a parenting textbook by different author.

**2** — _history of my local area_
```
I dug up Houma, Terrebonne
Author: Helen Emmeline Wurzlow
Published: H.E. Wurzlow, 1984
Subjects: History; Local History
```
judge: Local history of Houma, Terrebonne directly matches reader's request for area history.

**1** — _a book that will make me cry_
```
King of Scars
Author: Leigh Bardugo
Published: Marti Yayinlari, 2019
Subjects: Children's fiction; Magic, fiction; Kings, queens, rulers, etc., fiction; Adventure and adventurers, fiction; New York Times bestseller; Kings and rulers; Juvenile fiction; Imaginar
```
judge: Fantasy adventure that might move some readers; not designed to evoke tears.

**0** — _Sue Monk Kidd_
```
Snow Treasure
Author: Marie McSwigan
Published: Perfection Learning Prebound, 1942
Subjects: Juvenile fiction; Fiction; Courage; World War, 1939-1945; History; Children's fiction; World war, 1939-1945, fiction; Norway, fiction
```
judge: Reader wants Sue Monk Kidd; this is different author entirely.

**0** — _learning to grow vegetables in pots_
```
The Tiny Seed
Author: Eric Carle
Published: Knight, 1970
Subjects: Juvenile literature; Seeds; Plants; Dispersal; Plant life cycles; Juvenile fiction; Fiction; Development
```
judge: Children's picture book about seed dispersal, not vegetable gardening guide.

**0** — _a novel where the beekeeper is the detective_
```
Grey Bees
Author: Andreĭ Kurkov
Published: Deep Vellum Publishing, 2020
Subjects: Slavic philology; Middle-aged men; Fiction; Beekeepers; Ukraine Conflict, 2014-
```
judge: Beekeeper protagonist, but not a detective novel. Wrong genre.

**2** — _a funny book to cheer me up_
```
Pollyanna
Author: Eleanor Hodgman Porter
Published: ALMUZARA, 1912
Subjects: Aunts; Cheerfulness; Classic Literature; Conduct of life; Family; Fiction; History; Interpersonal relations
```
judge: Classic feel-good novel centered on cheerfulness and positive outlook.

**0** — _a funny book to cheer me up_
```
I'm Glad My Mom Died
Author: Jennette McCurdy
Published: Jannettte Mcury, 2022
Subjects: New York Times bestseller; Family & Relationships; Biography & Autobiography; Entertainment & Performing Arts; Biography & Autobiography / Personal Memoirs; Dysfunctional 
```
judge: Memoir about abuse and dysfunctional family, not humorous or uplifting.

**0** — _something gentle to read before bed_
```
The House of Sleep
Author: Jonathan Coe
Published: Gallimard, 1997
Subjects: Fiction; Literature; Sleep disorders; New York Times reviewed; Fiction, psychological; England, fiction; Sleep; Fiction, general
```
judge: Psychological fiction about sleep disorders, likely not gentle bedtime reading.

**0** — _an uplifting story after a hard year_
```
Your money or your life
Author: Vicki Robin
Published: New York, 2008
Subjects: Personal Finance; Finance, personal
```
judge: Personal finance guide, not an uplifting story or narrative work.

**0** — _Sue Monk Kidd_
```
Te lo do io il karate!
Author: Elisabetta Dami
Published: Sweet Cherry Publishing, 2005
Subjects: Karate; Detective and mystery stories; Journalists; Juvenile fiction; Mystery and detective stories; Mice; Fiction; Humorous stories
```
judge: Wrong author entirely. Reader asked for Sue Monk Kidd, not Elisabetta Dami.

**2** — _something gentle to read before bed_
```
Peace at Last
Author: Jill Murphy
Published: Macmillan Children's Books, 1998
Subjects: Children's fiction; Picture books; Sleep, fiction
```
judge: Picture book about sleep with calming title, ideal bedtime read.

**2** — _woodworking with hand tools_
```
The woodworker's bible
Author: Percy W. Blandford
Published: Popular Woodworking Books, 1976
Subjects: Crafts; Nonfiction; Woodwork; Carpentry
```
judge: Comprehensive woodworking guide covering hand tools and techniques.

**0** — _that book about a woman and bees in South Carolina_
```
Educated
Author: Tara Westover
Published: Random House US, 2018
Subjects: Subculture; Home schooling; Family; Anecdotes; Victims of family violence; Women college students; Christian biography; Rural conditions
```
judge: Memoir about homeschooling and family, not about bees or South Carolina.

**0** — _what to expect when you are pregnant_
```
Someone Like You
Author: Sarah Dessen
Published: Random House, 1998
Subjects: Pregnancy; Fiction; Unmarried mothers; Friendship; Children's fiction; Friendship, fiction; Unmarried mothers, fiction; Pregnancy, fiction
```
judge: Fiction novel about pregnancy, not informational guide as requested.

**1** — _something similar to Sherlock Holmes_
```
The Inheritance Games
Author: Jennifer Lynn Barnes
Published: Editorial Presença, 2020
Subjects: Children's fiction; New York Times bestseller; Young adult fiction, social themes, class differences; Young adult fiction, romance, contemporary; Young adult ficti
```
judge: Mystery/detective story for young adults, but not Sherlock Holmes-like deductive fiction

**0** — _Betty Crocker_
```
A new system of domestic cookery
Author: Maria Eliza Ketelby Rundell
Published: Robert M'Dermut, 1800
Subjects: American Cookery; Cookery; English Cookery; Cooking; American Cooking; English Cooking
```
judge: Different author and era. Reader wants Betty Crocker specifically.

**0** — _cosy mystery, nothing gruesome_
```
The House of Sleep
Author: Jonathan Coe
Published: Gallimard, 1997
Subjects: Fiction; Literature; Sleep disorders; New York Times reviewed; Fiction, psychological; England, fiction; Sleep; Fiction, general
```
judge: Psychological fiction about sleep disorders, likely dark/complex not cosy.

**0** — _a novel where the beekeeper is the detective_
```
A Second Roald Dahl Selection
Author: Roald Dahl
Published: Longman, 1987
Subjects: short stories; macabre; short story; diners; cannibalism; Bungarus; beekeepers; royal jelly
```
judge: Short story collection, not a detective novel with beekeeper protagonist.

**0** — _an uplifting story after a hard year_
```
Wasted
Author: Marya Hornbacher
Published: Little Brown and Company, 1998
Subjects: Biography; Patients; Health; Bulimia; Anorexia nervosa; Eating disorders; Biography & Autobiography; Nonfiction
```
judge: Memoir about eating disorders is clinical, not uplifting for struggling reader.

**2** — _second world war in the pacific_
```
The Caine mutiny
Author: Herman Wouk
Published: LITTLE BROWN AND COMPANY, 1951
Subjects: World War, 1939-1945; Fiction; American Naval operations; Fiction in English; Naval History; United States in fiction; World War, 1939-1945 in fiction; Trials (Mutiny)
```
judge: Novel set in WWII Pacific naval operations, directly matches reader's interest.

**2** — _books about the night sky for beginners_
```
Astronomy
Author: Fred Hoyle
Published: Doubleday, 1962
Subjects: Textbooks; Science textbooks; Astronomy; Astronomy textbooks; Pictorial works; History
```
judge: Classic astronomy textbook with illustrations, suitable for beginners learning about night sky.

**0** — _the one about a boy wizard at school_
```
Owen
Author: Kevin Henkes
Published: Greenwillow Books, 1993
Subjects: Fiction; Juvenile fiction; Parent and child; Blankets; Mice; Children's fiction; Parent and child, fiction; Cobijas
```
judge: Picture book about a mouse and blanket. No wizard or school elements.

**1** — _how to keep bees in a small garden_
```
The encyclopedia of country living
Author: Carla Emery
Published: Emery, 1975
Subjects: Home economics; homesteading; recipes; Handbooks, manuals; how-to books; Agriculture; Gardening; American Cookery
```
judge: Broad homesteading guide; likely includes beekeeping but not focused specifically on it.

**2** — _poems about loss_
```
Love Is a Dog from Hell
Author: Charles Bukowski
Published: Black Sparrow, 1977
Subjects: Poetry; Fiction; Poetry (poetic works by one author); American poetry; Poetry as Topic
```
judge: Bukowski poetry collection examining loss, heartbreak, and human suffering.
