|     | Knowledge-based |     |     |                     |            |             | Fast                 | Evolutionary |            |                 |     | MCTS |     |     |
| --- | --------------- | --- | --- | ------------------- | ---------- | ----------- | -------------------- | ------------ | ---------- | --------------- | --- | ---- | --- | --- |
|     |                 |     | for | General             |            |             | Video                |              | Game       | Playing         |     |      |     |     |
|     |                 |     |     | Diego               | Perez,     |             | Spyridon Samothrakis |              | and        | Simon Lucas     |     |      |     |     |
|     |                 |     |     |                     | School     | of Computer | Science              | and          | Electronic | Engineering     |     |      |     |     |
|     |                 |     |     |                     | University |             | of Essex,            | Colchester   | CO4        | 3SQ, UK         |     |      |     |     |
|     |                 |     |     | dperez@essex.ac.uk, |            |             | ssamot@essex.ac.uk,  |              |            | sml@essex.ac.uk |     |      |     |     |
Abstract—GeneralVideoGamePlayingisagameAIdomainin Obviously, algorithms that approach GVGP problems may
| which   | the usage | of game-dependent |     |      | domain  | knowledge | is very      |           |        |              |       |          |                |         |
| ------- | --------- | ----------------- | --- | ---- | ------- | --------- | ------------ | --------- | ------ | ------------ | ----- | -------- | -------------- | ------- |
|         |           |                   |     |      |         |           |              | still     | count  | on some kind | of    | domain   | knowledge,     | and the |
| limited | or even   | non existent.     |     | This | imposes | obvious   | difficulties |           |        |              |       |          |                |         |
|         |           |                   |     |      |         |           |              | questions | raised | above        | could | still be | asked. Indeed, | many    |
whenseekingtocreateagentsabletoplaysetsofdifferentgames.
|     |     |     |     |     |     |     |     | different | algorithms | can | be employed |     | for GVGP, | and chances |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ---------- | --- | ----------- | --- | --------- | ----------- |
Takenmorebroadly,thisissuecanbeusedasanintroductionto
the field of General Artificial Intelligence. This paper explores are that heuristics will still make a big difference. However,
theperformanceofavanillaMonteCarloTreeSearchalgorithm, by reducing the game-dependent knowledge, approaches are
| and analyzes | the | main | difficulties |     | encountered | when | tackling |     |     |     |     |     |     |     |
| ------------ | --- | ---- | ------------ | --- | ----------- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- |
forcedtobemoregeneral,andresearchconductedinthisfield
| this kind | of scenarios. |     | Modifications |     | are proposed |     | to overcome |     |     |     |     |     |     |     |
| --------- | ------------- | --- | ------------- | --- | ------------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
isclosertotheopendomainofGeneralArtificialIntelligence.
| these issues, | strengthening |     | the    | algorithm’s |     | ability | to gather and |     |          |               |     |         |         |            |
| ------------- | ------------- | --- | ------ | ----------- | --- | ------- | ------------- | --- | -------- | ------------- | --- | ------- | ------- | ---------- |
|               |               |     |        |             |     |         |               |     | The goal | of this paper | is  | to show | how and | why a well |
| discover      | knowledge,    | and | taking | advantage   |     | of past | experiences.  |     |          |               |     |         |         |            |
Results show that the performance of the algorithm is signif- knownalgorithm,MonteCarloTreeSearch(MCTS),struggles
icantly improved, although there remain unresolved problems in GVGP when there is no game specific information, and to
| that require  | further      | research.   |     | The framework |          | employed   | in this     |         |            |               |        |             |            |              |
| ------------- | ------------ | ----------- | --- | ------------- | -------- | ---------- | ----------- | ------- | ---------- | ------------- | ------ | ----------- | ---------- | ------------ |
|               |              |             |     |               |          |            |             | offer   | some       | initial ideas | on how | to overcome | this       | issue.       |
| research      | is publicly  | available   |     | and will      | be       | used in    | the General |         |            |               |        |             |            |              |
|               |              |             |     |               |          |            |             |         | The paper  | is structured | as     | follows:    | Section II | reviews the  |
| Video         | Game Playing | competition |     | at            | the IEEE | Conference | on          |         |            |               |        |             |            |              |
|               |              |             |     |               |          |            |             | related | literature | in GGP        | and    | GVGP,       | as well as | past uses of |
| Computational | Intelligence |             | and | Games         | in 2014. |            |             |         |            |               |        |             |            |              |
MCTSinthisfield.Then,SectionIIIdescribestheframework
I. INTRODUCTION used as a benchmark for this research. Section IV explains
Gamesaresplendidbenchmarkstotestdifferentalgorithms, the MCTS algorithm, a default controller and its limitations.
understandhownewadd-onsormodificationsaffecttheirper- Section V proposes a solution to the problems found in the
formance, and to compare different approaches. Quite often, previous section. Section VI details the experimental work
the research exercise is focused on one or several algorithms, and,finally,SectionVIIdrawssomeconclusionsandproposes
| measuring | their | performance |     | in one | particular | game. |     | future | work. |     |     |     |     |     |
| --------- | ----- | ----------- | --- | ------ | ---------- | ----- | --- | ------ | ----- | --- | --- | --- | --- | --- |
Inmostgamesusedbyresearchers,thereexiststhepossibil-
II. RELATEDRESEARCH
| ity of | adding a | significant | amount |     | of domain | knowledge | that |     |     |     |     |     |     |     |
| ------ | -------- | ----------- | ------ | --- | --------- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- |
helps the algorithms tested to achieve the game goals. Whilst One of the first attempts to develop and establish a general
not considered ’incorrect’, this addition can raise doubts as to game playing framework was carried out by the Stanford
whether the type of knowledge introduced is better suited to Logic Group of Stanford University, when they organized the
|              |     |                |     |       |             |     |              | first | AAAI | GGP competition | in  | 2005 | [8]. In this | competition, |
| ------------ | --- | -------------- | --- | ----- | ----------- | --- | ------------ | ----- | ---- | --------------- | --- | ---- | ------------ | ------------ |
| only certain | of  | the algorithms |     | under | comparison, |     | or even cast |       |      |                 |     |      |              |              |
doubt on how much of the success of an algorithm can be players(alsoreferredtointhispaperasagents,orcontrollers)
attributedtothealgorithmitselfortotheheuristicsemployed. would receive declarative descriptions of games at runtime
It could even be possible to argue that, in spite of applying (hence,thegameruleswereunknownbeforehand),andwould
the same heuristics in all algorithms, the comparison could use this information to play the game effectively. These com-
still be unbalanced in certain cases, as some heuristics could petitions feature finite and synchronous games, described in
benefit certain algorithms more than others. When comparing a Game Definition Language (GDL), and include games such
algorithms,itmaypossiblybefairertoalloweachalgorithmto as Chess, Tic-tac-toe, Othello, Connect-4 or Breakthrough.
useitsbestpossibleheuristic.Butshouldwenotthencompare Among the winners of the different editions, Upper Confi-
the quality of each heuristic separately as well? dence bounds for Trees (UCT) and Monte Carlo Tree Search
The objective of General Game Planning (GGP) and Gen- (MCTS) approaches deserve a special mention. CADIA-
eralVideoGamePlaying(GVGP)istoby-passtheadditionof Player, developed by Hilmar Finnsson in his Master’s the-
game specific knowledge, especially if the algorithm is tested sis[5],[6],wasthefirstMCTSbasedapproachtobedeclared
in games that have not been played before. This is the aim of winner of the competition, in 2007. The agent used a form of
theGeneralVideoGamePlayingCompetition[15],acomputer historicheuristicandparallelization,capturinggameproperties
game playing contest that will be held for the first time at the during the algorithm roll-outs.
IEEE Conference on Computational Intelligence and Games The winner of two later editions, 2009 and 2010, was
(CIG) 2014. another MCTS based approach, developed by J. Me´hat and

T. Cazenave [12]. This agent, named Ary, studied the con- Similar to the Atari 2600 domain, J. Levine et al. [10]
cept of parallelizing MCTS in more depth, implementing the recently proposed the creation of a benchmark for General
root parallel algorithm. The idea behind this technique is to Video Game playing that complements Atari 2600 in two
perform independent Monte-Carlo tree searches in parallel in ways: the creation of games in a more general framework,
different CPU. When the decision time allowed to make a and the organization of a competition to test the different
move is over, a master component decides which action to approaches to this problem. Additionally, in this framework,
take among the ones suggested by the different trees. the agent does not need to analyze a screen capture, as all
Other interesting MCTS-based players for GGP are Cen- information is accessible via encapsulated objects. It is also
turio, from Mo¨ller et al. [13], that combines parallelized worthwhile highlighting that the game rules are never given
MCTS with Answer Set Programming (ASP), and the agent to the agent, something that is usually done in GGP.
by Sharma et al. [17], that generates domain-independent This new framework is based on the work of Tom
knowledge and uses it to guide the simulations in UCT. Schaul[16],whocreatedaVideoGameDescriptionLanguage
In GGP, the time allowed for each player to pick a move (VGDL) to serve as a benchmark for learning and planning
can vary from game to game. The time allowed is usually problems. The first edition of the General Video Game AI
indicated in seconds. Therefore, any player will be able to (GVGAI)Competition,organizedbytheauthorsofthepresent
spend at least 1 second of decision time in choosing the next paper,andJulianTogeliusandTomSchaul,willbeheldatthe
action to make. Whilst this is an appropriate amount of time IEEE Conference on Computational Intelligence and Games
for the types of games that feature in GGP competitions, a (CIG) in 2014 [15]. This paper uses the framework of this
similar decision time cannot be afforded in video (real-time) competition as a benchmark, including some of the games
games.Heretheagentsperformactionsatamuchhigherrate, that feature in the contest.
making them appear almost continuous to a human player,
in contrast to turn-based games. Additionally, the real-time
III. THEGVGAIFRAMEWORK
component allows for an asynchronous interaction between This section describes the framework and games employed
the player and the other entities: the game progresses even if in this research.
the player does not take any action.
A. Games
During the last few years, important research has been car-
riedoutinthefieldofgeneralvideo-gameplaying,specifically The GVGAI competition presents several games divided
in arcade games where the games are clearly not turn-based into three sets: training, validation, and test. The first set
and where the time allowed for the agents to pick an action (the only one ready at the time this research was conducted)
is measured in milliseconds. Most of this research has been is public to all competitors, in order for them to train and
performedingamesfromtheAtari2600collection.Bellemare preparetheircontrollers,andistheoneusedintheexperiments
etal.[2]introducedtheArcadeLearningEnvironment(ALE), conducted for this paper. The objective of the validation set
aplatformandamethodologytoevaluateAIagentsindomain- is to serve as a hidden set of games where participants can
independent environments, employing 55 games from Atari execute their controllers in the server. Finally, the test set is
2600. In this framework, each observation consists of a single another set of secret games for the final evaluation of the
frame:atwodimensional(160×210)arrayof7-bitpixels.The competition.
agentactsevery5frames(12timespersecond),andtheaction Each set of games is composed of 10 games, and there
spacecontains the 18 discreteactionsallowed bythejoystick. are 5 different levels available for each one of them. Most
Theagent,therefore,mustanalyzethescreentoidentifygame games have a non-deterministic element in the behaviour of
objectsandperformamoveaccordingly,inadecisiontimeof their entities, producing slightly different playouts every time
close to 80ms. the same game and level is played. Also, all games have a
Several approaches have been proposed to deal with this maximumnumberofgamestepstobeplayed(2000),inorder
kind of environment, such as Reinforcement Learning and to avoid degenerate players never finishing the game. If this
MCTS [2]. M. Hausknecht et al. [9] employed evolutionary limit is violated, the result of the game will count as a loss.
neural networks to extract higher-dimensional representation The10gamesfromthetrainingsetaredescribedinTableI.
formsfromtherawgamescreen.YavarNaddaf,inhisMaster’s As can be seen, the games differ significantly in winning
thesis [14], extracted feature vectors from the game screen, conditions, different number of non-player characters (NPCs),
that were used in conjunction with Gradient-descent Sarsa(λ) scoring mechanics and even in the available actions for the
and UCT. Also in this domain, Bellemare et al. [7] explored agent. For instance, some games have a timer that finishes
theconceptofcontingencyawarenessusingAtari2600games. the game with a victory (as in Survive Zombies) or a defeat
Contingencyawarenessis“therecognitionthatafutureobser- (as in Sokoban). In some cases, it is desirable to collide with
vationisunderanagent’scontrolandnotsolelydeterminedby certain moving entities (as in Butterflies, or in Chase) but, in
the environment” [7, p. 2]. In this research, the authors show other games, those events are what actually kill the player (as
that contingency awareness helps the agent to track objects in Portals, or also in Chase). In other games, the agent (or
on the screen and improve on existing methods for feature avatar) is killed if it collides with a given sprite, that may
construction. only be killed if the avatar picks the action USE appropriately

| Game | Description |     |     |     |     | Score |     |     | Actions |
| ---- | ----------- | --- | --- | --- | --- | ----- | --- | --- | ------- |
Similar to traditional Space Invaders, Aliens features the player • 1pointisawardedforeachalienorpro-
LEFT,
(avatar) in the bottom of the screen, shooting upwards at aliens that tectivestructuredestroyedbytheavatar.
Aliens approachEarth,whoalsoshootbackattheavatar.Theplayerlosesif RIGHT,
|     |     |     |     |     |     | • −1pointisgiveniftheplayerishit. |     |     | USE. |
| --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | ---- |
anyalientouchesit,andwinsifallaliensareeliminated.
|     |     |     |     |     |     | • 2 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Theavatarmustdiginacavetofindatleast10diamonds,withthe points are awarded for each diamond LEFT,
|     |     |     |     |     |     | collected, | and 1 point every | time a new |     |
| --- | --- | --- | --- | --- | --- | ---------- | ----------------- | ---------- | --- |
aidofashovel,beforeexitingthroughadoor.Someheavyrocksmay RIGHT,
diamondisspawned.
Boulderdash fall while digging, killing the player if it is hit from above. There UP,
|     |     |     |     |     |     | • −1pointisgiveniftheavatariskilledby |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- |
areenemiesinthecavethatmightkilltheplayer,butiftwodifferent DOWN,
|     | enemiescollide,anewdiamondisspawned. |     |     |     |     | arockoranenemy. |     |     | USE. |
| --- | ------------------------------------ | --- | --- | --- | --- | --------------- | --- | --- | ---- |
The avatar must capture butterflies that move randomly around the LEFT,
|     |     |     |     |     |     | • 2 points | are awarded | for each butterfly |     |
| --- | --- | --- | --- | --- | --- | ---------- | ----------- | ------------------ | --- |
level. If a butterfly touches a cocoon, more butterflies are spawned. RIGHT,
| Butterflies |     |     |     |     |     | captured. |     |     |     |
| ----------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- |
The player wins if it collects all butterflies, but loses if all cocoons UP,
|     | areopened. |     |     |     |     |     |     |     | DOWN. |
| --- | ---------- | --- | --- | --- | --- | --- | --- | --- | ----- |
Theavatarmustchaseandkillscaredgoatsthatfleefromtheplayer. LEFT,
Ifagoatfindsanothergoat’scorpse,itbecomesangryandchasesthe • 1pointforkillingagoat. RIGHT,
| Chase |     |     |     |     |     | • −1pointforbeinghitbyanangrygoat. |     |     |     |
| ----- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- |
player.Theplayerwinsifallscaredgoatsaredead,butitlosesifis UP,
|     | hitbyanangrygoat. |     |     |     |     |     |     |     | DOWN. |
| --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | ----- |
LEFT,
Theavatarisafrogthatmustcrossaroad,fulloftracks,andariver, • 1pointforreachingthegoal.
RIGHT,
Frogs onlytraversablebylogs,toreachagoal.Theplayerwinsifthegoal • −2pointsforbeinghitbyatruck. UP,
isreached,butlosesifitishitbyatruckorfallsintothewater.
DOWN.
LEFT,
Theavatarmustshootatseveralmissilesthatfallfromthesky,before • 2pointsaregivenfordestroyingamissile. RIGHT,
Missile
they reach the cities they are directed towards. The player wins if it • −1pointforeachcityhit. UP,
Command isabletosaveatleastonecity,andlosesifallcitiesarehit. DOWN,
USE.
LEFT,
Theavatarmustfindthegoalwhileavoidinglasersthatkillhim.There
|         |                                                              |     |     |     |     | • 1pointisgivenforreachingthegoal. |     |     | RIGHT, |
| ------- | ------------------------------------------------------------ | --- | --- | --- | --- | ---------------------------------- | --- | --- | ------ |
| Portals | aremanyportalsthatteleporttheplayerfromonelocationtoanother. |     |     |     |     |                                    |     |     |        |
UP,
Theplayerwinsifthegoalisreached,andlosesifkilledbyalaser.
DOWN.
LEFT,
|     |     |     |     |     |     | • 1pointisgivenforeachboxpushedinto |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- |
The avatar must push boxes so they fall into holes. The player wins RIGHT,
| Sokoban |     |     |     |     |     | ahole. |     |     |     |
| ------- | --- | --- | --- | --- | --- | ------ | --- | --- | --- |
ifallboxesaremadetodisappear,andloseswhenthetimerrunsout. UP,
DOWN.
|     |     |     |     |     |     | • 1 point | is given for collecting | one piece |     |
| --- | --- | --- | --- | --- | --- | --------- | ----------------------- | --------- | --- |
Theavatarmuststayalivewhilebeingattackedbyspawnedzombies. LEFT,
Survive Itmaycollecthoney,droppedbybees,inordertoavoidbeingkilled ofhoney,andalsoforkillingazombie. RIGHT,
|     |     |     |     |     |     | • −1pointiftheavatariskilled,oritfalls |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- |
Zombies byzombies.Theplayerwinsifthetimerrunsout,andlosesifhitby UP,
intothezombiespawnpoint.
|       | azombiewhilehavingnohoney(otherwise,thezombiedies). |                 |                   |                  |               |                                        |                |                   | DOWN.  |
| ----- | --------------------------------------------------- | --------------- | ----------------- | ---------------- | ------------- | -------------------------------------- | -------------- | ----------------- | ------ |
|       |                                                     |                 |                   |                  |               | • 2 points                             | for killing an | enemy, 1 for col- | LEFT,  |
|       | The avatar                                          | must find a key | in a maze to open | a door           | and exit. The |                                        |                |                   |        |
|       |                                                     |                 |                   |                  |               | lectingthekey,andanotherpointforreach- |                |                   | RIGHT, |
|       | player is also                                      | equipped with   | a sword to kill   | enemies existing | in            | the                                    |                |                   |        |
| Zelda |                                                     |                 |                   |                  |               | ingthedoorwithit.                      |                |                   | UP,    |
maze.Theplayerwinsifitexitsthemaze,andlosesifitishitbyan DOWN,
|     | enemy. |     |     |     |     | • −1pointiftheavatariskilled. |     |     |     |
| --- | ------ | --- | --- | --- | --- | ----------------------------- | --- | --- | --- |
USE.
TABLE I: Games in the training set of the GVGAI Competition, employed in the experiments of this paper.
(in close proximity, as when using the sword in Zelda, or at a Each controller in this framework must implement two
greater distance, as when shooting in Aliens). Figure 1 shows methods, a constructor for initializing the agent (only called
some games from the training set. once), and an act function to determine the action to take at
These differences in game play make the creation of a every game step. The real-time constraints of the framework
simple game-dependent heuristic a relatively complex task, as determine that the first call must be completed in 1 second,
thedifferentmechanismsmustbehandledonagamepergame whileeveryactcallmustreturnamovetomakewithinatime
basis. Furthermore, a controller created following these ideas, budget of 40 milliseconds (or the agent will be disqualified).
wouldprobablyfailtobehavecorrectlyinotherunseengames Both methods receive a timer, as a reference, to know when
in the competition. the call is due to end, and a StateObservation object
|     |     |     |     |     | representing | the current | state of | the game. |     |
| --- | --- | --- | --- | --- | ------------ | ----------- | -------- | --------- | --- |
B. The framework
|     |     |     |     |     | The | state observation | object | is the window | the agent has |
| --- | --- | --- | --- | --- | --- | ----------------- | ------ | ------------- | ------------- |
All games are written in VGDL, a video-game description to the environment. This state can be advanced with a given
language that is able to fully define a game, typically in less action, allowing simulated moves by the agent. As the games
than 50 lines. For a full description of VGDL, the reader are generally stochastic, it is the responsibility of the agent to
should consult [16]. The framework used in the competition, determine how to trust the results of the simulated moves.
and in this paper, is a Java port of the original Python version The StateObservation object also provides informa-
of VGDL, originally developed by Tom Schaul. tionaboutthestateofthegame,suchasthecurrenttimestep,

Fig. 1: Four of the ten training set games: from top to bottom, left to right, Boulderdash, Survive Zombies, Aliens and Frogs.
score, or whether the player won or lost the game (or is still reward (Q(s,a)) obtained after applying a move a in the
ongoing). Additionally, it provides the list of actions available state s. On each iteration, or play-out, actions are simulated
in the game that is being played. Finally, two more involved from the root until either the end of the game or a maximum
pieces of information are available to the agent: simulation depth is reached. Figure 2 shows the four steps of
|             |                 |           |              |           |         |               |            | each iteration. |     |     |     |
| ----------- | --------------- | --------- | ------------ | --------- | ------- | ------------- | ---------- | --------------- | --- | --- | --- |
| • A history |                 | of avatar | events,      | sorted    | by time | step,         | that have  |                 |     |     |     |
| happened    |                 | in the    | game so      | far. An   | avatar  | event         | is defined |                 |     |     |     |
| as          | a collision     | between   | the          | avatar,   | or any  | sprite        | produced   |                 |     |     |     |
| by          | the avatar      | (such     | as bullets,  | shovel    | or      | sword),       | and any    |                 |     |     |     |
| other       | sprite          | in the    | game.        |           |         |               |            |                 |     |     |     |
| • Lists     | of observations |           | and          | distances | to      | the sprites   | in the     |                 |     |     |     |
| game.       | One             | list is   | given        | for each  | sprite  | type,         | and they   |                 |     |     |     |
| are         | grouped         | by        | the sprite’s | category. |         | This category | is         |                 |     |     |     |
determinedbytheapparentbehaviourofthesprite:static,
| non-static,    |       | NPCs,      | collectables | and        | portals   | (doors). |           |     |     |     |     |
| -------------- | ----- | ---------- | ------------ | ---------- | --------- | -------- | --------- | --- | --- | --- | --- |
| Although       | it    | might      | seem         | that the   | latter    | gives    | too much  |     |     |     |     |
| information    | to    | the agent, | the          | controller | still     | needs    | to figure |     |     |     |     |
| out how        | these | different  | sprites      | affect     | the game. | For      | instance, |     |     |     |     |
| no information |       | is given   | as to        | whether    | the NPCs  | are      | friendly  |     |     |     |     |
ordangerous.Theagentdoesnotknowifreachingtheportals
|     |     |     |     |     |     |     |     | Fig. | 2: MCTS algorithm | steps. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----------------- | ------ | --- |
wouldmakehimwinthegame,wouldkilltheavatar,orsimply
| teleport | it to a | different | location | within | the | same level. |     |     |     |     |     |
| -------- | ------- | --------- | -------- | ------ | --- | ----------- | --- | --- | --- | --- | --- |
IV. MONTECARLOTREESEARCH
|     |     |     |     |     |     |     |     | When the algorithm | starts, the | tree is formed only | by the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | ----------- | ------------------- | ------ |
MonteCarloTreeSearch(MCTS)isatreesearchalgorithm root node, which represents the current state of the game. In
that has had an important impact in Game AI since it was the first stage, Tree selection, the algorithm navigates through
introducedin2006byseveralresearchers.Anextensivesurvey the tree until it reaches a node that is not fully expanded
of MCTS methods is covered by Browne et al. in [4]. (this represents one of the tree’s children that has never been
MCTSestimatestheaveragevalueofrewardsbyiteratively explored). On each one of these selections, MCTS balances
sampling actions in the environment, building an asymmetric between exploration (actions that lead to less explored states)
tree that leans towards the most promising portions of the and exploitation (choosing the action with the best estimated
search space. Each node in the tree holds certain statistics reward). This is known as the MCTS tree policy, and one of
about how often a move is played from that state (N(s,a)), thetypicalwaysthisisperformedinMCTSisbyusingUpper
how many times that node is reached (N(s)) and the average Confidence Bounds (UCB1):

simulations cannot use any game specific information to bias
(cid:40) (cid:115) (cid:41)
lnN(s) the actions taken in this domain.
a∗ =argmax Q(s,a)+C (1)
N(s,a) ThereisanotherimportantproblemthatMCTSfacescontin-
a∈A(s)
uously when playing general video-games: in its vanilla form,
The balance between exploration and exploitation can be
MCTShasnowayofreusingpastinformationfromeventsthat
tempered by modifying C. Higher values of C give added
provided some score gain in the past, even during the same
weight to the second term of the UCB1 Equation 1, giving
game. Imagine that in the game Butterflies (see Table I), the
preference to those actions that have been explored less, at
avatarhassuccessfullycapturedsomebutterfliesinanyareaof
the expense of taking actions with the highest average reward
√ thelevel.Here,MCTSlearnedthatcertainactions(thatcaused
Q(s,a). A commonly used value is 2, as it balances both
the avatar to collide with a butterfly) provided a boost in the
facetsofthesearchwhentherewardsarenormalizedbetween
score, indirectly learning that colliding with butterflies was a
0 and 1.
good thing to do. However, these learnt facts are not used
Inthesecondphase,Expansion,anewnodeisaddedtothe
again to drive the agent to capture other butterflies that are
tree and the third stage, Monte Carlo simulation, is started.
beyond the horizon reachable by MCTS during the roll-outs.
Random actions, either uniformly or biased, are taken up to
Furthermore, there is a second important problem: some
the end of the play-out, where the state is analyzed and given
sprites in the game are never reached by the algorithm’s
a score or reward. This is known as the MCTS default policy
simulations.ImaginethegameofZelda,wheretheavatarmust
(and a roll-out is defined as the sequence of actions taken
collect a key before exiting the level to win the game. If the
in the Monte Carlo simulation step). In the final phase, Back
keyisbeyondthesimulationhorizon(inthiscase,ifitismore
propagation,therewardisbackpropagatedthroughallvisited
than 10 actions away), there is no incentive for the agent to
nodes up to the root, updating the stored statistics N(s,a),
collect it. Note that, if we were programming an agent that
N(s) and Q(s,a).
would only play this game, an obvious heuristic would be
OneofthemainadvantagesofMCTSisthatitisconsidered
to reduce the distance to the key. But in general video-game
tobeananytimealgorithm.Thismeansthatthealgorithmmay
playingthereisnowaytoknow(apriori)whatspritesshould
stop at any number of iterations and provide a reasonable,
be targeted by the avatar in the first place.
valid, next action to take. This makes MCTS a particularly
The next section focuses on a proposed solution to these
good choice for real-time games, where the time budget to
problems, by providing the algorithm with a knowledge base,
decide the next move is severely limited.
and biasing the Monte Carlo simulations to maximize the
Once all iterations have been performed, MCTS returns
knowledge gain obtained during the play-outs.
the next action the agent must take, usually according to the
statistics stored in the root node. Examples of these policies V. KNOWLEDGE-BASEDFASTEVOLUTIONARYMCTS
include taking the action chosen more often (a for the highest
Severalauthors[1],[3]haveevolvedheuristics,inanoffline
N(s,a)), the one that provides a highest average reward
manner, to bias roll-outs in MCTS. More recently, Lucas et
(Q(s,a)), or simply to apply Equation 1 at the root node.
al. [11] proposed an MCTS approach that uses evolution to
A. SampleMCTS controller adapt to the environment and increase performance. In this
approach, a vector of weights w is evolved online to bias the
ThevanillaMCTSalgorithmfeaturesintheGVGAIframe-
MonteCarlosimulations,usingafixedsetoffeaturesextracted
work as a sample controller. For this controller, the maximum
√
for the current game state. This section proposes an extension
depth of the tree is 10 actions, C = 2 and the score of each
of this work, by first employing any number of features, and
stateiscalculatedasafunctionofthegamescore,normalized
then dynamically creating a knowledge base that is used to
between the minimum and maximum scores ever seen during
better calculate the reward of a given state.
the play-outs. In case the game is won orlost in a given state,
therewardisalargepositiveornegativenumber,respectively.
A. Fast Evolutionary MCTS
B. Analysis The idea behind Fast Evolutionary MCTS is to embed the
Although detailed results of the sample controller are re- algorithm roll-outs within evolution. Every roll-out evaluates
ported later in Section VI, it is important to analyze first why a single individual of the evolutionary algorithm, providing
this controller only achieves a 21.6% victory rate. as fitness the reward calculated at the end of the roll-out. Its
Initially, the obvious reason for this low rate of success is pseudocode can be seen in Algorithm 1.
that the algorithm has no game specific information to bias The call in line 4 retrieves the next individual, or weight
the roll-outs, and the rewards depend only on the score and vector w, to evaluate, while its fitness is set in line 9. The
outcomeofthegame.Additionally,duetothereal-timenature vector w is used to bias the roll-out (line 7), following the
of the framework, roll-outs are limited in depth and therefore next process: mapping from state space S to feature space F.
the vast majority of the play-outs do not reach an end game A number of N features are extracted on each state found
state, preventing the algorithm from finding winning states. during the roll-out. Given a set of A available actions, the
Thisis,however,aproblemthatisnotpossibletoavoid:play- relativestrengthofeachaction(a )iscalculatedasaweighted
i
out depth is always going to be limited, and the Monte Carlo sum of feature values, as shown in Equation 2.

| Algorithm | 1 Fast | Evolutionary |     | MCTS Algorithm, |     | from [11], |     |     |     |     |     |     |     |     |     |
| --------- | ------ | ------------ | --- | --------------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
solutionstotheproblemsanalyzedattheendofSectionIV-B.
assuming one roll-out per fitness evaluation. Note that this score function is also the one that defines the
Input: v root state. fitness for the evolved weight vector w, and ultimately will
| 1:  | 0   |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2: Output: weight vector w, action a affect how the roll-outs are to be biased.
3: while within computational budget do In this domain, we define the concept knowledge base as
| w   | = EVO.GETNEXT() |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
4: the combination of two factors: curiosity plus experience.
5: Initialize Statistics Object S The former refers to discovering the effects of colliding with
6: v = TREEPOLICY(v ) other sprites, while the latter allows the agent to reward those
|     | l   |     | 0   |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
7: δ = DEFAULTPOLICY(s(v l ),D(w)) events that provided a score gain. Each piece of knowledge
8: UPDATESTATS(S,δ) corresponds to one event that, as defined in Section IV-B,
EVO.SETFITNESS(w,S)
9: represents the collision of the avatar - or a sprite produced by
return w = EVO.GETBEST() the avatar - with another sprite. Specifically, the knowledge
10:
|            |     |             |     |     |     |     | base | contemplates |          | only      | the types | of         | sprites   | used | to extract |
| ---------- | --- | ----------- | --- | --- | --- | --- | ---- | ------------ | -------- | --------- | --------- | ---------- | --------- | ---- | ---------- |
| 11: return | a=  | RECOMMEND(v |     | 0 ) |     |     |      |              |          |           |           |            |           |      |            |
|            |     |             |     |     |     |     | the  | features     | (NPC,    | resource, |           | non-static | object    | and  | portal).   |
|            |     |             |     |     |     |     | Each | one          | of these | knowledge |           | items      | maintains | the  | following  |
statistics:
|     |     |     | N        |           |     |     |     | Z :     | number  | of occurrences |         | of the | event      | i.  |             |
| --- | --- | --- | -------- | --------- | --- | --- | --- | ------- | ------- | -------------- | ------- | ------ | ---------- | --- | ----------- |
|     |     |     | (cid:88) |           |     |     |     | • i     |         |                |         |        |            |     |             |
|     |     | a i | =        | w ij ×f j |     | (2) |     |         |         |                |         |        |            |     |             |
|     |     |     |          |           |     |     |     | • x i : | average | of the score   | change, |        | calculated | as  | the differ- |
j=1
|     |     |     |     |     |     |     |     | ence | between | the game | score | before | and | after | the event |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ------- | -------- | ----- | ------ | --- | ----- | --------- |
The weights, initialized at every game step, are stored in a took place. It is important to note that an event does
| matrix W, | where | each | entry | w is the weighting |     | of feature |     |     |         |             |       |     |        |       |         |
| --------- | ----- | ---- | ----- | ------------------ | --- | ---------- | --- | --- | ------- | ----------- | ----- | --- | ------ | ----- | ------- |
|           |       |      |       | ij                 |     |            |     | not | contain | information | about | the | proper | score | change, |
j for action i. These relative action strengths are introduced this needs to be inferred by the controller. As multiple
intoasoftmaxfunctioninordertocalculatetheprobabilityof
simultaneouseventscantriggerascorechange,thelarger
selecting each action (see Equation 3). For more details about the value of Z , the more reliable x will be.
|                 |     |        |             |          |     |     |     |     |     | i   |     |     | i   |     |     |
| --------------- | --- | ------ | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| this algorithm, | the | reader | is referred | to [11]. |     |     |     |     |     |     |     |     |     |     |     |
ThesestatisticsareupdatedeverytimeMCTSmakesamove
|     |     |     |     |     |     |     | in  | a roll-out. | When | each | roll-out | finishes, | the | following | three |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ---- | -------- | --------- | --- | --------- | ----- |
e−ai
|     |     | P(a | )=  |     |     | (3) | values | are | calculated | in  | the final | state: |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | ---------- | --- | --------- | ------ | --- | --- | --- |
i (cid:80)A
e−aj
|     |     |     |     | j=1 |     |     |     | Score | change | ∆R: | the | difference | of  | the game | score |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------ | --- | --- | ---------- | --- | -------- | ----- |
•
Inthisresearch,thefeaturesextractedfromeachgamestate
|     |     |     |     |     |     |     |     | between | the | score value | at  | the beginning |     | and | at the end |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | ----------- | --- | ------------- | --- | --- | ---------- |
are the euclidean distances to the closest NPC, resource, non- of the play-out.
static object and portal (as described in Section III-B). Knowledge change ∆Z = (cid:80)N ∆(K ): a measure of
|     |     |     |     |     |     |     |     | •   |     |     |     | i=1 |     | i   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Note that each one of these features may be composed of curiositythatvaluesthechangeofallZ i intheknowledge
more than one distance, as there might be more than one type base, for each knowledge item i. ∆(K ) is calculated as
i
ofNPC,resource,portal,etc.Forinstance,inthegameChase, shown in Equation 4, where Z is the value of Z at the
|     |     |     |     |     |     |     |     |     |     |     |     | i0  |     |     | i   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the first feature would return two distances: to the closest beginning of the play-out and Z is the value of Z at
|     |     |     |     |     |     |     |     |     |     |     |     |     | iF  |     | i   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
scared and to the closest angry goat. The same amount of the end of the roll-out.
features will not be available in every game step of the same (cid:40)
|     |     |     |     |     |     |     |     |     |     |     | Z iF |     | : Z i0 | =0  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | ------ | --- | --- |
game, as there could be sprites that do not exist at some point ∆(K )= (4)
|     |     |     |     |     |     |     |     |     |     | i   | ZiF | −1  | : Otherwise |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- |
ofthegame:suchasenemyNPCsthathavenotbeenspawned
Zi0
yet, or depleted resources. Essentially,∆Z willbehigherwhentheroll-outsproduce
Therefore, the number N of features not only varies from moreevents.Eventsthathavebeenrarelyseenbeforewill
| game to | game, | but also | varies | from game | step to | game step. |     |         |        |         |           |     |           |     |           |
| ------- | ----- | -------- | ------ | --------- | ------- | ---------- | --- | ------- | ------ | ------- | --------- | --- | --------- | --- | --------- |
|         |       |          |        |           |         |            |     | provide | higher | values, | rewarding |     | knowledge |     | gathering |
Fast Evolutionary MCTS must therefore be able to adapt the from events less triggered in the past.
weight vector according to the present number of features at (cid:80)N
|     |     |     |     |     |     |     |     | • Distance | change | ∆D  | =   |     | ∆(D | i ): a measure | of  |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | --- | --- | --- | --- | -------------- | --- |
i=1
eachstep.WhileintheoriginalFastEvolutionaryMCTSalgo- change in distance to each sprite of type i. Equation 5
rithm the number of features was fixed, here the evolutionary defines the value of ∆(D ), where D is the distance
|           |      |              |     |                 |          |        |     |        |         |        |         | i   |        | i0        |        |
| --------- | ---- | ------------ | --- | --------------- | -------- | ------ | --- | ------ | ------- | ------ | ------- | --- | ------ | --------- | ------ |
| algorithm | maps | each feature |     | to a particular | position | in the |     |        |         |        |         |     |        |           |        |
|           |      |              |     |                 |          |        |     | to the | closest | sprite | of type | i   | at the | beginning | of the |
genome, increasing the length of the individual every time a play-out, and D is the same distance at the end of the
iF
| new feature | is discovered. |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
roll-out.
| In this | research, | as in | the original | Fast Evolutionary |     | MCTS |     |     |     |     |     |     |     |     |     |
| ------- | --------- | ----- | ------------ | ----------------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

|     |     |     |     |     |     |     |     |     |     | 1−  | D   | :Z  | =0  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
paper, the evolutionary algorithm used is a (1+1) Evolution  i F i0 OR
D i 0
Strategy(ES).Albeitasimplechoice,itproducesgoodresults. ∆(D )= D >0 and x >0 (5)
|     |     |     |     |     |     |     |     |     | i   |     |     | i0  |     | i   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0
: Otherwise
| B. Knowledge-based |     | Fast | Evolutionary | MCTS |     |     |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | ---- | ------------ | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Now that there is a procedure in place to guide the roll- Here,∆Dwillbehigheriftheavatar,inthecourseofthe
outs, the next step is to define a score function that provides roll-out, has reduced the distance from unknown sprites

(again, measuring curiosity), or from those that provided addingboththeknowledgebaseandevolutiontobiastheroll-
a score boost in the past (experience). outs provides a strong advantage to MCTS, but adding each
Oncethesethreevalueshavebeencalculated,thefinalscore one of these features separately does not impact the vanilla
forthegamestatereachedattheendoftheroll-outisobtained MCTSalgorithmsignificantly.Regardingthescores,KBFast-
as in Equation 6. Essentially, the reward will be the score Evo MCTS also leads on the average points achieved, with
difference ∆R, unless ∆R = 0. If this happens, none of the 13.5±1.2points,againsttheotheralgorithms(withresultsall
actions during the roll-out were able to change the score of rangingbetween 9 and 11 points).Nevertheless, asmentioned
the game, and the reward refers to the other two components. before, this particular result must be treated with care as
The values of α = 0.66 and β = 0.33 have been determined differentgamesvaryintheirscoresystem.Itisthereforemore
empirically for this research. relevant to compare scores on a game by game basis.
Table II shows the average victories and scores obtained in
(cid:40)
∆R : ∆R(cid:54)=0 everygame.Inmostgames,KBFast-EvoMCTSshowsabetter
Reward= (6)
performancethanVanillaMCTSinbothpercentageofvictories
α×∆Z+β×∆D : Otherwise
and scores achieved. In some cases, like in Boulderdash, the
Tosummarize,thenewscorefunctionprioritizestheactions increase in victory percentage is obtained when adding the
that lead to a score gain in the MCTS iterations. However, if knowledge base system, while in others, as in Zelda, it is
thereisnoscoregain,morerewardwillbegiventotheactions the evolution feature that gives this boost. On average in
that provide more information to the knowledge base, or that most cases, and also in some specific games such Missile
willgettheavatarclosertospritesthat,bycollidingwiththem CommmandandChase,itisbothevolutionandtheknowledge
in the past, seemed to produce a positive score change. base that cause the improvement. Special mention must be
made of Aliens, Butterflies and Chase, in which the KB Fast-
VI. EXPERIMENTS
Evo MCTS algorithm achieved a very high rate of victories.
The experimental work of this paper has been performed
Similarly,Aliens,ChaseandMissileCommandshowarelevant
on the 10 games explained in Table I. There are five different
improvement in average score.
levels for each one of the games, with variations on the
KB Fast-Evo MCTS fails, however, to provide good results
location of the sprites and, sometimes, with slightly different
incertaingames,wherelittle(asinSokobanandBoulderdash)
behavioursoftheNPCsprites.Thecompletesetofgamesand
ornoimprovementatall(likeinSurviveZombiesandFrogs)is
levels can be downloaded from the competition website [15].
observedcomparedwithVanillaMCTS.Thereasonsarevaried
Each one of the levels is played 5 times, giving a total of
as to why this algorithm does not achieve the results it did in
250gamesplayedforeachconfigurationtested.Fourdifferent
othergames.Clearly,oneisthefactthatthedistancesbetween
algorithms have been explored in the experiments:
sprites are euclidean, not considering obstacles. Shortest dis-
• Vanilla MCTS: the sample MCTS controller from the tances (i.e. using A*) would positively affect the performance
competition, as explained in Section IV-A. of the algorithm. This is, however, not trivial: path-finding
• Fast-EvoMCTS:FastEvolutionaryMCTS,asperLucas requires the definition of a navigable space. In the GVGAI
et al. [11], using dynamic adaptation of the number of frameworkthiscanbeinferredasemptyspacesinmostgames,
features, as explained in Section V-A. butinothers(particularlyinBoulderdash),theavatarmovesby
• KB MCTS: Knowledge-based (KB) MCTS as explained digging through dirt, creating new paths with each movement
in Section V-B, but employing uniformly random roll- (but dirt itself is not an obstacle).
outs in the Simulation phase of MCTS (i.e., no Fast Nevertheless, not all problems can be attributed to how the
Evolution is used to guide the Monte Carlo simulations). distancesarecalculated.Forinstance,inSokoban,whereboxes
• KB Fast-Evo MCTS: Knowledge-based Fast Evolution- are to be pushed, it is extremely important to consider where
ary MCTS, as explained in Section V-B, using both the box is pushed from. However, the algorithm considers
knowledge base and evolution to bias the roll-outs. collisions as non-directional events. It could be possible to
The experiments can be analyzed by considering two mea- includethisinformation(whichcollisionshappenedfromwhat
sures: the percentage of victories achieved and the score direction) in the model, but this would be an unnecessary
earned. As in the competition, it is considered that the former division for other games, where it is not relevant. Actually,
value takes precedence over the rankings (it is more relevant it would increase the number of features considered, causing
to win the game than to lose it with a higher score). Also, a higher computational cost for their calculation and a larger
comparing the percentage of victories across all games is search space for the evolutionary algorithm.
more representative than comparing scores, as each game has Another interesting case to analyze is Frogs. In this game,
a completely different score system. However, it is interesting the avatar usually struggles with crossing the road. This road
to compare scores on a game by game basis. istypicallycomposedofthreelaneswithmanytrucksthatkill
According to the total average of victories, KB Fast-Evo the agent when contacting with it. Therefore, the road needs
MCTS leads the comparison with 49.2% ± 3.2 of games to be crossed quickly, and most of the roll-outs are unable to
won. The other MCTS versions all obtained similar victory achieve this without colliding with a truck. The consequence
rates in the range 20% to 25%. This difference shows that of this is that most of the feedback retrieved suggests that

PercentageVictories AverageScore
Vanila Fast-Evo KBFast-Evo Vanila Fast-Evo KBFast-Evo
Game KBMCTS KBMCTS
MCTS MCTS MCTS MCTS MCTS MCTS
Aliens 8.0±5.4 4.0±3.9 4.0±3.9 100.0±0.0 36.72±0.9 38.4±0.8 37.56±1.0 54.92±1.6
Boulderdash 0.0±0.0 4.0±3.9 28.0±9.0 16.0±7.3 9.96±1.0 12.16±1.2 17.28±1.7 16.44±1.8
Butterflies 88.0±6.5 96.0±3.9 80.0±8.0 100.0±0.0 27.84±2.8 31.36±3.4 31.04±3.4 28.96±2.8
Chase 12.0±6.5 12.0±6.5 0.0±0.0 92.0±5.4 4.04±0.6 4.8±0.6 3.56±0.7 9.28±0.5
Frogs 24.0±8.5 16.0±7.3 8.0±5.4 20.0±8.0 -0.88±0.3 -1.04±0.2 -1.2±0.2 -0.68±0.2
MissileCommand 20.0±8.0 20.0±8.0 20.0±8.0 56.0±9.9 -1.44±0.3 -1.44±0.3 -1.28±0.3 3.24±1.3
Portals 12.0±6.5 28.0±9.0 16.0±7.3 28.0±9.0 0.12±0.06 0.28±0.09 0.16±0.07 0.28±0.09
Sokoban 0.0±0.0 0.0±0.0 4.0±3.9 8.0±5.4 0.16±0.1 0.32±0.1 0.7±0.2 0.6±0.1
SurviveZombies 44.0±9.9 36.0±9.6 52.0±10.0 44.0±9.9 13.28±2.3 14.32±2.4 18.56±3.1 21.36±3.3
Zelda 8.0±5.4 20.0±8.0 8.0±5.4 28.0±9.0 0.08±0.3 0.6±0.3 0.8±0.3 0.6±0.3
Overall 21.6±2.6 23.6±2.7 22.0±2.6 49.2±3.2 9.0±0.9 10.0±1.0 10.7±1.0 13.5±1.2
TABLE II: Percentage of victories and scores from each game. The results in bold are the best from each game. Each value
corresponds to the average result obtained by playing that particular game 25 times.
the action that moves the avatar into the first lane will most [2] MarcG.Bellemare,YavarNaddaf,JoelVeness,andMichaelBowling.
likely cause the player to lose the game. A typical behaviour TheArcadeLearningEnvironment:AnEvaluationPlatformforGeneral
Agents. JournalofArtificialIntelligenceResearch,47:253–279,2013.
observed in this game is the agent moving parallel to the road
[3] Amit Benbassat and Moshe Sipper. EvoMCTS: Enhancing MCTS-
without ever crossing it, trying to find gaps to cross, but too based Players Through Genetic Programming. In Proceedings of the
“scared” to actually try it. ConferenceonComputationalIntelligenceandGames(CIG),pages57–
64,2013.
VII. CONCLUSIONS [4] C.Browne,E.Powley,D.Whitehouse,S.Lucas,P.Cowling,P.Rohlf-
shagen, S. Tavener, D. Perez, S. Samothrakis, and S. Colton. A
This paper explored the performance and problems of a Survey of Monte Carlo Tree Search Methods. IEEE Transactions on
vanilla Monte Carlo Tree Search (MCTS) algorithm in the ComputationalIntelligenceandAIinGames,4:1:1–43,2012.
[5] HilmarFinnssonandYngviBjo¨rnsson. Simulation-basedApproachto
field of General Video Game Playing (GVGP). Several mod-
GeneralGamePlaying.InProceedingsofthe23rdNationalConference
ifications to the algorithm have been implemented in order onArtificialIntelligence,pages259–264,2008.
to overcome the problems, such as rewarding the discovery [6] HilmarFinnssonandYngviBjo¨rnsson. CADIA-Player:ASimulation-
Based General Game Player. IEEE Transactions on Computational
of new sprites, augmenting the knowledge of other elements
IntelligenceandAIinGames,1:1–12,2009.
in the game, and using past experience to ultimately guide [7] Marc Gendron-Bellemare, Joel Veness, and Michael Bowling. Investi-
the MCTS roll-outs. Results show a significant improvement gatingContingencyAwarenessusingAtari2600Games.InProceedings
oftheTwenty-SixthConferenceonArtificialIntelligence(AAAI),pages
in performance, both in percentage of victories and scores
864–871,2012.
achieved.Theseimprovementsworkbetterinsomegamesthan [8] Michael Genesereth, Nathaniel Love, and Barney Pell. General Game
in others, and reasons for this have also been suggested. Playing:OverviewoftheAAAICompetition. AIMagazine,26:62–72,
2005.
This work presages multiple future extensions, such as
[9] Matthew Hausknecht, Joel Lehman, Risto Miikkulainen, and Peter
the implementation of a general path-finding algorithm for Stone. A Neuroevolution Approach to General Atari Game Playing.
better distance measurements, or experimenting with different IEEE Transactions on Computational Intelligence and AI in Games,
algorithms to bias roll-outs. This study features a (1 + 1) DOI:10.1109/TCIAIG.2013.2294713:1–18,2013.
[10] John Levine, Clare B. Congdon, Michal B´ıda, Marc Ebner, Graham
Evolution Strategy to guide the Monte Carlo simulations, but Kendall, Simon Lucas, Risto Miikkulainen, Tom Schaul, and Tommy
more involved evolutionary techniques will be explored, as Thompson. GeneralVideoGamePlaying. DagstuhlFollow-up,6:1–7,
2013.
well as other approaches like gradient descent methods.
[11] Simon M. Lucas, Spyridon Samothrakis, and Diego Perez. Fast Evo-
GVGP, with the absence of game-dependent heuristics, has
lutionary Adaptation for Monte Carlo Tree Search. In Proceedings of
proven to be a challenging and fascinating problem. It re- EvoGames,pagetoappear,2014.
produces current open challenges in Reinforcement Learning, [12] JeanMe´hatandTristanCazenave. AParallelGeneralGamePlayer. KI
-KnstlicheIntelligenz,25:43–47,2011.
such as the absence of meaningful rewards (as explained for
[13] Maximilian M¨ller, Marius Thomas Schneider, Martin Wegner, and
Frogs, in Section VI). We hope that this research, and also Torsten Schaub. Centurio, a General Game Player: Parallel, Java- and
the new General Video Game Competition [15], helps to shed ASP-based. KI-KnstlicheIntelligenz,25:17–24,2011.
[14] Yavar Naddaf. Game-Independent AI Agents for Playing Atari 2600
some light on a problem that is still unresolved, and also to
ConsoleGames. Master’sthesis,UniversityofAlberta,2010.
bring more researchers to this topic. [15] Diego Perez, Spyridon Samothrakis, Julian Togelius, Tom Schaul, and
Simon Lucas. The General Video Game AI Competition, 2014.
ACKNOWLEDGMENT www.gvgai.net.
[16] Tom Schaul. A Video Game Description Language for Model-based
This work was supported by EPSRC grant EP/H048588/1.
or Interactive Learning. In Proceedings of the IEEE Conference on
ComputationalIntelligenceinGames,pages193–200,2013.
REFERENCES
[17] ShivenSharma,ZiadKobti,andScottGoodwin.KnowledgeGeneration
[1] Atif Alhejali and Simon M. Lucas. Using Genetic Programming to for Improving Simulations in UCT for General Game Playing. In
Evolve Heuristics for a Monte Carlo Tree Search Ms Pac-Man Agent. Proceedings of the 21st Australasian Joint Conference on Artificial
In Proceedings of the Conference on Computational Intelligence and Intelligence:AdvancesinArtificialIntelligence,pages49–55,2008.
Games(CIG),pages65–72,2013.