Document downloaded from:
http://hdl.handle.net/10251/83598
This paper must be cited as:
José Hernández-Orallo (2016). Evaluation in artificial intelligence: From task-oriented to
ability-oriented measurement. Artificial Intelligence Review. 1-51. doi:10.1007/s10462-016-
9505-7.
The final publication is available at
Copyright Springer Verlag (Germany)
Additional Information
The final publication is available at Springer via http://dx.doi.org/ 10.1007/s10462-016-9505-
7.

Artificial Intelligence Review manuscript No.
(will be inserted by the editor)
Evaluation in artificial intelligence
From task-oriented to ability-oriented measurement
Jos´e Hern´andez-Orallo
Received:October9,2014/Accepted:?
Abstract Theevaluationofartificialintelligencesystemsandcomponentsiscrucialfortheprogress
ofthediscipline.InthispaperwedescribeandcriticallyassessthedifferentwaysAIsystemsareeval-
uated,andtheroleofcomponentsandtechniquesinthesesystems.Wefirstfocusonthetraditional
task-oriented evaluation approach. We identify three kinds of evaluation: human discrimination,
problem benchmarks and peer confrontation. We describe some of the limitations of the many eval-
uation schemes and competitions in these three categories, and follow the progression of some of
thesetests.Wethenfocusonalesscustomary(andchallenging)ability-orientedevaluationapproach,
where a system is characterised by its (cognitive) abilities, rather than by the tasks it is designed to
solve.Wediscussseveralpossibilities:theadaptationofcognitivetestsusedforhumansandanimals,
thedevelopmentoftestsderivedfromalgorithmicinformationtheoryormoreintegratedapproaches
undertheperspectiveofuniversalpsychometrics.WeanalysesomeevaluationtestsfromAIthatare
better positioned for an ability-oriented evaluation and discuss how their problems and limitations
canpossiblybeaddressedwithsomeofthetoolsandideasthatappearwithinthepaper.Finally,we
enumerate a series of lessons learnt and generic guidelines to be used when an AI evaluation scheme
is under consideration.
Keywords AI evaluation · AI competitions · machine intelligence · cognitive abilities · universal
psychometrics · Turing Test
1 Introduction
The evaluation of any discipline must necessarily be linked to the purpose of the discipline. What is
the purpose of artificial intelligence (AI)? McCarthy’s pristine definition of AI sets this unambigu-
ously: “[AI is] the science and engineering of making intelligent machines” (McCarthy, 2007). As
a consequence, AI evaluation should focus on evaluating the intelligence of the artefacts it builds.
However,aswewillfurtherdiscussbelow,‘intelligencetests’(ofwhateverkind)arenottheeveryday
Jos´eHern´andez-Orallo
DSIC,UniversitatPolit`ecnicadeVal`encia,Spain
Tel.:+34963877007
E-mail:jorallo@dsic.upv.es

2 Jos´eHern´andez-Orallo
evaluation approach for AI. The explanation for this is that most AI research is better identified
by Minsky’s more pragmatic definition: “[AI is] the science of making machines capable of perform-
ing tasks that would require intelligence if done by [humans]” (Minsky, 1968, p.v). As a result, AI
evaluation focuses on checking whether machines do these tasks well.
This has led to an important anomaly of AI. AI artefacts solve these tasks without featuring
intelligence. Paradoxically, this is one of the reasons of AI success. Systems are designed for a
particular functionality and perform their task more predictably than humans, from driving cars
to supply chain planning. Frequently, some tasks are not considered AI problems any more, once
they are solved without full-fledged intelligence. This phenomenon is known as the “AI effect”
(McCorduck, 2004). It would be unfair, however, to deny that some current AI systems, especially
those that incorporate some learning potential, exhibit some intelligent behaviour.
Anyway, it is not the purpose of this paper to dig further into the time-worn debate between
weak (or soft) AI and strong (or hard) AI, and whether AI should solve problems as humans would
do or whether AI has to achieve what is usually referred to as human-level intelligence. We will
just focus on the distinction between narrow AI vs. general AI, but always recognising that both
approachesarevalidandgenuinepartsofAIresearch.ItisusefultohavespecialisedAIsystemsthat
solve specific tasks, as well as systems that have abilities so that they can solve new problems they
haveneverfacedbefore.Theintentionofstressingthisdualityisthatthisshouldnecessarilypervade
the evaluation procedures in AI. Specialised AI systems should require a task-oriented evaluation,
while general-purpose AI systems (also known as AGI systems, from the term ‘artificial general
intelligence’) should require an ability-oriented evaluation. In practice, however, we see that some
general-purpose AI systems are evaluated with a narrow set of tasks. Also, some general-purpose
components, such as planning or learning techniques, are integrated —or re-programmed— into
systems with specific goals, or become specialised after many training or interaction trials, losing
their plasticity as a result.
This paper focuses on the way evaluation is done in AI. As any science and engineering disci-
pline, measuring is crucial for AI. Disciplines progress when they have objective evaluation tools
to measure the elements and objects of study, assess the prototypes and artefacts that are being
built and examine the discipline as a whole. As we will discuss in subsequent sections, despite the
significant progress in the past couple of decades (with the generalisation of several AI benchmarks
and competitions) there still remains a large margin for improvement in the way AI systems are
evaluated. This is partially because we do not see AI evaluation as a measurement process (Hand,
2004).Also,itisprobablyacrucialmomenttooverhaulthewayAIevaluationisperformed,afterthe
recent progress in areasof AI that are distinct from the narrow AI approach, such as developmental
robotics(Asadaetal,2009),deeplearning(Areletal,2010),inductiveprogramming(Gulwanietal,
2015), artificial general intelligence (Goertzel and Pennachin, 2007), universal artificial intelligence
(Hutter, 2007), etc.
By overhauling AI evaluation, we aim to fill a gap, because, to our knowledge, there is no
overarching analysis about how evaluation is performed in AI and how it can be improved and
adapted to the challenges of the future. Some previous works discussing AI evaluation (Newell and
Simon,1976;Gaschnigetal,1983;Rothenbergetal,1987;GeissmanandSchultz,1988;Deckeretal,
1989; Langley, 1987, 2011; Buchanan, 1988; Simon, 1995; Baldwin and Yadav, 1995; Falkenauer,
1998; Langford, 2005; Legg and Hutter, 2007a; Whiteson et al, 2011; Drummond and Japkowicz,
2010; Anderson et al, 2011; Madhavan et al, 2009; Schlenoff et al, 2011) are relatively old, non-
comprehensive,restrictivetoaspecificareaofAI,limitedtooneparticularapproachand/orfocused
ontheexperimentalmethodologyratherthanwhatisbeingmeasuredandhow.Nonetheless,wewill

Evaluationinartificialintelligence 3
refer to many of these works in this text. In fact, whereas this paper aims to give a broad coverage
of AI evaluation, some of the works above can still be very useful as in-depth analysis for particular
domains or approaches.
Someideasfromtheoldanalysisstillholdtoday.Forinstance,CohenandHowe(1988)introduced
severalcriteriaforevaluatingresearchproblems,methods,implementations,experiments’design,and
evaluationoftheexperiments.Inthecriteriaforexperiments’design,weseeseveralofthetopicswe
will address in the paper: “1. How many examples can be demonstrated?” (are they sufficient and
qualitatively different and illustrative?), “2. Should the program’s performance be compared to a
standard?”, “3. What are the criteria for good performance?”, “4. Does the program purport to be
general(domain-independent)?”(dothedomainsbeingtestedconstitutearepresentativeclass?),and
“5. Is a series of related programs being evaluated?”. Other statements in (Cohen and Howe, 1988)
are not so up-to-date, and show that there has been an improvement in AI evaluation. For instance,
we found the recommendation “that editors, program committees, and reviewers should begin to
insist on evaluation”. Today this recommendation has been generalised. For instance, Conrad and
Zeleznikow (2013) report that more than 60% of the published papers in ICAIL (the International
ConferenceonArtificialIntelligenceandLaw)in1987didnothaveanyevaluationinfrontof20%in
2011.Asimilartrendisseeninjournalpapers,asthenumberofempiricalpaperswithoutevaluation
becomes marginal from 2005 and 2014 (Conrad and Zeleznikow, 2015). The most widespread book
onartificialintelligencealsoconfirmsthat“AIhasadvancedmorerapidlyinthepastdecadebecause
of greater use of the scientific method in experimenting with and comparing approaches” (Russell
and Norvig, 2009, p. 30). Hence, a lack of evaluation may no longer be the problem. However, there
isstillagreatdealofdisaggregation,manyad-hocprocedures,badhabitsandloopholesaboutwhat
is being measured and how it is being measured. This paper will focus on these issues.
Looking at what is being measured, it seems that we have to distinguish between AI systems
and AI components. Systems (such as AI agents, cognitive architectures or robots) can be evaluated
as they are, since they take some sort of problem (by a specification or by rewards) and can be
evaluated in terms of a utility function. Components (such as particular techniques, algorithms,
methods or tools) cannot be evaluated if there is no specification for the component. Hoever, even
in this case, their quality will ultimately be assessed in terms of their functionality as part of one or
moresystems.Forinstance,a(self-driving)carcanbeevaluatedasasystembutitscomponents(e.g.,
an engine or camera), even if they fulfil their own specifications perfectly, are ultimately evaluated
in the way they serve the whole performance of the car. Actually, a Formula One engine would be
inappropriateforafamilycar,andamonoscopiccamerawouldbeinappropriateforsomeself-driving
cars (but not others). Nevertheless, we can still characterise and measure an engine according to
several specifications, such as power, consumption and robustness. Similarly, in AI there are some
techniques that can be evaluated independently using several dimensions (e.g., a SAT solver), but
they really make sense when integrated into a system truly solving one or more problems.
InthispaperwewillmostlyrefertotheevaluationofAIsystems,asweconsiderAIcomponents
canbeevaluatedasisusuallydonein(therestof)computerscience(e.g.,analysingtheircompliance
with the specification and their computational complexity). Having said this, in AI there are cases
thatarearguablyhalf-waybetweenastandalonesystemandacomponent,suchasplanners,machine
learningmethods,naturallanguageprocessingtools,etc.Ourmaincriterionwillbetoconsiderthose
workingartefactsthatcansolveoneormoreproblemswithoutfurtherintegrationorprogramming.If
thereishumaninterventiononcetheevaluationstartsthenwearereallyevaluatingtheprogrammers,
integrators or curators, and not only the AI system (or component). A component that is able to
solve different problems by reprogramming will not be considered general-purpose. For instance, a

4 Jos´eHern´andez-Orallo
machine with a Lisp interpreter where a programmer can code specific applications, such as a chess
player or a theorem prover, is not a general-purpose system, even if the language is. Otherwise,
any Turing-complete programming language or platform would ultimately be considered a general-
purpose AI system, which would be absurd in terms of actual capabilities, autonomously.
Wewillstartwithasurveyoftask-orientedevaluationinAI,byfarmorecommoninAIresearch.
Thenotionofperformanceisrelativelyeasytodetermineasitisdirectlylinkedtothesetorclassof
problemsweareinterestedinfortheevaluation.Nonetheless,wewillidentifyseveralproblems,most
ofthemderivedfromtheconfusionofataskdefinitionwithitsevaluation.Anappropriatesampling
procedurefromtheclassofproblemsdefiningthetaskisnotalwayseasy.Wewillgivesomehintsto
derivebetterevaluationprotocols.Withthisperspectivewewillarguethatwhite-boxevaluation(by
algorithm inspection) is becoming less predominant in AI, and we will devote the rest of the paper
to black-box evaluation (by behaviour). We will distinguish three types of behavioural evaluation:
by human discrimination (performing a comparison against or by humans), problem benchmarks (a
repository or generator of problems) and by peer confrontation (1-vs-1 or multi-agent ‘matches’).
We will survey some of the competitions and repositories in these three categories, looking at the
progression of evaluation tools and the development of performance metrics. We will highlight some
problems in how these evaluation schemes are developed and used.
In a second part of the paper, we will pay attention to the more elusive and challenging problem
of ability-oriented evaluation. The three types of evaluation seen for task-oriented evaluation are
not directly applicable, as we now do not want to evaluate systems for what they do but for what
they are able to (learn to) do. In other words, we are looking for signs or indications that show that
the system has a certain ability. One idea that has been around since the inception of AI is to use
human (or animal) intelligence tests, such as the IQ-tests used in psychometrics. During over a cen-
tury, psychometrics and comparative psychology have developed rigorous experimental techniques
and derived many theories of (the evaluation of) human intelligence, identifying several factors em-
pirically, which can be arranged multidimensionally or hierarchically, and usually linked to abilities,
such as verbal skills, short-term or long-term memory, spatial skills or general intelligence. Each
particular test tries to identify a series of exercises that are representative (necessary and sufficient)
for a given ability. We will briefly discuss their use and possible adaptation for the evaluation of AI
systems. A quite different, and less anthropocentric, approach is based on algorithmic information
theory(AIT),atheorydevelopedinthepastdecadesbridgingcomputationandinformationtheory,
where the length and steps taken by an algorithmic solution to a problem are key to analyse its
complexity and the patterns it contains. For instance, Kolmogorov complexity, the length of the
shortest algorithm describing a sequence, is one of key notions in AIT. Using AIT, problem classes
and their difficulty are derived from computational principles. In this way, we are sure about what
we are actually evaluating. Also, exercise generators can be derived from first principles. A more
unified view integrating different evaluation paradigms and procedures found in many disciplines is
alsodescribed,knownas‘universalpsychometrics’,linkedtothenotionof‘universaltest’,atestthat
canbeappliedtoanykindofsystem.WealsoseesomeexamplesfromareasinAI(transferlearning,
inductive programming, cognitive architectures and developmental robotics) where general-purpose
systems have been evaluated.
Finally, in a third part of the paper, we look at a more gradual view bridging the task-oriented
and ability-oriented types of evaluation in terms of task classes that go from specific to general.
The discussion also includes some guidelines about how competitions and problem generators can
be improved, integrated or overhauled for a more robust and efficient AI system evaluation. This is
followed by the conclusions.

Evaluationinartificialintelligence 5
2 Task-oriented evaluation
AI is a successful discipline. The range of applications has been greatly enlarged over the years.
We have successful applications in computer vision, speech recognition, music analysis, machine
translation,textsummarisation,informationretrieval,roboticnavigationandinteraction,automated
vehicles, game playing, prediction, estimation, planning, automated deduction, expert systems, etc.
(see, e.g., Russell and Norvig, 2009). Most of these application problems are specific. This implies
that the goals are clear and that researchers can focus on the problem. This does not mean that we
are not allowed to use more general principles, components and techniques to solve many of these
problems, but that the task is sufficiently specific so that systems can be specialised for these tasks.
For instance, robotic navigation of a Mars rover can share some of the techniques with a driverless
car on Earth, but the final application is extremely specialised in both cases.
Thisspecialisationleadstoanapplication-specific(task-oriented)evaluation.Infact,goingfrom
an abstract problem to a specific task is encouraged: “refine the topic to a task”, provided it is
“representative” (Cohen and Howe, 1988). Given a precise definition of the task, we only need to
define a notion of performance from it. Clearly, we measure performance, and not intelligence. In
fact, many of the most successful AI systems solve each problem in a way that is different from the
way humans solve the same problem. Also, AI systems usually include a great amount of built-in
programming and knowledge for the task. It is not unfair to say that we evaluate the researchers
thathavedesignedthesystemratherthanthesystemitself.Forinstance,wecansaythatitwasthe
research team after Deep Blue (Campbell et al, 2002) (with the help of a powerful computer) who
actually defeated Kasparov. Things have changed significantly in the way AlphaGo (Silver et al,
2016) defeated Lee Sedol, the world’s top Go player at the time, but the system is still strongly
specialised for the game, although the components used (deep neural networks and reinforcement
learning) are general-purpose.
Disregarding who is praiseworthy for each new successful application, AI systems that address
specialised problems with a clear performance should be easy to evaluate. The reality is not that
straightforward,mostlybecausetherearemanydifferent(andusuallyad-hoc)evaluationapproaches.
Let us examine them.
2.1 Types of performance measurement in AI
An application domain, as described above, can be characterised by a set of problems, tasks or
exercises M. In order to evaluate each exercise µ ∈ M we can get a measurement R(π,µ) of the
performance of system π. Measurements can be imperfect. Also, the system, the problem or the
measurement may be non-deterministic. As a result, it is usual to work with the expected value of
the performance of π as E[R(π,µ)].
The definition of M and R does not specify how we want to aggregate the results when M has
more than one problem. The most common approaches are1
1 Worst-caseperformanceandbest-caseperformancearespecialcasesofarank-basedaggregation(usingthecumu-
lativedistributionofresults),withotherpossibilitiessuchasthemedian,thefirstdecile,etc.Rank-basedaggregation,
especially worst-case performance, is more robust to systems getting good scores on many easy problems but doing
poorlyonthedifficultproblems.

| 6            |               |     |     |     | Jos´eHern´andez-Orallo |     |
| ------------ | ------------- | --- | --- | --- | ---------------------- | --- |
| – Worst-case | performance2: |     |     |     |                        |     |
minE[R(π,µ)]
|     |     |     | Φ min (π,M)= |     |     | (1) |
| --- | --- | --- | ------------ | --- | --- | --- |
µ∈M
| – Best-case | performance: |     |     |     |     |     |
| ----------- | ------------ | --- | --- | --- | --- | --- |
(π,M)=maxE[R(π,µ)]
|     |     |     | Φ max |     |     | (2) |
| --- | --- | --- | ----- | --- | --- | --- |
µ∈M
| – Average-case | performance: |     |     |     |     |     |
| -------------- | ------------ | --- | --- | --- | --- | --- |
X
|     |     | Φ(π,M,p)= |     | p(µ)·E[R(π,µ)] |     | (3) |
| --- | --- | --------- | --- | -------------- | --- | --- |
µ∈M
| where p | is a probability | distribution | on M. |     |     |     |
| ------- | ---------------- | ------------ | ----- | --- | --- | --- |
ItisassumedthatthemagnitudesofRfordifferentπ ∈M arecommensurate.Forinstance,ifRcan
range between 0 and 1 for problem µ but ranges between 0 and 10,000 for problem µ , the latter
|     |     |     | 1   |     | 2   |     |
| --- | --- | --- | --- | --- | --- | --- |
will have a much higher weight and will dominate the aggregation. This is not necessarily wrong,
e.g.,iftheyaremeasuredwiththesameunit(e.g.,euros).Ingeneral,however,R isaconstructthat
needs to be normalised. The choice of a performance metric that is sufficiently normalised such that
the results are commensurate is not always easy, but possible to some extent (see, e.g., Whiteson
et al, 2011).
At this point, it is pertinent to make a comment about the well-known no-free-lunch (NFL)
theorems(WolpertandMacready,1995;Wolpert,1996,2012),asthesetheoremsareusuallymisun-
derstood.Thesetheoremsstatethatgivenallpossibleproblems,undersomeparticulardistributions,
no method can work better than any other on average. The argument to support this interpretation
is that, considering all problems, if method π is better than method π for one problem then π
|     |     |     | A   |     | B   | B   |
| --- | --- | --- | --- | --- | --- | --- |
will be better than π for another problem. Some people have even interpreted that research in
A
AI (including search and optimisation problems in computer science) is futile. However, the NFL
theorems can only be applied when the assumptions hold. The conditions state that M must be
infinite and include all possible problems. Also, the problems can be shuffled without affecting the
probability, which can be expressed as “block uniformity” (Igel and Toussaint, 2005), for which the
uniform distribution would be a special case. Nonetheless, these conditions are not plausible if the
problems are taken from the real world. It is unrealistic to assume that the problems we face are
taken from a series of random bits, or that a problem, and its opposite problem (whatever it is)
are equally probable. Many other distributions are much more plausible. A universal distribution
(Solomonoff, 1964; Li and Vit´anyi, 2008), e.g., which is consistent with the idea that problems are
generated by physical laws, processes, living creatures, etc., states that random (incompressible)
problems are less likely. So, for many distributions p, the conditions of the NFL do not hold and
we find that there can be methods π and π such that: Φ(π ,M,p) > Φ(π ,M,p). In fact, there
|     |     |     | A B | A   | B   |     |
| --- | --- | --- | --- | --- | --- | --- |
can be optimal methods for inductive inference (Lattimore and Hutter, 2013), some free lunches for
co-evolution(WolpertandMacready,2005),andotherareas,althoughitseemsthatforoptimisation
| the free lunches | are very | small (Everitt | et al, 2014). |     |     |     |
| ---------------- | -------- | -------------- | ------------- | --- | --- | --- |
After this clarification, it is relevant to determine how R is going to be obtained. For relatively
simple solutions, we can analyse the code or the algorithm of the system π. If the code can be well
understood then its computational properties and behaviour can be clearly determined. We use the
term ‘white-box’ evaluation when R is inferred through program inspection or algorithm analysis.
2
Note that this formula does not have the size of the instance as a parameter, and hence it is not comparable to
theusualviewofworst-caseanalysisofalgorithms.

Evaluationinartificialintelligence 7
White-box evaluation is powerful because we can obtain R theoretically for a given agent π and a
problem class M (provided both are defined theoretically). One common type of problem that is
evaluatedwithawhite-boxapproachtakesplacewhenthesolutiontotheproblemhastobecorrect
or optimal (i.e., perfect). In this case, the performance metric R is defined in terms of time and/or
space resources. This is the case of classical computational complexity theory. Worst-case analysis
(equation 1) is more common than average-case analysis (equation 3), although the latter has also
become popular recently (Knuth, 1973; Levin, 1986; Goldreich and Vadhan, 2007). Nonetheless,
many AI problems are so challenging nowadays that perfect solutions are no longer considered as
a constraint. Instead, approximate solvers are designed to optimise a performance metric that is
defined in terms of the level of error of the solution and the time and/or space resources. In this
case, the use of an average-case analysis is more common, although worst-case analysis (equation 1)
can also be studied under some paradigms (e.g., Probably Approximately Correct learning, Valiant,
1984). In agent theory, the behaviour of the agent (and its properties) can be analysed under some
paradigms such as Belief-Desire-Intention (BDI) agents (see, e.g., a testability approach, Winikoff
and Cranefield, 2014). The theoretical analysis of ‘white-box’ evaluation has also been applied to
games. For instance, in board games, algorithms can be derived and analysed whether they are
optimal, such as noughts and crosses (tic-tac-toe) and English draughts (checkers), the latter solved
byJonathanSchaeffer(Schaefferetal,2007).Finally,ingametheory,theexpectedpay-offplaysthe
role of R and optimal strategies can be determined for some simple games, as well as equilibria and
other properties. In games, some results can be obtained independently of the opponent, but others
are only true if we also know the algorithm that the other players are using (so it becomes a double
‘white-box’ approach to evaluation).
As AI systems become more sophisticated, white-box assessment becomes more difficult, if not
impossible, because the unpredictability of complex systems. Many AI systems incorporate many
different techniques and have stochastic behaviours. This is also in agreement with a view of AI as
anexperimentalscience(Buchanan,1988;Simon,1995).Asaresult,ablack-boxapproachistaken3.
This means that R is obtained exclusively from the behaviour of the system in an empirical way. In
this case, average-case evaluation is usual4.
There are many kinds of black-box assessment in AI, but we can group them into three main
categories:
– Human discrimination: The assessment is made by and/or against humans through observation,
scrutinyand/orinterview.Althoughitcanbebasedonaquestionnaireoraprocedure,theassess-
ment is usually informal and subjective. In AI, this kind of evaluation is not very usual, except
fortheTuringTestandvariants,aswewilldiscusslateron,despitebeingmorecommoninother
disciplines dealing with behaviour, such as psychology, ethology and comparative psychology.
– Problembenchmarks:Theassessmentisperformedagainstacollectionorrepositoryofproblems
(M).ThisapproachisveryfrequentinAI,wherewehaveproblemlibraries,repositories,corpora,
etc.Unlikeotherareaswherethisapproachisalsocommon(suchaspsychologyandcomparative
psychology), most tests and repositories in AI are public. However, as the public access to the
benchmark before the evaluation can lead to “evaluation overfitting”, there have also been some
3 Thedistinctionbetweenwhiteandblackboxcanbeenrichedtoconsiderthoseproblemswherethesolutionmust
beaccompaniedbyaverification,prooforexplanation(Hern´andez-Orallo,2000b;Alpcanetal,2014).
4 Although it is not uncommon, as we will see, that the set of problems from M are chosen by the research team
thatisevaluatingitsownmethod,sotheprobabilitytochoosefromM canbebiasedinsuchawaythatitisactually
abest-caseevaluation.

8 Jos´eHern´andez-Orallo
occasionalevaluationsinAIfollowinga“secretgeneralizedmethodology”(Whitesonetal,2011).
For instance, M can be generated in real time using a problem generator, which actually defines
M and p.
– Peer confrontation: The assessment is performed through a series of (1-vs-1 or n-vs-n) matches.
The result is relative to the other participants. Given this relative value, in order to allow for a
numericalcomparison,sophisticatedperformancemetricscanbederived(e.g.,theElosystemin
chess, Elo, 1978). Whereas this is the common approach in several domains such as games and
multi-agent systems, other AI domains can use this format, especially if systems are evaluated
according to the best one in terms of resources or accuracy, in a competitive way, or when the
evaluation is set up as a challenge, because it is difficult to give a score but the best of two
systems can still be objectively determined.
Thecombinationofsomeoftheaboveisalsocommonforevaluation.Inaddition,therearedomains
in AI that can be evaluated with the three types above. For instance, common-sense reasoning
can be analysed by human discrimination (interviewing), benchmarks (comprehension tests) and
confrontation (competitions such as Jeopardy!, a TV quiz, Ferrucci et al, 2010). In what follows, we
analyse each of the three categories in more detail.
2.2 Evaluation by human discrimination
In this first category we include the evaluation approaches that are performed by a comparison
with or by humans. The Turing Test (Turing, 1950; Oppy and Dowe, 2011) is a case in which there
is both comparison against humans and evaluation by human judges. While the ‘imitation game’
was introduced by Turing as a philosophical instrument in his response to nine objections against
machine intelligence, the game has been (mis-)understood as an actual test, the Turing test, ever
since, with the standard interpretation of one human, one machine pretending to be a human, and
a human interrogator through a teletype acting as a judge. The latter must tell which one is the
machine and the human.
Not only has the game been taken as an actual test, but it has had several implementations,
such as the Loebner Prize5, held every year since 1991. Despite the criticisms of how this prize is
conductedanditsinterpretationthroughtheyears,therehavebeenmoreimplementations.In2014,
Kevin Warwick organised a similar competition that took place at the Royal Society in London.
Even if the results were not significantly different to previous results of the Loebner Prize (or even
whatWeizenbaum’sELIZAwasabletodofiftyyearsago,Weizenbaum,1966),theover-reactionand
publicity of this outcome were preposterous. The reputation of the implementations of the Turing
Test was (further) stained with statements such as this: “If a computer is mistaken for a human
more than 30% of the time during a series of five minute keyboard conversations it passes the test.
No computer has ever achieved this, until now. Eugene managed to convince 33% of the human
judges (30 judges took part [...]) that it was human.” (Warwick, 8 June 2014). And Warwick went
on: “We are therefore proud to declare that Alan Turing’s Test was passed for the first time. [...]
This milestone will go down in history as one of the most exciting”.
Is the imitation game a valid test? Even assuming that the times and thresholds are stricter
than the previous incarnations, the Turing Test has many problems as an intelligence test. First, it
is a test of humanity, relative to human characteristics (i.e., anthropocentric). It is neither gradual
5 http://www.loebner.net/Prizef/loebner-prize.html.

Evaluationinartificialintelligence 9
nor factorial and needs human intervention (it cannot be automated). If done properly, it may take
too much time. Even so, as we have seen, it can be gamed by non-intelligent chatterbots. As a
result, the Turing Test is neither a sufficient nor a necessary condition for intelligence. Despite the
criticism, the Turing Test still has many advocates (Proudfoot, 2011). It is also an inspiration for
countlessphilosophicaldebatesandhasledtoconnectionswithotherconceptsinAIorcomputation
(Hern´andez-Orallo et al, 2012b).
Inanycase,TuringisnottobeblamedbyafailureoftheTuringTestasausefultesttoevaluate
AI systems. Turing did not conceive the test as a practical test to measure intelligence up to and
beyond human intelligence. He is not to blame for a philosophical construct that has had a great
impact in the philosophy and understanding of machine intelligence, but a negative impact on its
measurement.
Does this mean that we should discard the idea of evaluating AI systems by human judges or by
comparing with humans? Not at all. Recently, there have been many variants of the Turing Test:
Total Turing Tests (Schweizer, 1998), Visual Turing Tests including sensory information, Toddler
Turing Tests (Alvarado et al, 2002), robotic interfaces, virtual worlds, etc. (Mueller and Minnery,
2008; Hingston, 2010). These may be useful for chatterbot evaluation, personal assistants, robots
and videogames. For instance, it is within the area of videogames where the notion of ‘believability’
has appeared, which is understood as the property of a bot of looking ‘believable’ as a human
(Livingstone, 2006; Hingston, 2012). This term is interesting, as it clearly detaches these tests from
theevaluationofintelligence.Invideogames,thereareapplicationswherewewantbotsthatcanfool
opponentsintothinkingthattheyarejustanotherhumanplayer.Otherhighlysubjectiveproperties
may also be of interest: enjoyability, resilience, aggressiveness, fun, etc.
Finally, there is a kind of test that is related to the Turing Test, the so-called CAPTCHA
(Completely Automated Public Turing test to tell Computers and Humans Apart) (von Ahn et al,
2004;vonAhnetal,2008).Itissaidtobea‘reverseTuringTest’becausethegoalistotellcomputers
and humans apart in order to ensure that an action or access is only performed by a human (e.g.,
making a post, registering in a service, etc.). CAPTCHAs are quick and practical, omnipresent
nowadays. However, they are designed according to the tasks that are solved by the current state of
AI technology. At present, for instance, a common CAPTCHA is a series of distorted letters, which
are usually easy to recognise by humans but not by machines (e.g., current OCR systems struggle).
Logically, when character recognition systems and other techniques improve, current CAPTCHAs
are broken (see, e.g., Bursztein et al, 2014), and CAPTCHAs need to be updated to more distorted
words or to other tasks that are beyond AI technology. Similarly, the detection of bots in social
networks (sybils) and crowdsourcing platforms rely on tests that are variants of CAPTCHAs, the
Turing Test, or the observation and analysis of user profiles and behaviour (Chu et al, 2010; Wang
et al, 2012).
Table 1 includes a selection of evaluation schemes under the human-discrimination category. As
it is not possible to go into the details of all of them because of brevity, let us choose some that
are most representative. Of particular interest is the BotPrize competition, which has been held
since 2008. This contest awards the bot that is deemed more believable (playing like a human) by
6 http://www.loebner.net/Prizef/loebner-prize.html
7 http://www.reading.ac.uk/news-and-events/releases/PR583836.aspx
8 http://botprize.org/
9 http://www.robochatchallenge.com/
10 http://www.captcha.net/
11 http://www.human-competitive.org

10 Jos´eHern´andez-Orallo
EvaluationScheme Description
LoebnerPrize6 GeneralTuringTestimplementation.
U.ofReadingTT20147 OccasionalTuringTestimplementation(KevinWarwick).
BotPrize8 Contestaboutbotbelievabilityinvideogames.(Livingstone,2006;Hingston,2012)
RoboChatChallenge9 Chatteringbotscompetition.
CAPTCHAs10 Spottingbotsinapplicationsrequiringhumans.(vonAhnetal,2004;vonAhnetal,2008)
Humiesawards11 Human-competitiveresultsusinggeneticandevolutionarycomputation.(Koza,2010)
GraphicsTuringTest Computer-generatedvirtualworldvs.arealcamera.(McGuigan,2006;Borgetal,2012)
Table 1 Listofsomeevaluationschemesinthehuman-discriminationcategory.
theother(human)players.Thecompetitionusesafirst-personshootervideogame,theDeathMatch
game type, as used in Unreal Tournament 2004. It is important to clarify that the bots do not
process the image but receive a description of it through textual messages in a specific language
through the GameBots2004 interface (Pogamut). For the competition, chatting is disabled (as it is
notachatbotcompetition).Thereisa“judginggun”andthehumanjudgesalsoplay,tryingtoplay
normally(aprizeforthejudgesexistsforthosethatareconsideredmore“human”byotherjudges).
Some questions have been raised about how well the competition evaluates the believability of
theparticipants.Forinstance,believabilityissaidtobebetterassessedfromathird-personperspec-
tive (judging recorded video of other players without playing) than with a first-person perspective
(Togelius et al, 2012). The reason is that third-person human judges can concentrate on judging
instead on not being killed or aiming at high scores. Actually, this third-person perspective was
included in the 2014 competition using a crowdsourcing platform (Llargues-Asensio et al, 2014)
so that the two judging systems were incorporated: the First-Person Assessment (FPA), using the
BotPrize in-game judging system, and the Third-Person Assessment (TPA), using a crowdsourcing
platform. Another issue that could be considered in the future is a richer (and more challenging)
representation of the environment, closer to the way humans perceive the images of the game (such
as the graphical processing required for the Arcade Learning Environment Bellemare et al, 2013) or
theGeneralVideoGameCompetition(Schaul,2014)wewillmentionlateron.LiketheTuringtest,
the more time the system is evaluated the more accurate the evaluation can be. Nevertheless, in
BotPrize, repetitions are used, because each game can lead to different situations according to some
randomcomponentsofthegamethatarenotcontrolledbytheplayerorthejudges.Becauseofthis,
in the 2014 competition, each player was judged around 25 times. Finally, about the results, we see
how brittle the notion of humannesss or believability can be. The most human bot got a humanness
score of 52.2% while the most human human just got slightly better with a score of 53.3%.
There are other competitions and awards to evaluate the progress of a domain or area in AI
that are evaluated by a human committee, with a loose set of criteria about how a system is to be
evaluated. For instance, the Humies awards (Koza, 2010), also shown in Table 1, grant a prize to
thosemethodsshowingreplicableresultsthatareabletosolveaproblemina“human-competitive”
way, “a long-standing problem for which there has been a succession of increasingly better human-
createdsolutions”.Theevaluationisperformedbyajudgingcommittee,whoshortlistsafewfinalists
that have to present their results. Accordingly, it is not exactly AI systems what are evaluated but
AI researchers. Nonetheless, this allows for the evaluation of components and new tools if these are
shown to solve several “human-competitive” problems with them.
Finally,asasummaryofthelimitationsandpotentialsofthehuman-discriminationcategory,we
first acknowledge that some variants are being useful, based on the advantage that the intelligence

Evaluationinartificialintelligence 11
and expertise of the evaluator can be used in a less strict way than other kinds of evaluation.
However, the format differs significantly from a standard Turing Test. For instance, the human-
discrimination approach to evaluation can be just solved by a more traditional interview format
withaprocedureorstoryline(asinpsychologyinterviews),orbyanevaluationthroughobservation
(using a committee of dedicated judges). This casts doubts about whether evaluation by imitation
using the standard interpretation of the Turing Test is practical for task-oriented evaluation in AI.
Itistheconceptthatisuseful,anddeservesbeingadaptedtoseveralapplications,wheretheproper
settingforobservation,interactionandinterrogationhastobeanalysedinordertohaveanaccurate
and practical assessment.
2.3 Evaluation through problem benchmarks
In this very common approach to evaluation, M is defined as a set of problems. This fits equation 3
perfectly. Necessarily, the quality of the evaluations depends on M and how exhaustively this set is
explored.Thereareotherissuesthatcouldcompromisethequalityofthemeasurement.Forinstance,
whenM isapublicproblemrepositoryandisnotverylarge,wefindthatthesystemscanspecialise
for M. Also, the solutions may also be available beforehand, or can be inferred by humans, so the
systems can embed part of the solutions. In fact, a system can succeed in a benchmark with a small
sizeof M byusing atechniqueknown asthe“bigswitch”, i.e.,thesystemrecogniseswhich problem
is facing and uses the hardwired solution for that specific exercise. Things can become worse if the
selection of examples from M is made by the researchers themselves (e.g., the usual procedure in
machinelearningofselecting10or20datasetsfromtheUCIrepository,theUniversityofCalifornia
IrvineMachineLearningRepository,BacheandLichman,2013,aswewilldiscussbelow).Ingeneral,
the size of M and a bona fide attitude to research somewhat limit these concerns. Nonetheless, it
is generally acknowledged that most systems actually embed what the researchers have learnt from
M. In a way, again, these benchmarks actually evaluate the researchers, not their systems.
Theabove-mentionedproblemisknownas‘evaluationoverfitting’(Whitesonetal,2011),‘method
overfitting’ (Falkenauer, 1998) or “clever methods of overfitting” (Langford, 2005). For instance, we
can evaluate a self-driving cars in a small parking lot or a restricted part of a city, and we may get
one car with excellent performance in this area in particular. To avoid or reduce this problem, it
is much better if M is very large or infinite, or at least the problems are not disclosed until eval-
uation time (the part of the city, or even the city, is not known in advance). Problem generators
are an alternative. However, it is not always easy to generate a large M of realistic problems (e.g.,
in a car driving domain). Generators can be based on the use of some prototypes with parameter
variations or distortions. These prototypes can be “based on reality”, so that the generator “takes
asinputarealdomain,analysesitautomaticallyandgeneratesdeformations[...]thatfollowcertain
high-level characteristics” (Drummond and Japkowicz, 2010). More powerful and diverse generators
can be defined by the use of problem representation languages. A general and elegant approach is
to determine a probabilistic or stochastic generator (e.g., a grammar) of problems, which directly
defines the probability p for the average-case performance equation 3. Nonetheless, it is not easy to
make a generator that can rule out unusable or Frankenstein-like problems. As an alternative, when
a generative model is inappropriate, virtual simulators inspired in real life can be used (V´azquez
et al, 2014).
When the set of problems is large or generated, we clearly cannot evaluate AI systems efficiently
with the whole set M. So we need to do some sampling of M. It is at this point when we need

12 Jos´eHern´andez-Orallo
to distinguish the benchmark or problem definition from an effective evaluation. Assume we have
a limited number of exercises n that we can administer. The goal will be to reduce the variance
of the measurement given n. One naive approach is to sort M by decreasing p and evaluate the
system with the first n exercises. This maximises the accumulated mass for p for a given n. One
problem about this procedure is that it is highly predictable. Systems will surely specialise on the
first n exercises. For instance, in the self-driving car domain, all systems would specialise for the
mostimportantroutes,whichareprobablyafewmotorwaysandcityavenues.Also,thisapproachis
not very meaningful when R is non-deterministic and/or not completely reliable. Repeated testing
may be necessary, which raises the question of whether to explore a higher n or to perform more
repetitions.
Random sampling using p seems to be a more reasonable alternative. As said above, if R is non-
deterministicand/orsubjecttomeasurementerror,thenrandomsamplingcanbewithreplacement.
IfM andpdefinethebenchmark,isprobability-proportionalsamplingonpthebestwaytoevaluate
systems?Theanswerisno,ingeneral.Again,intheself-drivingcardomain,wewouldprobablyhave
the cars evaluated on important routes only. There are better ways of approximating equation 3.
The idea is to sample in such a way that the diversity of the selection is increased. For instance, for
cars, all kinds of roads and streets should be considered. This ‘diversity-driven sampling” is related
to several kinds of sampling, such as importance sampling (Srinivasan, 2002), stratified sampling
(Cochran, 2007) and other forced Monte Carlo procedures. The key issue is that we use a different
probabilitydistributionforsampling.Althoughtherearemanywaysofobtaininga‘diverse’sample,
we just highlight two main approaches that can be useful for AI evaluation:
– Information-driven sampling: Assume that we have a similarity function sim(µ ,µ ), which in-
1 2
dicates how similar (or correlated) exercises µ and µ in M are. In this case, we need to sample
1 2
on M such that the accumulated mass on p is high and that diversity is also high. The rationale
is that if µ and µ are very similar, using one of them can ‘fill the gap’ of the other, and we
1 2
can assume as if both µ and µ had been explored, actually accumulating p(µ )+p(µ ). One
1 2 1 2
possible way of doing this is by stratified sampling using clustering (not to be confused with
cluster sampling). Information-driven sampling suffers from the need of defining the similarity
function sim. An alternative is to derive m features that describe the exercises, so creating an
m-dimensional space where distances and other topological information can be used to support
thenotionofdiversity(andperformingclustering).Forinstance,ifwewanttoevaluateroutesfor
self-drivingcars,wemightclusteradatabaseofroutesbytheirdistanceandtheirtrafficdensity.
If five clusters of a minimum cardinality are found, we can just sample a few routes from each
cluster. An example of this procedure is shown in Figure 1 (left).
– Difficulty-driven sampling. A set M can contain very easy and very challenging problems. Using
easyproblemsforgoodsystemsordifficultproblemsforbadsystemsisnotveryoptimal.Theidea
tooptimisetheevaluationistochoosearangeofdifficultiesforwhichtheevaluationresultsmay
beinformative(ortogivehigherprobabilitytoexercisesinsidethisrange),asinFigure1(right).
ThisprocedureisdonetoagreaterorlesserdegreeinmanyevaluationsandbenchmarksinAI.In
fact,morechallengingproblemsareusuallyaddedovertheyears,asthesystemsareabletosolve
the easy problems (which soon become ‘toy problems’). One of the crucial points of difficulty-
driven sampling is the definition of a difficulty function d:M →R+. Ideally, we would like that
for every π,Φ(π,µ ,p) > Φ(π,µ ,p) iff d(µ ) < d(µ ). In practice, this condition is too strong,
1 2 1 2
and more flexible characterisations are expected, such as that for every π, and two difficulties a
andbsuchthata≤bwehavethatΦ(π,M ,p)≥Φ(π,M ,p)(whereM denotesalltheexercises
a b a

Evaluationinartificialintelligence 13
| 0.1 |     |     | 0.1                                       |     |     |
| --- | --- | --- | ----------------------------------------- | --- | --- |
|     |     |     | lllllllllllllllllllllllllllllll l lllllll |     |     |
llllllllll l
ll
| 8.0 lllllllllllllllllllllllllllll l lllllll | ll  |                                         | 8.0  |          |     |
| ------------------------------------------- | --- | --------------------------------------- | ---- | -------- | --- |
| l                                           |     | l                                       |      | ll lll   |     |
|                                             |     | l ll l llll llllll l ll l               |      |          |     |
|                                             | ll  | ll lll lll lll lllllll ll llll lll ll l |      | ll ll    |     |
|                                             |     | l lll l l l l l                         |      |          |     |
| 6.0                                         |     |                                         | 6.0  | lllll ll |     |
|                                             |     |                                         | ]R[E | ll ll    |     |
x 2
| 4.0 |             | l                                                                                     | 4.0 | l l |     |
| --- | ----------- | ------------------------------------------------------------------------------------- | --- | --- | --- |
|     |             | l l l lll lll l l l llllll l l l l ll lll l l l lll l l ll l lllll lllllllll l l l ll |     |     |     |
|     |             | ll l l llllll l llllllll llll l l l ll l l l l l l ll llll ll ll l ll l l ll lll      |     | l l |     |
|     | l           | lllll l l l lll l l l ll l l lll l l l ll ll l ll ll lll l l l l l l ll l l           |     |     |     |
|     | l l         | l l l lll l ll l l                                                                    |     | l l |     |
|     | ll ll       | l l l l ll l l l l                                                                    |     |     |     |
| 2.0 | llll        |                                                                                       | 2.0 | l   |     |
|     | lll l l lll | l                                                                                     |     |     |     |
|     | l           |                                                                                       |     |     | l   |
lllllllllllllllllllllllllllllllllllllllllll lll
| 0.0 |         |             | 0.0 |     | llll  |
| --- | ------- | ----------- | --- | --- | ----- |
| 0.0 | 0.2 0.4 | 0.6 0.8 1.0 | 0 5 | 10  | 15 20 |
x
|     |     | 1   |     | d   |     |
| --- | --- | --- | --- | --- | --- |
Fig. 1 Left: a figurative repository M with |M| = 300 exercises shown with empty black circles. Two features x1
andx2 areusedtodescribethemostrelevantcharacteristicsoftheexercises(accordingtodiversity).Thesefeatures
are used to cluster them into five groups. Next, stratified sampling using these clusters is performed with a sample
sizeofn=50.Clustersareofdifferentsize(60,20,70,110,30)but10samples(showninsolidredcircles)aretaken
from each cluster. Because of the constant number of examples per cluster, in order to estimate Φ, measurements
for under-represented clusters are multiplied by their size. Right: a repository of |M|=100 exercises. A measure of
difficulty d has been derived that is monotonically decreasing with (estimated) expected performance (for a group
of agents or for the problem overall). Only n=30 exercises are sampled in the area where the results may be most
informative.
inM ofdifficultya).Thiscouldstillbetoostrongandwemayusearelaxedversionsuchthatfor
every π, there is a t such that for all a and b≥a+t: Φ(π,M ,p)≥Φ(π,M ,p). In experimental
|     |     |     | a   | b   |     |
| --- | --- | --- | --- | --- | --- |
sciences,wehaveapopulation-basedviewofdifficultysuchthatd(µ)ismonotonicallydecreasing
on E [Φ(π,µ,p)], where Ω is a population of subjects, agents or systems that are evaluated
π∈Ω
for the same problem class. In fact, Item Response Theory (Embretson and Reise, 2000) in
psychometrics follows this approach. Finally, we can derive the difficulty of a problem as a
function of the complexity of the problem itself. The complexity metric can be specific to the
application, such as the complexity for mazes in (Bagnall and Zatuchna, 2005; Zatuchna and
Bagnall,2009)orgrid-worlddomainsin(Sturtevant,2012),oritcanbeamoregeneralapproach
(e.g., Kolmogorov complexity). Note that some of the definitions of difficulty above would not
be possible for a set M and distribution p if the conditions of the NFL theorem held.
Both the information-driven sampling and the difficulty-driven sampling can be made adaptive
(Seber and Salehi, 2013), common in population surveys and many experimental sciences. However,
whenevaluatingperformance,itisdifficulty-drivensamplingthathasbeenusedmoresystematically
in the past, especially in psychometrics. In psychometrics, difficulty is inferred from a population of
subjects (in the case of AI, this could be a set of solvers or algorithms). Instead of difficulty, items
are analysed by proficiency, represented by θ, a corresponding concept to difficulty from the point
of view of the solver (higher problem difficulty requires higher agent proficiency).
Item response theory (IRT) (Embretson and Reise, 2000) estimates mathematical models to
infer the associated probability and informativeness estimations for each item. When R is discrete
orbounded,oneverycommonmodelisthethree-parameterlogisticmodel,wheretheitemresponse

| 14  |     |     |     |     |     | Jos´eHern´andez-Orallo |     |
| --- | --- | --- | --- | --- | --- | ---------------------- | --- |
|     | 0.1 |     |     |     | 01  |                        |     |
8.0
5
6.0
| p   |     |     |     | X   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
4.0
0
2.0
|     | 0.0 |     |     |     | 5−   |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- |
|     | −2  | 0 2 | 4   | 6   | −2 0 | 2 4 | 6   |
|     |     |     | q   |     |      | q   |     |
Fig. 2 Left:itemresponsefunction(orcurve)forabinaryscoreitemwiththefollowingparametersforthelogistic
model:discriminationa=1.5,itemlocationb=3,andchancec=0.1.Thediscriminationisshownby theslopeof
thecurveatthemidpoint:a(1−c)/4(indottedred),thelocationisgivenbyb(indashedgreen)andthechanceis
givenbythe horizontallineatc(indashed-dottedgrey,at0.1),whichisveryclosetothezero-proficiencyexpected
result p(θ)=z (here shown in dashed-dotted blue, at 0.11). Right: A linear model for a continuous score item with
parameterz=−1andλ=1.2.Thedashed-dottedlineshowsthezero-abilityexpectedresult.
function (or curve) corresponds to the probability that an agent with proficiency θ gives a correct
| response | to an | item. This model | is characterised | as follows: |     |     |     |
| -------- | ----- | ---------------- | ---------------- | ----------- | --- | --- | --- |
1−c
p(θ)(cid:44)c+
1+e−a(θ−b)
where a is the discrimination (the maximum slope of the curve), b is the difficulty or item location
(thevalueofθ leadingtoaprobabilityhalf-waybetweencand1,i.e.,(1+c)/2),andcisthechance
or asymptotic minimum (the value that is obtained by random guess, as in multiple choice items).
The zero-ability expected result is given when θ =0, which is exactly z =c+ 1−c . Figure 2 (left)
1+eab
| shows | an example | of a logistic | item | response curve. |     |     |     |
| ----- | ---------- | ------------- | ---- | --------------- | --- | --- | --- |
ForcontinuousR,iftheyarebounded,thelogisticmodelabovemaybeappropriate.Onotheroc-
casions,especiallyifRisunbounded,alinearmodelmaybepreferred(Mellenbergh,1994;Ferrando,
2009):
X(θ)(cid:44)z+λθ+(cid:15)
where z is the intercept (zero-ability expected result), λ is the loading or slope, and (cid:15) is the mea-
surement error. Again, the slope λ is positively related to most measures of discriminating power
(Ferrando, 2012). Figure 2 (right) shows an example of a linear item response curve.
Working with item response models is very useful for the design of tests, because if we have a
collection of items, we can choose the most suited one for the subject (or population) we want to
evaluate. According to the results that the subject has obtained on previous items, we may choose
more difficult items if the subject has succeeded on the easy ones, we may look for those items

Evaluationinartificialintelligence 15
Item Theta SE -3.......-2........-1.........0........+1........+2........+3
0 0.00* 1.00* --------------------X--------------------
1 4.00* 1.00* . -------------------->
2 0.11 0.52 -----------I----------
3 0.20 0.45 ----------C--------- 4 -0.04 0.35 -------I------- l
5 0.05 0.32 ------C------
6 -0.13 0.29 ------I------
7 -0.07 0.27 ------C----- 8 -0.18 0.25 -----I-----
9 -0.25 0.25 -----I-----
10 -0.18 0.23 -----C----
11 -0.27 0.23 -----I---- 12 -0.21 0.22 ----C-----
13 -0.26 0.22 ----I----
14 -0.34 0.22 ----I-----
15 -0.37 0.22 -----I---- 16 -0.33 0.20 ----C----
17 -0.29 0.19 ----C---
18 -0.33 0.19 ----I----
19 -0.38 0.19 ----I---- 20 -0.34 0.18 ----C----
21 -0.30 0.18 ----C---
22 -0.27 0.17 ----C---
23 -0.29 0.17 ----I---
24 -0.26 0.17 ---C--- l
25 -0.28 0.16 ----I---
26 -0.30 0.16 ----I---
27 -0.27 0.16 ---C--- −0.4 −0.2 0.0 0.2
28 -0.25 0.15 ---C---
29 -0.23 0.15 ---C--- 30 -0.21 0.15 ---C---
0.1
8.0
6.0
4.0
2.0
0.0
Proficiency
erocS
l
Fig. 3 AnexampleofanIRT-basedadaptivetest(freelyadaptedfrom(Weiss,2011,Fig.8)).Left:theprocessand
proficiencies (thetas) used until convergence. The final proficiency calculated by the test was −0.21 with a standard
errorof0.15.Right:Theresultsshownonaplot.TheblackcurveshowsaEuclideankernelsmoothingwithaconstant
of0.1.
that are most discriminating (i.e., most informative) in the area we have doubts, etc. Note that
discrimination is not a global issue: a curve may have a very high slope at a given point, so it is
highly discriminating in this area, but the curve will almost be flat when we are far from this point.
Conversely, if we have a low slope, then the item covers a wide range of difficulties but the result of
the item will not be so informative as for a higher slope.
Figure 3 shows an example of an adaptive test using IRT. The sequence of exercise difficulties
is shown on the left. The plot on the right shows that averaging the results (especially here, as the
outcomeofR isdiscrete,either0or1)makestheestimationofΦmoredifficultwithanon-adaptive
test. These tools, although generally unknown in AI research, could be very useful for its evaluation
in the future.
Table 2 includes a selection of evaluation schemes in the problem benchmarks category. We see
the variety of repositories, challenges and competitions. As it is impossible to survey all of them in
detail, we will focus on some of them, such as the ICAPS (International Conference on Automated
Planning and Scheduling) competitions, the evaluation in the area of machine learning (including
data mining, data science and KDD challenges), reinforcement learning and videogames.
The ICAPS competitions around the areas of planning and scheduling have taken place since
they were started by Drew McDermott in 1998. The recent planning competitions (as for the 2014
edition,Vallatietal,2015)featureseveraltracks,fordeterministic,learning,continuousprobabilistic
and discrete probabilistic domains. The original deterministic track now contains many domains
(both sequential and temporal). The actual domains used for the evaluation are only disclosed
during the evaluation, which in some recent editions has been done with the organisers running the
planners themselves instead of the participants. Since all participatns use the same machine, the
sole criterion is set that the problem must be (maybe optimally, depending on the subtracks) solved
within a given amount of time. In the learning version the planners can take advantage of previous
problems of the same domain, using random variants generated with the use of a distribution for
each domain. Repetitions are used to account for variations in the actual computation resources

16 Jos´eHern´andez-Orallo
| EvaluationScheme |     | Description |     |     |
| ---------------- | --- | ----------- | --- | --- |
CADEATPSystemCompetition12 Theoremproving(SutcliffeandSuttner,2006)(Sutcliffe,2009).
TerminationCompetition13 Terminationoftermrewritingprograms(March´eandZantema,2007).
Thereinforcementlearningcompetition14 Reinforcementlearning(Whitesonetal,2010;Dimitrakakisetal,2014).
| Syntax-guidedsynthesiscompetition15      |     | Programsynthesis(Aluretal,2013). |     |     |
| ---------------------------------------- | --- | -------------------------------- | --- | --- |
| InternationalAerialRoboticsCompetition16 |     | Pilotlessaircraftcompetition.    |     |     |
DARPAGrandChallenge17
Autonomousgroundvehicles.
DARPAUrbanChallenge18
Driverlessvehicles.
| DARPACyberGrandChallenge19 |     | Computersecurity.                        |     |     |
| -------------------------- | --- | ---------------------------------------- | --- | --- |
| DARPASavetheday20          |     | RescueRoboticchallenge(Jacoffetal,2003). |     |     |
Theplanningcompetition21 Planning(LongandFox,2003;Vallatietal,2015).
UCI22 andKEEL23 repositories Machinelearning(BacheandLichman,2013)(Alcala´etal,2010).
| PRTools24 |     | Patternrecognitionproblemrepository. |     |     |
| --------- | --- | ------------------------------------ | --- | --- |
KDD-cupchallenges25 andkaggle26 Machinelearninganddataminingcompetitions.
Challengesinmachinelearning27 Miscellaneousmachinelearningchallenges
Plagiarismdetection28 Plagiarism,authorshipandsocialsoftwaremisuse(Potthastetal,2013).
TheGeneralVideoGameCompetition29
Generalvideogameplayers(Schaul,2014;Perezetal,2015).
| HutterPrize30 | andrelatedbenchmarks31 |     |     |     |
| ------------- | ---------------------- | --- | --- | --- |
Textcompression.
Pedestrianbenchmarks Pedestriandetection(Ger´onimoandL´opez,2014)
ImageClef,LanguageImageRetrieval32 Cross-languageannotationandretrievalofimages(Caputoetal,2014)
Europarl33,SEtimes34,theeuromatrix35 Machinetranslationcorpora(Starkieetal,2006).
NISTOpenMTandDARPATIDESMT36 Automatictranslationbetweenhumanlanguages
| Linguisticdataconsortiumcorpora37 |     | NLPcorpora.          |     |     |
| --------------------------------- | --- | -------------------- | --- | --- |
| AngryBirdsAICompetition38         |     | Angrybirdsvideogame. |     |     |
TheArcadeLearningEnvironment39 Atari2600videogames(reinforcementlearning)(Bellemareetal,2013).
GPbenchmarks40 Geneticprogramming(McDermottetal,2012;Whiteetal,2013).
Pathfindingbenchmarks41
Gridworlddomains(mazes)(Sturtevant,2012).
FIRAHuroCup42
Humanoidrobotcompetitions(Andersonetal,2011)
Table 2 Listofsomeevaluationschemesintheproblem-benchmarkscategory.
(e.g., cloud platforms). The probabilistic versions have different performance metrics and a more
reduced set of domains, but the general principles of the competition are similar.
In machine learning and related areas (data mining and data science), because of its recent
relevance, the number of repositories and competitions is growing incessantly. Let us start our
analysiswithoneofthemostwidespreadrepositoriesincomputerscience,theUCImachinelearning
repository(BacheandLichman,2013).Mostofthediscussionbelowisapplicabletootherrepositories
| and, to | some extent, to competitions | and challenges | in machine | learning. |
| ------- | ---------------------------- | -------------- | ---------- | --------- |
The UCI repository includes many supervised (classification and regression) and some unsuper-
viseddatasets.Therepositoryispubliclyavailableandisregularlyusedinmachinelearningresearch.
Therepositoryisamulti-domaincollectionofdatasetsthathasbeenveryusefulinthedevelopment
of several machine learning techniques in the past two decades, such as ensemble methods (Zhou,
2012)ormeta-learning(Brazdiletal,2008),wheretheevaluationovermultipledatasetswascrucial
| to compare | the improvements | of new algorithms, | parameters | and tools. |
| ---------- | ---------------- | ------------------ | ---------- | ---------- |

Evaluationinartificialintelligence 17
Theusageprocedure,whichisreferredtoas“TheUCItest”(Maci`aandBernad´o-Mansilla,2014)
or the “de facto approach” (Drummond and Japkowicz, 2010; Japkowicz and Shah, 2011), follows
the general form of equation 3 where M is the repository, p is the choice of datasets and R is one
particular performance metric (accuracy, AUC, Brier score, F-measure, MSE, etc., Ferri et al, 2009;
Hern´andez-Orallo et al, 2012a). With the chosen datasets, several algorithms (where one or more
are usually introduced by the authors of the research work) can be evaluated by their performance
on the datasets. The aggregation over several datasets according equation 3, however, is not very
common in machine learning, as there is the general belief that averaging the results for several
datasets is wrong, as results are not commensurate (see, e.g., Demˇsar, 2006). We already discussed
thisissueinsection2.1andsawthattherearewaystonormalisetheperformancemetricorusesome
utility measure instead (e.g., what are the costs, in euros, of false positives and false negatives for
each dataset) such that they can be aggregated. Nonetheless, statistical tests are the predominant
and encouraged approach to evaluation validation by the machine learning research community.
“TheUCItest”canbeseenasabona-fidemixoftheproblembenchmarkapproachandthepeer
confrontation approach. Even if there is a repository (M), only a few problems are chosen, and can
becherry-picked(pischangingandarbitrary).Also,astheresearchers’algorithmmustbecompared
tootheralgorithmstoshowthatthenewoneisbetter,afewcompetingalgorithmsarechosen,which
12 http://www.cs.miami.edu/˜tptp/CASC/
13 http://termination-portal.org/wiki/Termination_Competition_2014
14 http://www.rl-competition.org/
15 http://www.sygus.org/
16 http://www.aerialroboticscompetition.org/
17 http://archive.darpa.mil/grandchallenge04/index.htm
18 http://archive.darpa.mil/grandchallenge/
19 http://www.cybergrandchallenge.com/
20 http://www.theroboticschallenge.org/
21 http://ipc.icaps-conference.org/
22 http://archive.ics.uci.edu/ml/
23 http://sci2s.ugr.es/keel/datasets.php
24 http://prtools.org/
25 http://www.sigkdd.org/kddcup/index.php
26 http://www.kaggle.com/
27 http://www.chalearn.org/
28 http://pan.webis.de/
29 http://www.gvgai.net/
30 http://prize.hutter1.net/
31 http://mattmahoney.net/dc/text.html
32 http://www.imageclef.org/
33 http://www.statmt.org/europarl/
34 http://www.statmt.org/setimes/
35 http://matrix.statmt.org/matrix/info
36 http://www.nist.gov/itl/iad/mig/openmt.cfm
37 https://www.ldc.upenn.edu/new-corpora
38 https://aibirds.org/
39 http://www.arcadelearningenvironment.org/
40 http://gpbenchmarks.org/
41 http://www.movingai.com/benchmarks/
42 http://www.fira.net/contents/sub03/sub03_1.asp

18 Jos´eHern´andez-Orallo
can also be cherry-picked, without much effort on fine-tuning their best parameters. Finally, as the
results are analysed by statistical tests, cross-validation or other repetition approaches are used to
reduce the variance of R(π,µ,p) so that we have fewer “ties”. This procedure frequently leads to
claimsaboutnewmethodsbeingbetterthantherest.Manyoftheseclaimsarebasedonattempting
many techniques and variants until some of them are better than state of the art. As a result,
many results are usually affected by parameter overfitting, especially when using cross-validation
(Rao et al, 2008) or are reproducible, but not replicable (Drummond, 2009), i.e., whenever a few
things are changed (about the kind of data, the application domain or the parameter tuning) the
improvementcompletelyvanishes.Nonetheless,theUCIrepositoryisnottoblameforthissituation,
butratherthemethodologywherestatisticalsignificanceforafewdatasetsisgivenmorevaluethan
a commensurate average aggregate performance on a large collection of datasets.
As a result, there have been suggestions of a better use of the UCI repository. These suggestions
imply an improvement of the procedure but also of the repository itself. For instance, UCI+, “a
mindfulUCI”(Maci`aandBernad´o-Mansilla,2014),proposesthecharacterisationoftheproblemsin
theUCIrepositorybyasetofcomplexitymeasuresfrom(HoandBasu,2002).Thischaracterisation
canbeusedtomakesamplesthataremorediverseandrepresentative.Also,theydiscussthenotionof
a problem being ‘challenging’, trying to infer a notion of ‘difficulty’. In the end, an artificial dataset
generator is proposed to complement the original UCI dataset. It is a distortion-based generator
(similar to Soares’s UCI++, Soares, 2009). Finally, Maci`a and Bernad´o-Mansilla (2014) suggest
ideas about sharing and arranging the results of previous evaluations so that each new algorithm
canbecomparedimmediatelywithmanyotheralgorithmsusingthesameexperimentalsetting.This
ideaof‘experimentdatabase’(Vanschorenetal,2012)hasalreadybeensetup.Openml43 (vanRijn
et al, 2013; Vanschoren et al, 2014) is an open science platform that integrates machine learning
data, software and results.
Although some of these improvements are in the line of better sampling approaches (more rep-
resentative and more effective), there are still many issues about the way these repositories are
constructed and used. The complexity measures could be used to derive how representative a prob-
lem is with respect to the whole distribution in order to make a more adequate sampling procedure.
Also, a pattern-based generator instead of a distortion-based generator could give more control of
what is generated and its difficulty. This could be done with a stochastic generative grammar for
differentkindsofpatterns,asisusuallydonewithartificialdatasets,usingGaussiansorgeometrical
constructs. Finally, if results are aggregated according to equation 3, the experimental setting and
the use of repetitions should be overhauled. For instance, by using 20 different problems with 10
repetitionsusing cross-validation (avery common settingin machine learning experiments)wehave
less information than by using 200 different problems with 1 repetition. Choosing the least informa-
tive procedure only makes sense because of the way results are fitted into the statistical tests and
also because repetitions usually involve less effort than preparing a large number of datasets.
Inotherwords,abenchmarkcanbewellusedornot,depending,e.g.,onhowdatasetsarechosen,
and the evaluation procedures used for comparing algorithms. As an alternative, independently-
organised competitions or challenges are a fairer way of comparing progress in machine learning
and related areas. For instance, several KDD challenges are organised at several conferences (such
as the KDD Cup by ACM SIGKDD44 and the Discovery Challenges at ECML/PKDD45). In these
43 http://openml.org/.
44 http://www.kdd.org/kdd-cup.
45 http://www.ecmlpkdd2015.org/discovery-challenges.

Evaluationinartificialintelligence 19
competitions, a single dataset or domain is used and the prize is given to the participant that
is able to integrate better statistical and machine learning tools to get the best results using the
chosenmetric.Thiscanworkwithautomatedsubmissionproceduresandleaderboards,suchasthose
displayed by Kaggle46 and other platforms. These competitions, if performed for a wide range of
problems at a time, could be a way of controlling some of the methodological problems of how the
repositories, such as the UCI, are used. Nevertheless, these will be still evaluating AI teams instead
of systems. This is perhaps motivated by the view that machine learning algorithms are usually
seen as components rather than systems. One way or another, there are no current competitions of
non-interactive machine learning where the systems are really compared.
Reinforcement learning can be considered a part of machine learning, which is usually very
different in terms of techniques and evaluation procedures, compared to what happens in other
non-interactive machine learning areas, used in data mining and data science applications. In rein-
forcement learning, one of the main features is that the quality of a system is evaluated according
toanaggregatedmetricofthereceivedrewardsduringasession,episodeortrial.Thereinforcement
learning competition47 has been running intermittently since 2004. In 2014, three domains were
included (helicopter, polyathlon and invasive species) (Dimitrakakis et al, 2014), although in the
past there have been many other domains. Teams must be registered in a server and must provide
a system that works under the RL-glue standard interface, by modifying a basic RL agent that
is given with the training pack for each domain. Participants can take part in each of the three
domains independently. Trials are repeated for robustness and leaderboards are built according to
the aggregated reward metric.
One relevant insight extracted from the recent editions is that some “seemingly ‘easy’ domains
have old approaches which remain quite hard to beat. It is consequently a difficult task to find
new, sufficiently challenging domains; in the last competition [the invasive species problem was
tried], which is an apparently very complex problem, but for which a very simple approach seemed
to perform the best in the competition. In the end, the metrics [used] to test the algorithms are
quite important, as different metrics may put different algorithms on top. To give an example, if
we measure the total reward over a very large number of time steps, we may favor algorithms that
are asymptotically optimal but which perform badly in the short term. In the end, a single number
can’t say very much” (Dimitrakakis, 2016).
One domain that is very well-suited for RL-like agents is videogames. Since very realistic video-
games in 3D with complex textures are generally beyond the state of the art for artificial vision
systems, several proposals have been undertaken where simple arcade games are used instead. For
instance, the Arcade Learning Environment48 integrates many Atari 2600 videogames (Bellemare
et al, 2013). The screen consists of 160×210 pixels, with a 128-colour palette and 18 actions. About
55differentgamescanbeusedaschallengesfor“reinforcementlearning,modellearning,model-based
planning,imitationlearning,transferlearning,andintrinsicmotivation”(Bellemareetal,2013).For
instance, the score can be processed and mapped to the reward input of a generic RL interface (as
done by Mnih et al (2015), showing fantastic performance), although using a very large number of
training sessions.
One of the key issues of the Arcade Learning Environment is how scores are integrated and
compared. Bellemare et al (2013) discuss three ways in which scores can be normalised: comparing
46 http://www.kaggle.com.
47 http://www.rl-competition.org/.
48 http://www.arcadelearningenvironment.org/.

20 Jos´eHern´andez-Orallo
to a reference score, e.g., a random agent, normalising with a baseline set, e.g., using several agents
to get some kind of average baseline score and inter-algorithm normalisation, i.e., setting the best
algorithm for each game to 1 and then normalising the rest. Also, there is an interesting discussion
on how scores for different games are aggregated. First, it is very important that they are first
normalised, otherwise they will not be commensurate. Second, several options exist, such as an
average score, median score or a score distribution (a quantile plot).
TheGeneralVideoGameCompetition49 (Perezetal,2015)isbasedontheVideoGameDescrip-
tionLanguage(Schaul,2014),alanguagethatallowsarcadevideogamestobedefinedatanabstract
level, describing objects and dynamics in a two-dimensional space. The analysis of the screen can
be done at a more abstract level, without necessarily using an artificial vision approach at regular
screenshots, as in the Arcade Learning Environment. Some games are simplified versions of popular
games (e.g., Pacman) whereas others have been created on purpose. Games are defined in a rein-
forcement learning setting, but there are several reward schemas: binary (there is an all-or-nothing
reward, depending on whether the agent achieves a final goal), incremental (a more traditional cu-
mulative reward system) and discontinuous (somewhat in between). The competition is organised
in three stages: training, validation and test. The games for the last stage are not known by the
participants to prevent evaluation overfitting. Overall, this is a very significant effort towards more
general artificial intelligence. Consequently it will be discussed again in section 3. Nevertheless, it is
not, for the moment, an ability-oriented approach, since it just aggregates results for several games
without identifying the relevance of several abilities in each of them.
In a very different setting, several DARPA challenges for autonomous vehicles, security and
rescue robots have been held in the past years. Some of them strongly rely on hardware and the
quality of sensors according to the particular application, such as the DARPA Urban Challenge for
autonomous vehicles50 and the DARPA Save the Day (http://www.theroboticschallenge.org/)
forRescueRobots.Amoresoftware-orientedchallengeisDARPACyberGrandChallenge51 wherea
purpose-built computer competes against the circuit’s greatest experts in CTF (Capture the Flag),
atournamentcircuitwhereexpertsreverseengineersoftware,probeitsweaknesses,searchfordeeply
hidden flaws, and create securely patched replacements. In general, DARPA integrates very specific
and challenging domains that require a strong commitment for participation in terms of resources
and the integration of techniques.
In the area of natural language processing (NLP), machine translation area is one where there is
anabundanceofcorpora52 aswellasseveralevaluationeffortscoordinatedbytheNationalInstitute
of Standards and Technology (NIST OpenMT) and DARPA (TIDES MT)53. The particular focus
of each evaluation series has changed over the years but they are aimed at the general problem of
automatic translation between human languages. One of the issues in these evaluations is how to
rankalternativetranslations,usingvolunteerhumanassessments,andthederivationofmetricsfrom
them. In a way, because of the human factor, some of these tasks can also be considered a case of
the previous subsection (human discrimination), but in a much more controlled scenario, given the
corpora.
49 http://www.gvgai.net/.
50 http://archive.darpa.mil/grandchallenge/.
51 http://www.cybergrandchallenge.com/.
52 http://www.statmt.org/europarl/, http://www.statmt.org/setimes/, http://matrix.statmt.org/matrix/
info.
53 http://www.nist.gov/itl/iad/mig/openmt.cfm.

Evaluationinartificialintelligence 21
Finally,somebenchmarksintegratemanydomains,butatthesametimeareveryspecific(likethe
DARPAchallenges).ImageClef,theCLEFCrossLanguageImageRetrievalTrack54,isabenchmark
for the evaluation of cross-language annotation and retrieval of images (Caputo et al, 2014). The
goalistheannotationandretrievalofimagesinvariousdomains.The2014editionconsistedoffour
tasks: domain adaptation, scalable concept image annotation, liver CT image annotation and robot
vision.Thesechangedintootherfivetasksin2015:imageannotation,medicalclassification,medical
clustering and liver annotation.
Overall, even if the repositories and competitions in machine learning and other domains seen
above may have particular issues, many of the benchmarks and competitions in Table 2 suffer from
the same problems about how representative the problems are (if M is small), how representative
the sample is (if M is large) or whether there are some kinds of problems that can be solved with
specific approaches that dominate the sample. Other issues are the estimation of task difficulty and
whether M is able to really discriminate between a set of AI systems. Also, none of the benchmarks
in AI are adaptive.
2.4 Evaluation by peer confrontation
In the evaluation by peer confrontation, we evaluate a system by letting it compete against another
system.Thisusuallymeansthatamatchisplayedbetweenpeers.Thisisusualforgames(including
gametheory)andalsocommoninmulti-agentresearch.Theresultsofeachmatch(possiblyrepeated
withthesamepeer)mayserveasanestimationofwhichofthetwosystemsisbest(andhowmuch).
Nonetheless,themainproblemaboutthisapproachisthattheresultsarerelativetotheopponents.
This is natural in games, as people are said to be good or bad at chess, for instance, depending on
whom they are compared to.
Despitethisrelativecharacteroftheevaluation,wecanstillseetheaverageperformanceaccord-
ing to equation 3. In order to do this, we must first identify the set of opponents Ω. Then, the set
of problems M is enriched (or even substituted) by the parametrisation of each single game (e.g.,
chess)withdifferentcompetitorsfromΩ.In1-vs-1matcheswehavethat|M|=|Ω|−1(ifwedonot
consider a match between a system and itself). In other multi-agent situations where many agents
play at the same time, |M| can grow combinatorially on |Ω|.
Nonetheless,forAIresearch,ourmainconcernisaboutrobustnessandstandardisationofresults.
For instance, how can we compare results between two different competitions if opponents are
different? If these competitions are performed year after year, how can we compare progress? If
there are common players, we can use rankings, such as the Elo ranking (Elo, 1978), or more
sophisticated rating systems (Smith, 2002; Masum and Christensen, 2003), to see whether there
is progress. In fact, it would be very informative for AI competitions based on peer confrontation
to keep all participants from previous editions in subsequent editions. However, this comes with a
drawback, as systems could specialise to the kind of opponents that are expected in a competition.
If a high percentage of competitors are inherited from previous editions, specialisation to those old
(and bad) systems could be common.
Itisinsightfultothinkhowmanyoftheseissuesareaddressedinsportcompetitions.Forinstance,
some tournaments adapt their matches according to previous information (by round, by ranking,
etc.). In fact, a league may be redundant (for the same reasons why the information-driven or
54 http://www.imageclef.org/.

22 Jos´eHern´andez-Orallo
difficulty-driven sampling are introduced) and other tournament arrangements are more effective
with almost the same robustness and far fewer matches.
As an alternative, games and multi-agent environments could be evaluated against standardised
opponents.However,howcanwechooseasetofstandardisedopponents?Iftheopponentsareknown,
the systems can be specialised to the opponents. For instance, in an English draughts (checkers)
competition, we could have players being specialised to play against Chinook, the proven optimal
player (Schaeffer et al, 2007). Again, this ends up again in the design of an opponent generator.
This of course does not mean a random player (which is usually very bad), but players that can
play well. One option is to use old systems where some parameters are changed. Alternatively, a
morefar-reachingapproachistodefineanagentlanguageandgenerateplayers(programs)withthat
language. As it is expected that this generation will not achieve very good players (otherwise we
wouldbefacingaverysimpleproblem),apossiblesolutionistogivemoreinformationandresources
to these standardised opponents to make them more competitive (e.g., in some applications these
opponents could have more sophisticated sensor mechanisms or some extra information about the
match that regular players do not have).
Given the set Ω composed of old opponents or generated opponents, we need to assess whether
Ω is sufficiently challenging and whether it is able to discriminate the participants. For instance,
some competitions in AI finally award a champion, but there is the feeling that the result is mostly
arbitraryandcausedbyluck,ashappenswithmanysportcompetitions55.Howcanweassesswhether
the set Ω has sufficient difficulty and discriminating power? This is of course a hard problem, which
has recently been analysed in (Hern´andez-Orallo, 2014), which is not only applicable to multi-agent
systems but for the assessment of any kind of task, by calculating the size of the simplest policy
that solves the problem.
Table 3 shows a sample of evaluation schemes based on peer confrontation. Once again, because
of obvious space constraints, we will just choose some representative and interesting cases from the
table. First, we will see the Computer Olympiad56, an event that congregates many board games,
which has been held intermittently since 1989. After so many years, systems are very sophisticated
andcompletelyspecialisedtoonegame,whichusuallyrequiresaverygoodintegrationofknowledge
about the game and heuristics. One relevant issue of the evolution of the olympiad is that some
games soon disappeared from the competitions, such as checkers, because an optimal solution was
found for the game (and hence the competition lost interest for computers).
TheGeneralGameCompetition,whichhasbeenrunyearlysince2005,canbeseenasareaction
to the computer olympiad and the classical approach to solve specific games, as represented by the
superhuman results in particular games such as chess, draughts, poker and others. According to
the webpage57, “general game players are systems able to accept descriptions of arbitrary games at
runtimeandabletousesuchdescriptionstoplaythosegameseffectivelywithouthumanintervention.
In other words, they do not know the rules until the games start”. Games are described in the
language GDL (Game Description Language). The description of the game is given to the players.
Different kinds of games are allowed, such as noughts and crosses (tic tac toe), chess, in static
or dynamic worlds, with complete or partial information, with varying number of players, with
simultaneous or alternating plays, two-player or single-player, etc. For the competition, games are
chosen—non-randomly,i.e.,manuallybytheorganisers—fromthepoolofgamesalreadydescribed
55 Statisticaltestsarenotusedtodeterminewhenacontestantcanbesaidtobesignificantlybetterthananother.
56 http://www.icga.org/.
57 http://games.stanford.edu/.

Evaluationinartificialintelligence 23
EvaluationScheme Description
Robocup58 andFIRA59 Robotics(robotfootball/soccer)(Kitanoetal,1997;Kim,2004).
GeneralgameplayingAAAIcompetition60 GeneralgameplayingusingGDL(Geneserethetal,2005).
WorldComputerChessChampionship61 Chess.
ComputerOlympiad62 Boardgames.
AnnualComputerPokerCompetition63 Poker.
TradingAgentsCompetition64 Tradingagents(Wellmanetal,2004;KetterandSymeonidis,2012).
WarlightAIChallenge65 Strategygames(Warlight).
Table 3 Listofsomeevaluationschemesinthepeer-confrontationcategory.
inGDLandnewgamesarealsonewlyintroducedforthecompetition.Asaresult,gamespecialisation
is difficult. The competition is run like many sports tournaments, with players participating in
qualifying rounds (where they may be tested in single player games, e.g., Sudoku) or in two-player
gamesagainstasampleplayer.Thosequalifiedparticipateinpreliminaryrounds,wherescoresfrom
single-player games and results against the other competitors are aggregated. The top four players
from them qualify for the semifinal and final rounds. Results are basically win/loss/tie. This loses
the information about partial situations during the game or the tactics that have been used during
the game.
DespitebeingoneofthemostinterestingAIcompetitions,thereisstillsomemarginforimprove-
ment. For instance, a more sophisticated analysis of how difficult and representative each problem
is would be useful. For instance, several properties about the adequacy of an environment or game
for peer-confrontation evaluation could be identified and analysed depending on the population of
opponents that is being considered. Also, rankings (e.g., using the Elo system mentioned above)
could be calculated, and former participants could be kept for the following competitions, so there
aremore participants(and more overlap betweencompetitions).Amore radicalchange wouldbeto
learn without the description of the game, as a reinforcement learning problem (where the system
learns the rules from many matches).
The RoboCup Soccer competition66 is clearly a competition where team confrontation takes
place,andbothintrateamcooperationandinterteamcompetitionarerequired.Evenifitisarobotic
competiton, the modalities are played with the same hardware (or with some strict hardware cate-
goriesorspecifications),sothatthecompetitioncanfocusonAItechniques,andnotonsensorimotor
hardware optimisation. Obviously, in many categories, the competition becomes more challenging
not because new domains are introduced or because the rules are changed, but rather because the
other opponents improve. The competition is held as usual sports tournaments, either as a league
or with rounds.
Summing up our observations on peer confrontation problems, we see that the dependency on
the set Ω makes this kind of evaluation more problematic. Nonetheless, as AI research is becoming
58 http://www.robocup.org/
59 http://www.fira.net
60 http://games.stanford.edu/
61 http://www.icga.org/
62 http://www.icga.org/
63 http://www.computerpokercompetition.org/
64 http://tradingagents.eecs.umich.edu/
65 http://theaigames.com/competitions/warlight-ai-challenge/rules
66 http://www.robocup.org/.

24 Jos´eHern´andez-Orallo
more socially oriented, with significantly more presence of multi-agent systems and game theory,
an effort has to be undertaken to make this kind of evaluation more systematic, instead of the
plethoraofarrangementsthatweseeinsports,forinstance.Basically,theissueisthattheorganisers
want to limit the number of confrontations when the number of participants is high, but may
be reluctant to use a schema based on rounds because it can be highly unreliable. Apart from
the Elo ranking (Elo, 1978) and other rating systems mentioned above (Smith, 2002; Masum and
Christensen, 2003), some recent studies have analysed partially completed sports competitions, and
how possible and necessary winners can be derived from partial tournaments, and arrange pending
pairwise comparisons accordingly (Aziz et al, 2015).
2.5 Highlights and directions of the evaluation of specialised AI systems
Given the three kinds of evaluation in the previous subsections (for which there may be some
overlap, as we have seen), we now give a more comprehensive view of the key issues about all these
initiatives.Thefirstthingthatwerealiseisthattheevaluationeffortsareextremelyscatteredacross
AIdisciplines,anditisquitecommontofindduplicatedefforts,forwhichsolutionshavetobefound
again and again. Apparently, there is limited exchange of experiences between them, just because
the domains are different. It is then very useful to look at some organisations that can serve to
centralise and exchange insights and lessons-learnt across several domains, apart from identifying
new needs for benchmarks and competitions.
ExamplesoftheseorganisationsaregovernmentinstitutionssuchasNISTandDARPA,scientific
organisations such as AAAI, or on-purpose associations, such as chalearn67, focused on machine
learning and with an associated book series. In particular, NIST has performed a continued effort
towards the evaluation of several domains in AI. In fact, the series of workshops on Performance
Metrics for Intelligent Systems, held from 2000 to 2012 at the National Institute of Standards &
Technology (Meystel, 2000; Messina et al, 2001; Evans and Messina, 2001; Meystel et al, 2003b,a;
Gordon, 2007; Madhavan et al, 2009; Schlenoff et al, 2011) is the most relevant continuous effort
in the analysis of the state of the art, the methodology and the progress of application-oriented
evaluation in AI.
Whilethefirsttwoworkshopsdiscussedapossibleanalysisofthe“SpaceofIntelligence”(Meystel,
2000), their participants were “not looking for and [were] not interested in a nouveau Turing test”
(Messina et al, 2001). The preference for task-oriented evaluation soon prevailed: “the more that we
can make it clear that we are interested in performance, rather than intelligence, per se, the bet-
ter off we will be” (Simmons, 2000). From 2003 onwards the workshops focused almost exclusively
on “performance measures” for “practical problems in commercial, industrial, and military appli-
cations” (Meystel et al, 2003b), covering, e.g., self-driving cars, robotic rescue systems, distributed
control, human-robot interaction, soldier-worn sensor systems, Mars rovers, mining robots, smart
grid systems, manufacturing robots, healthcare systems, etc.
Several interesting conclusions were drawn from the workshop reports through the years. For
instance, there was the perception that the focus on applications have been created a schism with
artificial intelligence. A session of the 2007 workshop focused on “(re-)establishing or increasing
collaborative links between artificial intelligence and intelligent systems” (Gordon, 2007), as the
latterweresupposedtobeconcernedaboutcontrolandrobotics.Actually,therewasaperceptionof
67 http://www.chalearn.org/.

Evaluationinartificialintelligence 25
progress but, because the great amount of systems fell in the category of “cyber physical systems”,
it is difficult to tell in many domains whether this comes from better hardware, better sensors or
betterAImethods.Also,thespecialisationofmanymetricstothedomainandthelackofcontinuity
in some evaluation procedures were recognised as one of the limitations.
Apart from the PerMIS workshop, NIST has been organising several workshops where different
AI domains are investigated and evaluated, and the evaluation procedures have more continuity
and standardisation. For instance, NIST holds the Text Analysis Conference (TAC68), to encourage
researchinNLPbyprovidingtestsandcommonevaluationprocedures.Thetracksmaychangeeach
year, but the conference has usually covered question answering, recognising textual entailment,
summarisation and knowledge base population. A similar series is the Text Retrieval Conference
(TREC69) encourages research in information retrieval from large text collections. The tracks take
participants that submit their results and are evaluated, not as a real competition but more like a
certification, where each participant is given a report about the shortcomings of their results. The
tracks are adjusted year after year according to the results and the discussion of committees during
theconferencesand onthemailing list.Thisis notdifferent fromthewayother competitions evolve
in other organisations. However, NIST is an organisation that is specialised in evaluation and some
methodological issues are ensured and shared across different domains.
From the more academic associations in artificial intelligence, there has been a renewed interest
in establishing a series of tasks that could serve as a real evaluation of the progress of AI (You,
2015; Marcus et al, 2016). The suggestion is to include several diverse tasks. For instance, the
Winograd Schema Challenge is a commonsense reasoning task70 (Levesque et al, 2012; Levesque,
2014;Morgensternetal,2016)introducedbyHectorLevesqueinhis2013IJCAIResearchExcellence
Address.Thetaskfeaturesquestionssuchas.“Thetrophywouldnotfitinthebrownsuitcasebecause
it was too big (small). What was too big (small)? Answer 0: the trophy, Answer 1: the suitcase”.
This task and others (some resembling physically embodied versions of the Turing Tests) could be
chosen by IJCAI and AAAI for possible inclusion into a regular “Turing Championship” or “Turing
Olympics”(You,2015).Itisunclearwhyandhowthisisverydifferentfromwhathasbeenregularly
done by NIST, apart from less emphasis on hardware and robotics.
3 Towards ability-oriented evaluation
AI is successful in many ways nowadays but it took a long time to flourish in applications (e.g.,
driverless cars, machine translators, game bots, etc.). Most of them correspond to specific tasks and
require task-oriented evaluation. Other tasks that are still not solved by AI technology are already
evaluated in this way and will be successful one day. However, if instead of AI applications we think
aboutAIsystems,weseethattherearesomekindsofAIsystemsforwhichtask-orientedevaluation
is not appropriate. For instance, cognitive robots, artificial pets, assistants, avatars, smartbots, etc.,
are not designed to cover one particular application but are expected to be customised by the user
for a variety of tasks. In order to cover this wide range of (previously unseen) tasks, these systems
musthavesomeabilitiessuchasreasoningskills,inductivelearningabilities,verbalabilities,motion
abilities,etc.Hence,thismeansthatapartfromtask-orientedevaluationmethodswemayalsoneed
ability-oriented evaluation techniques.
68 http://www.nist.gov/tac/.
69 http://trec.nist.gov/.
70 http://commonsensereasoning.org.

26 Jos´eHern´andez-Orallo
ThingsaremoreconspicuouswhenwelookattheevaluationoftheprogressofAIasadiscipline.
If we look at AI with Minsky’s 1968 definition seen in the introduction, i.e., by achievement of
tasks that would require intelligence, AI has progressed very significantly. For instance, one way of
evaluating AI progress is to look at a task and check in which category an AI system is placed:
optimal if no other system can perform better, strong super-human if it performs better than all
humans,super-humanifitperformsbetterthanmosthumans,par-humanifitperformssimilarlyto
mosthumans,andsub-humanifitperformsworsethanmosthumans(Rajani,2011).Notethatthis
approachdoesnotimplythatthetaskisnecessarilyevaluatedwithahuman-discriminativeapproach.
Having these categories in mind, we can see how AI has scaled up for many tasks, even before AI
had a name. For instance, calculation became super-human in the 19th century, cryptography in
the1940s,simplegamessuchas noughtsandcrossesbecame optimalin1960s,morecomplexgames
(draughts, bridge) a couple of decades later, printed (non-distorted) character recognition in the
1970s, statistical inference in the 1990s, chess in the 1990s, speech recognition in the 2000s, and TV
quizzes, driving a car, technical translation, Texas hold ’em poker in the 2010s. According to this
evolution,theprogressofAIhasbeenimpressive(Bostrom,2014).Theuseofhumanintelligenceas
a baseline has been used in competitions (such as the humies awards71) or to define ratios, where
median human performance is set at a zero scale, such as the so-called Turing-ratio (Masum et al,
2002; Masum and Christensen, 2003), with values greater than 0 for super-human performance and
values lower than 0 for sub-human performance.
However,letusfirstrealisethatnosystemcando(orcanlearntodo)allofthesethingstogether.
The big-switch approach may be useful for a few of them (e.g., a robot with an advanced computer
vision system that detects whether it is facing a chess board or a bridge table and then switch to
theappropriateprogramtoplaythegamethatithasjustrecognised).Second,ifwelookatAIwith
McCarthy’s definition seen in the introduction, i.e., by making intelligent machines, things are less
encouraging. Not only has the progress been more limited, but also there is a huge controversy for
quantifying this progress (in fact, some argue that machines are more intelligent today than fifty
years ago while others say that there has been no progress at all other than computational power).
Hence, worse than having a poor progress or no progress at all, we regard with contempt that we
do not have effective evaluation mechanisms to evaluate this progress. It seems that none of the
evaluationschemesseenintheprevioussectionareabletoevaluatewhethertheAIsystemsoftoday
are more intelligent than the AI systems of yore. Also, for developmental robotics and other areas
of AI where systems are supposed to improve their performance with time, we want to know if a
6-month-old robot has progressed over its initial state, in the same way that we see how abilities
increase and crystallise with humans, from toddlers to adults. Ability-oriented evaluation, and not
task-oriented evaluation, seems to have a better chance of answering this question.
Tomakethepointunequivocal,wecouldevengobeyondMcCarthy’sdefinitionofAIwithoutthe
useof‘intelligence’anddefinethisviewofAIasthescience and engineering of making machines do
tasks they have never seen and have not been prepared for beforehand. Clearly, this view puts more
emphasis on learning, but it also makes it crystal clear that task-oriented evaluation, as have been
performed for years, would not fit the above definition.
It would be unfair to forget to acknowledge that some attempts seen in the previous section
have made an effort for a more general AI evaluation. The general game competition seen in the
previoussectionisoneexampleofhowsomethingsarechanginginevaluation.Usersandresearchers
are becoming tired of a big-switch approach. They yearn for and conceive systems that are able to
71 www.human-competitive.org.

Evaluationinartificialintelligence 27
cover more and more general task classes. Nonetheless, it is still a limited generalisation, which is
too based on a very specific range of tasks. Many good players at the General Game Competition
would be helpless at any game of the Arcade Learning Environments, and vice versa. Actually, only
some reinforcement learning (and perhaps genetic programming) systems can at least participate
in (adaptations to) of both games —excelling in them would not be possible though without an
important degree of specialisation.
IntherestofthissectionwewillintroducewhatanabilityisandhowtheycanbeevaluatedinAI.
Thetitleofthissection(startingwith‘Towards’)suggeststhatwhatfollowsismoreinterdisciplinary
andcontainsproposalsthatarenotwellconsolidatedyet,orthatmayevengointhewrongdirection.
Nonetheless, let us be more lenient and have in mind that ability-based evaluation is much more
challenging than task-specific evaluation.
3.1 Cognitive abilities
We must first clarify that we are talking about cognitive abilities, in the same way that in the
previous section we referred to cognitive tasks. Some AI applications require physical abilities, most
especiallyinrobotics,butAIdealswithhowthesensorsandactuatorsarecontrolled,notabouttheir
strength, consumption, etc. After this clarification, we can define a cognitive ability as a property of
individuals that allows them to perform well in a range of information-processing tasks.Atfirstsight
thisdefinitionmayjustlooklikeachangeofperspective(fromproblemstosystems).However,what
we see now is that the ability is required, and performance is worse without featuring the ability. In
other words, the ability is necessary but it does not have to be sufficient (e.g., spatial abilities are
necessary but not sufficient for driving a car). Also, the ability is assumed to be general, to cover
a range of tasks. Actually, general intelligence would be one of these cognitive abilities, one that
covers all cognitive tasks: “general intelligence is a very broad trait that encompasses quickness and
quality of response to all cognitive tasks” (Strickler, 1973).
Themajorissueaboutabilitiesisthattheyare‘properties’,andassuchtheyhavetobeconcep-
tualised and identified. While tasks can be seen as measuring instruments, abilities are constructs.
In psychology, many different cognitive abilities have been identified and have been arranged in dif-
ferent ways (Schaie, 2010). For instance, one well-known comprehensive theory of human cognitive
abilities is the Cattell-Horn-Carroll theory (Keith and Reynolds, 2010). Figure 4 shows a graphical
representation of these abilities. The top level represents the g factor or general intelligence, the
middle level identifies a set of broad abilities and the bottom level may include many narrow abili-
ties. Again, this top level seems to saturate all tasks: “g is common to all cognitive tasks including
learning tasks” (Alexander and Smales, 1997).
Interestingly,thisisnotsurprisingfromanAIstandpoint.Thebroadabilitiesseemtocorrespond
to subfields in AI. For instance, looking at any AI textbook (e.g., Russell and Norvig, 2009), we can
enumerate areas such as problem solving, use of knowledge, reasoning, learning, perception, natural
language processing, etc., that would roughly correspond to some of the cognitive abilities in Figure
4.
Can we evaluate broad abilities as we did for specific tasks? Application-specific (task-oriented)
approaches will not do. But is ability-oriented evaluation ready for this? The answer, as we will see
below,isthatthistypeofevaluationisstillinaveryincipientstageinAI.Thereareseveralreasons
for this. First, general (ability-oriented) evaluation is more challenging. Second, we no longer have a
clear definition of the task(s). In fact, defining the ability depends on a conceptualisation, and from

| 28  |     |     |     |     |     | Jos´eHern´andez-Orallo |
| --- | --- | --- | --- | --- | --- | ---------------------- |
)III mutarts(
lareneG
gg
)II mutarts(
daorB
|     | GGcc | GGff GGqq | GGrrww GGssmm | GGllrr GGvv | GGaa GGss | GGtt |
| --- | ---- | --------- | ------------- | ----------- | --------- | ---- |
)I mutarts(
|     | worraN cccc11 ............ | ccff11 ............ ccqq11 ............ | rrccww11 ............ ssccmm11 ............ | ccllrr11 ............ ccvv11 ............ | ccaa11 ............ ccss11 ............ | cctt11 ............ |
| --- | -------------------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- | --------------------------------------- | ------------------- |
|     | 11                         | 11 11                                   | 11 11                                       | 11 11                                     | 11 11                                   | 11                  |
|     | cccc2222                   | ccff2222 ccqq2222                       | rrccww2222 ssccmm2222                       | ccllrr2222 ccvv2222                       | ccaa2222 ccss2222                       | cctt2222            |
Fig.4 Cattell-Horn-Carroll’sthreestratummodel.ThebroadabilitiesareCrystallisedIntelligence(Gc),FluidIntel-
ligence (Gf), Quantitative Reasoning (Gq), Reading and Writing Ability (Grw), Short-Term Memory (Gsm), Long-
Term Storage and Retrieval (Glr), Visual Processing (Gv), Auditory Processing (Ga), Processing Speed (Gs) and
Decision/ReactionTime/Speed(Gt).
there we need to find a set of representative exercises that require the ability. And third, there have
not been too many general AI systems to date, so task-oriented evaluation has seemed sufficient for
the evaluation of AI systems so far. However, things are changing as new kinds of AI systems (e.g.,
| developmental | robotics) | are becoming | more general. |     |     |     |
| ------------- | --------- | ------------ | ------------- | --- | --- | --- |
Before starting with some approaches in the direction of ability-oriented evaluation, it can be
arguedthatsomeexistingevaluationschemesinAIarealreadyability-oriented.Forexample,evenif
theplanningcompetitionfeaturesasetoftasks,itgoesaroundtheabilityofplanning,whichismore
general than any particular task. However, the systems are not able to determine when planning is
required for a range of problems. In other words, the ability is not a resource of the system, but
the very goal of the system. In the end, it is the researchers who incorporate planning modules in
several application-specific systems, and not the systems that independently enable their planning
| abilities to            | solve a new problem. |                         |     |     |     |     |
| ----------------------- | -------------------- | ----------------------- | --- | --- | --- | --- |
| 3.2 The anthropocentric |                      | approach: psychometrics |     |     |     |     |
Psychometrics was developed by Galton, Binet, Spearman and many others at the end of the 19th
centuryandfirsthalfofthe20thcentury.Anearlyconceptthatarosewastheneedofdistinguishing
tasksrequiringveryspecificknowledgeorskillsfromgeneralabilities.Forinstance,an“idiotsavant”
couldhavealotofknowledgeorcouldhavedevelopedasophisticatedskillduringtheyearsforsome
specific domain, but could be obtuse for other problems. On the contrary, a very able person with
no previous knowledge could perform well in a range of tasks, provided they are culture-fair. This
distinctiontookseveraldecadestoconsolidate.Inaway,thisbearsresemblancewiththenarrowvs.
| general dilemma | in AI. |     |     |     |     |     |
| --------------- | ------ | --- | --- | --- | --- | --- |
Psychometricsisconcernedaboutmeasuringcognitiveabilities,personalitytraitsandotherpsy-
chological properties (Sternberg (ed.), 2000). Factors differ from abilities, in principle, in that they
are obtained through testing and further analysed through systematic approaches based on factor
analysis. Some factors have been equated and named after existing abilities while others are ‘dis-
covered’ and receive new technical names. Several indices can be derived from a battery of tests by
aggregatingabilitiesandfactors.Onejointindexthatisusuallydeterminedfromsomeofthesetests
is known as IQ (Intelligence Quotient). Although IQ was originally normalised by the subject’s age
(hence its name), its value for adults today is normalised relative to an adult population, assum-
ing a normal distribution with mean µ=100 and standard deviation σ=15. This corresponds to a

Evaluationinartificialintelligence 29
more sophisticated (normalised) aggregation of results for several items, which again resembles our
equation 3.
IQ tests incorporate items of variable difficulty. Item difficulty is determined by the percentage
of subjects that are able to solve the item, or using functional models in Item Response Theory
(Lord, 1980; Embretson and Reise, 2000), as seen in the previous section. Note that this difficulty
assessment is relative to the population and not derived from the nature of the item itself.
IQ tests are easy to administer, fast and accurate, and they are used by companies and govern-
ments, essential in education and pedagogy. IQ tests are generally culture-fair through the use of
abstract exercises (except for the verbal comprehension abilities).
As they work reasonably well for humans, their use for evaluating machines has been suggested
many times, even since the early days of AI, with the goal of constructing “a single program that
wouldtakeastandardintelligencetest”(Newell,1973).Morerecently,theirusehasbeenvindicated
by Bringsjord and Schmimanski (Bringsjord and Schimanski, 2003; Bringsjord, 2011), under the
so-called ‘Psychometric AI’ (PAI), as “the field devoted to building information-processing entities
capable of at least solid performance on all established, validated tests of intelligence and men-
tal ability, a class of tests that includes not just the rather restrictive IQ tests, but also tests of
artistic and literary creativity, mechanical ability, and so on”. It is important to clarify that PAI
is a redefinition or new roadmap for AI —not an evaluation methodology— and does not further
develop or adapt IQ tests for AI systems. In fact, PAI does not explicitly claim that IQ tests (or
other psychometric tests) are the best way to evaluate AI systems, but it is said that an “agent is
intelligent if and only if it excels at all established, validated tests of intelligence” (later broadened
to any other psychometric test) (Bringsjord and Schimanski, 2003; Bringsjord, 2011). The question
of whether these tests are a necessary and sufficient condition for machines and the limitations of
PAI as a guide for AI research have been recently discussed in (Besold, 2014).
Not surprisingly, this claim that IQ tests are the best way to evaluate AI systems has recently
come from human intelligence research. Detterman, editor of the Intelligence Journal, wrote an
editorial(Detterman,2011)wherehesuggestedthatWatson(thethenrecentwinneroftheJeopardy!
TV quiz (Ferrucci et al, 2010)) should be evaluated with IQ tests. The challenge is very explicit: “I,
theeditorialboardofIntelligence,andmembersoftheInternationalSocietyforIntelligenceResearch
will develop a unique battery of intelligence tests that would be administered to that computer
and would result in an actual IQ score” (Detterman, 2011). Detterman established two levels for
the challenge, a first level, where the type of IQ tests can be seen beforehand by the AI system
programmer, and a second level, where the types of tests would have not seen beforehand. Only
computers passing the second level “could be said to be truly intelligent” (Detterman, 2011). The
needfortwolevelsseemsrelatedtothebig-switchapproachandtheproblemoverfittingissue,which
wehavealreadymentionedinprevioussectionsforAIevaluationschemes.Itisappositeatthispoint
to recall that academic and professional IQ tests and many other standardised psychological tests
are never made public, because otherwise people could practise on them and game the evaluation.
Notethatthenon-disclosureofthetestsuntilevaluationtimeissomethingthatweonlyfindinvery
few evaluation schemes in the previous section.
Detterman was unaware that almost a decade before, in 2003, Sanghi and Dowe (Sanghi and
Dowe,2003)implementedasmallprogram(lessthan1,000linesofcode)whichcouldscorerelatively
well on many IQ tests, as shown in Table 4. The program used a big-switch approach and was
programmed to some specific kinds of IQ tests the authors had seen beforehand. The authors still
made the point unequivocally: this program is not intelligent and can pass IQ tests.

30 Jos´eHern´andez-Orallo
|     |     | Test             | IQScore | HumanAverage |
| --- | --- | ---------------- | ------- | ------------ |
|     |     | A.C.E.IQTest     | 108     | 100          |
|     |     | EysenckTest1     | 107.5   | 90-110       |
|     |     | EysenckTest2     | 107.5   | 90-110       |
|     |     | EysenckTest3     | 101     | 90-110       |
|     |     | EysenckTest4     | 103.25  | 90-110       |
|     |     | EysenckTest5     | 107.5   | 90-110       |
|     |     | EysenckTest6     | 95      | 90-110       |
|     |     | EysenckTest7     | 112.5   | 90-110       |
|     |     | EysenckTest8     | 110     | 90-110       |
|     |     | IQTestLabs       | 59      | 80-120       |
|     |     | TestedichIQTest  | 84      | 100          |
|     |     | IQTestfromNorway | 60      | 100          |
|     |     | Average          | 96.27   | 92-108       |
Table 4 ResultsbyarudimentaryprogramforpassingIQtests(from(SanghiandDowe,2003)).
While it must be conceded that the results only reach the first level of Detterman’s challenge
—sothereisatestadministrationissue(i.e.,anevaluationflaw)—therearesomeweaknessesabout
human IQ tests that would also arise if a system passed the second level as well. In particular, “the
editorial board of Intelligence, and members of the International Society for Intelligence Research”
couldbetemptedtodeviseorchoosethoseIQteststhataremore‘machine-unfriendly’.IfAIsystems
eventually passed some of them, the battery could be refined again and again, in a similar way as
howCAPTCHAsareupdatedwhentheybecomeobsolete.Inotherwords,thisselection(orbattery)
of IQ tests would need to be changed and made more elaborate year after year as AI technology
advances. Also, the limitations of this approach if AI systems ever become more intelligent than
| humans | are notorious. |     |     |     |
| ------ | -------------- | --- | --- | --- |
The main problem about IQ tests is that they are anthropocentric, i.e., they have been devised
for humans and take many things for granted. For instance, most assume that the subject can
understand natural language to read the instructions of the exercise. On top of that, they are
specialised to some human groups. For instance, tests are significantly different when evaluating
small children, people with disabilities, etc. Also, the relation between items and abilities have been
studiedduringthepastcenturyexclusivelyusinghumans,soitisnotclearthatasetofitemswould
measure the same ability for a human or for a machine. For instance, is it reasonable to expect that
well-established tests of choice reaction time be correlated with intelligence in machines as they are
correlated in humans (Deary et al, 2001)? Or, what makes a set of psychometric tests different from
a set of “human intelligence tasks” in Amazon Mechanical Turk (Buhrmester et al, 2011)? For a
more complete discussion about why IQ tests are not ready for AI evaluation, the reader is referred
to a response (Dowe and Hern´andez-Orallo, 2012) to Detterman’s editorial.
HavingsaidallthisanddespitethelimitationsofIQtestsforAIevaluation,theiruseisbecoming
more popular in the past decade (including robotics, Schenck, 2013) and systems whose results are
like those of Table 4 are becoming common (for a survey, see Hern´andez-Orallo et al, 2016, for an
PEBL72).
| open library | of IQ tests, | see |     |     |
| ------------ | ------------ | --- | --- | --- |
As just said, one of the problems of IQ tests is that they are specialised for humans. In fact,
standardised adult IQ tests do not work with people with disabilities or children of different ages.
In a similar way, we do not expect animals to behave well on a standard human IQ test, starting
72
pebl.sourceforge.net/battery.html.

Evaluationinartificialintelligence 31
from the fact that they will not be able to read the text. This leads us to the consideration of
howcognitiveabilitiesareevaluatedinanimals.Comparativepsychologyandcomparativecognition
(Shettleworth, 2010; Shettleworth et al, 2013) are the main disciplines that perform this evaluation.
For a time, much research about cognitive abilities in animals was performed on apes. The term
‘chimpocentric’ was introduced as a criticism about tests that had gone from being anthropocentric
to being chimpocentric. Nonetheless, in the past decades, the perspective is much more general
and any species may be a subject of study for comparative psychology: mammals (apes, cetaceans,
dogs and mice), birds and some cephalopods. The evaluation focuses on “basic processes”, such as
perception,attention,memory,associativelearningandthediscriminationofconcepts,andrecently
on more sophisticated instrumental or social abilities (Shettleworth et al, 2013).
One of the most distinctive features of animal evaluation is the use of rewards, as instructions
cannot be used. This setting is very similar to the way reinforcement learning works. Animal evalu-
ationhasalsobroughtattentiontotherelevanceoftheinterface.Clearly,thesametestmayrequire
very different interfaces for a dolphin and a bonobo.
Human evaluation and animal evaluation have become more integrated in the past years, and
testing procedures half way between psychometrics and comparative cognition are becoming more
usual. For instance, several kinds of skills are evaluated in human children and apes in (Herrmann
etal,2007).Inrecentyears,manyabilitiesthatwereconsideredexclusivelyhumanhavebeenfound
to some extent in many animals (Wasserman and Zentall, 2006; Shettleworth, 2010).
Does the enlargement from humans to the whole animal kingdom suggest that these tests for
animals can be used for machines? While the lower ranges of the studied abilities and the use
of rewards can facilitate its application to AI systems significantly (at least for some autonomous
systemsincognitiverobotics,autonomousdevelopment,‘animats’,roboticpetsandotherAIsystems
thataredesignedtoresemblehumanoranimalbehaviours),westillhavemanyissuesaboutwhether
theycanbeappliedinAI(atleastdirectly).First,theselectionoftasksandabilitiesisnotsystematic.
Second,manyofthetasksthatareappliedtoanimalswouldbetooeasyformachines(e.g.,memory).
And third, others would be too difficult (e.g., orientation, recognition and interaction in the real
world). Nonetheless, there seems to be an increasing interest for the evaluation of the so-called
‘animats’(AIsystemsthatareinspiredbyorresembleananimal,WilliamsandBeer,2010)andthe
evaluation procedures for animals are the first candidates to try.
3.3 Evaluation using AIT
A radically different approach to AI evaluation started in the late 1990s. If intelligence was viewed
as a “kind of information processing” (Chandrasekaran, 1990) then it seemed reasonable to look at
information theory for an “essential nature or formal basis of intelligence and the proper theoretical
frameworkforit”(Chandrasekaran,1990).Thiswasfinallydonewithalgorithmicinformationtheory
(AIT), and the related notions of Solomonoff universal probability (Solomonoff, 1964), Kolmogorov
complexity (Li and Vit´anyi, 2008) and Wallace’s Minimum Message Length (MML) (Wallace and
Boulton, 1968; Wallace and Dowe, 1999), which we describe below.
There are several good properties about algorithmic information theory for evaluation. First,
severaldefinitionsofinformationandcomplexitycanbedefinedexclusivelyincomputationalterms,
actuallyrelativetoaUniversalTuringMachine(UTM),afundamentalanduniversalmodelofeffec-
tivecomputation.Forinstance,theKolmogorovcomplexityofanobject(expressedasabinarystring)
relative to a UTM is defined as the shortest program (for that machine) that describes/outputs the

32 Jos´eHern´andez-Orallo
object. Even if these definitions depend on the UTM that is used, the invariance theorem states
that their values will only differ with respect to other UTM up to a constant that only depends on
the two different UTMs (because one can emulate the other) (Li and Vit´anyi, 2008). The notion
of algorithmic probability, introduced by Solomonoff, allows a universal distribution to be defined
for each UTM, which is just the probability of objects as outputs of a UTM fed by a fair coin.
While, in general, this means that compressible strings are more likely than incompressible ones, it
can be shown that every computable probability distribution can be approximated by a universal
distribution. In a way, Solomonoff, the father of algorithmic probability (Solomonoff, 1964) gave a
theoretical backing to Occam’s razor. There are reasons to think that many phenomena and, as a
result, many of the problems that we face every day, follow a universal distribution. This is directly
linked to equation 3 again, and the discussion about the choice of the probability p. Also, we have
the relevant fact, which is very significant for evaluation as well, that universal distributions are
immune to the no-free-lunch theorems, where system performance can differ very significantly for
induction (Lattimore and Hutter, 2013; Hibbard, 2009). And finally, Kolmogorov complexity and
algorithmic probability are two sides of the same coin, which led to a formal connection of compres-
sion and inductive inference. One particular common (Bayesian) interpretation can be made under
the Minimum Message Length, where the best hypothesis is the one that minimises the length of
the theory and the length of coding the evidence using the theory. It has been acknowledged that
Solomonoff “solved the problem of induction” (Solomonoff, 1996; Dowe, 2013).
Of course, not everything in AIT is straightforward. For instance, some of these concepts lead
to incomputable functions, although approximations exist, such as Levin’s Kt (Levin, 1973). In
Levin’s Kt, it is not only the size of the program that is minimised but also the logarithm of the
computational steps taken by the program to produce the string. In this way, finding the program
that minimises the sum of these two terms becomes computable, and also has some important
connections with heuristics (Levin, 2013), and artificial intelligence (Solomonoff, 1984).
Chaitin suggested the application of AIT to the “definition of intelligence and measures of its
varioscomponents”(Chaitin,1982),buttheuseofAITfor(artificial)intelligenceevaluationstarted
with a variant of the Turing Test that featured compression problems (Dowe and Hajek, 1997,
1998), to make the test more sufficient. While one of the goals of this work was to criticise Searle’s
Chinese room73, this is one of the first intelligence test proposals using AIT. At roughly the same
time, a formal definition of intelligence in the form of a so-called C-test was derived from AIT
(Hern´andez-Orallo and Minaya-Collado, 1998; Hern´andez-Orallo, 2000a). Figure 5 shows examples
of sequences that appear in this test. They clearly resemble some exercises found in IQ tests, such
as Thurstone letter series (Thurstone, 1938a). The major differences are that (1) sequences are
obtained by a generator (a UTM with some post-conditions about the generated sequence, ensuring
the unquestionability of the series continuation and less dependency on the reference machine) and
(2) the fact that each sequence is accompanied by a theoretical assessment of difficulty (a variant of
Levin’s Kt complexity). Note the implications for evaluation of such a test, as exercises are derived
from first principles (instead of being contrived by psychometricians) and the difficulty of these
exercisesisintrinsic,andnotbasedonhowdifficulthumansfindthem.Finally,thesesequenceswere
used to define a test by aggregating results in a way that highly resembles our recurrent equation 3,
73 TheChineseroomargumentwasintroducedby(Searle,1980)toargueagainstthepossibilityofamachinehaving
amind,bycomparingacomputerprocessinginputsandoutputsassymbolswithapersonknowingnoChineseina
roomreceivingmessagesinChinesethathavetobeanswered,alsoinChinese,usingaseriesofbookstomapinputs
to outputs. Given the relevance of machine learning in AI nowadays, among other things, the argument has mostly
fadedtoday.

Evaluationinartificialintelligence 33
k=9 :a,d,g,j,... Answer:m
k=12:a,a,z,c,y,e,x,... Answer:g
k=14:c,a,b,d,b,c,c,e,c,d,...Answer:d
Fig. 5 Severalseriesofdifferentcomplexity9,12,and14usedintheC-test(Hern´andez-Orallo,2000a).
where M is formally defined as including all possible sequences (following some conditions) and the
probability is defined to cover a range of difficulties, leading to a difficulty-driven sampling as in
Figure 1 (right).
Somepreliminaryexperimentalresultsshowedthathumanperformancecorrelatedwiththeabso-
lutedifficulty(k)ofeachexerciseandalsowithIQtestresultsforthesamesubjects.Thisencourages
theuseofthisapproachforIQ-testre-engineering.Withtheaimofamorecompletetest,someexten-
sionsoftheC-testweresuggested,suchastransformingittoworkwithinteractiveagents(“cognitive
agents [...] with input/output devices for a complex environment” (Hern´andez-Orallo and Minaya-
Collado, 1998) where “rewards and penalties could be used instead” (Hern´andez-Orallo, 2000b)).
Despite its explanatory power about IQ tests, this line of research was held back by some literal
views of compression as intelligence, and even the proposal of tests of intelligence based on the
compression of text (Mahoney, 1999). The use of AIT for measuring intelligence was more sharply
dashedin2003(atleasttowardsgeneralintelligencetestsresemblingIQtestsusedformachines)by
the evidence that very simple —non-intelligent— programs could pass IQ tests (Sanghi and Dowe,
2003), as we have discussed in section 3.2 (see Table 4).
Nonetheless,theextensiontointeractiveagentswasperformedanyway.Interestingly,whenagents
andenvironmentsareconsideredintermsofequation3,wejustfindaperformanceaggregationover
a set of environments, exactly as had been formulated several times in the past: “intelligence is the
abilityofadecision-makingentitytoachievesuccessinavarietyofgoalswhenfacedwitharangeof
environments”(Fogel,1991).Notethatthisroughlycorrespondstothepsychometricviewofgeneral
intelligence as key to performance in a range (or all) cognitive tasks. A crucial aspect was then to
define this range of environments, i.e., the choice of the distribution in equation 3. One option was
to include all environments. In order to do this in a meaningful, elegant way (and get rid of any
no-free lunch theorem), AIT and reinforcement learning were combined (Legg and Hutter, 2007b).
Equation 3 was instantiated with all environments as tasks with a universal distribution for p, i.e.,
p(µ)=2−K(µ), with K(µ) being the Kolmogorov complexity of each environment µ.
These proposals present several problems. First, some constructions are not computable, so ap-
proximations need to be used. Second, most environments are not really discriminative, and all
agents will score the same, will just ‘die’ or be stuck after a few steps. Third, overweighting very
smallenvironments(bytheuseofauniversaldistributionoracomplexitylimit)makesthedefinition
very dependent on the reference machine chosen as environment generator. Finally, time (or speed)
is not considered for the environment or for the agent. For more details about these (and other)
issues and some possible solutions, the reader is referred to (Hibbard, 2009) and (Hern´andez-Orallo
and Dowe, 2010, secs. 3.3 and 4). Taking into account these solutions, some actual tests have been
developed (Insa-Cabrera et al, 2011b,a; Legg and Veness, 2013). While the results may still be use-
ful to rank some state-of-the-art machines, if they are not compared to humans (or animals), as we
discuss in the following section, the validation (or more precisely the refutation) of these tests as
true intelligence tests cannot be done.
Summingup,theAITapproachischaracterisedbythedefinitionoftestsfromformalinformation-
based principles. This is in stark contrast to other approaches where tasks are collected, refined by

34 Jos´eHern´andez-Orallo
trial-and-error or invented in a more arbitrary way. Most of the approaches to AI evaluation using
AIT seen above have aimed at defining and measuring general intelligence, which is placed at the
very top of the hierarchy of abilities (and hence at the opposite extreme from a specialised task-
oriented evaluation). However, many interesting things can happen if AIT is applied at other layers
of the hierarchy, for general cognitive abilities other than intelligence, as suggested in (Hern´andez-
Orallo, 2000c) for the passive case and hinted in (Hern´andez-Orallo and Dowe, 2010, secs. 6.5 and
7.2) for the dynamic cases, with the use of different kinds of videogames as environments (two of
the most recently introduced benchmarks and competitions are in this direction, Bellemare et al,
2013; Schaul, 2014; Perez et al, 2015). Finally, the information-theoretic approach is not isolated
from some of the approaches seen so far in section 2. Actually, some hybridisations and integrated
approacheshavebeenproposed(Hern´andez-Oralloetal,2011;Insa-Cabreraetal,2012),apartfrom
the compression-enriched Turing Tests (Dowe and Hajek, 1997, 1998), already mentioned above.
TherehavebeenotherapproachesthatarerelatedtoAITwiththeaimofdefiningameasure(or
theory)ofintelligenceinmathematicalterms,suchasSmith’s“uniformlyasymptoticallycompetitive
intelligence”,relyingonanenumerationalgorithmlookingforpolicies(Smith,2006),orYampolskiy’s
‘efficiencytheory’(Yampolskiy,2015,chap.9).Also,AITconnectedverywellwithearlyperspectives
about the principle of simplicity under the so-called structural complexity (Krueger and Osherson,
1980).PsychologyandcognitionbegantoincludemoreexplicitreferencestoKolmogorovcomplexity
and related notions in AIT (Chater, 1999; Chater and Vit´anyi, 2003; Feldman, 2003; Leeuwenberg
and Van Der Helm, 2012) and derived into the analyses or derivation of more tests using ideas from
Kolmogorov complexity (Stranneg˚ard et al, 2013b,a; Schmid and Ragni, 2015; Nizamani, 2015).
The tasks and tests generated with AIT have been restricted to interactive (learning) systems
in a sequential context, tasks resembling those found in IQ tests or in a reinforcement learning
setting,asseenabove.However,forAIpractitionersinotherareassuchasmoretraditionalmachine
learning, datasets could be generated using principles from AIT, quantifying the complexity of the
best hypothesis for a dataset (including noise or not). Although not based on AIT, this has been
applied with a statistical approach rather than an informational approach based on the distortion
ofactualdatasets,asmentionedintheprevioussection(Soares,2009;Maci`aandBernad´o-Mansilla,
2014). Of more interest can be the creation of datasets whose dependence on previous datasets
is based on an algorithmic modification rather than statistical distortions. We will discuss this in
section 3.5 in the context of transfer learning (sequences of tasks) and inductive programming.
3.4 Universal psychometrics
The previous sections show a fragmentation of techniques and problems. This fragmentation origi-
natedbythekindofmeasurementweareinterestedin(task-orientedorability-oriented,collectedor
AIT-derivedtests)butmostespeciallybythekindofsubjectthatisbeingmeasured.In(Hern´andez-
OralloandDowe,2010),thenotionof‘universaltest’isintroduced,asatestthatisapplicableto“any
biological or artificial system that exists at this time or in the future”: human, non-human animal,
enhanced human, machine, hybrid or collective. The stakes were set high, as the tests should work
without knowledge about the subject, derive from computational principles, be unbiased (species,
culture,language, ...),require no human intervention, be practical,producea meaningfulscore,and
be anytime (the more time we have for the test the higher the reliability of the score). Note that in
order to apply the same test to several subjects we are allowed to customise the interface, provided
the features and difficulty of the items are permitted to remain unaltered. Also, we need to think

Evaluationinartificialintelligence 35
about the speed of the subject, and adapt to it accordingly. Also, the capabilities of the subject can
be quite varied, so the ranges of difficulty need to adapt to the agent. That suggests that universal
tests must necessarily be adaptive.
A first framework for universal, anytime intelligence tests is introduced in (Hern´andez-Orallo
and Dowe, 2010), where a class of environment is carefully chosen to be discriminative. The test
starts with very simple environments and adapts to the subject’s performance and speed. In this
regard, this resembles a difficulty-driven sampling as described in section 2.3. The set of tasks
(environments) was developed upon some of the ideas about using AIT for intelligence evaluation,
as seen in section 3.3. Some experiments were performed (Insa-Cabrera et al, 2011b,a) using the
environment class defined in (Hern´andez-Orallo, 2010). Difficulty was estimated using a variant of
Levin’s Kt. As a way of checking whether the results were meaningful, the same test compared Q-
learning(WatkinsandDayan,1992)withhumans.Twodifferentinterfacesweredesignedonpurpose.
ThetestgaveconsistentresultsforQ-learningandhumanswhenconsideredseparately,butwereless
reasonablewhenputtogether.Theexperimentalsettingsfeaturedmanylimitations(simplifications,
non-adaptiveness,absenceof noise,low-complexity patterns,no incrementality, nosocialbehaviour,
etc.)and,probablybecauseofthis,theresultsdidnotshowtheactualdifferencebetweenQ-learning
and humans. Despite the limited results, the experiment had quite a repercussion (Kleiner, 2011;
Biever, 2011; Yonck, 2012). Nonetheless, the tests were a first effort towards a universal test and
highlighted some of the challenges.
One possible explanation for all these limitations is that universal intelligence tests may simply
be impossible (Smith, 2006; Edmondson, 2012) or the very notion of a general intelligent system
unfeasible (Melkikh, 2014), supported by the no-free-lunch theorems (Wolpert and Macready, 1995;
Wolpert,1996,2012).Anotherlessextremeexplanationisbasedontheconcernaboutageneratorof
environments lacking richness of interaction and social behaviours. In other words, an environment
thatisrandomlygeneratedwillhaveanextremelylowprobabilityofshowingsomesocialbehaviour,
which is a distinctive trait of human intelligence. This has suggested other ways of generating the
environments and ways of incorporating other agents into them (e.g., the Darwin-Wallace distri-
bution, Hern´andez-Orallo et al, 2011), but it is still an open research question how to adapt these
ideas to the measurement of social intelligence and multi-agent systems (Insa-Cabrera et al, 2012;
Insa-Cabrera, 2016).
Thefragmentationofapproachesandtheneedofsolvingmanyoftheaboveissueshassuggested
the introduction of a new perspective, dubbed ‘universal psychometrics’ (Hern´andez-Orallo et al,
2014). Universal psychometrics focuses on the measurement of cognitive abilities for the ‘machine
kingdom’,whichcomprisesany(cognitive)system,individualorcollective,eitherartificial,biological
or hybrid. This comprehensive view is born with many hurdles ahead. Evaluation is always harder
the less we know about the subject. The less we take for granted about the subjects the more
difficultitistoconstructatestforthem.Forinstance,humanintelligenceevaluation(psychometrics)
works because it is highly specialised for humans. Similarly, animal testing works (relatively well)
because tests are designed in a very specific way to each species. And some of the AI evaluation
schemes we have already seen work because they are specialised for some kind of AI systems that
aredesignedforsomespecificapplications.InthecaseofAI,whowouldtrytotackleamoregeneral
problem (evaluating any system) instead of the actual problem (evaluating machines)? The answer
to this question is that the actual problem for AI is the universal problem. Notions such as ‘animat’
(Williams and Beer, 2010), machine-enhanced humans (Cohen, 2013), human-enhanced machines
(von Ahn, 2009), other kind of hybrids and, most especially, collectives (Quinn and Bederson, 2011)
ofanyoftheformer,suggestthatthedistinctionbetweenanimals,humansandmachinesisnotonly

36 Jos´eHern´andez-Orallo
Machine Kingdom
Animal Kingdom
Homo sapiens
Universal Psychometrics
Fig. 6 Therealmofevaluablesubjectsforuniversalpsychometrics.
inappropriate, but no longer useful to advance in the evaluation of cognitive abilities. The notion of
‘machinekingdom’,asillustratedinFigure6,isnotverysurprisingtothecurrentscientificparadigm
but clarifies which class of subjects is most comprehensive.
Universal psychometrics attempts to integrate and standardise a series of concepts. A subject
is seen as a physically computable (resource-bounded) interactive system. Cognitive tasks are seen
as physically computable interactive systems with a score function. Interfaces are defined between
subjects and tasks (observations-outputs, actions-inputs). Cognitive abilities are seen as properties
over a set of cognitive tasks (or task classes). As a result, the separation between task-specific and
ability-specific becomes a progressive thing, depending on the generality of the task class. Distribu-
tions are defined over task classes and results as aggregated performance on a task class (again, a
generalised version of equation 3). Difficulty functions are computationally defined from each task.
Overall, some of these elements found in psychometrics, comparative cognition and AI evaluation
are overhauled here with the theory of computation and AIT. As a result, cognitive abilities are
no longer what the cognitive tests measure, as in human psychometrics, so adapting the (in)famous
statementthat“measurableintelligenceissimplywhatthetestsofintelligencetest”(Boring,1923),
but they are properties that emanate from (general) classes of tasks, perfectly defined in computa-
tional terms. As a consequence, the relation between abilities can be explored experimentally, but
also theoretically, and measures are absolute and not relativised wrt. a population (except for social
abilities). This could imply some revitalisation of the white-box approach, especially for those AI
systems that can be formally described in a theoretical way (e.g., some results in Hutter, 2007 and
Hibbard, 2009 take a white-box evaluation approach).
This view of a cognitive ability is consistent with its association with a “class of cognitive tasks”
(Carroll, 1993) that must be ‘representative’ for the ability. From the association between abilities
andclassesoftasks,weseethatbymergingtwocognitivetaskclasseswegetamoregeneralcognitive
taskclass,andamoregeneralability.Typically,thisisstudiedinahierarchicalway,startingwiththe
so-calledelementarycognitivetasks(Carroll,1993,page11)(closelyrelatedtothenotionofprimary
mental abilities of Thurstone, 1938b). This redraws our dilemma between task-oriented and ability-
oriented into a gradual hierarchy from specific tasks to general abilities, with general intelligence at
the very top (including all possible cognitive tasks, i.e., all interactive Turing machines with a score
function). The questions about how to sample from a task class for an effective evaluation can be
generalised from our discussion in section 2.
Thissetsadualviewofcognitivetasksontheonehandandcognitivesystemsontheotherhand,
wherebothspaces (theability space andthe machinekingdom)can be explored.Interestingly,both
cognitive tasks and cognitive systems are defined as interactive systems, reflecting a duality world-
agent.Onesingularityofcognitivesystems(aswellastheenvironmentstheyarein)isthattheycan

Evaluationinartificialintelligence 37
evolve with time, and their abilities can change. In other words, it seems that some abilities need to
be constructed on top of other previously consolidated abilities, and this seems to be independent
of the subject to some extent, in the same way that it seems difficult to be able to multiply without
being able to add. A theoretical analysis of ability interdependency, how they can develop and the
notionofpotentialintelligence,arestillinaveryincipientstage(Hern´andez-OralloandDowe,2013).
As we have discussed, human-based psychometrics is inadequate for most of the systems and
techniques and AI. In particular, much of our notion of ‘ability’ is very anthropocentric, and does
not reflect well the diversity of artificial systems. This is why universal psychometrics is not meant
to be an extension of human psychometrics. Again, this does not mean that all techniques from
psychometricsshouldbeexcluded,astherearesomeinterestingprinciplesandtoolsthere(e.g.,item
response theory, adaptive tests, etc.), as already discussed in this paper. The key idea of universal
psychometricsistocategoriseallthepossiblekindsofsystems,artificialornaturalandseehowthey
can be evaluated, without any anthropocentric assumption.
Ofcourse,therecanbeobjectionsanddisagreementsaboutthewaymanyofthekeyconceptsin
universalpsychometricsshouldbeunderstoodanddefined.Forinstance,thedefinitionofsubjectsas
interactivecognitivesystemsmayapparentlyexcludesomeAIsystems,althoughthesecouldstillbe
embeddedifaninput-outputorrewardsetting.Therecanalsobeobjectionsaboutwhatauniversal
test should look like (Dowe and Hern´andez-Orallo, 2014). But a more integrated view of cognitive
abilities for humans, animals, robots, agents, ‘animats’, hybrids, swarms, etc., is not only possible
but useful.
3.5 Highlights and directions of the evaluation of general-purpose AI systems
The previous subsections take ideas from areas outside artificial intelligence (e.g., psychology and
information theory) and suggest their use for a more ability-oriented evaluation of AI systems. This
may give the impression that there have not been efforts to evaluate more abstract abilities (and
general-purpose competence) in AI. In this section we briefly discuss some of the most promising
areas where this kind of evaluation has been attempted, its limitations and how it can be merged
with the ideas seen in the previous subsections.
Machine learning was discussed in section 2.3, as a kind of task-oriented evaluation. We already
saw that some areas in machine learning, such as reinforcement learning, and some recent proposals
andcompetitionssuchastheArcadeLearningEnvironment(Bellemareetal,2013)andtheGeneral
Video Game Competition (Perez et al, 2015) go in the direction of more general systems. The basic
ideaisthatbycombininganevaluationbasedonautilityfunctionthatdependsonthetaskathand
(as usual in reinforcement learning) and the use of many tasks, by aggregation, some kind of more
general ability is expected to be measured. Related to reinforcement learning, new platforms and
projects are sprouting every year, with an extra focus on general-purpose evaluation. For instance,
Project Malmo74 is a platform developed by Microsoft over the well-known video game sandbox
Minecraft,wheredifferenttaskscanbecreatedandAIagentscanbeprogrammedforthem(Johnson
etal,2016;Abeletal,2016).TheOpenAIGym75 isacollectionoftasks,whichintegratestheAtari
games in the Arcade Learning Environment but also features some continuous control tasks, and
provides an interface with some deep reinforcement learning libraries (Duan et al, 2016).
74 http://blogs.microsoft.com/next/2016/03/13/project-malmo-using-minecraft-build-intelligent-technology.
75 https://gym.openai.com/

38 Jos´eHern´andez-Orallo
Nevertheless,bychangingafewminorthingsinonegameortask,weobservethatcurrentsystems
have to learn everything from scratch. Basically, the concepts that are learnt are not sufficiently
abstract, or general, to be brought from one situation or another (apart from the fact that the
systems are usually rebooted between tasks). Transfer learning (Pan and Yang, 2010), seen as “the
improvement of learning in a new task through the transfer of knowledge from a related task that
has already been learned” (Torrey and Shavlik, 2009), is presented as a solution for this. Other
related areas are domain adaptation (Jiang, 2008), multi-task learning (Caruana, 1997; Thrun,
1996), lifelong learning (Thrun and Pratt, 2012), incremental learning (Khreich et al, 2012) and
activelearning(Settles,2012).Theseareashavebeenveryactiveinthepasttwodecades,andthere
is already a number of benchmarks, competitions and challenges. However, the challenges (such as
the‘unsupervisedandtransferlearningchallenge’Mesniletal,2012)usuallyevaluateteamsinstead
of systems, or the systems are restricted to some kind of specific technology.
Theevaluationmetricsinmostofthesechallengesaredefinedintermsofperformance,asthereis
no easy way to check how able the system is to build new concepts that can be generalised between
tasks, or how previous information is used. In order to do this, other areas of machine learning,
such as inductive programming (Gulwani et al, 2015) may be more advantageous, because several
systems are general-purpose but also based on declarative languages, so that the invented abstract
constructs can be inspected. Hence, it is easier to analyse what kind of concepts are required from
sometasksforothertasks.Someevaluationbenchmarksalongtheselineshavebeenproposedaround
a repository76, but they are still in an incipient stage. One interesting point of this approach is that
the systems are not anthropocentric; they are just conceptual learners. Combining the previous
notionsoftransferlearning,multi-tasklearningandincrementallearningwithconstructive(Turing-
complete) languages, there is the question of how to generate a set of tasks that depend on the
creation of previous concepts, such as, for instance, learning multiplication after learning addition.
Thiscanbewellmodelledininductiveprogramminginsuchawaythatthedifficultyofthetransfer
or abstraction needed can be measured with a principled approach such as algorithmic information
theory (instead of a distortion-based approach or a subjective evaluation of the conceptual jump
according to the difficulty found by humans). Similarly, the way in which concepts are learnt from
natural language in a continual way, as in NELL (Carlson et al, 2010), is closely related to the
analysisofwhetherthesystemisabletocapturemorecomplexandabstractconceptsincrementally.
Developmental robotics (Cangelosi et al, 2015) is an area that is particularly focused on the
progress of an AI system from sensorimotor interaction towards more abstract concepts, in an in-
cremental way. At the current stage of the area, there are not many objective tests, benchmarks
and competitions to evaluate cognitive development in general, other than some qualitative assess-
ments (e.g., Tables 6.4, 7.5 and 8.2 in Cangelosi et al, 2015), which are usually very antropocentric.
Quantitative tests focus mostly on the sensorimotor part77. The RoCKIn competitions (Amigoni
et al, 2015), and the associated challenges (RoCKIn@home and RoCKIn@work, even if they define
specificbenchmarkssuchastheassistanceofagingpeopleinahouseenvironmentortheassembling
of mechanical parts in a factory, distinguish between tasks and functionalities (such as object per-
ception, navigation, speech understanding and object manipulation). However, none of these tasks
or evaluation frameworks fully delve into the concept development part, in terms of a proper cumu-
76 http://www.inductive-programming.org/repository.html.
77 http://www.robot.uji.es/EURON/en/index.htm, http://rockinrobotchallenge.eu/Benchmarking_Robotics.
pdf.

Evaluationinartificialintelligence 39
lative or lifelong learning. Going beyond “one cognitive faculty” (Cangelosi et al, 2015, p. 272) is
one of the future challenges.
Finally, cognitive architectures is one of the areas in artificial intelligence (or more precisely, in
cognitivescience)thathasbeentraditionallyassociatedwiththecreationofgeneral-purposesystems,
with a clear orientation in modelling human performance (so they are clearly anthropocentric),
althoughforsomeofthemthegoalhasbeentoreallydisplaysomeintelligencebehaviouringeneral,
which is nowadays more generally referred to as (interactive) “cognitive sytems” (Langley, 2012).
Let us have a look at the way evaluation is performed in this area.
First, there are several ways of comparing —at the conceptual level— the existing cognitive
architecutres (e.g., Sloman and Scheutz, 2002). However, we are interested in integrated evaluation
benchmarks, especially those that can identify abilities. For instance, the DARPA’s Brain-Inspired
—later Biologically-Inspired— Cognitive Architectures program (BICA) maintains a comparison
table of architectures78. There, we can find several “common general paradigms” that can be used
to compare them, such as problem solving, decision making, analogy, language processing, working
memory, perceptual illusions, implicit memory, metacognition, social tasks, personality and motiva-
tionaldynamics.AsimilarlistalsoclassifyingtheexistingarchitecturescanbefoundattheCogArch
webpage79. Similarly, the ‘Newell test’ (Anderson and Lebiere, 2003), which is not an actual test
but a set of criteria for architectures or theories of cognition, distilled into 12 criteria for cogni-
tion: “flexible behavior, real-time performance, adaptive behavior, vast knowledge base, dynamic
behavior, knowledge integration, natural language, learning, development, evolution, and brain re-
alization”, integrating two lists from (Newell, 1980, 1990), hence the name. Adams et al (2012)
identify several ‘competency areas’ for an AGI system: perception, memory, attention, social in-
teraction, planning, motivation, actuation, reasoning, communication, learning, emotion, modelling
self/other, building/creation and use of quantities, many of which match with subdisciplines in AI.
Focusingontestsinsteadoftaxonomies,themostremarkableproposalistheso-called‘cognitive
decathlon’, briefly suggested by Vere (1992), and also linked to the Newell test by Anderson and
Lebiere (2003). The actual cognitive decathlon was completed by Mueller et al (2007), with two
other tests (Challenge Scenarios and Biovalidity Assessment) that jointly “cover a core set of cogni-
tive, perceptual, and motor skills typical for a two-year-old human child”. The Cognitive Decathlon
features25‘levels’insixcategories:vision,search,manualcontrolandlearning,knowledgelearning,
language and concept learning, and simple motor control. There is a partial implementation of the
cognitive decathlon (Mueller, 2010), but no systematic evaluation has been performed using it.
The cognitive decathlon is a very interesting case because there is a clear mapping between
the abilities and the proposed tests. This is not the case for other approaches. For instance, the
competencies mentioned above by (Adams et al, 2012) are replaced by different test scenarios that
are assumed to cover all the competencies: general video-game learning, preschool learning, reading
comprehension, story or scene comprehension, school learning and the ‘Wozniak Test’ (walk into an
unfamiliar house and make a cup of coffee). More recently, the so-called “I-athlon” (Adams et al,
2016) also groups a set of tasks, but taking more inspiration from the state of the art in AI than
developmental psychology. However, an ability-oriented evaluation is lost after these integrations.
Basically, this is the trivial way of preventing specialisation: including many tasks in a battery
and increasing the breadth of the tasks (Goertzel et al, 2009; Wang, 2010; Rohrer, 2010). But test
generalisation is not the same as an ability-oriented evaluation.
78 http://bicasociety.org/cogarch/architectures.htm.
79 http://cogarch.org/index.php/Capabilities.

40 Jos´eHern´andez-Orallo
Overall,wehaveseenthatability-orientedandgeneral-purposeevaluationapproacheshaveboth
been pursued in artificial intelligence, but the actual proposals are still very incipient, and more re-
searchanddiscussionisneeded80.Theywillrequiresomeoftheideasseenintheprevioussubsections
for a more foundational approach in the way abilities are identified and their difficulty quantified,
especially as many of these systems cannot be evaluated with anthropocentric tests. Also, we have
seen that general-purpose evaluation is not only about autonomous robots, agents, cognitive archi-
tectures or other kinds of cognitive systems, but it can also be applied to non-interactive systems,
such as inductive programming and other constructive learning approaches, which of course lack
other abilities that are required in an interactive world.
4 Lessons learnt and guidelines for AI practitioners
Thereisa(hierarchical)continuumbetweentask-orientedevaluationandability-orientedevaluation.
As a result, it would be misleading to consider that the progress in AI should be analysed by the
progressofspecificsystems(asitisusuallydoneandmentionedintheprevioussection)butitwould
also be a mistake to evaluate AI exclusively by the progress of general-purpose systems. Populating
the space of benchmarks, tests and competitions is a kind of work that, to an extent, started many
decades ago. However, the analysis of one particular test is of limited insight for AI researchers if
there is a complete lack of understanding of the relationships of tasks and abilities in this space.
For instance, in this continuum of benchmarks, we need to realise that if “one system demonstrates
impressive natural language processing and another demonstrates impressive perception [this] does
notimplythatthesecapabilitiescanbeintegratedinordertoautomatetheperceptualandlinguistic
aspects”(Brundage,2016).Thewaysometasksandabilitiesarerelatedinhumansisnotexpectedto
extrapolate to AI. Also, the way many subareas in AI have developed may reflect the pace in which
the applications and technologies appeared historically in AI, and may not reflect a fundamental
way in which AI system should be evaluated. In other words, both natural intelligence research
and artificial intelligence research have been highly anthropocentric in the creation of benchmarks
and the analysis of their relationship. This means that whenever a group of AI researchers think
of creating a new competition, extreme care should be put on not reusing previous benchmarks or
tests (from psychology or AI) if it is not clear what the tasks are really measuring and how they
relate to other tasks.
Another important lesson learnt when one analyses the myriad of benchmarks and competitions
inAIisthatitisnotalwaysclearthesubjectweareevaluating,anAIsystem,someofitscomponents
or the way a team of AI practitioners have put everthing together to score well for the competition.
Aswehavementionedinthispaper,manycompetitions,especiallyinmachinelearning(andrelated
areas such as data mining and data science), evaluate teams who are able to integrate many AI
tools, by thoroughly studying the problem. It is the human intelligence of the team what these
competitions usually award, rather than the real value of the tools used. This relates to our initial
discussion that evaluating components instead of systems is more prone to misinterpretations since
components must be integrated by an AI practitioner.
Even if the benchmark, test or competition aims at evaluating AI systems (and not AI teams),
the most important issue is “evaluation overfitting”, which is usually counteracted by changes in
the evaluation year after year, or the addition of more tasks, instead of properly analysing why this
80 http://users.dsic.upv.es/˜flip/EGPAI2016/.

Evaluationinartificialintelligence 41
overfitting happens. Even with these tactics, AI practitioners can still use a “big switch” approach
and get good results on average.
Of more logistic character, a general comment shared by their organisers is that competitions
are hard to organise and maintain. Technology changes quickly, and servers, platforms must be
updated almost every year. It is also difficult to encourage participation, given the large number of
competitions. As a result, the integration of many benchmarks and competitions and the support
from prestigious organisations and governments is key so that the best AI researchers are involved
in creating the future evaluation initiatives and the best AI researchers also participate in them.
Given the caveats mentioned above about AI evaluation, we now enumerate a series of generic
guidelines for AI practitioners willing to create or improve a benchmark or competition:
– The definition of Ω, the set of possible systems that can be evaluated (or that can be opponents
in peer confrontation evaluation), must be clarified from the beginning, as well as whether the
AI systems are fully autonomous or require the integration and finetuning of AI researchers. If
humans are considered, the way in which they are admitted and how they are instructed must
be defined. The more general Ω is the less we can assume about the evaluation process. If Ω is
heterogeneous (e.g., a universal test), different interfaces must be considered.
– The definition of M, the set of possible tasks, and its associated distribution p configure what
we are measuring. This can be built from a set of problems or using a generator. This pair
hM,pi has to be representative of a task (in task-oriented evaluation) or an ability (in ability-
oriented evaluation). If it is a peer confrontation evaluation, M will be enlarged with as many
combinations between game (environment) and agents in Ω are possible. The distribution p will
be updated accordingly.
– The definition of R and its aggregation Φ must ensure that the values R(µ) for all µ ∈ M are
going to be commensurate and that the aggregation is bounded. An analysis about expected
measurement error is useful at this point. The robustness of R depending on the length or time
leftforeachepisodewillindicatewhetherrepetitionsareneededtoreducethemeasurementerror
given by R(µ).
– As much as possible, the similarity between tasks or a set of features describing them should be
identified.Anintrinsicdifficultyfunction(evenifapproximate)isalwaysveryuseful.Showingthe
distribution of difficulty for M can be highly informative. If difficulty is available, item response
curves could be prepared.
– The sampling method must be as efficient as possible, by using, e.g., an information-driven
samplingorarangeofdifficultiesifwehaveanon-adaptiveevaluation.Forthepeer-confrontation
evaluation, the arrangement of matches can be designed beforehand if the evaluation is not
adaptive. Similarly, the procedure for an adaptive evaluation must also be carefully designed to
ensure measurement robustness. Simulations can be useful to estimate this.
– Information about how the evaluation is performed (including R, Φ and some illustrative prob-
lems) can be disclosed to the systems that are being evaluated (or to their designers). However,
Ω, M and p should not be disclosed. If possible, the problems should not be disclosed after the
evaluationeither,askeepingthemsecretmakesitpossibletocomparewiththesameproblemsfor
different subjects or at different times (e.g., we can evaluate progress of a system or a discipline
during a period).
– After the evaluation, results must be analysed beyond the mere calculation of the aggregated
results.Itemresponsefunctionsandagentresponsefunctions(Hern´andez-Oralloetal,2014)can
be constructed empirically from the results and compared with the theoretical functions or any

42 Jos´eHern´andez-Orallo
other information about Ω and M. Discrepancies or anomalies may suggest that the evaluation
scheme has to be revised. Results of the evaluation must become public at the highest possible
detail, so they can be analysed and compared by other researchers and participants (following,
e.g.,thenotionof‘experimentdatabase’(Vanschorenetal,2012),suchasinthemachinelearning
community81).
Theabovelistisverydifferentfromarecentlistof“principlesfordesigninganAIcompetition”such
that they induce real progress in AI (Shieber, 2016). Shieber is more elementary and general in the
desiderata, focusing on the rules of the competition (‘occasionality’, ‘flexibility’ and ‘absoluteness’)
and the nature of the task (‘reasonableness’). It is of course an open question to what extent any or
both of the above recommendation lists will be followed on a regular basis for AI evaluation. It can
be argued whether AI evaluation has been a priority for AI in the past, but it seems that it has not
been recognised as an imperative problem or a mainstream area of research. Anyhow, the question
of AI evaluation remains and there is space for significant improvement, even for the most specific
sets Ω and M (bottom-left part of Figure 7). At the other end, measuring intelligence and doing
it universally is a key ingredient for understanding what intelligence is (and, of course, to devise
intelligent artefacts). Many interesting questions and applications lay in the middle of Figure 7, as
AI evaluation is no longer limited to task-specific evaluation of AI systems or to evaluating progress
in AI. Instead, AI is becoming able to evaluate systems that learn to solve instead of systems that
are programmed to solve.
5 Conclusions
We started this paper looking at the way AI evaluation is commonly performed, through task-
oriented evaluation, mostly with a black-box approach. We identified several problems and limita-
tions, and we noticed that there is still a huge margin for improvement available in the way AI
systems are evaluated. The key issues are the set of tasks M and their distribution p, as well as
distinguishing the definition of the problem class (aggregation) from an effective sampling proce-
dure (testing procedure). Then we switched to ability-oriented evaluation, a much more immature
approach, but that may have a more relevant role in the future. The notion and evaluation of abili-
ties is more elusive than the notion and evaluation of tasks. We have argued that this requires the
integration of several perspectives that are currently scattered efforts in AI, psychometrics, AIT
and comparative cognition. The different areas, philosophies, tools, foundations, terminologies and
the different kinds of subjects to be evaluated can be unified with an integrated perspective known
as universal psychometrics. Here, the exploration of the machine kingdom mirrors the exploration
of the set of possible cognitive abilities/tasks, from the duality environment-agent (task-subject).
In both spaces we aim at becoming more general, which is where evaluation is more challenging
(see Figure 7). This resembles the duality in the theory of computation (e.g., problem classes and
automata classes).
Inanycase,andwithanyoftheapproachesseensofar,amorescientifictheoryofAIevaluation
is being required for many applications (CAPTCHAs, social networks, agent certification, etc.) and
it will be more and more common in a future with a plethora of bots, robots, artificial agents,
avatars, control systems, ‘animats’, hybrids, collectives, etc. It is also crucial for alternatives to the
concept of ‘human-level intelligence’ and for the assessement of the plausibility of futuristic ideas
81 http://openml.org/.

Evaluationinartificialintelligence 43
ability-
oriented
Challenging
Easy
task-
oriented
subject-
universal
specific
Fig. 7 Tests become more or less challenging depending on the generality of the class of subjects considered (from
subject-specifictouniversal)andtheclassofabilities(fromtask-orientedevaluationtoability-orientedevaluation).
such as ‘the technological singularity’ (Eden et al, 2013), especially because some of the prophecies
and forecasts disregard that the first thing to consider about the future of intelligence is to have
metrics to detect whether AI progresses and in which direction.
Summing up, AI requires an accurate, effective, non-anthropocentric, meaningful and computa-
tional way of evaluating its progress, by evaluating its artefacts. This paper has just portrayed the
state of the art of AI evaluation, its challenges and the avenues for future work.
Acknowledgements IthanktheorganisersoftheAEPIASummerSchoolOnArtificialIntelligence,heldinSeptem-
ber2014,forgivingmetheopportunitytogivealectureon‘AIEvaluation’.Thispaperwasbornoutofandevolved
through that lecture. The information about many benchmarks and competitions discussed in this paper have been
contrasted with information from and discussions with many people: M. Bedia, A. Cangelosi, C. Dimitrakakis, I.
Garcia-Varea, K. Hofmann, W. Langdon, E. Messina, S. Mueller, M. Siebers and C. Soares. Figure 4 is courtesy of
F. Mart´ınez-Plumed. Finally, I thank the anonymous reviewers, whose comments have helped to significantly im-
prove the balance and coverage of the paper. This work has been partially supported by the EU (FEDER) and
theSpanishMINECOundergrantsTIN2013-45732-C4-1-P,TIN2015-69175-C4-1-RandbyGeneralitatValenciana
PROMETEOII2015/013.
References
AbelD,AgarwalA,DiazF,KrishnamurthyA,SchapireRE(2016)Exploratorygradientboostingforreinforcement
learningincomplexdomains.arXivpreprintarXiv:160304119 37
AdamsS,ArelI,BachJ,CoopR,FurlanR,GoertzelB,HallJS,SamsonovichA,ScheutzM,SchlesingerM,Shapiro
SC, Sowa J (2012) Mapping the landscape of human-level artificial general intelligence. AI Magazine 33(1):25–42
39
AdamsSS,BanavarG,CampbellM(2016)I-athlon:Towardsamulti-dimensionalTuringtest.AIMagazine37(1):78–
84 39
vonAhnL(2009)Humancomputation.In:DesignAutomationConference,2009.DAC’09.46thACM/IEEE,IEEE,
pp418–419 35
vonAhnL,BlumM,LangfordJ(2004)Tellinghumansandcomputersapartautomatically.Communicationsofthe
ACM47(2):56–60 9,10
vonAhnL,MaurerB,McMillenC,AbrahamD,BlumM(2008)RECAPTCHA:Human-basedcharacterrecognition
viawebsecuritymeasures.Science321(5895):1465 9,10
Alcal´a J, Fern´andez A, Luengo J, Derrac J, Garc´ıa S, S´anchez L, Herrera F (2010) Keel data-mining software tool:
Data set repository, integration of algorithms and experimental analysis framework. Journal of Multiple-Valued
LogicandSoftComputing17:255–287 16
AlexanderJRM,SmalesS(1997)Intelligence,learningandlong-termmemory.PersonalityandIndividualDifferences
23(5):815–825 27

44 Jos´eHern´andez-Orallo
AlpcanT,EverittT,HutterM(2014)Canwemeasurethedifficultyofanoptimizationproblem?IEEEInformation
TheoryWorkshop(ITW) 7
AlurR,BodikR,JuniwalG,MartinMMK,RaghothamanM,SeshiaSA,SinghR,Solar-LezamaA,TorlakE,Udupa
A(2013)Syntax-guidedsynthesis.In:FormalMethodsinComputer-AidedDesign(FMCAD),2013,IEEE,pp1–17
16
Alvarado N, Adams SS, Burbeck S, Latta C (2002) Beyond the Turing Test: Performance metrics for evaluating a
computersimulationofthehumanmind.In:DevelopmentandLearning,2002.Proceedings.The2ndInternational
Conferenceon,IEEE,pp147–152 9
AmigoniF,BastianelliE,BerghoferJ,BonariniA,FontanaG,HochgeschwenderN,IocchiL,KraetzschmarG,Lima
P,MatteucciM,MiraldoP,NardiD,SchiaffonatiV(2015)Competitionsforbenchmarking:Taskandfunctionality
scoringcompleteperformanceassessment.IEEERobotics&AutomationMagazine22(3):53–61 38
AndersonJ,LebiereC(2003)TheNewelltestforatheoryofcognition.BehavioralandBrainSciences26(5):587–601
39
Anderson J, Baltes J, Cheng CT (2011) Robotics competitions as benchmarks for AI research. The Knowledge
EngineeringReview26(01):11–17 2,16
Arel I, Rose DC, Karnowski TP (2010) Deep machine learning - a new frontier in artificial intelligence research.
ComputationalIntelligenceMagazine,IEEE5(4):13–18 2
AsadaM,HosodaK,KuniyoshiY,IshiguroH,InuiT,YoshikawaY,OginoM,YoshidaC(2009)Cognitivedevelop-
mentalrobotics:asurvey.AutonomousMentalDevelopment,IEEETransactionson1(1):12–34 2
Aziz H, Brill M, Fischer F, Harrenstein P, Lang J, Seedig HG (2015) Possible and necessary winners of partial
tournaments.JournalofArtiificialIntelligenceResearchpp493–534 24
BacheK,LichmanM(2013)UCImachinelearningrepository.URLhttp://archive.ics.uci.edu/ml 11,16
Bagnall AJ, Zatuchna ZV (2005) On the classification of maze problems. In: Foundations of Learning Classifier
Systems,Springer,pp305–316 13
BaldwinD,YadavSB(1995)Theprocessofresearchinvestigationsinartificialintelligence-aunifiedview.Systems,
ManandCybernetics,IEEETransactionson25(5):852–861 2
BellemareMG,NaddafY,VenessJ,BowlingM(2013)Thearcadelearningenvironment:Anevaluationplatformfor
generalagents.JournalofArtificialIntelligenceResearch47:253–279 10,16,19,34,37
BesoldTR(2014)Anoteonchancesandlimitationsofpsychometricai.In:KI2014:AdvancesinArtificialIntelligence,
Springer,pp49–54 29
BieverC(2011)UltimateIQ:onetesttorulethemall.NewScientist211(2829,10September2011):42–45 35
BorgM,JohansenSS,ThomsenDL,KrausM(2012)PracticalimplementationofagraphicsTuringTest.In:Advances
inVisualComputing,Springer,pp305–313 10
BoringEG(1923)Intelligenceastheteststestit.NewRepublicpp35–37 36
BostromN(2014)Superintelligence:Paths,dangers,strategies.OxfordUniversityPress 26
Brazdil P, Carrier CG, Soares C, Vilalta R (2008) Metalearning: applications to data mining. Springer Science &
BusinessMedia 16
BringsjordS(2011)Psychometricartificialintelligence.JournalofExperimental&TheoreticalArtificialIntelligence
23(3):271–277 29
Bringsjord S, Schimanski B (2003) What is artificial intelligence? Psychometric AI as an answer. In: International
JointConferenceonArtificialIntelligence,pp887–893 29
BrundageM(2016)Modelingprogressinai.AAAI2016WorkshoponAI,Ethics,andSociety 40
BuchananBG(1988)Artificialintelligenceasanexperimentalscience.Springer 2,7
BuhrmesterM,KwangT,GoslingSD(2011)Amazon’smechanicalturkanewsourceofinexpensive,yethigh-quality,
data?PerspectivesonPsychologicalScience6(1):3–5 30
Bursztein E, Aigrain J, Moscicki A, Mitchell JC (2014) The end is nigh: generic solving of text-based captchas. In:
Proceedingsofthe8thUSENIXconferenceonOffensiveTechnologies,USENIXAssociation,pp3–3 9
CampbellM,HoaneAJ,HsuF(2002)DeepBlue.ArtificialIntelligence134(1-2):57–83 5
CangelosiA,SchlesingerM,SmithLB(2015)Developmentalrobotics:Frombabiestorobots.MITPress 38,39
Caputo B, Mu¨ller H, Martinez-Gomez J, Villegas M, Acar B, Patricia N, Marvasti N, U¨sku¨darlı S, Paredes R,
Cazorla M, et al (2014) Imageclef 2014: Overview and analysis of the results. In: Information Access Evaluation.
Multilinguality,Multimodality,andInteraction,Springer,pp192–211 16,21
CarlsonA,BetteridgeJ,KisielB,SettlesB,HruschkaJrER,MitchellTM(2010)Towardanarchitecturefornever-
endinglanguagelearning.In:AAAI,vol5,p3 38
CarrollJB(1993)Humancognitiveabilities:Asurveyoffactor-analyticstudies.CambridgeUniversityPress 36
CaruanaR(1997)Multitasklearning.MachineLearning28(1):41–75 38
ChaitinGJ(1982)Go¨del’stheoremandinformation.InternationalJournalofTheoreticalPhysics21(12):941–954 32

Evaluationinartificialintelligence 45
Chandrasekaran B (1990) What kind of information processing is intelligence? In: The foundation of artificial
intelligence—asourcebook,CambridgeUniversityPress,pp14–46 31
ChaterN(1999)Thesearchforsimplicity:afundamentalcognitiveprinciple?TheQuarterlyJournalofExperimental
Psychology:SectionA52(2):273–302 34
ChaterN,Vita´nyiP(2003)Simplicity:Aunifyingprincipleincognitivescience?Trendsincognitivesciences7(1):19–
22 34
ChuZ,GianvecchioS,WangH,JajodiaS(2010)Whoistweetingontwitter:human,bot,orcyborg?In:Proceedings
ofthe26thannualcomputersecurityapplicationsconference,ACM,pp21–30 9
CochranWG(2007)Samplingtechniques.JohnWiley&Sons 12
CohenPR,HoweAE(1988)HowevaluationguidesAIresearch:Themessagestillcountsmorethanthemedium.AI
Magazine9(4):35 3,5
Cohen Y (2013) Testing and cognitive enhancement. Tech. rep., National Institute for Testing and Evaluation,
Jerusalem,Israel 35
Conrad JG, Zeleznikow J (2013) The significance of evaluation in AI and law: a case study re-examining ICAIL
proceedings.In:Proceedingsofthe14thIntlConfonArtificialIntelligenceandLaw,ACM,pp186–191 3
Conrad JG, Zeleznikow J (2015) The role of evaluation in ai and law. In: Proceedings of the 15th International
ConferenceonArtificialIntelligenceandLaw,pp181–186 3
DearyIJ,DerG,FordG(2001)Reactiontimesandintelligencedifferences:Apopulation-basedcohortstudy.Intel-
ligence29(5):389–399 30
DeckerKS,DurfeeEH,LesserVR(1989)Evaluatingresearchincooperativedistributedproblemsolving.Distributed
ArtificialIntelligence2:487–519 2
Demˇsar J (2006) Statistical comparisons of classifiers over multiple data sets. The Journal of Machine Learning
Research7:1–30 17
DettermanDK(2011)AchallengetoWatson.Intelligence39(2-3):77–78 29
DimitrakakisC(2016)Personalcommunication 19
Dimitrakakis C, Li G, Tziortziotis N (2014) The reinforcement learning competition 2014. AI Magazine 35(3):61–65
16,19
DoweDL(2013)IntroductiontoRaySolomonoff85thmemorialconference.In:DoweDL(ed)AlgorithmicProbability
andFriends.BayesianPredictionandArtificialIntelligence,LectureNotesinComputerScience,vol7070,Springer
BerlinHeidelberg,pp1–36 32
DoweDL,HajekAR(1997)AcomputationalextensiontotheTuringTest.In:Proceedingsofthe4thConferenceof
theAustralasianCognitiveScienceSociety,UniversityofNewcastle,NSW,Australia 32,34
Dowe DL, Hajek AR (1998) A non-behavioural, computational extension to the Turing Test. In: Intl. Conf. on
ComputationalIntelligence&multimediaapplications(ICCIMA’98),Gippsland,Australia,pp101–106 32,34
DoweDL,Hern´andez-OralloJ(2012)IQtestsarenotformachines,yet.Intelligence40(2):77–81 30
DoweDL,Hern´andez-OralloJ(2014)Howuniversalcananintelligencetestbe?AdaptiveBehavior22(1):51–69 37
DrummondC(2009)Replicabilityisnotreproducibility:norisitgoodscience.In:Proc.oftheEvaluationMethods
forMachineLearningWorkshopatthe26thICML,Montreal,Canada 18
Drummond C, Japkowicz N (2010) Warning: statistical benchmarking is addictive. Kicking the habit in machine
learning.JournalofExperimental&TheoreticalArtificialIntelligence22(1):67–80 2,11,17
DuanY,ChenX,HouthooftR,SchulmanJ,AbbeelP(2016)Benchmarkingdeepreinforcementlearningforcontinuous
control.arXivpreprintarXiv:160406778 37
EdenAH,MoorJH,SorakerJH,SteinhartE(2013)Singularityhypotheses:Ascientificandphilosophicalassessment.
Springer 43
EdmondsonW(2012)TheintelligenceinETI-Whatcanweknow?ActaAstronautica78:37–42 35
EloAE(1978)Theratingofchessplayers,pastandpresent,vol3.BatsfordLondon 8,21,24
EmbretsonSE,ReiseSP(2000)Itemresponsetheoryforpsychologists.L.Erlbaum 13,29
EvansJM,MessinaER(2001)Performancemetricsforintelligentsystems.NISTSpecialPublicationSPpp101–104
24
EverittT,LattimoreT,HutterM(2014)Freelunchforoptimisationundertheuniversaldistribution.In:Evolutionary
Computation(CEC),2014IEEECongresson,IEEE,pp167–174 6
FalkenauerE(1998)Onmethodoverfitting.JournalofHeuristics4(3):281–287 2,11
FeldmanJ(2003)Simplicityandcomplexityinhumanconceptlearning.TheGeneralPsychologist38(1):9–15 34
FerrandoPJ(2009)Difficulty,discrimination,andinformationindicesinthelinearfactoranalysismodelforcontinuous
itemresponses.AppliedPsychologicalMeasurement33(1):9–24 14
Ferrando PJ (2012) Assessing the discriminating power of item and test scores in the linear factor-analysis model.
Psicol´ogica33:111–139 14

46 Jos´eHern´andez-Orallo
FerriC,Herna´ndez-OralloJ,ModroiuR(2009)Anexperimentalcomparisonofperformancemeasuresforclassification.
PatternRecognitionLetters30(1):27–38 17
FerrucciD,BrownE,Chu-CarrollJ,FanJ,GondekD,KalyanpurAA,LallyA,MurdockJ,NybergE,PragerJ,etal
(2010)BuildingWatson:AnoverviewoftheDeepQAproject.AIMagazine31(3):59–79 8,29
FogelDB(1991)Theevolutionofintelligentdecisionmakingingaming.CyberneticsandSystems22(2):223–236 33
Gaschnig J, Klahr P, Pople H, Shortliffe E, Terry A (1983) Evaluation of expert systems: Issues and case studies.
Buildingexpertsystems1:241–278 2
GeissmanJR,SchultzRD(1988)Verification&validation.AIExpert3(2):26–33 2
GeneserethM,LoveN,PellB(2005)Generalgameplaying:OverviewoftheAAAIcompetition.AIMagazine26(2):62
23
Ger´onimo D, L´opez AM (2014) Datasets and benchmarking. In: Vision-based Pedestrian Protection Systems for
IntelligentVehicles,Springer,pp87–93 16
GoertzelB,PennachinC(eds)(2007)Artificialgeneralintelligence.Springer 2
Goertzel B, Arel I, Scheutz M (2009) Toward a roadmap for human-level artificial general intelligence: Embedding
HLAIsystemsinbroad,approachable,physicalorvirtualcontexts.ArtificialGeneralIntelligenceRoadmapInitia-
tive 39
GoldreichO,VadhanS(2007)Specialissueonworst-caseversusaverage-casecomplexity–editors’foreword.compu-
tationalcomplexity16(4):325–330 7
GordonBB(2007)Reportonpaneldiscussionon(re-)establishingorincreasingcollaborativelinksbetweenartificial
intelligence and intelligent systems. In: Messina ER, Madhavan R (eds) Proceedings of the 2007 Workshop on
PerformanceMetricsforIntelligentSystems,pp302–303 24
Gulwani S, Herna´ndez-Orallo J, Kitzelmann E, Muggleton SH, Schmid U, Zorn B (2015) Inductive programming
meetstherealworld.CommunicationsoftheACM58(11):90–99 2,38
HandDJ(2004)Measurementtheoryandpractice.AHodderArnoldPublication 2
Hern´andez-OralloJ(2000a)BeyondtheTuringTest.JLogic,Language&Information9(4):447–466 32,33
Hern´andez-OralloJ(2000b)Onthecomputationalmeasurementofintelligencefactors.In:MeystelA(ed)Performance
metrics for intelligent systems workshop, National Institute of Standards and Technology, Gaithersburg, MD,
U.S.A.,pp1–8 7,33
Hern´andez-OralloJ(2000c)Thesis:Computationalmeasuresofinformationgainandreinforcementininferencepro-
cesses.AICommunications13(1):49–50 34
Hern´andez-OralloJ(2010)A(hopefully)non-biaseduniversalenvironmentclassformeasuringintelligenceofbiological
and artificial systems. In: et al MH (ed) Artificial General Intelligence, 3rd Intl Conf, Atlantis Press, Extended
reportathttp://users.dsic.upv.es/proy/anynt/unbiased.pdf,pp182–183 35
Hern´andez-OralloJ(2014)Onenvironmentdifficultyanddiscriminatingpower.AutonomousAgentsandMulti-Agent
Systemspp1–53,DOI10.1007/s10458-014-9257-1,URLhttp://dx.doi.org/10.1007/s10458-014-9257-1 22
Hern´andez-OralloJ,DoweDL(2010)Measuringuniversalintelligence:Towardsananytimeintelligencetest.Artificial
Intelligence174(18):1508–1539 33,34,35
Hern´andez-OralloJ,DoweDL(2013)Onpotentialcognitiveabilitiesinthemachinekingdom.MindsandMachines
23:179–210 37
Hern´andez-Orallo J, Minaya-Collado N (1998) A formal definition of intelligence based on an intensional variant of
Kolmogorovcomplexity.In:Proc.IntlSymposiumofEngineeringofIntelligentSystems(EIS’98),ICSCPress,pp
146–163 32,33
Hern´andez-Orallo J, Dowe DL, Espan˜a-Cubillo S, Hern´andez-Lloreda MV, Insa-Cabrera J (2011) On more realistic
environment distributions for defining, evaluating and developing intelligence. In: Schmidhuber J, Tho´risson K,
LooksM(eds)ArtificialGeneralIntelligence,LNAI,Springer,vol6830,pp82–91 34,35
Hern´andez-OralloJ,FlachP,FerriC(2012a)Aunifiedviewofperformancemetrics:Translatingthresholdchoiceinto
expectedclassificationloss.TheJournalofMachineLearningResearch13(1):2813–2869 17
Hern´andez-OralloJ,Insa-CabreraJ,DoweDL,HibbardB(2012b)TuringTestswithTuringmachines.In:Voronkov
A(ed)Turing-100,EPiCSeries,vol10,pp140–156 9
Hern´andez-OralloJ,DoweDL,Hern´andez-LloredaMV(2014)Universalpsychometrics:Measuringcognitiveabilities
inthemachinekingdom.CognitiveSystemsResearch27:50âĂŞ74 35,41
Hern´andez-OralloJ,Mart´ınez-PlumedF,SchmidU,SiebersM,DoweDL(2016)Computermodelssolvingintelligence
testproblems:Progressandimplications.ArtificialIntelligence230:74–107 30
HerrmannE,CallJ,Hern´andez-LloredaMV,HareB,TomaselloM(2007)Humanshaveevolvedspecializedskillsof
socialcognition:Theculturalintelligencehypothesis.ScienceVol317(5843):1360–1366 31
HibbardB(2009)Biasandnofreelunchinformalmeasuresofintelligence.JournalofArtificialGeneralIntelligence
1(1):54–61 32,33,36

Evaluationinartificialintelligence 47
HingstonP(2010)AnewdesignforaTuringTestforbots.In:ComputationalIntelligenceandGames(CIG),2010
IEEESymposiumon,IEEE,pp345–350 9
HingstonP(2012)BelievableBots:CanComputersPlayLikePeople?Springer 9,10
Ho TK, Basu M (2002) Complexity measures of supervised classification problems. Pattern Analysis and Machine
Intelligence,IEEETransactionson24(3):289–300 18
HutterM(2007)Universalalgorithmicintelligence:Amathematicaltop→downapproach.In:GoertzelB,Pennachin
C(eds)ArtificialGeneralIntelligence,CognitiveTechnologies,Springer,Berlin,pp227–290 2,36
Igel C, Toussaint M (2005) A no-free-lunch theorem for non-uniform distributions of target functions. Journal of
MathematicalModellingandAlgorithms3(4):313–322 6
Insa-CabreraJ(2016)Towardsauniversaltestofsocialintelligence.PhDthesis,DepartamentdeSistemesInform´atics
iComputaci´o,UPV 35
Insa-CabreraJ,DoweDL,Espan˜a-CubilloS,Hern´andez-LloredaMV,Hern´andez-OralloJ(2011a)Comparinghumans
andAIagents.In:SchmidhuberJ,Tho´rissonK,LooksM(eds)ArtificialGeneralIntelligence,LNAI,Springer,vol
6830,pp122–132 33,35
Insa-CabreraJ,DoweDL,Hern´andez-OralloJ(2011b)Evaluatingareinforcementlearningalgorithmwithageneral
intelligencetest.In:JALozanoJMJAGamez(ed)CurrentTopicsinArtificialIntelligence.CAEPIA2011,LNAI
Series7023,Springer 33,35
Insa-Cabrera J, Benacloch-Ayuso JL, Hern´andez-Orallo J (2012) On measuring social intelligence: Experiments on
competition and cooperation. In: Bach J, Goertzel B, Ikl´e M (eds) AGI, Springer, Lecture Notes in Computer
Science,vol7716,pp126–135 34,35
Jacoff A, Messina E, Weiss BA, Tadokoro S, Nakagawa Y (2003) Test arenas and performance metrics for urban
search and rescue robots. In: Intelligent Robots and Systems, 2003.(IROS 2003). Proceedings. 2003 IEEE/RSJ
InternationalConferenceon,IEEE,vol4,pp3396–3403 16
JapkowiczN,ShahM(2011)EvaluatingLearningAlgorithms.CambridgeUniversityPress 17
Jiang J (2008) A literature survey on domain adaptation of statistical classifiers. URL:
http://sifakacsuiucedu/jiang4/domainadaptation/survey 38
JohnsonM,HofmannK,HuttonT,BignellD(2016)TheMalmoplatformforartificialintelligenceexperimentation.
In:InternationalJointConferenceonArtificialIntelligence(IJCAI) 37
KeithTZ,ReynoldsMR(2010)Cattell–Horn–Carrollabilitiesandcognitivetests:Whatwe’velearnedfrom20years
ofresearch.PsychologyintheSchools47(7):635–650 27
KetterW,SymeonidisA(2012)Competitivebenchmarking:Lessonslearnedfromthetradingagentcompetition.AI
Magazine33(2):103 23
KhreichW,GrangerE,MiriA,SabourinR(2012)AsurveyoftechniquesforincrementallearningofHMMparameters.
InformationSciences197:105–130 38
KimJH(2004)Soccerrobotics,vol11.Springer 23
KitanoH,AsadaM,KuniyoshiY,NodaI,OsawaE(1997)Robocup:Therobotworldcupinitiative.In:Proceedings
ofthefirstinternationalconferenceonAutonomousagents,ACM,pp340–347 23
KleinerK(2011)Whoareyoucallingbird-brained?Anattemptisbeingmadetodeviseauniversalintelligencetest.
TheEconomist398(8723,5March2011):82 35
KnuthDE(1973)Sortingandsearching,volume3ofTheArtofComputerProgramming.Addison-Wesley 7
KozaJR(2010)Human-competitiveresultsproducedbygeneticprogramming.GeneticProgrammingandEvolvable
Machines11(3-4):251–284 10
KruegerJ,OshersonD(1980)Onthepsychologyofstructuralsimplicity.In:JusczykPW,KleinRM(eds)TheNature
ofThought:EssaysinHonorofD.O.Hebb,PsychologyPress,pp187–205 34
LangfordJ(2005)Clevermethodsofoverfitting.MachineLearning(Theory),http://hunchnet 2,11
LangleyP(1987)Researchpapersinmachinelearning.MachineLearning2(3):195–198 2
LangleyP(2011)Thechangingscienceofmachinelearning.MachineLearning82(3):275–279 2
LangleyP(2012)Thecognitivesystemsparadigm.AdvancesinCognitiveSystems1:3–13 39
LattimoreT,HutterM(2013)NofreelunchversusOccam’srazorinsupervisedlearning.In:AlgorithmicProbability
andFriends.BayesianPredictionandArtificialIntelligence,Springer,pp223–235 6,32
LeeuwenbergELJ,VanDerHelmPA(2012)Structuralinformationtheory:Thesimplicityofvisualform.Cambridge
UniversityPress 34
LeggS,HutterM(2007a)Testsofmachineintelligence.In:LungarellaM,IidaF,BongardJ,PfeiferR(eds)50Years
of Artificial Intelligence, Lecture Notes in Computer Science, vol 4850, Springer Berlin Heidelberg, pp 232–242,
DOI10.1007/978-3-540-77296-5 22,URLhttp://dx.doi.org/10.1007/978-3-540-77296-5_22 2
LeggS,HutterM(2007b)Universalintelligence:Adefinitionofmachineintelligence.MindsandMachines17(4):391–
444 33

48 Jos´eHern´andez-Orallo
Legg S, Veness J (2013) An approximation of the universal intelligence measure. In: Algorithmic Probability and
Friends.BayesianPredictionandArtificialIntelligence,Springer,pp236–249 33
LevesqueHJ(2014)Onourbestbehaviour.ArtificialIntelligence212:27–35 25
LevesqueHJ,DavisE,MorgensternL(2012)Thewinogradschemachallenge.In:KR 25
LevinLA(1973)Universalsequentialsearchproblems.ProblemsofInformationTransmission9(3):265–266 32
LevinLA(1986)Averagecasecompleteproblems.SIAMJonComputing15:285–286 7
LevinLA(2013)Universalheuristics:HowdohumanssolveâĂĲunsolvableâĂİproblems?In:DoweDL(ed)Algorith-
mic Probability and Friends. Bayesian Prediction and Artificial Intelligence, Lecture Notes in Computer Science,
vol7070,Springer,pp53–54 32
LiM,Vita´nyiP(2008)AnintroductiontoKolmogorovcomplexityanditsapplications(3rded.).Springer 6,31,32
LivingstoneD(2006)Turing’stestandbelievableAIingames.ComputersinEntertainment(CIE)4(1):6 9,10
Llargues-AsensioJM,PeraltaJ,ArrabalesR,Gonz´alez-Bed´ıaM,CortezP,L´opez-Pen˜aAL(2014)Artificialintelligence
approaches for the generation and assessment of believable human-like behaviour in virtual characters. Expert
SystemswithApplications 10
Long D, Fox M (2003) The 3rd international planning competition: Results and analysis. J Artif Intell Res (JAIR)
20:1–59 16
LordFM(1980)Applicationsofitemresponsetheorytopracticaltestingproblems.Mahwah,NJ:Erlbaum 29
Maci`aN,Bernad´o-MansillaE(2014)TowardsUCI+:Amindfulrepositorydesign.InformationSciences261:237–262
17,18,34
MadhavanR,TunstelE,MessinaE(2009)PerformanceEvaluationandBenchmarkingofIntelligentSystems.Springer,
September 2,24
MahoneyMV(1999)Textcompressionasatestforartificialintelligence.In:ProceedingsoftheNationalConference
onArtificialIntelligence,AAAI,pp970–970 33
March´eC,ZantemaH(2007)Theterminationcompetition.In:TermRewritingandApplications,Springer,pp303–
313 16
MarcusG,RossiF,VelosoM(2016)BeyondtheTuringtest(specialissue).AIMagazine37(1):3–101 25
MasumH,ChristensenS(2003)Theturingratio:Aframeworkforopen-endedtaskmetrics.In:JournalofEvolution
andTechnology,Citeseer 21,24,26
MasumH,ChristensenS,OppacherF(2002)Theturingratio:Metricsforopen-endedtasks.In:GECCO,Citeseer,
pp973–980 26
McCarthyJ(2007)Whatisartificialintelligence.Tech.rep.,StanfordUniversity,http://www-formal.stanford.edu/
jmc/whatisai.html 1
McCorduckP(2004)Machineswhothink.AKPeters/CRCPress 2
McDermott J, White DR, Luke S, Manzoni L, Castelli M, Vanneschi L, Ja´skowski W, Krawiec K, Harper R, Jong
KD,O’ReillyUM(2012)Geneticprogrammingneedsbetterbenchmarks.In:Proceedingsofthe14thinternational
conferenceonGeneticandevolutionarycomputationconference,ACM,Philadelphia,pp791–798 16
McGuiganM(2006)GraphicsTuringTest.arXivpreprintcs/0603132 10
Melkikh AV (2014) The no free lunch theorem and hypothesis of instinctive animal behavior. Artificial Intelligence
Research3(4):p43 35
MellenberghGJ(1994)Generalizedlinearitemresponsetheory.PsychologicalBulletin115(2):300 14
MesnilG,DauphinY,GlorotX,RifaiS,BengioY,GoodfellowIJ,LavoieE,MullerX,DesjardinsG,Warde-Farley
D, et al (2012) Unsupervised and transfer learning challenge: a deep learning approach. JMLR: Workshop and
ConferenceProceedings27:97âĂŞ111,2012ICMLWorkshoponUnsupervisedandTransferLearning27:97–110 38
MessinaE,MeystelA,ReekerL(2001)PerMIS2001,whitepaper.In:MeystelAM,MessinaER(eds)Measuringthe
performanceandintelligenceofsystems:proceedingsofthe2001PerMISWorkshop,September4,2001,National
InstituteofStandardsandTechnology(NIST)SpecialPublication982,Gaithersburg,MD,U.S.A.,pp3–15 24
Meystel A (2000) Permis 2000 white paper: Measuring performance and intelligence of systems with autonomy. In:
Meystel AM, Messina ER (eds) Measuring the performance and intelligence of systems: proceedings of the 2000
PerMISWorkshop,August14–16,2000,NationalInstituteofStandardsandTechnology(NIST)SpecialPublication
970,Gaithersburg,MD,U.S.A.,pp1–34 24
Meystel A, Albus J, Messina E, Leedom D (2003a) Performance measures for intelligent systems: Measures of tech-
nologyreadiness.Tech.rep.,DTICDocument 24
Meystel A, Albus J, Messina E, Leedom D (2003b) Permis 2003 white paper: Performance measures for intelligent
systems - measures of technology readiness. In: Meystel AM, Messina ER (eds) Measuring the performance and
intelligenceofsystems:proceedingsofthe2003PerMISWorkshop,NationalInstituteofStandardsandTechnology
(NIST)SpecialPublication1014,Gaithersburg,MD,U.S.A. 24
MinskyML(ed)(1968)SemanticInformationProcessing.MITPress 2

Evaluationinartificialintelligence 49
Mnih V, Kavukcuoglu K, Silver D, Rusu AA, Veness J, Bellemare MG, Graves A, Riedmiller M, Fidjeland AK,
OstrovskiG,etal(2015)Human-levelcontrolthroughdeepreinforcementlearning.Nature518(7540):529–533 19
MorgensternL,DavisE,Ortiz-JrCL(2016)Planning,executing,andevaluatingtheWinogradschemachallenge.AI
Magazine37(1):50–54 25
MuellerS,JonesM,MinneryB,HilandJM(2007)Thebicacognitivedecathlon:Atestsuiteforbiologically-inspired
cognitiveagents.In:ProceedingsofBehaviorRepresentationinModelingandSimulationConference,Norfolk 39
MuellerST(2010)ApartialimplementationoftheBICAcognitivedecathlonusingthepsychologyexperimentbuilding
language(PEBL).InternationalJournalofMachineConsciousness2(02):273–288 39
Mueller ST, Minnery BS (2008) Adapting the Turing Test for embodied neurocognitive evaluation of biologically-
inspired cognitive agents. In: Proc. 2008 AAAI Fall Symposium on Biologically Inspired Cognitive Architectures
9
NewellA(1973)Youcan’tplay20questionswithnatureandwin:Projectivecommentsonthepapersofthissympo-
sium.In:VisualInformationProcessing,ed.W.Chase,NewYork:AcademicPress,pp283–308 29
NewellA(1980)Physicalsymbolsystems.Cognitivescience4(2):135–183 39
NewellA(1990)Unifiedtheoriesofcognition.Cambridge,MA:HarvardUniversity 39
NewellA,SimonHA(1976)Computerscienceasempiricalinquiry:Symbolsandsearch.CommunicationsoftheACM
19(3):113–126 2
Nizamani AR (2015) Reasoning with bounded cognitive resources. PhD thesis, Department of Applied Information
Technology,ChalmersUniversityofTechnology&UniversityofGothenburg,Sweden 34
Oppy G, Dowe DL (2011) The Turing Test. In: Zalta EN (ed) Stanford Encyclopedia of Philosophy, pp Stanford
University,http://plato.stanford.edu/entries/turing–test/ 8
Pan SJ, Yang Q (2010) A survey on transfer learning. Knowledge and Data Engineering, IEEE Transactions on
22(10):1345–1359 38
PerezD,SamothrakisS,TogeliusJ,SchaulT,LucasS,Cou¨etouxA,LeeJ,LimCU,ThompsonT(2015)The2014
general video game playing competition. IEEE Transactions on Computational Intelligence and AI in Games 16,
20,34,37
Potthast M, Hagen M, Gollub T, Tippmann M, Kiesel J, Rosso P, Stamatatos E, Stein B (2013) Overview of the
5thinternationalcompetitiononplagiarismdetection.CLEF2013EvaluationLabsandWorkshopWorkingNotes
Papers,23-26September,Valencia,Spain 16
ProudfootD(2011)AnthropomorphismandAI:Turing’smuchmisunderstoodimitationgame.ArtificialIntelligence
175(5):950–957 9
Quinn AJ, Bederson BB (2011) Human computation: a survey and taxonomy of a growing field. In: Proceedings of
theSIGCHIConferenceonHumanFactorsinComputingSystems,ACM,pp1403–1412 35
RajaniS(2011)Artificialintelligence–manormachine.IntllJournalofInformationTechnology4(1):173–176 26
RaoRB,FungG,RosalesR(2008)Onthedangersofcross-validation.anexperimentalevaluation.In:SDM,SIAM,
pp588–596 18
vanRijnJN,BischlB,TorgoL,GaoB,UmaashankarV,FischerS,WinterP,WiswedelB,BertholdMR,Vanschoren
J(2013)Openml:acollaborativescienceplatform.In:MachineLearningandKnowledgeDiscoveryinDatabases,
Springer,pp645–649 18
Rohrer B (2010) Accelerating progress in artificial general intelligence: Choosing a benchmark for natural world
interaction.JofArtificialGeneralIntelligence2(1):1–28 39
Rothenberg J, Paul J, Kameny I, Kipps JR, Swenson M (1987) Evaluating expert system tools: A framework and
methodology–workshops.Tech.rep.,DTICDocument 2
RussellS,NorvigP(2009)ArtificialIntelligence:AModernApproach.PrenticeHall 3,5,27
SanghiP,DoweDL(2003)AcomputerprogramcapableofpassingIQtests.In:4thIntl.Conf.onCognitiveScience
(ICCS’03),Sydney,pp570–575 29,30,33
Schaeffer J, Burch N, Bjornsson Y, Kishimoto A, Muller M, Lake R, Lu P, Sutphen S (2007) Checkers is solved.
Science317(5844):1518 7,22
SchaieKW(2010)Primarymentalabilities.CorsiniEncyclopediaofPsychology 27
Schaul T (2014) An extensible description language for video games. Computational Intelligence and AI in Games,
IEEETransactionsonPP(99):1–1,DOI10.1109/TCIAIG.2014.2352795 10,16,20,34
Schenck C (2013) Intelligence tests for robots: Solving perceptual reasoning tasks with a humanoid robot. Master’s
thesis,IowaStateUniversity 30
Schlenoff C, Scott H, Balakirsky S (2011) Performance evaluation of intelligent systems at the national institute of
standardsandtechnology(nist).Tech.rep.,DTICDocument 2,24
SchmidU,RagniM(2015)Comparingcomputermodelssolvingnumberseriesproblems.In:ArtificialGeneralIntel-
ligence,Springer,pp352–361 34
SchweizerP(1998)ThetrulytotalTuringTest.MindsandMachines8(2):263–272 9

50 Jos´eHern´andez-Orallo
SearleJR(1980)Minds,brains,andprograms.TheBehavioralandBrainSciences3:417–457 32
SeberGAF,SalehiMM(2013)Adaptiveclustersampling.In:AdaptiveSamplingDesigns,Springer,pp11–26 13
SettlesB(2012)Activelearning.SynthesisLecturesonArtificialIntelligenceandMachineLearning6(1):1–114 38
ShettleworthSJ(2010)Cognition,evolution,andbehavior.OxfordUniversityPress 31
ShettleworthSJ,BloomP,NadelL(2013)FundamentalsofComparativeCognition.OxfordUniversityPress 31
Shieber SM (2016) Principles for designing an AI competition, or why the Turing test fails as an inducement prize.
AIMagazine37(1):91–96 42
Silver D, Huang A, Maddison CJ, Guez A, Sifre L, Van Den Driessche G, Schrittwieser J, Antonoglou I, Panneer-
shelvamV,LanctotM,etal(2016)Masteringthegameofgowithdeepneuralnetworksandtreesearch.Nature
529(7587):484–489 5
Simmons R (2000) Survivability and competence as measures of intelligent systems. In: Meystel AM, Messina ER
(eds) Measuring the performance and intelligence of systems: proceedings of the 2000 PerMIS Workshop, August
14–16,2000,NationalInstituteofStandardsandTechnology(NIST)SpecialPublication970,Gaithersburg,MD,
U.S.A.,pp162–163 24
SimonHA(1995)Artificialintelligence:anempiricalscience.ArtificialIntelligence77(1):95–127 2,7
SlomanA,ScheutzM(2002)Aframeworkforcomparingagentarchitectures.ProceedingsofUKCI2 39
SmithWD(2002)Ratingsystemsforgameplayers,andlearning.NEC,Princeton,NJ,TechReppp93–104 21,24
SmithWD(2006)Mathematicaldefinitionof“intelligence”(andconsequences).Unpublishedreport 34,35
Soares C (2009) UCI++: Improved support for algorithm selection using datasetoids. In: Advances in Knowledge
DiscoveryandDataMining,Springer,pp499–506 18,34
SolomonoffR(1996)Doesalgorithmicprobabilitysolvetheproblemofinduction.Information,StatisticsandInduction
inSciencepp7–8 32
SolomonoffRJ(1964)Aformaltheoryofinductiveinference.PartI.Informationandcontrol7(1):1–22 6,31,32
Solomonoff RJ (1984) Optimum sequential search. Oxbridge Research, Cambridge, Mass http://world.std.com/
˜rjs/optseq.pdf 32
SrinivasanR(2002)Importancesampling:Applicationsincommunicationsanddetection.Springer 12
StarkieB,vanZaanenM,EstivalD(2006)TheTenjinnomachinetranslationcompetition.In:GrammaticalInference:
AlgorithmsandApplications,Springer,pp214–226 16
Sternberg(ed)RJ(2000)Handbookofintelligence.CambridgeUniversityPress 28
Stranneg˚ard C, Amirghasemi M, Ulfsbu¨cker S (2013a) An anthropomorphic method for number sequence problems.
CognitiveSystemsResearch22âĂŞ23(0):27–34 34
Stranneg˚ard C, Nizamani A, Sj¨oberg A, Engstr¨om F (2013b) Bounded Kolmogorov complexity based on cognitive
models.In:Ku¨hnbergerKU,RudolphS,WangP(eds)ArtificialGeneralIntelligence,LectureNotesinComputer
Science,vol7999,SpringerBerlinHeidelberg,pp130–139 34
StricklerRE(1973)Changeinselectedcharacteristicsofstudentsbetweenninthandtwelfthgradeasrelatedtohigh
schoolcurriculum. 27
Sturtevant N (2012) Benchmarks for grid-based pathfinding. Transactions on Computational Intelligence and AI in
Games4(2):144–148,URLhttp://web.cs.du.edu/˜sturtevant/papers/benchmarks.pdf 13,16
Sutcliffe G (2009) The TPTP Problem Library and Associated Infrastructure: The FOF and CNF Parts, v3.5.0.
JournalofAutomatedReasoning43(4):337–362 16
SutcliffeG,SuttnerC(2006)TheStateofCASC.AICommunications19(1):35–48 16
ThrunS(1996)Islearningthen-ththinganyeasierthanlearningthefirst?Advancesinneuralinformationprocessing
systemspp640–646 38
ThrunS,PrattL(2012)Learningtolearn.SpringerScience&BusinessMedia 38
ThurstoneLL(1938a)Primarymentalabilities.Psychometricmonographs 32
ThurstoneLL(1938b)Primarymentalabilities.Psychometricmonographs 36
TogeliusJ,YannakakisGN,KarakovskiyS,ShakerN(2012)Assessingbelievability.In:BelievableBots,Springer,pp
215–230 10
TorreyL,ShavlikJ(2009)Transferlearning.HandbookofResearchonMachineLearningApplications3:17–35 38
TuringAM(1950)Computingmachineryandintelligence.Mind59:433–460 8
ValiantLG(1984)Atheoryofthelearnable.CommunicationsoftheACM27(11):1134–1142 7
VallatiM,ChrpaL,GrzesM,McCluskeyTL,RobertsM,SannerS(2015)The2014internationalplanningcompeti-
tion:Progressandtrends.AIMagazine36(3):90–98 15,16
Vanschoren J, Blockeel H, Pfahringer B, Holmes G (2012) Experiment databases. Machine Learning 87(2):127–158
18,42
VanschorenJ,vanRijnJN,BischlB,TorgoL(2014)Openml:networkedscienceinmachinelearning.ACMSIGKDD
ExplorationsNewsletter15(2):49–60 18

Evaluationinartificialintelligence 51
V´azquez D, L´opez AM, Mar´ın J, Ponsa D, Gero´nimo D (2014) Virtual and real world adaptation for pedestrian
detection.PatternAnalysisandMachineIntelligence,IEEETransactionson36(4):797–809,DOI10.1109/TPAMI.
2013.163 11
VereSA(1992)Acognitiveprocessshell.BehavioralandBrainSciences15(03):460–461 39
WallaceCS,BoultonDM(1968)Aninformationmeasureforclassification.ComputerJournal11(2):185–194 31
WallaceCS,DoweDL(1999)MinimummessagelengthandKolmogorovcomplexity.ComputerJournal42(4):270–283,
specialissueonKolmogorovcomplexity. 31
WangG,MohanlalM,WilsonC,WangX,MetzgerM,ZhengH,ZhaoBY(2012)SocialTuringTests:Crowdsourcing
sybildetection.arXivpreprintarXiv:12053856 9
WangP(2010)Theevaluationofagisystems.In:ProceedingsoftheThirdConferenceonArtificialGeneralIntelli-
gence,Citeseer,pp164–169 39
Warwick K (8 June 2014) Turing Test success marks milestone in computing history. University or Reading Press
Release 8
WassermanEA,ZentallTR(2006)Comparativecognition:Experimentalexplorationsofanimalintelligence.Oxford
UniversityPress 31
WatkinsCJCH,DayanP(1992)Q-learning.Machlearning8(3):279–292 35
WeissDJ(2011)Betterdatafrombettermeasurementsusingcomputerizedadaptivetesting.JournalofMethodsand
MeasurementintheSocialSciences2(1):1–27 15
WeizenbaumJ(1966)ELIZA–acomputerprogramforthestudyofnaturallanguagecommunicationbetweenman
andmachine.CommunicationsoftheACM9(1):36ÃćÂĂÂŞ45 8
Wellman M, Reeves D, Lochner K, Vorobeychik Y (2004) Price prediction in a trading agent competition. J Artif
IntellRes(JAIR)21:19–36 23
White DR, McDermott J, Castelli M, Manzoni L, Goldman BW, Kronberger G, Ja´skowski W, O’Reilly UM, Luke
S (2013) Better GP benchmarks: Community survey results and proposals. Genetic Programming and Evolvable
Machines14:3–29,DOI10.1007/s10710-012-9177-2 16
WhitesonS,TannerB,WhiteA(2010)TheReinforcementLearningCompetitions.TheAImagazine31(2):81–94 16
WhitesonS,TannerB,TaylorME,StoneP(2011)Protectingagainstevaluationoverfittinginempiricalreinforcement
learning.In:AdaptiveDynamicProgrammingAndReinforcementLearning(ADPRL),2011IEEESymposiumon,
IEEE,pp120–127 2,6,8,11
Williams PL, Beer RD (2010) Information dynamics of evolved agents. In: From Animals to Animats 11, Springer,
pp38–49 31,35
WinikoffM,CranefieldS(2014)Onthetestabilityofbdiagentsystems.JArtifIntellRes(JAIR)51:71–131 7
WolpertDH(1996)Thelackofaprioridistinctionsbetweenlearningalgorithms.NeuralComputation8(7):1341–1390
6,35
WolpertDH(2012)Whatthenofreelunchtheoremsreallymean;howtoimprovesearchalgorithms.Tech.rep.,Santa
feInstituteWorkingPaper 6,35
WolpertDH,MacreadyWG(1995)Nofreelunchtheoremsforsearch.Tech.rep.,SFI-TR-95-02-010(SantaFeInsti-
tute) 6,35
Wolpert DH, Macready WG (2005) Coevolutionary free lunches. Evolutionary Computation, IEEE Transactions on
9(6):721–735 6
YampolskiyRV(2015)ArtificialSuperintelligence:AFuturisticApproach.CRCPress 34
YonckR(2012)Towardastandardmetricofmachineintelligence.WorldFutureReview4(2):61–70 35
YouJ(2015)Beyondtheturingtest.Science347(6218):116–116 25
Zatuchna Z, Bagnall A (2009) Learning mazes with aliasing states: An LCS algorithm with associative perception.
AdaptiveBehavior17(1):28–57 13
ZhouZH(2012)Ensemblemethods:foundationsandalgorithms.CRCPress 16