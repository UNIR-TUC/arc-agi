|                  |     | Communicating |           |     | Natural     |          | Programs |                      |
| ---------------- | --- | ------------- | --------- | --- | ----------- | -------- | -------- | -------------------- |
|                  |     |               | to Humans |     | and         | Machines |          |                      |
| SamuelAcquaviva⇤ |     |               | YewenPu⇤  |     | MartaKryven |          |          | TheodorosSechopoulos |
† †
|     | MIT           |     | AutodeskResearch |                  |     | MIT |     | MIT        |
| --- | ------------- | --- | ---------------- | ---------------- | --- | --- | --- | ---------- |
|     | CatherineWong |     |                  | GabrielleEEcanow |     |     |     | MaxwellNye |
†
|     |     | MIT                 |     |     | MIT |     |                   | MIT |
| --- | --- | ------------------- | --- | --- | --- | --- | ----------------- | --- |
|     |     | MichaelHenryTessler |     |     |     |     | JoshuaB.Tenenbaum |     |
|     |     |                     | MIT |     |     |     | MIT               |     |
Abstract
|     | The                                                | Abstraction | and Reasoning | Corpus | (ARC) | is  | a set of procedural  | tasks that |
| --- | -------------------------------------------------- | ----------- | ------------- | ------ | ----- | --- | -------------------- | ---------- |
|     | testsanagent’sabilitytoflexiblysolvenovelproblems. |             |               |        |       |     | WhilemostARCtasksare |            |
easyforhumans,theyarechallengingforstate-of-the-artAI.Whatmakesbuilding
intelligentsystemsthatcangeneralizetonovelsituationssuchasARCdifficult?
Wepositthattheanswermightbefoundbystudyingthedifferenceoflanguage:
Whilehumansreadilygenerateandinterpretinstructionsinagenerallanguage,
computersystemsareshackledtoanarrowdomain-specificlanguagethattheycan
|     | preciselyexecute. |     | WepresentLARC,theLanguage-completeARC:acollectionof |     |     |     |     |     |
| --- | ----------------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- |
naturallanguagedescriptionsbyagroupofhumanparticipantswhoinstructeach
otheronhowtosolveARCtasksusinglanguagealone,whichcontainssuccessful
|     | instructionsfor88%oftheARCtasks. |     |     |     | Weanalyzethecollectedinstructionsas |     |     |     |
| --- | -------------------------------- | --- | --- | --- | ----------------------------------- | --- | --- | --- |
‘naturalprograms’,findingthatwhiletheyresemblecomputerprograms,theyare
|     | distinctintwoways: |     | First,theycontainawiderangeofprimitives;Second,they |     |     |     |     |     |
| --- | ------------------ | --- | --------------------------------------------------- | --- | --- | --- | --- | --- |
frequentlyleveragecommunicativestrategiesbeyonddirectlyexecutablecodes.
Wedemonstratethatthesetwodistinctionspreventcurrentprogramsynthesistech-
niquesfromleveragingLARCtoitsfullpotential,andgiveconcretesuggestions
onhowtobuildthenext-generationprogramsynthesizers.
1 Introduction
Humanssolvearangeofproceduraltaskssuchascooking,tyingshoes,andprogramming. Although
currentAIsystemsachievesuper-humanproficiencyatcertainnarrowlyspecifiedtasks[1,2],their
reasoning is domain-specific and fails to generalize to novel situations [3]. The Abstraction and
ReasoningCorpus(ARC)introducedby[4]presentsasetofproceduraltasksconstructedexpresslyto
benchmarkfundamentalcapacitiesassociatedwithhumangeneralintelligence,includingabstraction,
generalization,objectcategories,andproceduralanalogies [3,5–10]. Specifically,ARCrequiresone
toinferaprocedureconsistentwithasmallnumberofabstractinput-outputexamplesandapplyitto
anewinputtogenerateanunseenanswer,seeFigure1.
How do we build systems that are capable of solving general, procedural tasks such as ARC?
Traditionalapproachesofprogramsynthesis[11–14]andsemanticparsing[15–20]assumethetasks
are DSL-closed – for any task, there exists a program, written in a predefined Domain Specific
Language(DSL),thatsolvesthetask. TheARCbenchmarkisuniquelydesignedtobeDSL-open
⇤and denoteequalcontributions
†
36thConferenceonNeuralInformationProcessingSystems(NeurIPS2022)TrackonDatasetsandBenchmarks.

Figure1: FourARCtasks,thegoalistocorrectlyinfertheunseenoutputfromthegivenexamples.
–itdoesnotcomewithapredefinedDSLcapableofrepresentingitstasksintuitively. Thisisboth
reasonable–mostreallifetasks,suchascookingandassemblingfurniture,areDSL-open–and
challenging–howcanonebuildanintelligentsystemthatcansolvetasksfromfewexampleswithout
aDSL?Toillustrate,whatmightaDSLthatwouldallowonetoprogramalltheARCtasksinFigure
1looklike? Thisquestionisdifficulttoanswer;arecentKagglecompetitionfoundthatthebestAI
systemssolveatmost20%ofthetasks,while[21]foundthatmosthumanseasilysolveover80%2.
Given that humans greatly outperform the best AI systems in solving ARC tasks, studying the
human’scognitiveprocesses(forinstance,whichsetofconceptsdohumanusetorepresentthese
tasks?) canshedlightonhowtobuildsimilarlyintelligentsystems. Asthesethoughtprocessesare
notobservabledirectly,westudynaturalprograms–instructionsthathumansgivetoeachother,as
awindowintotheselatentcognitiveprocesses. Likecomputerprograms,theseinstructionscanbe
reliablyinterpreted(byanotherhuman)toproducetheintendedoutput. Unlikecomputerprograms,
whichmustbestatedinaspecificstyle,naturalprogramscanbestatedinanyform–suchasverbal
instructionsorinput-outputexamples–aslongasanotherhumancanexecutethem. Inthiswork,
westudyaparticularformofnaturalprograms,thatofnaturallanguageinstructions. Weshowthat
analyzingthesenaturalprograms–withexplicitcomparisonstocomputerprograms–canbothshed
lightonhowhumanscommunicateandinterpretprocedures[22–25]andinformhowonemaybuild
AIsystemsforchallenging,DSL-opendomainssuchasARC.
Figure2: FourLARCtasks,correspondingtothoseofFigure1. Thegoalistoproducethecorrect
outputgivenonlythelanguageinstructions. 88%oftheARCtaskscanbecommunicatedthisway.
Whataresomeofthecommunicativestrategiesusedbyhumanshere?
We present the Language-complete Abstraction and Reasoning Corpus (LARC) 3 of natural
languageinstructionselicitedfromatwo-playercommunicationgame,where88%oftheARCtasks
canbesuccessfullycommunicated. LARCtasksarelanguage-complete: Thesuccessfulinstructions
containalltherelevantinformation,eveninabsenceoftheoriginalinput-outputexamples(seeFigure
2). Thisisimportantinseveralways: First,onecanuseLARCtostudyhowhumansuselanguageto
communicateabstractprocedures,ashumansclearlyhavethecapacitytobothgenerateandexecute
2Humanswereevaluatedonasubsetofthetrainingtasks;theKagglecompetitionusedaprivatetestset.
3https://github.com/samacqua/LARC
2

Figure3: Threekindsof“programs”: instruction(top-left),programming(top-right),synthesis(bot).
thesenaturalprograms;Second,onecandirectlyseewhatconceptsanintelligentsystemmustbe
awareof(suchascolorsandnumbers);Third,aspeoplereadilygeneratenaturalprograms,studying
themwillprovideinsightsonbuildinginteractivesystems.
WeperformlinguisticanalysisonLARC,findingthathumansreadilyleveragealgorithmicconcepts
withoutbeingexplicitlyinstructedtodoso. Theseconceptsrangefromdomaingeneralones,such
asloops,todomain-specificconceptssuchasflood-fill. However,naturalprogramsinLARCare
distinctfromtypicalcomputerprogramsintwoways: (1)naturalprogramsuseamuchwiderrange
ofconceptscomparedtoatypicalDSL;(2)naturalprogramscontainclarificationsandvalidations
in greater quantity than directly executable procedures. We apply standard program synthesis
algorithmsonLARC,findingthatwhileexistingapproachescanbenefitfromtheadditionallanguage
annotations,thetwoaforementioneddistinctionsposesignificantchallengestostandardprogram
synthesis approaches. We conclude by providing concrete suggestions on how to build the next
generationprogramsynthesizers.
2 CommunicatingandInterpretingPrograms
Inprogramming,aprogrammerconstructsaprograminasuitablelanguage,whichisthenexecuted
onaninterpreter,producingabehaviour. Forinstance,apersoncaninstructanotherpersontocarry
outacertaintask(Fig. 3top-left),ordirectlyprogramamachinetosolvetasksusingcode(Fig. 3
top-right). Aprogramsynthesizertakesinaninstruction,andreformulatesitascode,insulatingthe
personfromtheprogrammingprocess(Fig. 3bot). Wetreatallthreeasactsofprogramming.
Howdowebuildsystemsthatcanbecommunicatednaturallytosolvechallengingtasks? Typically,
one follows a “DSL-first” approach, where one first defines a programming language and builds
a corresponding interpreter capable of executing programs written in this language. Then, one
naturalizes the initial DSL using synthesis, allowing end-users to describe tasks using natural
language [15–18,26,27], or by giving examples [12,13,28]. While this DSL-first workflow has
yieldedimpressiveresults, theDSLitselfisalsoasinglepointoffailure. Itisdifficulttodesign
DSLwiththerightscope,sothatitbothexpressiveandnon-redundant[29–31]. Onemustensure
thattheDSLalignsreasonablytohumaninstructions[32,33],whilesimultaneouslybeingefficient
whenusedbythesynthesizer[12,34]. ThesechallengesmayexplainwhyARC,andotherDSL-open
domains(whereproceduraltasksaregivenintheabsenceofanarrowDSL),aredifficulttotackle.
In this work, we adopt the Wizard-of-Oz approach [35–37] by using a human as an interpreter
of natural language instructions (Fig 3 top-left). We define a natural program as instructions
constructedbyapersonthatcanbeinterpretedbyanotherpersontoproduceaspecificoutput. This
programisnatural–itcanbeunderstoodbyspeakersofthelanguage4withoutapriorconsensus–but
behavesasaprogram,inthatitproducesadefinitiveoutput,whichcanbeunambiguouslychecked
forcorrectness. Forinstance, theoriginalARC[4]tasksarenaturalprograms: Givenaprogram
consistingofinput-outputexamples,afellowhumancanreadilyinterpretthisprogramtoproducean
4languagehereistobeunderstoodlooselyasanymediumofcommunicationbetweenpeople
3

outputonanewinput,whichcanbecheckedforcorrectness. Bystartingwith(linguistic)natural
programs,onecandirectlyobservethesetofconceptsandstrategiesnecessarytomasteradomain
(suchasARC),withoutcommittingtoaspecificinterpreter.
Figure4: adescriberinstructsabuilderhowtosolveanARCtaskusinganaturalprogram
3 LARC:Language-completeAbstractionandReasoningCorpus
WepresentadatasetthataugmentstheoriginalARCtasksfrom[4]withlanguage-completeinstruc-
tions:theycanbedemonstrablyinterpretedbyotherhumanstocorrectlyproducetheintendedoutputs
withoutanyadditionalcontexts(i.e. intheabsenceoftheoriginalinput-outputexamples). Thus,
LARCtasks(Fig2),liketheircounterpartsinARC,meetthedefinitionofnaturalprogramwhile
containingonlynaturallanguagedescriptions. Tocollectthisdataset,weintroduceacommunication
game:humandescribersproducelinguisticinstructionsfromthegiveninput-outputexamplesofARC,
theseinstructionsaretheninterpretedbyhumanbuilders(intheabsenceoftheoriginalinput-output)
on a new instance of the same task (Fig. 4). We deployed this experiment using a novel bandit
algorithmtoefficientlycollectverifiablenaturalprograms. Thefinaldatasetaugments88%ofthe
originalARCtasks(354/400)withatleastoneverifiablenaturalprogramdescriptionthatcould
besuccessfullyinterpretedbyanotherhumanparticipanttosolvethetask. Fig.5(C-D)showsthe
distributionofsuccessratesforparticipantsactingasdescribersandbuildersovertime.
3.1 Humanannotationdetails
Werecruited373subjectsviaAmazonMechanicalTurkwhowerepaidfor45minutesofwork. Fifty
individualswereexcludedforfailingtocompletethetask,sothefinalanalysisincluded323subjects.
Thestudywasapprovedbyourinstitution’sInstitutionalReviewBoard,didnotcollectpersonally
identifiableinformation,anddidnotposeriskstoparticipants. Subjectswerepaid$6.00anda$0.25
bonusforeverysuccessfulcommunication. Subjectsaveraged5.5communications,bringingtheir
expectedhourlywageto$9.83. ForinterfaceandconsentformseeAppendixA.2. 5
3.2 Two-playercommunicationgame
Foreachtask,aparticipantmaybeassignedoneoftworoles:adescriberorabuilder. Thedescriber
playstheroleofahumansynthesizer,whoreformulatesinput-outputexamples(ofARC)tonatural
language descriptions. The builder plays a role of a human interpreter, who must construct the
correctoutputonanewinputwithoutaccesstotheoriginalexamples(Fig4). Thedescriptionis
structuredintothreesectionstoincentivizeconsistency: (1)whatthebuildershouldexpecttosee
intheinput,(2)theoutputgridsize,and(3)whatthebuildershoulddotocreatetheoutput(Fig2).
Afterthedescriptionwassubmitted,weverifythedescriber’sownunderstandingbyaskingthemto
buildit,anddiscardingthesubmissionifthedescriberfails. Thedescriberwasshownallprevious
verified descriptions for a task, alleviating challenge of solving the task from scratch. Builders
construct/drawtheoutputusingactionsdefinedinARC,suchaspaint(color,x,y),copy/paste,
andfloodfill. Alldrawingsequencesarerecordedandcanbeplayedback.
3.3 TheBanditAlgorithmforDataCollection
Collectingvalidlinguisticnaturalprogramsrequiressignificanthumanefforts: Foreachtask(of
varyingdifficulties),naturalprogramsmustfirstbeproposedbyanumberofdescribers,andthen
5seehttps://arxiv.org/abs/2106.07824forfullpaperwithappendixattachedattheend
4

Figure5: A.Describerimprovesatverifyingtheirowndescriptionsasatheydescribemoretasks.B.Builders
donotimproveatconstructingthecorrectoutputsastheybuildmoretasks(likelyduetohavingnocontrolover
thequalitiesoftheirgivendescriptions).C.Rateofdescribersverifyingtheirowndescriptions(avg75%).D.
Therateofbuildersconstructingthecorrectoutput,(avg50%).
1500
1000
500
0
EMAN
RUOLOC
no fo erauqs dirg REBMUN roloc emas htiw dna ni ot tupni llec nrettap MxN era/si deroloc enil taht NUONORP epahs ezis kcolb uoy dluohs hcae neht pot tfel ypoc fi thgir ekam llif egnahc ro dnuorgkcab lliw tuptuo lla tnereffid ht-N morf rehto renroc wor mottob ton
words
tnuoc
Figure6: Wordsusedinsuccessfullybuiltdescriptions,sortedbytheirfrequencyinthecorpus(total642
uniquewords).Thewordsweresingularized.Colorsnames,numbers,andpronounsweregroupedtogether.
validated byanumberofbuilders,wherebothcanmakemistakes. Thus,Anaivedata-collection
processthatsimplycollectsafixednumberofdescriptionsandbuildspertaskwillbeexpensive. To
addressthischallenge,weformulatethefollowingbanditproblem: multi-bandit–eachofthe400
ARCtasksisadifferentbandit;infinite-arm–givenatask,eachnaturallanguagedescription(there
areinfinitelymany)isadifferentarm;best-armidentification–onceanaturalprogramisproposed,
wemustvalidateit. Wedevelopanovelbanditalgorithm(AppendixB)tosolvethisproblem,asto
ourbestknowledge,noknownbanditalgorithmcanbedirectlyapplied. ForeachMTurkparticipant,
ourbanditalgorithmdynamicallyallocatesasetofdescribingandbuildingeffortsfortheirsession.
Asaresult,theLARCdatasetwasannotatedfor$3667,whereasanaivelycollecting20annotations
pertaskwouldcostatleast$10,800.
4 CommunicationStrategiesinNaturalPrograms
Whataresomestrategieshumansusetoproducerobustlyinterpretableinstructions? Toanswerthis
question,wecuratealinguisticallytaggeddatasetoftaggedphrasesfromsuccessfuldescriptions
underthelensofcomputerprograms. Weannotatethesephraseswithtagscorrespondingtogeneral
conceptsfromalgorithmsandcoreknowledge[38].Intotal,wemanuallylabel532randomlysampled
phrases(22%ofthephrasecorpus)using17conceptualtags(inwhichmultipletagscanbeapplied
toeachphrase);Figure 7A.showsafrequencyofthesetags. FordetailsseeAppendixA.3.
4.1 SimilaritiesofComputerandNaturalPrograms
GeneralAlgorithmicConcepts LARCcontainsalgorithmicconceptssimilartothosefoundina
typicalprogramminglanguage(i.e. python). Forinstance,tag_logicisabooleancheck(i.e. “thebox
isblue”),tag_arrayreferencesasetofsimilarobjects(i.e. “youshouldseefourredshapes”),and
tag_loopissimilartoloops(“keepgoinguntil”). Humansgeneratetheseconceptswithoutbeing
directlyinstructedtodoso,suggestingthathumansreasonaboutARCtasksalgorithmically.
DomainSpecificConcepts SimilartoacomputerDSL,LARCcontainsconceptsthatdistinguish
itfromotherdomains. Wefocusontheobjectsystemofcoreknowledge[38],definedbycohesion,
5

Figure7: A.Thefrequenciesofalltagsoccurringinhumanphrases. Eachphrasecanhavemultipletags.
B.Morethanhalfofthephrasesdescribedobjects,ofwhich,75%describedspatialrelations. C.Relative
frequenciesofcode(procedures)againstnon-code(example,framing,clarification,validation). D.Relative
frequenciesofcoreknowledgetopicsinphrasesthatreferencedobjects.
persistence,andinfluenceviacontact,whichtheARCcorpuswasexplicitlydesignedtoleverage.
Wefindabouthalfofthephrasesreferencedobjects,andthreequartersofthesedescribedspatial
relations (Fig.7B). Majority of operations on objects (Fig.7D) are visual_graphical_transform
whereasonly5%ofarephysical_interaction. Presumably,graphicaltransformationsareeasierto
representintheinput-outputformatofARC.
4.2 DifferencesofComputerandNaturalPrograms
Weoutlinetwo(related)waysnaturalprogramsdifferfromcomputerprograms.First,insteadofusing
anarrowDSLwithfewprimitives,naturalprogramsusealarge,diversesetofprimitivefunctions.
Second,insteadofstatingapreciseprocedureverbatim,naturalprogramsrelyonarangeofadditional
strategiestoensurethattheycanbeinterpretedprecisely.
NaturalProgramsInvokeaLargeNumberofConcepts SinceLARCislanguage-complete,
analyzingthewordsusedinLARCservesasagoodproxyfortheunderlyingconceptspresentinthe
ARCdomain. Similarto[21],wefindthathumansuseawiderangeofconcepts(Fig6). Thisisa
testamentofthegeneralcapabilitiesofthehumaninterpreter: thedescribersreadilyinvokethese
conceptsfromthebuilders,withtheconfidencethattheycanbecorrectlyinterpreted. Givenalarge
numberofconcepts,effectivelyindicatingthesetofrelevantconcepts(foragiventask)becomes
nontrivial: Whilehumandescribersandbuilderscanmakeuseofgenericwordsuchas‘bumpinto’,
computer programmers must be extremely careful in selecting the exact concept using a precise
language(i.e. move_until_touches_block).
NaturalProgramsCommunicateInformationBeyondProcedures Westudytherelativefre-
quenciesofdirectlyexecutablecommands,tag_procedure,incontrasttonotdirectlyexecutablemeta
informationsuchastag_framing–commentsaboutwhichconceptsarerelevant,tag_validation–
checkstoensurecorrectexecution,andtag_clarifications–restatingthesameprocedureindifferent
words. Themoststrikingfindingisthatprocedure,framing,andvalidationoccuratroughlythesame
frequency(seeFig.7C).Incontrast,only14%ofthecodesarecommented[39].
The high frequency of framing tags suggests that describers anticipate the large number of con-
ceptsthatthebuildercanoperateover,andcarefullyframetheinstructiontoinvoketheappropriate
ones. Thedescriberoftenassumesthedirectlyexecutableportion(i.e. tag_procedure)asinherently
ambiguous,assuggestedbyfrequentuseoftag_validationsandtag_clarificationsfollowingthese
procedures. Specifically,validationgivesachecktothebuildertotestiftheircurrentinterpretation
is correct. Clarification amends the initial ambiguous explanation with another explanation, nar-
rowingthenumberofpossibleinterpretations. Theseareevidencesthat,unlikecommunicationin
computerprogramsoveranarrowandunambiguousDSL,communicationinnaturalprogramsare
fundamentallyexpressiveyetambiguous,requiringextraeffortstomaintainprecision.
6

5 ExecutingNaturalProgramsusingProgramSynthesis
WeevaluatewhethercurrentDSL-firstprogramsynthesismethods(Fig3,bot)canexecutenatural
programs as well as humans do. We consider three kinds of natural programs: (1) Input-output
examples from the original ARC corpus (IO); (2) IO in conjunction with successful language
instructionsinLARC(IO+NL);And(3)languagealone(NL-only)–sameastheMTurkbuildertask.
5.1 ProgramSynthesis
In(symbolic)programsynthesis[14,19,40],thesynthesizertakesinanaturalprogram,andreformu-
latesitascodeoveraDSLtobeexecuted. WehavemanuallycraftedaDSLbasedlooselyonthe
conceptspresentintheLARCcorpusandbuiltitscorrespondinginterpreter(seeAppendixA.4)6.
Wepresentourbestsynthesisresultshere. Foradditionalmodels(usingaCNNencoder,asequence
decoder[19])seeA.5. PreliminarystudieswithcodexandclipseeA.6andA.7.
GenerateandCheckUsingIO IfthegivennaturalprogramcontainsIOexamples,thestandard
symbolicprogramsynthesisapproach[13,14]followsthegenerateandcheckstrategy.Letnatprogbe
anaturalprogram,thesynthesizerreturnsprogramsprogfromaDSLfromthefollowingdistribution:
P (prog natprog) P (prog natprog)1[prog IO]
synth gen
| / | `
P is the generative distribution: given a natural program, it proposes program prog from the
gen
DSL.1[prog IO]isthechecker: itvalidatesprogbyexecutingitontheinterpreter,ensuringthat
`
prog(x) = y forallinput-outputpairs(x,y) IO. Thekeystrengthofthisapproachliesinits
2
generalizability: IfaproposedprogramcanbecheckedagainstallIOexamples,itisverylikelyto
generalizetoannewinstanceofthesametaskduetotheinductivebiasoftheDSL.
GenerationModels OurP gen (prog natprog)generatesprogramsintwoparts: aneuralmodel
|
outputs a tree bigram over the grammar of the DSL [41], then a dedicated Ocaml enumerator
deterministically enumerates programs from a probabilistic context free grammar fitted to this
bigramdistributionindecreasingprobability[34]. Forsimplicity,wereportresultsofunconditioned
generatorsP (prog)(i.e. afittedprior)whenlanguageisabsent,andlanguage-conditionedmodels
gen
P (prog NL)whenlanguageispresent. Thisway,wecanusethesameP (prog NL)model
gen gen
| |
forbothIO+NLandNL-onlytasksinthetestset,asitdoesnotdependonIO.Similarto[42,43],we
firstbootstrapourgenerativemodelswith10“seed”programs,discovereduninformedenumeration.
LeveragingLanguage Weuseapre-trainedmodel(T5,[44])torepresentlanguagebytakingan
averageofitsencodedtokens. Toencouragethelearningofcompositionalrelationshipsbetween
languageandprogram,weusepseudo-annotation,similartorecentmethodsthathaveleveraged
synchronousgrammars[18,33,43,45]. First, weprovidelinguisticcommentsforeachprimitive
function in the program DSL (e.g. flood_fill(color) with fill with the color). Then, during
training,weobtainadditionalpairedlanguageandprogramexamplesbysubstitutingprimitivesof
artificialprogramswiththeircorrespondingcomments7. FormoreexamplesseeAppendixA.4.
DistantSupervision LARC,similartoSCONE[46],fallsunderthechallengeofdistantsupervision:
eachtrainingtaskonlycontainsthecorrectoutput,butnottheground-truthprogramresponsiblefor
generatingit. Weadopttheiterativeapproachusedin[19,34,42,43]todiscoversuitableprograms
duringthetrainingphase,byalternatively(1)generatingalargesampleofprogramsusingP and
gen
(2)fittingabetterP fromgoodprogramsinthegeneratedsamples.
gen
5.2 Results
Wesplitthe400tasksinto200trainingtasks(withorwithoutvalidlanguagedescriptions)and183
testingtasks(theremaining200filteredforhavingvalidlanguagedeceptions). Wethentrainthe
modelsfor10hourseachusingiterativelearning. Wetestonthe183testtasksbyfirstusingthe
neuralmodeltoproposeabigrampertask,thenenumeratingthebigramfor720seconds. Wekeep
6thisisatremendousengineeringeffort,consistingof103primitivescomparedto33ofSCONE
7for instance, (lambda (to_original_grid_overlay (remove_color(grid_to_block x)
yellow) false)) becomesplaceblockoninputgridremovecolorfromblockyellow
7

trainingtasksdiscovered
no-pseudo pseudo
IO 15/200 -
IO+NL 13/200 21/200
testingtaskssolved
no-pseudo pseudo
NL-only 1/183 0/183
IO 18/183 -
IO+NL 16/183 22/183
Table1: Executingdifferentkindsof Figure8:Numberoftesttaskssolvedforthethree
natural programs (IO – Input-output kindsofnaturalprograms,IO,NL,IO+NL,with
examplesfromtheoriginalARCcor- andwithoutpseudoannotations,asafunctionof
pus,IO+NL–IOinconjunctionwith enumeration time. There are error bars as the
successful language instructions in bigramenumeratorisdeterministic. Itispossible
LARC,NL-only–sameastheMTurk (butnotlikely)thatre-trainingthesemodelswill
buildertask)usingprogramsynthesis. haveaneffectduetotherandomnessofsampling
Here,"pseudo"meanstheNLtraining pseudo-annotated programs. All models vastly
hasbeenpre-trainedongeneratedsyn- underperformswhencomparedtoahuman,but
thetic language to code pairs. Train naturalprogramsconsistingofNL+IOfairsbest.
tasksdiscoveredunderdistantsupervi-
sion(top). Testtaskssolved(bot).
thetop-3mostlikelyprogramsthatalsosatisfytheIOexamplesifthenaturalprogramcontainsIO.
Wethencheckif anyof thetop3programs satisfiestestinput-output. SeeTable1and Figure 8.
Overall,weconcludethatwhilelanguagedefinitelyhelpscurrentapproaches,theoverallresults(best
12%)arestillcomicallybad.
QuantitativeFindings IO+NL+psuedoperformsbest, solving22/183ofthetestingtasks. We
believethisduetopsuedo-annotationbeingabletogenerateaninfinitenumber(albeitlowquality)of
artificialNL-progpairs. Wenotethathavingtheabilitytocheckifaproposedprogramiscorrect
underIOiscrucialforthesuccessofcurrentprogramsynthesizers,withnomorethan1tasksolved
withNL-only. LikethevalidationphrasesinLARC,theinput-outputexamplesinIOserveasaform
ofvalidationfortheenumerativesynthesizer. Thisfindingcorroborateswith[47].
QualitativeFindings Weinvestigateinwhatwaydoeslanguageaffectsynthesis.Foreachprimitive
inourDSL,weaskhowmanytimesmorelikelyisitgoingtoappearincorrectprogramsgenerated
withthelanguage-conditionedbigramvstheunconditionedone. Weplotthisratioonalogscalefor
allprimitivesthatwereusedinground-truthprograms,seeFigure9. Wenotethatformostofthe
frequentlyusedprimitives,thelanguage-conditionedgeneratorismorelikelytogeneratethecorrect
primitivesthantheunconditionedgenerator.
5.3 Challenges
Thebiggestchallengeisscoping. SinceLARCisDSL-open,wewereinaviciouscycleofconstantly
adding more primitives and refactoring the DSL. Even now, we cannot guarantee our DSL can
represent all LARC tasks. Second challenge is referencing: with 103 primitives, selecting the
relevantprimitivesbecomescrucial8. Finally,currentNL-to-codeapproaches–liketheoneswe
used–assumeaclose,1-to-1paraphrase-likemappingbetweenlanguageandprocedure,which
misinterpretcrucialframingandvalidationstatementsthatoccursinabundanceinLARC.
8ifwecanmagicallyselect10,thesearchspaceis105insteadof1035foraprogramoflength5
8

Figure9: Relativeoddsofusingthecorrectprimitiveforatask,language-conditionedvsuncondi-
tionedgeneration. Numberinparenthesisdenotesthetotalnumberoftimesaprimitiveisused.
6 RelatedWorks
Taskorienteddialoguesystems LARCasadatasetbelongstothefamilyoftaskorienteddialogue
systems[35–37,48,49]. OnecanviewnaturalprogramsinLARCasasingle-turn,task-oriented
dialogue,wherethedescribergivesanaturalisticinstructionwithaspecific,check-abletaskinmind.
Further, LARC uses the Wizard-of-Oz style of data collection – leveraging a human interpreter
withoutcommittingtobuildingaworkingsystem–acommonframeworktocollectdataindialogue
systems. LARCdiffersfromtheseexistingdatasetsmainlyinthediversityofitstasks(Section4)
whichcontainawiderangeofabstractconceptsratherthanbeinglimitedtospecificdomainssuchas
databasemanipulations[37,49].
Embodiedinstructionfollowing Embodiedinstructionfollowingconsistingofanembodiedagent
(oftenaavatarinavideogame)beingabletocarryoutasequenceofcommandswhenpromptedwith
naturallanguageinstructions[27,50–53]. Thesecommandscanoftenbehierarchical[17,50,53],
whicharenaturallyrepresentedasprograms. LARCagaindiffersfromtheseworksduetotherange
ofabstractconcepts,whereasaforementionedworkstypicallyfollowsaDSL-closedassumption.
As a result of a narrower range of concepts, a paraphrasal strategy that simply translate natural
languageintocodehasbeenfairlysuccessfulinpriorworksthataimtobuildaninstructionfollowing
system[27,49,51]. LARC gives strongevidence thatadditionalgroundingstrategies needtobe
modeledtotrulycapturetherichnessofnaturallanguageinstructions(forinstance,considertheset
ofstrategiesusedinFig2).
7 ConclusionandFutureWorks
We present LARC, a DSL-open yet Language-complete dataset, highlighting the difference of
between human-to-human and human-to-machines communications. By annotating successful
communications(datasetoflinguistically-tagged-phrases),wefindthathumanscommunicateusing
awiderangeofconceptsandcommunicativestrategies,whicharedifficulttointerpretusingexisting
techniques. WehopeLARCcanhelpdifferentcommunities(AI,ProgrammingLanguage,Cognitive
Science, etc) understand and build intelligent, communicative systems. Specifically, we believe
that defining concepts upfront (DSL-first) is not scalable. Instead, they should be learned and
taught(byend-users). Tofullyharnessthepowerofnaturallanguage,weneedtolookbeyondthe
simplisticnotionthatlanguagehavinga1-1relationshipwithdirectexecution,andentertaindifferent
communicativestrategies[54]. Webelievedatasets[51,55,56]thatsharetheproperties–namely,
DSL-openandlanguage-complete–arecrucialtobridgingthegapsbetweenhuman-humanand
human-machinecommunications. Lastly,itwillbebeneficialtoadaptfoundationalmodels[57–59]–
withsomeconventionalunderstandingsoflanguage,vision,andcode–towardsspecificdomains.
9

LimitationsandPotentialNegativeImpacts LARCconsistsofasingle,constrainedtaskformat
inahighlycontrolledsetting. Thelong-termgoalofthisworkisto‘reverse-engineer’howhumans
thinkandcommunicate,andsuchsystemsraiseconcernsregardingvaluealignmentsofusers,for
instance,non-expertsoperatingsafety-criticalequipmentusingnaturallanguage.
References
[1] DavidSilver,ThomasHubert,JulianSchrittwieser,IoannisAntonoglou,MatthewLai,ArthurGuez,Marc
Lanctot,LaurentSifre,DharshanKumaran,ThoreGraepel,etal. Masteringchessandshogibyself-play
withageneralreinforcementlearningalgorithm. arXivpreprintarXiv:1712.01815,2017.
[2] AdamLerer,HengyuanHu,JakobFoerster,andNoamBrown.Improvingpoliciesviasearchincooperative
partiallyobservablegames. InProceedingsoftheAAAIConferenceonArtificialIntelligence,volume34,
pages7187–7194,2020.
[3] BrendenMLake,TomerDUllman,JoshuaBTenenbaum,andSamuelJGershman. Buildingmachines
thatlearnandthinklikepeople. Behavioralandbrainsciences,40,2017.
[4] FrançoisChollet. Onthemeasureofintelligence. arXivpreprintarXiv:1911.01547,2019.
[5] MicheleneTHChi,RobertGlaser,andMarshallJFarr. Thenatureofexpertise. PsychologyPress,2014.
[6] HarryFHarlow. Theformationoflearningsets. Psychologicalreview,56(1):51,1949.
[7] BrendenMLakeandStevenTPiantadosi. Peopleinferrecursivevisualconceptsfromjustafewexamples.
ComputationalBrain&Behavior,3(1):54–65,2020.
[8] FredericCharlesBartlett. Remembering: Astudyinexperimentalandsocialpsychology. Cambridge
UniversityPress,1932.
[9] LucasTian,KevinEllis,MartaKryven,andJoshTenenbaum. Learningabstractstructurefordrawingby
efficientmotorprograminduction. AdvancesinNeuralInformationProcessingSystems,33,2020.
[10] TaniaLombrozo.Thestructureandfunctionofexplanations.Trendsincognitivesciences,10(10):464–470,
2006.
[11] EmilioParisotto,Abdel-rahmanMohamed,RishabhSingh,LihongLi,DengyongZhou,andPushmeet
Kohli. Neuro-symbolicprogramsynthesis. arXivpreprintarXiv:1611.01855,2016.
[12] KevinEllis,MaxwellNye,YewenPu,FelixSosa,JoshTenenbaum,andArmandoSolar-Lezama. Write,
execute,assess:Programsynthesiswitharepl. InAdvancesinNeuralInformationProcessingSystems,
pages9165–9174,2019.
[13] ArmandoSolar-Lezama,LiviuTancau,RastislavBodik,SanjitSeshia,andVijaySaraswat. Combinatorial
sketchingforfiniteprograms. InACMSigplanNotices,volume41,pages404–415.ACM,2006.
[14] JacobDevlin,JonathanUesato,SuryaBhupatiraju,RishabhSingh,Abdel-rahmanMohamed,andPushmeet
Kohli. Robustfill:Neuralprogramlearningundernoisyi/o. ICML,2017.
[15] YoavArtziandLukeZettlemoyer.Weaklysupervisedlearningofsemanticparsersformappinginstructions
toactions. TransactionsoftheAssociationforComputationalLinguistics,1:49–62,2013.
[16] XiYe,QiaochuChen,IsilDillig,andGregDurrett. Optimalneuralprogramsynthesisfrommultimodal
specifications. arXivpreprintarXiv:2010.01678,2020.
[17] YushiWang,JonathanBerant,andPercyLiang. Buildingasemanticparserovernight. InProceedingsof
the53rdAnnualMeetingoftheAssociationforComputationalLinguisticsandthe7thInternationalJoint
ConferenceonNaturalLanguageProcessing(Volume1:LongPapers),pages1332–1342,2015.
[18] AlanaMarzoev,SamuelMadden,MFransKaashoek,MichaelCafarella,andJacobAndreas. Unnatural
language processing: Bridging the gap between synthetic and natural language data. arXiv preprint
arXiv:2004.13645,2020.
[19] KelvinGuu,PanupongPasupat,EvanZheranLiu,andPercyLiang. Fromlanguagetoprograms:Bridging
reinforcementlearningandmaximummarginallikelihood. arXivpreprintarXiv:1704.07926,2017.
[20] SumithKulal,PanupongPasupat,KartikChandra,MinaLee,OdedPadon,AlexAiken,andPercyLiang.
Spoc:Search-basedpseudocodetocode. arXivpreprintarXiv:1906.04908,2019.
[21] AysjaJohnson, WaiKeenVong, BrendenMLake, andToddMGureckis. Fastandflexible: Human
programinductioninabstractreasoningtasks. arXivpreprintarXiv:2103.05823,2021.
[22] ElizabethSSpelke,KarenBreinlinger,JanetMacomber,andKristenJacobson. Originsofknowledge.
Psychologicalreview,99(4):605,1992.
[23] W.McCarthy,R.D.Hawkins,C.Holdaway,H.Wang,andJFan. Learningtocommunicateaboutshared
proceduralabstractions. InProceedingsofthe43rdAnnualConferenceoftheCognitive ScienceSociety,
2021.
10

[24] Herbert H Clark, Robert Schreuder, and Samuel Buttrick. Common ground at the understanding of
demonstrativereference. Journalofverballearningandverbalbehavior,22(2):245–258,1983.
[25] HerbertHClarkandDeannaWilkes-Gibbs. Referringasacollaborativeprocess. Cognition,22(1):1–39,
1986.
[26] YoavArtzi,DipanjanDas,andSlavPetrov. Learningcompactlexiconsforccgsemanticparsing. 2014.
[27] SidaIWang,PercyLiang,andChristopherDManning. Learninglanguagegamesthroughinteraction.
arXivpreprintarXiv:1606.02447,2016.
[28] YewenPu,KevinEllis,MartaKryven,JoshTenenbaum,andArmandoSolar-Lezama. Programsynthesis
withpragmaticcommunication. AdvancesinNeuralInformationProcessingSystems,33,2020.
[29] SumitGulwani,JoséHernández-Orallo,EmanuelKitzelmann,StephenHMuggleton,UteSchmid,and
BenjaminZorn. Inductiveprogrammingmeetstherealworld. CommunicationsoftheACM,58(11):90–99,
2015.
[30] BobbyRBruce,TianyiZhang,JaspreetArora,GuoqingHarryXu,andMiryungKim. Jshrink:in-depth
investigationintodebloatingmodernjavaapplications. InProceedingsofthe28thACMJointMeetingon
EuropeanSoftwareEngineeringConferenceandSymposiumontheFoundationsofSoftwareEngineering,
pages135–146,2020.
[31] CésarSoto-Valero,NicolasHarrand,MartinMonperrus,andBenoitBaudry. Acomprehensivestudyof
bloateddependenciesinthemavenecosystem. EmpiricalSoftwareEngineering,26(3):1–44,2021.
[32] PercyLiang. Learningexecutablesemanticparsersfornaturallanguageunderstanding. Communications
oftheACM,59(9):68–76,2016.
[33] Richard Shin, Christopher H Lin, Sam Thomson, Charles Chen, Subhro Roy, Emmanouil Antonios
Platanios,AdamPauls,DanKlein,JasonEisner,andBenjaminVanDurme. Constrainedlanguagemodels
yieldfew-shotsemanticparsers. arXivpreprintarXiv:2104.08768,2021.
[34] KevinEllis,CatherineWong,MaxwellNye,MathiasSable-Meyer,LucCary,LucasMorales,LukeHewitt,
ArmandoSolar-Lezama,andJoshuaBTenenbaum. Dreamcoder:Growinggeneralizable,interpretable
knowledgewithwake-sleepbayesianprogramlearning. arXivpreprintarXiv:2006.08381,2020.
[35] JohnFKelley. Aniterativedesignmethodologyforuser-friendlynaturallanguageofficeinformation
applications. ACMTransactionsonInformationSystems(TOIS),2(1):26–41,1984.
[36] PawełBudzianowski,Tsung-HsienWen,Bo-HsiangTseng,InigoCasanueva,StefanUltes,OsmanRa-
madan,andMilicaGašic´. Multiwoz–alarge-scalemulti-domainwizard-of-ozdatasetfortask-oriented
dialoguemodelling. arXivpreprintarXiv:1810.00278,2018.
[37] Tsung-HsienWen,DavidVandyke,NikolaMrksic,MilicaGasic,LinaMRojas-Barahona,Pei-HaoSu,
StefanUltes,andSteveYoung.Anetwork-basedend-to-endtrainabletask-orienteddialoguesystem.arXiv
preprintarXiv:1604.04562,2016.
[38] ElizabethS.SpelkeandKatherineD.Kinzler. Coreknowledge. DevelopmentalScience,10(1):89–96,
2007.
[39] YuanHuang,NanJia,JunhuaiShu,XinyuHu,XiangpingChen,andQiangZhou. Doesyourcodeneed
comment? Software:PracticeandExperience,50(3):227–245,2020.
[40] ArmandoSolarLezama. ProgramSynthesisBySketching. PhDthesis,2008.
[41] KarimLariandSteveJYoung.Theestimationofstochasticcontext-freegrammarsusingtheinside-outside
algorithm. Computerspeech&language,4(1):35–56,1990.
[42] EyalDechter,JonMalmaud,RyanPAdams,andJoshuaBTenenbaum. Bootstraplearningviamodular
conceptdiscovery. InTwenty-ThirdInternationalJointConferenceonArtificialIntelligence,2013.
[43] WongCatherine,LevinEllis,JacobAndreas,andJoshuaTenenbaum. Leveragingnaturallanguagefor
programsearchandabstractionlearning. Thirty-eighthInternationalConferenceonMachineLearning,
2021.
[44] ColinRaffel,NoamShazeer,AdamRoberts,KatherineLee,SharanNarang,MichaelMatena,YanqiZhou,
WeiLi,andPeterJLiu. Exploringthelimitsoftransferlearningwithaunifiedtext-to-texttransformer.
arXivpreprintarXiv:1910.10683,2019.
[45] Robin Jia and Percy Liang. Data recombination for neural semantic parsing. arXiv preprint
arXiv:1606.03622,2016.
[46] ReginaldLong,PanupongPasupat,andPercyLiang. Simplercontext-dependentlogicalformsviamodel
projections. arXivpreprintarXiv:1606.05378,2016.
[47] JacobAustin,AugustusOdena,MaxwellNye,MaartenBosma,HenrykMichalewski,DavidDohan,Ellen
Jiang,CarrieCai,MichaelTerry,QuocLe,etal. Programsynthesiswithlargelanguagemodels. arXiv
preprintarXiv:2108.07732,2021.
11

[48] Shin’yaNakajimaandJamesFAllen. Astudyonprosodyanddiscoursestructureincooperativedialogues.
Phonetica,50(3):197–210,1993.
[49] JacobAndreas,JohnBufe,DavidBurkett,CharlesChen,JoshClausman,JeanCrawford,KateCrim,Jordan
DeLoach,LeahDorner,JasonEisner,etal. Task-orienteddialogueasdataflowsynthesis. Transactionsof
theAssociationforComputationalLinguistics,8:556–571,2020.
[50] RyanVolum,SudhaRao,MichaelXu,GabrielADesGarennes,ChrisBrockett,BenjaminVanDurme,
OliviaDeng,AkankshaMalhotra,andBillDolan. Craftanironsword:Dynamicallygeneratinginteractive
gamecharactersbypromptinglargelanguagemodelstunedoncode. InTheThirdWordplay: When
LanguageMeetsGamesWorkshop,2022.
[51] AlaneSuhr,ClaudiaYan,JacobSchluger,StanleyYu,HadiKhader,MarwaMouallem,IrisZhang,and
YoavArtzi. Executinginstructionsinsituatedcollaborativeinteractions. arXivpreprintarXiv:1910.03655,
2019.
[52] PeterAnderson,QiWu,DamienTeney,JakeBruce,MarkJohnson,NikoSünderhauf,IanReid,Stephen
Gould, andAntonVanDenHengel. Vision-and-languagenavigation: Interpretingvisually-grounded
navigationinstructionsinrealenvironments. InProceedingsoftheIEEEconferenceoncomputervision
andpatternrecognition,pages3674–3683,2018.
[53] PratyushaSharma,AntonioTorralba,andJacobAndreas.Skillinductionandplanningwithlatentlanguage.
arXivpreprintarXiv:2110.01517,2021.
[54] Theodore R Sumers, Mark K Ho, Robert D Hawkins, Karthik Narasimhan, and Thomas L Griffiths.
Learningrewardsfromlinguisticfeedback. arXivpreprintarXiv:2009.14715,2020.
[55] Royi Lachmy, Valentina Pyatkin, and Reut Tsarfaty. Draw me a flower: Grounding formal abstract
structuresstatedininformalnaturallanguage. arXivpreprintarXiv:2106.14321,2021.
[56] AnjaliNarayan-Chen,PrashantJayannavar,andJuliaHockenmaier. Collaborativedialogueinminecraft.
In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages
5405–5415,2019.
[57] MarkChen,JerryTworek,HeewooJun,QimingYuan,HenriquePonde,JaredKaplan,HarriEdwards,
YuraBurda,NicholasJoseph,GregBrockman,etal. Evaluatinglargelanguagemodelstrainedoncode.
arXivpreprintarXiv:2107.03374,2021.
[58] Jean-BaptisteAlayrac,JeffDonahue,PaulineLuc,AntoineMiech,IainBarr,YanaHasson,KarelLenc,
ArthurMensch,KatieMillican,MalcolmReynolds,etal. Flamingo:avisuallanguagemodelforfew-shot
learning. arXivpreprintarXiv:2204.14198,2022.
[59] AdityaRamesh,PrafullaDhariwal,AlexNichol,CaseyChu,andMarkChen.Hierarchicaltext-conditional
imagegenerationwithcliplatents. arXivpreprintarXiv:2204.06125,2022.
[60] AlecRadford,JongWookKim,ChrisHallacy,AdityaRamesh,GabrielGoh,SandhiniAgarwal,Girish
Sastry,AmandaAskell,PamelaMishkin,JackClark,etal. Learningtransferablevisualmodelsfrom
naturallanguagesupervision.InInternationalConferenceonMachineLearning,pages8748–8763.PMLR,
2021.
[61] YizaoWang,Jean-YvesAudibert,andRémiMunos.Infinitelymany-armedbandits.InAdvancesinNeural
InformationProcessingSystems,2008.
Checklist
1. Forallauthors...
(a) Dothemainclaimsmadeintheabstractandintroductionaccuratelyreflectthepaper’s
contributionsandscope? [Yes]
(b) Didyoudescribethelimitationsofyourwork? [Yes]
(c) Didyoudiscussanypotentialnegativesocietalimpactsofyourwork? [Yes]
(d) Haveyoureadtheethicsreviewguidelinesandensuredthatyourpaperconformsto
them? [Yes]
2. Ifyouareincludingtheoreticalresults...
(a) Didyoustatethefullsetofassumptionsofalltheoreticalresults? [N/A]
(b) Didyouincludecompleteproofsofalltheoreticalresults? [N/A]
3. Ifyouranexperiments(e.g. forbenchmarks)...
(a) Didyouincludethecode,data,andinstructionsneededtoreproducethemainexperi-
mentalresults(eitherinthesupplementalmaterialorasaURL)?[Yes]SeeSupplement
12

(b) Didyouspecifyallthetrainingdetails(e.g.,datasplits,hyperparameters,howthey
werechosen)? [Yes]SeeSupplement
(c) Didyoureporterrorbars(e.g.,withrespecttotherandomseedafterrunningexper-
imentsmultipletimes)? [No]Eachsynthesisstudytakes30hoursandisexpensive,
andwearenotexpresslyclaimingresultsoftheform“ourapproachisgood”butonly
providingsuggestionsonwhatmay/maynotwork
(d) Didyouincludethetotalamountofcomputeandthetypeofresourcesused(e.g.,type
ofGPUs,internalcluster,orcloudprovider)? [Yes]SeeSupplement
4. Ifyouareusingexistingassets(e.g.,code,data,models)orcurating/releasingnewassets...
(a) Ifyourworkusesexistingassets,didyoucitethecreators? [Yes]ARC[4]
(b) Didyoumentionthelicenseoftheassets? [Yes]SeeSupplement
(c) DidyouincludeanynewassetseitherinthesupplementalmaterialorasaURL?[Yes]
SeeSupplement
(d) Didyoudiscusswhetherandhowconsentwasobtainedfrompeoplewhosedatayou’re
using/curating? [Yes]SeeSupplement
(e) Didyoudiscusswhetherthedatayouareusing/curatingcontainspersonallyidentifiable
informationoroffensivecontent? [Yes]SeeSupplement. Wedonotcollectpersonally
identifiableorsensitiveinformation
5. Ifyouusedcrowdsourcingorconductedresearchwithhumansubjects...
(a) Didyouincludethefulltextofinstructionsgiventoparticipantsandscreenshots,if
applicable? [Yes]SeeSupplement
(b) Did you describe any potential participant risks, with links to Institutional Review
Board(IRB)approvals,ifapplicable? [Yes]SeeSupplement
(c) Didyouincludetheestimatedhourlywagepaidtoparticipantsandthetotalamount
spentonparticipantcompensation? [Yes]SeeSupplement
Acknowlegements
The authors would like to thank Eric Lu for inspiring the wonderful communication game that
catalyzedourwork.
13