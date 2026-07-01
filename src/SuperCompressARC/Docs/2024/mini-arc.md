Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small
Transformer Models
Paul Fletcher-Hill
November 12, 2024
Abstract
easy ones and 400 hard ones), but the competition
evaluates each submission against 100 secret puzzles
The Abstraction and Reasoning Corpus (ARC)[3] is a that may or may not share concepts with the publicly
benchmark to test the ability of artificial systems to available puzzles.
adapt, learn on the fly, and quickly acquire new skills Many of the most successful systems thus far have
in a human-like way. The benchmark has proven chal- made use of fine-tuned LLMs and a combination of
lenging for even the most advanced language models test-time training and program synthesis to try to
to solve. In this paper, I explain a novel approach to solve puzzles. Two teams recently beat 50% on the
solving ARC puzzles that uses (1) small (67M param) 2024ARCPrizeleaderboardusingsimilarapproaches.
Transformer models trained exclusively on ARC puz-
zles, (2) test-time training (TTT), and (3) refinement.
1.2 Motivation
I demonstrate that a system combining these three
strategies is able to solve 41% of puzzles from a sub-
While considerable progress is being made with the
set of the public ARC evaluation dataset that fit the
approaches described above, I wanted to see if similar
model dimensions. This result is notable because it
performancecouldbeachievedwithoutusinglanguage
is achieved with a very small model and without the
models at all. Intuitively, I do not "think" in terms
use of search, language models, or program synthesis.
of language when solving ARC puzzles myself, so I
Code is available at: https://github.com/
wondered whether an artificial system could reason
pfletcherhill/mini-arc
and identify patterns without language too.
1 Introduction
2 Related Work
1.1 The Abstraction and Reasoning
2.1 Test-Time Training (TTT)
Corpus (ARC)
Test-time training (TTT) is a meta learning strategy
The Abstraction and Reasoning Corpus (ARC)[3] is
being used by multiple groups to quickly adapt pre-
a benchmark published in 2019 by Francois Chol-
trained models for new puzzles when they see them.
let which aims to test models’ ability to reason and
MindsAI, the group currently holding the top spot on
efficiently learn new patterns. The benchmark is com-
the 2024 ARC Prize leaderboard, has written about
posed of 2D puzzles that are relatively simple for
test-time training[4] and their use of it.
humans to solve but which have stumped even the
most advanced LLMs. Additionally, a recent paper[1] demonstrates that
ARCpuzzlesarestructuredasalistofinput/output test-timetrainingcanimproveperformancebyafactor
grids,eachofwhichdemonstratesomesortoftransfor- of 6 when using a fine-tuned 8B parameter LLM to
mation. The goal is to infer the transformation from solveARCpuzzles. Thisgroupachieved53%accuracy
thelistofinput/outputgridsandthenapplythesame on the ARC public evaluation dataset.
transformationtoanewinputgrid. Thebenchmarkis
verydiverseintermsofthetransformations,including 2.2 2D Transformer Models
concepts like counting, gravity, rotation, and more.
The ARC Prize[2] is a competition to build an arti- Li et al.[11] tried using Vision Transformer models
ficial system that solves not-seen-before ARC puzzles. to solve ARC puzzles. The group showed that a
There are 800 publicly available ARC puzzles (400 vanilla Vision Transformer failed to solve most ARC
1

Figure1: Mini-ARCsolvesARCpuzzlesusingcustomizedsmallTransformermodels. Thesearerealpredicted
outputs for two ARC puzzles as they progress through the Transformer encoder layers.
3 Mini-ARC
| puzzles       | but that     | a custom | positional | encoding   | scheme |                |           |              |            |             |             |           |
| ------------- | ------------ | -------- | ---------- | ---------- | ------ | -------------- | --------- | ------------ | ---------- | ----------- | ----------- | --------- |
| improved      | performance. |          |            |            |        |                |           |              |            |             |             |           |
|               |              |          |            |            |        | 3.1 Model      |           | Architecture |            |             |             |           |
|               |              |          |            |            |        | At the core    | of        | Mini-ARC     | is a       | specialized | Transformer |           |
|               |              |          |            |            |        | model          | designed  | for          | processing | ARC         | puzzles.    | The       |
|               |              |          |            |            |        | model consists |           | of:          |            |             |             |           |
| 2.3 Synthetic |              | ARC      | Puzzle     | Generation |        |                |           |              |            |             |             |           |
|               |              |          |            |            |        | 1. An          | embedding | layer        | to convert | discrete    |             | colors to |
vectors
| Efficient | generation | of new | ARC | puzzle | for training |     |     |     |     |     |     |     |
| --------- | ---------- | ------ | --- | ------ | ------------ | --- | --- | --- | --- | --- | --- | --- |
purposes has been critical. It’s very tricky to come 2. A custom positional encoding scheme that repre-
|            |             |                 |          |     |              | sents | a cell’s   | 2-D     | position | as well | as it’s | position |
| ---------- | ----------- | --------------- | -------- | --- | ------------ | ----- | ---------- | ------- | -------- | ------- | ------- | -------- |
| up with    | new puzzles | and             | generate | new | puzzles that |       |            |         |          |         |         |          |
| are seeded | from        | existing public | ones.    |     |              | in    | the larger | context |          |         |         |          |
Michael Hodel published RE-ARC[7], which in- 3. A stack of Transformer self-attention encoder
| cludes | a domain-specific |     | language | (DSL) | for ARC | layers |     |     |     |     |     |     |
| ------ | ----------------- | --- | -------- | ----- | ------- | ------ | --- | --- | --- | --- | --- | --- |
puzzlesaswellasgeneratorandsolverprogramsusing
thatDSLforthe400ARCPublicTrainingSetpuzzles. 4. Afinallayertoprojecttheoutputbacktodiscrete
colors
| RE-ARC   | is an extremely | helpful      |          | resource | for anyone |                                 |     |     |     |     |             |     |
| -------- | --------------- | ------------ | -------- | -------- | ---------- | ------------------------------- | --- | --- | --- | --- | ----------- | --- |
| training | models          | to solve ARC | puzzles. |          |            |                                 |     |     |     |     |             |     |
|          |                 |              |          |          |            | Itrainedtwomodelsforevaluation: |     |     |     |     | Mini-ARC-12 |     |
Recently, another group published BARC[10], and Mini-ARC-v12. See Table 1 for the specifica-
which includes further Python programs to gener- tions and hyperparameters used for each model. The
ate more ARC puzzles. The BARC Heavy dataset [9] onlydifferencebetweenthetwoisthatMini-ARC-v12
includes 200k puzzles, each with many example input uses a modified embedding scheme which compresses
and output grids to construct tasks from. each 12x12 grid into a 6x6 grid using 2x2 patches[5],
2

Figure 2: Here is an illustration of how ARC Puzzle cfb2ce5a is encoded into a sequence along with a
| placeholder | output   | grid and passed       | through | the Mini-ARC model. |                      |     |     |     |
| ----------- | -------- | --------------------- | ------- | ------------------- | -------------------- | --- | --- | --- |
|             |          |                       |         | 3.1.1               | Input Representation |     |     |     |
|             | Table 1: | Model Hyperparameters |         |                     |                      |     |     |     |
Specification Mini-ARC-12 Mini-ARC-v12 In order to maximize in-context learning, an entire
|                  |      |            |            | puzzle, | including all training | input             | and output | grids      |
| ---------------- | ---- | ---------- | ---------- | ------- | ---------------------- | ----------------- | ---------- | ---------- |
| Max grid         | dim. | 12x12      | 12x12      |         |                        |                   |            |            |
|                  |      |            |            | as well | as the test input      | grid, is included | in         | the input. |
| Total parameters |      | 67,320,715 | 67,343,755 |         |                        |                   |            |            |
Sequence len. 1,440 468 While ARC puzzles have many context pairs and
|           |      |       |       | grids can | be from 1x1    | to 30x30, I  | chose to  | limit the |
| --------- | ---- | ----- | ----- | --------- | -------------- | ------------ | --------- | --------- |
| Embedding | dim. | 512   | 512   |           |                |              |           |           |
|           |      |       |       | sequence  | length for the | sake of more | efficient | experi-   |
| FFN dim.  |      | 3,072 | 3,072 |           |                |              |           |           |
Encoder layers 16 16 mentation. Mini-ARC-12 and Mini-ARC-v12 expect
|           |       |     |     | the input | to be nine      | 12x12 grids:   | four  | input and |
| --------- | ----- | --- | --- | --------- | --------------- | -------------- | ----- | --------- |
| Attention | heads | 16  | 16  |           |                 |                |       |           |
|           |       |     |     | output    | train grids and | one test input | grid. |           |
Allgridsarepaddedto12x12usingapaddingtoken
|     |     |     |     | (0) and       | all missing training | pairs       | are padded | with      |
| --- | --- | --- | --- | ------------- | -------------------- | ----------- | ---------- | --------- |
|     |     |     |     | 12x12 grids   | as well. Since       | the padding | token      | is 0, the |
|     |     |     |     | color classes | are all bumped       | up by       | one before | being     |
encoded.
which is a common technique for Vision Transform- The input tokens are flattened to make an input
ers—thus the "v" in the model name. Therefore, sequence of all nine grids and a 12x12 output grid is
the input sequence for Mini-ARC-v12 is considerably added to the end. By default the placeholder output
shorter than the un-patched input sequence for Mini- gridisalearnedparameter,butthemodelsalsoaccept
| ARC-12. |     |     |     | a custom | starting output | grid as an | argument | to the |
| ------- | --- | --- | --- | -------- | --------------- | ---------- | -------- | ------ |
3

forward pass. See Figure 2 for an illustration of how 3.2.1 ARC-HTML
the full sequence is constructed and Table 1 for the
In addition to the RE-ARC and BARC puzzles, I
specific sequence lengths for each model.
wanted to generate puzzles similar in complexity to
the ARC Evaluation Set. I initially tried writing
3.1.2 Embedding and Positional Encoding
Python functions for each puzzle in the ARC Public
The sequence is embedded into 512-dimensional space Evaluation Set, but that proved extremely tedious.
using a learned embedding parameter. Each token is I then wrote out descriptions of some of the trans-
augmented with a positional encoding that includes: formations represented in the puzzles in English and
(1) its grid row, (2) its grid column, (3) whether its tried prompting LLMs to write Python programs for
part of an input or output grid, and (4) which grid me. I even provided the RE-ARC DSL to LLMs and
pairitsapartof. Themodelhasalearnedembedding triedhavingthemusethat. NoneofthePython-based
for each of those four attributes. approaches proved successful or efficient. Either the
LLM-generatedprogramsrequiredheavyeditsorthey
3.1.3 Attention and Masking didn’t work at all.
Eventually, I tried prompting LLMs to generate
The full embedded sequence is passed through 16
HTML documents for each puzzle. I thought that the
Transformer encoder layers with self-attention mecha-
transformations and patterns included in the puzzles
nisms. Apaddingmaskpreventspaddingtokensfrom
might be easier for LLMs to write with HTML, CSS,
attending to other tokens. And a causal attention
andJavascript,becausethey’reoftenprojectionsof3D
mask prevents the input sequence from attending to
objects in a 2D space, similar to HTML pages. While
the output sequence. Input tokens can attend to any
it was still burdensome to write English descriptions
tokens in the input sequence, and output tokens can
of the puzzles, I wrote 40 descriptions and ChatGPT
attend to all tokens—both the input and output.
and Claude wrote extensive HTML documents for
each one. I prompted them with a sample HTML
3.2 Training Data and Synthetic Data document including containers for the input/output
Generation grids. Then I wrote a script to load each HTML
document, takeanimageofthewebpage, andparseit
The ARC benchmark includes two datasets: the Pub-
pixel-by-pixel to turn each into ARC puzzles. Specific
lic Training Set with 400 easy puzzles and the Public
prompts ensured that the HTML documents kept the
Evaluation Set with 400 hard puzzles. While the
grids at certain sizes so that I could scrape them
point of the ARC Prize is to focus on generalization,
consistently.
I needed more data in order to train the Mini-ARC
Fromthe40HTMLdocuments,Igenerated360,000
models.
puzzles. Though only 16,117 of those from 30 HTML
Iendedupwithatrainingdatasetof830,648puzzles
documents fit inside the 12x12 grid limitation I im-
and an evaluation dataset of 167,880 puzzles. All of
posed for training Mini-ARC. The 40 HTML docu-
thesepuzzleshave4orfewertrainingpairsanddonot
ments and prompts are available on Github.
include any grids larger than 12x12. While the ARC
public datasets differ in complexity, the training and
evaluation datasets I used for training are the same 3.3 Training
difficulty and complexity. The evaluation dataset is
Training of the Mini-ARC models was done using
just a random subset of the total dataset that was
supervised learning on 4-8 A100 GPUs on Modal[8]
kept out of training.
over multiple days. Both Mini-ARC-12 and Mini-
The dataset is a combination of three sources:
ARC-v12 were trained for at least 150,000 steps with
1. 290,025 RE-ARC puzzles, which represent the effective batch sizes ranging from 32 to 192.
400 patterns and transformations from the ARC
Public Training Set
3.4 Test-Time Training
2. 524,506 BARC puzzles, which represent 200,000
In addition to the pre-trained models, I also set up
patterns and transformations from the BARC
a test-time training scheme where models could be
Heavy dataset
fine-tuned for each individual puzzle as they solve it.
3. 16,117 ARC-HTML puzzles, which represent 30 For each puzzle, I created a new dataset by sampling
patternsandtransformationsfromtheARCPub- pairs from the context input and output grid pairs in
lic Evaluation Set the puzzle. So a puzzle with 4 context pairs would
4

Figure 3: ARC-HTML is a prompting strategy for getting LLMs to generate HTML documents where each
| page load | yields         | a new       | ARC     | puzzle.   |       |           |           |     |     |     |     |     |     |
| --------- | -------------- | ----------- | ------- | --------- | ----- | --------- | --------- | --- | --- | --- | --- | --- | --- |
|           |                |             |         |           |       |           | 4 Results |     |     |     |     |     |     |
| generate  | 48 fine-tuning |             | puzzles | by taking | all   | the per-  |           |     |     |     |     |     |     |
| mutations | of every       | combination |         | of at     | least | 3 puzzles |           |     |     |     |     |     |     |
IevaluatedeachMini-ARCmodelandstrategyagainst
| from the | group. |     |     |     |     |     |                  |        |     |            |     |            |     |
| -------- | ------ | --- | --- | --- | --- | --- | ---------------- | ------ | --- | ---------- | --- | ---------- | --- |
|          |        |     |     |     |     |     | a representative | subset | of  | the public | ARC | Evaluation |     |
Attesttime,Itrainacopyofthepre-trainedmodel
|     |     |     |     |     |     |     | Dataset. | This version | of Mini-ARC |     | models | is  | limited |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | ----------- | --- | ------ | --- | ------- |
on the puzzle-specific dataset using supervised learn- to puzzles with grids up to 12x12 in size and up to 4
ing. Trainingproceedsuntileitheranaccuracycut-off
|             |            |     |      |             |     |       | pairs of training |            | pairs. Therefore, |     | I tested | against | a        |
| ----------- | ---------- | --- | ---- | ----------- | --- | ----- | ----------------- | ---------- | ----------------- | --- | -------- | ------- | -------- |
| is achieved | (typically |     | 99%) | or a number | of  | steps | is                |            |                   |     |          |         |          |
|             |            |     |      |             |     |       | subset of         | the public | evaluation        |     | dataset  | that    | fit that |
exceeded. The batch size, learning rate, number of criteria, which was 114 puzzles of the 400 available.
| steps, and | accuracy | cut-off | are | all tunable | arguments. |     |              |            |            |             |            |         |       |
| ---------- | -------- | ------- | --- | ----------- | ---------- | --- | ------------ | ---------- | ---------- | ----------- | ---------- | ------- | ----- |
|            |          |         |     |             |            |     | See Appendix | A          | for a list | of all      | 114 puzzle | IDs.    |       |
|            |          |         |     |             |            |     | I used three | metrics    | to         | evaluate    | each       | model:  |       |
|            |          |         |     |             |            |     | 1. Score     | - how      | many of    | the puzzles |            | did the | model |
|            |          |         |     |             |            |     | predict      | correctly? |            |             |            |         |       |
3.5 Refinement
|     |     |     |     |     |     |     | 2. Accuracy-howmanypixelsdidthemodelpredict |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | --- |
correctly?
TheMini-ARCmodelsweretrainedwithtwosettings:
predicting from scratch and refinement. The forward 3. Closeness-howmanyofthepuzzlesdidthemodel
|             |            |         |          |           |          |          | predict    | within | 95% accuracy? |       |       |             |     |
| ----------- | ---------- | ------- | -------- | --------- | -------- | -------- | ---------- | ------ | ------------- | ----- | ----- | ----------- | --- |
| pass of     | the models | accepts | an       | optional  | output   | argu-    |            |        |               |       |       |             |     |
| ment, which | if         | present | will     | be used   | as the   | starting |            |        |               |       |       |             |     |
|             |            |         |          |           |          |          | Each model | was    | evaluated     | using | three | strategies: |     |
| point for   | the output |         | grid. If | no output | argument |          | is         |        |               |       |       |             |     |
passed,theoutputgridissetfromalearnedparameter 1. Zero-shot prediction
on the model.
|     |     |     |     |     |     |     | 2. TTT | prediction | - TTT | for | up to | 15 epochs | with |
| --- | --- | --- | --- | --- | --- | --- | ------ | ---------- | ----- | --- | ----- | --------- | ---- |
During training, I set 25% of the training steps accuracy cut-off of 99.5%
| to focus    | on refinement. |     | In         | those steps, | I generated |          |         |           |            |               |       |     |          |
| ----------- | -------------- | --- | ---------- | ------------ | ----------- | -------- | ------- | --------- | ---------- | ------------- | ----- | --- | -------- |
|             |                |     |            |              |             |          | 3. TTT  | + Refined | prediction |               | - TTT | for | up to 15 |
| a partial   | solution       | to  | the puzzle | and          | fed it      | into the |         |           |            |               |       |     |          |
|             |                |     |            |              |             |          | epochs, | then      | 2 rounds   | of refinement |       |     |          |
| model along | with           | the | input.     | Partial      | solutions   | were     |         |           |            |               |       |     |          |
createdbyaddingvaryingamountsofnoisetothereal As you can see in Table 2, ARC-Mini-12 performs
outputs. The aim was to be able to refine a puzzle significantly better than ARC-Mini-v12 in all cate-
solution over multiple passes. gories. 90%+ accuracy was achieved by all strategies,
5

|     |     |           |           | Table     | 2: Mini-ARC | Performance |              |            |     |     |     |     |
| --- | --- | --------- | --------- | --------- | ----------- | ----------- | ------------ | ---------- | --- | --- | --- | --- |
|     |     | Metric    |           |           |             | Mini-ARC-12 | Mini-ARC-v12 |            |     |     |     |     |
|     |     | Zero-shot | Score     |           |             | 26 (22.8%)  |              | 11 (9.6%)  |     |     |     |     |
|     |     | Zero-shot | Accuracy  |           |             | 93.0%       |              | 90.7%      |     |     |     |     |
|     |     | Zero-shot | Closeness |           |             | 56 (49.1%)  |              | 44 (38.6%) |     |     |     |     |
|     |     | TTT       | Score     |           |             | 43 (37.7%)  |              | 17 (14.9%) |     |     |     |     |
|     |     | TTT       | Accuracy  |           |             | 95.0%       |              | 93.1%      |     |     |     |     |
|     |     | TTT       | Closeness |           |             | 78 (68.4%)  |              | 65 (57.0%) |     |     |     |     |
|     |     | TTT       | + Refined | Score     |             | 47 (41.2%)  |              | 20 (17.5%) |     |     |     |     |
|     |     | TTT       | + Refined | Accuracy  |             | 95.3%       |              | 93.3%      |     |     |     |     |
|     |     | TTT       | + Refined | Closeness |             | 80 (70.2%)  |              | 64 (56.1%) |     |     |     |     |
whichisimpressive,especiallyforthezero-shotpredic- mans—we are able to execute the transformations we
tions. Thoughkeepinmindthataccuracycalculations contemplatementallywithouthavingtowritePython
includepaddingtokensintheoutputgrids. Assuming programs to try them—but it is still a disadvantage
paddingtokensareeasiertopredictthanothertokens, relative to other approaches.
thisaccuracycalculationfavorspuzzleswithsmallout-
| put grids. | The combination |     | of test-time |     | training | and |     |     |     |     |     |     |
| ---------- | --------------- | --- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
refinement achieves the best result, solving 41.2% of 5.3 Future Work and Other Ideas
| puzzles | in the dataset. |     |     |     |     |             |          |           |        |           |             |     |
| ------- | --------------- | --- | --- | --- | --- | ----------- | -------- | --------- | ------ | --------- | ----------- | --- |
|         |                 |     |     |     |     | 5.3.1       | Scale    | Up Models |        | to Larger | Grids       |     |
|         |                 |     |     |     |     | The current | Mini-ARC |           | models | are       | constrained | by  |
5 Discussion
|          |         |     |     |     |     | their 12x12                               | grid     | size   | requirement. |      | In the future, | I      |
| -------- | ------- | --- | --- | --- | --- | ----------------------------------------- | -------- | ------ | ------------ | ---- | -------------- | ------ |
|          |         |     |     |     |     | would like                                | to train | larger | models       | with | the same       | archi- |
| 5.1 Data | Leakage |     |     |     |     |                                           |          |        |              |      |                |        |
|          |         |     |     |     |     | tecturetoevaluateagainstlargerARCpuzzles. |          |        |              |      |                | While  |
Because the ARC-HTML portion of the Mini-ARC Mini-ARC-12 performed better than Mini-ARC-v12
|          |             |         |      |         |        | at this size, | using | Vision | Transformer-style |     | patch | em- |
| -------- | ----------- | ------- | ---- | ------- | ------ | ------------- | ----- | ------ | ----------------- | --- | ----- | --- |
| training | dataset was | derived | from | puzzles | in the | ARC           |       |        |                   |     |       |     |
Public Evaluation Set, there is a concern about data bedding will be important as we scale up to keep the
|            |                   |     |               |     |             | sequence     | length | reasonable. |       | Without | using patches, |       |
| ---------- | ----------------- | --- | ------------- | --- | ----------- | ------------ | ------ | ----------- | ----- | ------- | -------------- | ----- |
| leakage    | when benchmarking |     | performance   |     | against     | the          |        |             |       |         |                |       |
|            |                   |     |               |     |             | the sequence | length | for         | 30x30 | puzzles | would be       | 9,000 |
| ARC Public | Evaluation        |     | Set. However, |     | the overlap |              |        |             |       |         |                |       |
betweenthepuzzlesusedtogeneratetheARC-HTML tokens, whichwouldrequireamuchlargermodel. For
|         |                 |     |            |         |            | comparison, | thecurrentcontextwindowforOpenAI’s |     |     |     |     |     |
| ------- | --------------- | --- | ---------- | ------- | ---------- | ----------- | ---------------------------------- | --- | --- | --- | --- | --- |
| dataset | and the puzzles |     | in the 114 | limited | evaluation |             |                                    |     |     |     |     |     |
datasetisonly10puzzles. Ididnotconsidergridsizes GPT-3.5 is 16,385 tokens.
whenpickingpuzzlesfortheARC-HTMLdataset,and
| many of  | them exceed | the | 12x12         | grid constraint |     | for   |          |     |                 |     |     |     |
| -------- | ----------- | --- | ------------- | --------------- | --- | ----- | -------- | --- | --------------- | --- | --- | --- |
|          |             |     |               |                 |     | 5.3.2 | ARC-HTML |     | Experimentation |     |     |     |
| training | Mini-ARC-12 | and | Mini-ARC-v12. |                 |     |       |          |     |                 |     |     |     |
Acrossthe10puzzlesthatdooverlap,theMini-ARC The ARC-HTML approach demonstrated that ad-
| performance | is similar |     | to the rest | of the | evaluation |             |     |         |     |             |             |     |
| ----------- | ---------- | --- | ----------- | ------ | ---------- | ----------- | --- | ------- | --- | ----------- | ----------- | --- |
|             |            |     |             |        |            | vanced LLMs | are | capable | of  | formalizing | ARC puzzles |     |
dataset, solving a maximum of 4 of the 10 puzzles in in HTML, CSS, and Javascript documents. In ad-
any of the performance metrics. See a full report of dition to prompting LLMs to follow instructions for
| the results | on these | 10 puzzles | in  | Appendix | C.  |          |          |           |        |               |                   |     |
| ----------- | -------- | ---------- | --- | -------- | --- | -------- | -------- | --------- | ------ | ------------- | ----------------- | --- |
|             |          |            |     |          |     | specific | puzzles, | we        | should | try prompting | them              | to  |
|             |          |            |     |          |     | generate | HTML     | documents |        | for new       | puzzles entirely. |     |
5.2 Limitations This strategy is similar to BARC[10], which cleverly
|     |     |     |     |     |     | prompted | LLMs | to generate |     | new puzzles | from | seed |
| --- | --- | --- | --- | --- | --- | -------- | ---- | ----------- | --- | ----------- | ---- | ---- |
One limitation of a system like Mini-ARC compared programs, where the seed programs represented a
to strategies that use LLMs is that Mini-ARC has to diverse set of grid transformations. But since ARC-
handle both reasoning and program execution, while HTML puzzles are HTML documents, in order to get
LLMs are able to write code to test their programs an even more diverse dataset, we could prompt LLMs
and offload computation to a computer. The combi- to modify ARC-HTML puzzles using all possible CSS
| nation of | these modes | in  | Mini-ARC | is similar | to  | hu- transformations. |     |     |     |     |     |     |
| --------- | ----------- | --- | -------- | ---------- | --- | -------------------- | --- | --- | --- | --- | --- | --- |
6

5.3.3 Meta Learning [7] Michael Hodel. Addressing the abstraction and
|                 |        |            |          |             |           |               | reasoning | corpus | via    | procedural | example |             | genera- |
| --------------- | ------ | ---------- | -------- | ----------- | --------- | ------------- | --------- | ------ | ------ | ---------- | ------- | ----------- | ------- |
| Since test-time |        | training   | has      | proven      | to        | be effective, |           |        |        |            |         |             |         |
|                 |        |            |          |             |           |               | tion,     | 2024.  |        |            |         |             |         |
| future work     | should | be         | done     | to evaluate | meta      | learning      |           |        |        |            |         |             |         |
| strategies[6].  |        | Currently, | training | and         | test-time | train-        |           |        |        |            |         |             |         |
|                 |        |            |          |             |           |               | [8] Modal | Labs.  | Modal: | Serverless | cloud   | infrastruc- |         |
ing occur independently, but theoretically we should ture for ai, ml, and data applications, 2024.
| be trying | to minimize |     | the loss | after | test-time | training |     |     |     |     |     |     |     |
| --------- | ----------- | --- | -------- | ----- | --------- | -------- | --- | --- | --- | --- | --- | --- | --- |
rather than zero-shot prediction, which may result in [9] Wen-Ding Li, Keya Hu, Carter Larsen, Yuqing
a different state for the model parameters. Wu,SimonAlford,CalebWoo,SpencerM.Dunn,
HaoTang,MichelangeloNaim,DatNguyen,Wei-
|     |     |     |     |     |     |     | Long | Zheng, | Zenna | Tavares, | Yewen | Pu, | and |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------ | ----- | -------- | ----- | --- | --- |
6 Conclusion
|     |     |     |     |     |     |     | Kevin | Ellis. | Barc heavy, | 2024. |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------ | ----------- | ----- | --- | --- | --- |
ThispaperintroducesMini-ARC,acollectionofsmall [10] Wen-Ding Li, Keya Hu, Carter Larsen, Yuqing
| Transformer | models | trained |     | on ARC | puzzles | as well |     |     |     |     |     |     |     |
| ----------- | ------ | ------- | --- | ------ | ------- | ------- | --- | --- | --- | --- | --- | --- | --- |
Wu,SimonAlford,CalebWoo,SpencerM.Dunn,
as test-time training (TTT) and refinement strate- HaoTang,MichelangeloNaim,DatNguyen,Wei-
gies for solving ARC puzzles. Using Mini-ARC-12, Long Zheng, Zenna Tavares, Yewen Pu, and
I show a best performance of 41% across a subset Kevin Ellis. Combining induction and transduc-
of the ARC evaluation dataset, which is filtered to tion for abstract reasoning, 2024.
| puzzles     | that | fit the Mini-ARC |         | grid   | size       | constraints. |             |     |        |     |       |         |     |
| ----------- | ---- | ---------------- | ------- | ------ | ---------- | ------------ | ----------- | --- | ------ | --- | ----- | ------- | --- |
|             |      |                  |         |        |            |              | [11] Wenhao | Li, | Yudong | Xu, | Scott | Sanner, | and |
| This result | is   | notable          | because | of the | relatively | small        |             |     |        |     |       |         |     |
size of the models (67M params) and because it was Elias Boutros Khalil. Tackling the abstraction
achieved without the use of language models, search, and reasoning corpus with vision transformers:
or program synthesis. the importance of 2d representation, positions,
|     |     |     |     |     |     |     | and objects, |     | 2024. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | --- | --- | --- | --- |
References
|          |          |       |     |         |       |          | A Mini-ARC |     |     |     | Evaluation |     |     |
| -------- | -------- | ----- | --- | ------- | ----- | -------- | ---------- | --- | --- | --- | ---------- | --- | --- |
| [1] Ekin | Akyürek, | Mehul |     | Damani, | Linlu | Qiu, Han |            |     |     |     |            |     |     |
Dataset
| Guo,YoonKim,andJacobAndreas. |     |     |     |     |     | Thesurpris- |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
ingeffectivenessoftest-timetrainingforabstract
Herearethe114puzzlesfromthepublicARCEvalua-
| reasoning, |     | 2024. |     |     |     |     |     |     |     |     |     |     |     |
| ---------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionSetthatfitwithintheMini-ARCsizeconstraints:
[2] Francois Chollet, Mike Knoop, Bryan Lan- 3b4c2228, fc754716, 5d2a5c43, e5790162, 4e469f39,
|       |      |          |     |          |      |        | 6ea4a07e, | bf32578f, | ef26cbf6, |     | ca8de6ea, | 5783df64, |     |
| ----- | ---- | -------- | --- | -------- | ---- | ------ | --------- | --------- | --------- | --- | --------- | --------- | --- |
| ders, | Greg | Kamradt, |     | Hansueli | Jud, | Walter |           |           |           |     |           |           |     |
Reade, and Addison Howard. Arc prize 9c56f360, d017b73f, 626c0bcc, c35c1b4c, c48954c1,
2024.https://kaggle.com/competitions/arc-prize- b15fca0b, 4acc7107, ac605cbb, f0afb749, c8b7cc0f,
|       |       |         |     |     |     |     | da2b0fe3, | ae58858e, | e99362f0, |     | 67c52801, | 66e6c45b, |     |
| ----- | ----- | ------- | --- | --- | --- | --- | --------- | --------- | --------- | --- | --------- | --------- | --- |
| 2024, | 2024. | Kaggle. |     |     |     |     |           |           |           |     |           |           |     |
|       |       |         |     |     |     |     | 48131b3c, | 2685904e, | 90347967, |     | a406ac07, | 60c09cac, |     |
[3] François Chollet. On the measure of intelligence, 332efdb3, b1fc8b8e, 506d28a5, dc2aa30b, 8fbca751,
2019.
|            |             |               |         |              |        |              | 17cae0c1, | e633a9e5, | ed74f2f2, |     | ecaa0ec1, | 68b67ca3, |     |
| ---------- | ----------- | ------------- | ------- | ------------ | ------ | ------------ | --------- | --------- | --------- | --- | --------- | --------- | --- |
|            |             |               |         |              |        |              | f45f5ca7, | cfb2ce5a, | 7ee1c6ea, |     | 48f8583b, | aa300dc3, |     |
| [4] Jack   | Cole        | and           | Mohamed | Osman.       |        | Dataset-     |           |           |           |     |           |           |     |
|            |             |               |         |              |        |              | 9f27f097, | 4cd1b7b2, | 31adaf00, |     | e345f17b, | 2072aba6, |     |
| induced    |             | meta-learning |         | (and other   |        | tricks): Im- |           |           |           |     |           |           |     |
|            |             |               |         |              |        |              | 9c1e755f, | f3e62deb, | c7d4e6ad, |     | a8610ef7, | 84db8fc4, |     |
| proving    | model       | efficiency    |         | on arc,      | 2023.  |              |           |           |           |     |           |           |     |
|            |             |               |         |              |        |              | 31d5ba1a, | 7953d61e, | bbb1b8b6, |     | 0692e18c, | 782b5218, |     |
|            |             |               |         |              |        |              | 0c786b71, | 575b1a71, | 2c737e39, |     | 94414823, | 137f0df0, |     |
| [5] Alexey |             | Dosovitskiy,  |         | Lucas        | Beyer, | Alexan-      |           |           |           |     |           |           |     |
|            |             |               |         |              |        |              | 6f473927, | 00576224, | a59b95c0, |     | b942fd60, | 4852f2fa, |     |
| der        | Kolesnikov, |               | Dirk    | Weissenborn, |        | Xiaohua      |           |           |           |     |           |           |     |
|            |             |               |         |              |        |              | 6ad5bdfd, | d19f7514, | 8b28cd80, |     | 27f8ce4f, | ea9794b1, |     |
| Zhai,      | Thomas      | Unterthiner,  |         | Mostafa      |        | Dehghani,    |           |           |           |     |           |           |     |
MatthiasMinderer,GeorgHeigold,SylvainGelly, 73182012, 917bccba, d2acf2cb, 8e2edd66, e5c44e8f,
|             |            |              |       |               |     |           | ce039d91, | 15696249, | f3cdc58f, | 73c3b0d8, |           | 34b99a2b, |     |
| ----------- | ---------- | ------------ | ----- | ------------- | --- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --- |
| Jakob       | Uszkoreit, |              | and   | Neil Houlsby. |     | An image  |           |           |           |           |           |           |     |
|             |            |              |       |               |     |           | b0722778, | e7dd8335, | 1acc24af, |           | e133d23d, | 69889d6e, |     |
| is          | worth      | 16x16 words: |       | Transformers  |     | for image |           |           |           |           |           |           |     |
|             |            |              |       |               |     |           | 9110e3c5, | 12eac192, | c074846d, |           | 64a7c07e, | 8ba14f53, |     |
| recognition |            | at scale,    | 2021. |               |     |           |           |           |           |           |           |           |     |
|             |            |              |       |               |     |           | e872b94a, | e6de6e8f, | 85fa5666, |           | 8597cfd7, | 7e02026e, |     |
[6] Chelsea Finn, Pieter Abbeel, and Sergey Levine. 32e9702f, 59341089, 03560426, 3979b1a8, aa18de87,
Model-agnostic meta-learning for fast adaptation af24b4cc, e69241bd, be03b35f, 27a77e38, 0becf7df,
of deep networks, 2017. 3d31c5b3, 7c8af763, 6df30ad6, ed98d772.
7

B ARC-HTML Prompt C ARC-HTML Data Leakage In-
vestigation Results
The10puzzlesthatarebothinthe114Mini-ARCeval-
HereisanexamplepromptusedforgeneratingHTML
uation dataset and the ARC-HTML 12x12 training
files from English descriptions of ARC puzzles via
dataset are: 0becf7df, 00576224, 0c786b71, 03560426,
LLMs:
137f0df0, 17cae0c1, 12eac192, 15696249, 332efdb3,
32e9702f. See Table 3 for performance on these 10
puzzles.
Update this HTML document using the
instructions below. Think step-by-
step and make sure the HTML is Table3: Mini-ARCDataLeakagePuzzlePerformance
correct. The name of the puzzle is
Metric Mini-ARC-12 Mini-ARC-v12
<Insert Puzzle ID>.
Zero-shot Score 3 (30.0%) 2 (20.0%)
<Insert HTML template here> Zero-shot Accuracy 93.0% 91.1%
Zero-shot Closeness 5 (50.0%) 3 (30.0%)
General Instructions:
- Change the HTML title to the name of TTT Score 4 (40.0%) 3 (30.0%)
the puzzle TTT Accuracy 91.2% 94.7%
- On each page load, pick a random TTT Closeness 5 (50.0%) 5 (50.0%)
number of pairs (between 2-5) and
add more pairs using the pattern TTT + Refined Score 4 (40.0%) 3 (30.0%)
already in the document for the TTT + Refined Accuracy 91.7% 94.9%
first two pairs. The number of TTT + Refined Closeness 5 (50.0%) 5 (50.0%)
pairs should be determined randomly
with each page load.
- Each pair should include two
containers, sized 30x30 pixels each
. Each container will contain an
input grid div and an output grid
div, which the puzzle instructions
will help you define. Do not change
the size of the containers, only
the sizes of the grids inside them.
- Do not modify the CSS for main, pair
, container classes. None of the
classes in the template should be
changed.
- Pick a background color for all the
grids, which should be black for
60% of puzzles
Instructions for puzzle <Insert Puzzle
ID>:
<Insert Puzzle Description>
Warnings:
- Do not use a canvas, because the
output will be blurry.
- Make sure your script does not cause
an infinite loop or cause the page
to crash when the HTML is loaded.
- Just return the HTML document for
saving in a file.
8