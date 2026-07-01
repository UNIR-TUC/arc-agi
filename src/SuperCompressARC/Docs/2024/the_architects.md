The LLM ARChitect: Solving ARC-AGI
Is A Matter of Perspective
Daniel Franzen∗†1, Jan Disselhoff∗†2, and David Hartmann†3
1dfranzen.it@gmail.com
2JanDissel.it@gmail.com
3Lambda, Inc., davidh@lambdal.com
Abstract
Largelanguagemodels(LLMs)havemadeimpressiveprogress,but
they still struggle with abstract reasoning tasks like the Abstraction
and Reasoning Corpus (ARC-AGI). Prior approaches have not been
able to achieve high scores on ARC-AGI, suggesting a gap between
current AI systems and human-level reasoning. In this work we ap-
proach the problem using tailored data augmentation techniques, op-
timizing the data format as well as training performance, and lever-
aging our generative model both as a predictor and as a classifier for
good solutions. We are able to maximize performance despite limited
computational resources, achieving 53.5 (56.51) points on the private
evaluation set during the Kaggle ARC Prize 2024 Contest. Addition-
ally, we are able to solve 72.5 out of 100 randomly split-off tasks from
the public evaluation set (while merging the other 300 tasks into our
training data). The code of our submission is publicly available in the
Kaggle competition2 and in the Github repository of this paper.3
∗Contributing equally to the Kaggle ARC Prize 2024.
†Contributing equally to this paper.
1Run finished after kaggle submission deadline
2Team “the ARChitects”: https://www.kaggle.com/competitions/arc-prize-2024
3Github Repository: https://github.com/da-fr/arc-prize-2024
1

1 Introduction
Large Language Models (LLMs) have demonstrated remarkable capabili-
ties across diverse tasks, from natural language processing to code gener-
ation. However, the fundamental question of whether these systems can
truly "reason" remains contentious in the artificial intelligence community.
TheAbstractionandReasoningCorpus(ARC-AGI)[1],designedspecifically
to evaluate genuine intelligence in AI systems, provides a stark illustration
of this challenge. Although the tasks appear simple to human test takers,
bothclassicalalgorithmicapproaches[2]andmodernneuralarchitectures[3]
have struggled to achieve high performance on ARC-AGI, painting a bleak
picture for artificial reasoning capabilities.
Yet, recent developments suggest this picture might be incomplete. The
rapid evolution of language models has produced increasingly capable sys-
tems at surprisingly small scales, as demonstrated by models like LLaMA-
3.2-3B [4] and Nvidia NeMo-Minitron-8B [5]. Moreover, growing evidence
suggests that many perceived limitations of these models may be artifacts
of implementation or limitations of language rather than fundamental ca-
pability gaps. For example, Allen-Zhu and Li [6] show that models often
have awareness of their mistakes, even when unable to correct them. Small
mistakes in data modelling can significantly inhibit finetuning performance,
not because the problem is too complex, but because a simpler structure
might be available for the model to learn [7]. Finally, a growing body of
research [8, 9, 10] establishes that apparent failures can frequently be traced
to tokenization issues rather than reasoning deficits.
It seems that models often possess the requisite capabilities but struggle to
access them effectively, leading to a counter-intuitive conclusion: The chal-
lenge for LLMs lies not in the absence of reasoning ability, but in creating
conditions that allow these capabilities to emerge.
Building on these insights, we developed an approach specifically tailored to
the ARC-AGI dataset. Our method solves 72.5 task out of 100 examples
of the public evaluation dataset and 56.5 points in a late submission to the
Kaggle competition, suggesting that efficient finetuning, proper tokenization
and tailored algorithmic support can indeed unlock the latent reasoning ca-
pabilities of these systems.
2 Pipeline Overview
Our approach focuses on efficiently fine-tuning a large language model to
solve the Abstraction and Reasoning Corpus (ARC-AGI) tasks. The ap-
proach can be separated into several distinct components: an expanded
dataset,secondaryfine-tuningattest-time,aninferencemethodoptimizedto
ARC-AGI and a heuristic candidate selection algorithm based on the predic-
2

Figure 1: High Level overview of our pipeline. We retrain an LLM on public ARC-AGI
data, which is then finetuned an additional time on the hidden test cases. Subsequently,
this model predicts several solution candidates from which we select two using an algo-
rithmic approach. All blue rectangles are calculated in the 12h timelimit of the kaggle
notebook. At several parts in our pipeline we "augment" (Aug) the data by applying
transformations to the examples.
tion scores. We also use data augmentations at train, inference and scoring,
substantially increasing our score. The key components of our pipeline are
outlined below:
Datasets:
In addition to the official ARC-AGI dataset, which is divided into “pub-
lic training”, “public evaluation”, and “private evaluation” subsets, we also
utilize the Re-ARC dataset by Hodel [11]. This extended dataset is de-
rived from the public training dataset using custom generators written in
a domain-specific language (DSL), allowing for the creation of additional,
diverse examples. Since the Re-ARC dataset is a superset of the public
ARC-AGI public training dataset, we choose to replace the public training
data with Re-ARC.
We additionally use Concept-ARC and ARC-Heavy [12, 13], which we intro-
duce in Section 3.1.
Data Representation:
In order to represent the ARC-AGI puzzles in a denser format, we modify
the tokenizer and embedding layers by reducing the number of tokens sig-
nificantly to only 64 symbols, allowing us to encode the data succinctly and
transparently for the LLM. More detailed information regarding our data
modeling approach can be found in Section 3.2.
Augmentation:
While data augmentations are commonly used to increase dataset sizes, our
3

approach uses augmentations at every part of the pipeline. Specifically, we
use rotations, transpositions, and permutations of the colors and the order
of examples, as these preserve the underlying task structure. We use these
augmentations not only to increase the amount of training data, but also to
increase the variety in our predictions and to provide a way to filter wrong
solutions. This contributes significantly to the performance, and will be
| discussed | more extensively |     | in Section | 3.3. |     |     |     |
| --------- | ---------------- | --- | ---------- | ---- | --- | --- | --- |
Models:
We use the augmented data to fine-tune decoder-only LLMs. Being con-
strained by Kaggle’s 2 Nvidia T4 GPUs, the models have to meet two main
requirements: a maximal memory usage of 16GB during training and infer-
ence,andaminimumcontextsizeof8000tokenstohandleinferenceonlarger
problems. We want to point out two models that worked particularly well
in our case: Mistral-NeMo-Minitron-8B-Base [5] and an uncensored version
| of Llama-3.2-3B-instruct |             | [14]. |              |     |     |     |     |
| ------------------------ | ----------- | ----- | ------------ | --- | --- | --- | --- |
| Preliminary              | & Secondary |       | Fine-Tuning: |     |     |     |     |
OurmodelsareinitiallytrainedonRe-ARCand75%oftheARC-AGIpublic
evaluation dataset, leveraging LoRA adapters [15] as a fine-tuning method.
Thefine-tunedmodelissubsequentlyuploadedtoKaggle,whereitundergoes
additionalfine-tuningonthehiddendataset. Formorecomprehensivedetails
| on training | and hyperparameters, |     |     | please refer | to Section | 3.4. |     |
| ----------- | -------------------- | --- | --- | ------------ | ---------- | ---- | --- |
| Candidate   | Generation:          |     |     |              |            |      |     |
After fine-tuning has completed, the final model is used to generate solution
candidates. We found that both greedy and multinomial sampling provided
suboptimal performance for the benchmark. Instead, we designed a cus-
tom generator that leverages depth-first search (DFS) to extract all possible
completions with a sampling probability exceeding a specified cutoff value.
This DFS approach is applied to both the original task as well as its "aug-
mented" versions, resulting in a list of candidates for each task. This not
onlyimprovedourscore, butalsotheinferencetimeneeded. Foranin-depth
| discussion | on inference | see | Section | 3.5. |     |     |     |
| ---------- | ------------ | --- | ------- | ---- | --- | --- | --- |
| Candidate  | Selection:   |     |         |      |     |     |     |
Finally, given the generated list of candidates, we use the aggregated log-
|         | assigned | by  | the fine-tuned |     | model to | select two of them | for |
| ------- | -------- | --- | -------------- | --- | -------- | ------------------ | --- |
| softmax | scores   |     |                |     |          |                    |     |
submission. To make sure that these values are informative we compute
them over several augmentations of the task. This selection algorithm is
highly effective, provided that the correct candidate is among those gener-
ated. For a detailed explanation of our candidate selection process, refer to
Section 3.6.
4

Model Baseline + Fine-tune + Candidate + AugScore + DFS
Llama-mix 21.0 35.5 55.5 57.5 63.5
Nemo-mix 26.0 40.5 57.5 69.0 72.5
Table 1: Performance on 100 randomly split-off examples of the ARC-AGI evaluation
dataset, adding parts of our pipeline. Baseline score shows performance of our network
after preliminary finetuning when taking two samples from generation. Fine-tune adds
test-timetrainingontheexamplesofthepuzzles. Candidateselectionusesaugmentation
and greedy sampling to generate a candidate for each of 8 different augmentation of the
task, using log softmax scores for selection. AugScore additionally uses the sum of the
log softmax probabilites from 8 different augmentations as score. Finally, DFS uses our
custom DFS scheme to find all candidates with a sampling probability larger than 10%,
at the same time increasing the augmentations during inference from 8 to 16 per task
(to reflect the speedups gained by the DFS sampling). Scores were measured on the 100
problems of the evaluation dataset that were not merged into the training set.
3 Methods
Our approach to solving ARC-AGI combines data expansion, multi-stage
fine-tuning of language models, and specialized solution evaluation. Below,
we explain how these components work together to improve the model’s
performance while keeping computational costs manageable.
3.1 Datasets
The Abstraction and Reasoning Corpus (ARC-AGI) introduced by Chollet
[1] challenges the idea that language models cannot effectively generalize
from a small number of examples, often referred to as few-shot prompting.
The original ARC-AGI dataset consists of 900 reasoning tasks, divided into
400 training tasks, 400 public evaluation tasks, and 100 private evaluation
tasks. Each task involves grids of varying sizes, ranging from 1x1 to 30x30,
utilizing a palette of ten distinct colors.
Importantly, each task contains only a hand full instances of the respective
problem, as illustrated in Figure 2(a). Each instance consists of two grids:
one representing the input of the problem and the other representing the
expected output. The objective is to infer the underlying mechanics from
a few examples and apply this understanding to a new, unseen instance as
illustrated in the figure.
Hodel[11]introducedthere-ARCdataset, whichre-implementsall400tasks
of the public training dataset. Their code can be used to generate an ar-
bitrary amount of training data for these tasks. An example of code and
generated data can be seen in Figure 2.
In addition to ARC-AGI and its re-implementation re-ARC, we make use
of Concept-ARC, which is a second dataset containing similar problems in-
5

|     | Figure 2: | Examples of | Re-ARC challenges | and the | corresponding | code. |
| --- | --------- | ----------- | ----------------- | ------- | ------------- | ----- |
volving "basic spatial and semantic concepts" from the same domain as the
| original | ARC-AGI | dataset. | It contains | 176 additional | tasks | [12]. |
| -------- | ------- | -------- | ----------- | -------------- | ----- | ----- |
Our best scoring model also included ARC-Heavy [13], which uses LLMs to
| generate | a large       | amount of | synthetic tasks. |     |     |     |
| -------- | ------------- | --------- | ---------------- | --- | --- | --- |
| 3.2      | Data Modeling |           |                  |     |     |     |
InordertoapplyLLMstoARC-AGIpuzzles,weneedtotokenizethedatain
a manner suitable for our model. This process requires careful consideration
| of two | main challenges: |     |     |     |     |     |
| ------ | ---------------- | --- | --- | --- | --- | --- |
First, due to the limited context size in typical LLM architectures, an in-
creaseofinferencetimeanddeclineinperformanceonlongcontexttasks[16],
we require a representation that minimizes the number of tokens the model
needs to process. Secondly, it is widely recognized that numerous common
failure modes in Large Language Models (LLMs) stem from tokenization [8,
9, 10]. For instance, standard tokenization techniques group numbers (some
but not all combinations) of one, two or three succeeding digits into dedi-
cated “grouped-digit tokens” [17]. These kinds of merges would complicate
| the | puzzles unnecessarily. |     |     |     |     |     |
| --- | ---------------------- | --- | --- | --- | --- | --- |
To address this, we opted to simplify the token set available to the model.
In particular, we reduced the number of tokens available from over 120.000
| to  | or less tokens | (see Table | 2). |     |     |     |
| --- | -------------- | ---------- | --- | --- | --- | --- |
64
6

| Token    | Category | Tokens |     |                 | Purpose  |            |         |           |
| -------- | -------- | ------ | --- | --------------- | -------- | ---------- | ------- | --------- |
| Alphabet |          | A-Z,   | a-z | (excl. I,O,i,o) | Learned  | pre-prompt |         | tokens    |
| Numbers  |          | 0-9    |     |                 | Encoding | the        | 10      | colors    |
| Newline  | token    |        |     |                 | Signals  | end        | of each | grid line |
\n
| Input/Output |                | I,      | O     |         | Signals          | start | of problem | input/output  |
| ------------ | -------------- | ------- | ----- | ------- | ---------------- | ----- | ---------- | ------------- |
| Begin        | token          | ⟨bos⟩   |       |         | Inserted         | once  | at         | the beginning |
| End token    |                | ⟨eos⟩   |       |         | Inserted         | after | each       | output        |
| Padding      | token          | ⟨pad⟩   |       |         | Internal         | usage | (e.g.      | batching)     |
|              | Table 2:       | Reduced | Token | Set for | ARC-AGI-specific |       | LLM        | Model         |
| Visual       | Representation |         | of    | a Task  | Instance:        |       |            |               |
| Compact      | String         | Format  | of    | same    | Instance:        |       |            |               |
a z
|     |     | <bos> | A   | ... Z | ... I | 2 1 | 1 0 1 | 2 1 \n |
| --- | --- | ----- | --- | ----- | ----- | --- | ----- | ------ |
|     |     |       |     |       |       | 1 2 | 1 0 2 | 2 2    |
\n
|     |     |     |     |     |     | 2 1 | 1 0 1 | 1 1 \n |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------ |
|     |     |     |     |     |     |     | O 1   | 1 1 \n |
|     |     |     |     |     |     |     | 1     | 3 1 \n |
|     |     |     |     |     |     |     | 1     | 1 1    |
\n
<eos>
Figure3: Ourstandardtokenizationapproach. Notethatweuseonetokenpercellinstead
ofcompressingtheproblemmore. Wealsotrytonotincludeanyunnecessarydelimiters.
The Pre-prompt(green) is only included for the first example. Depending on the model
andruntheremightbesomesmallchangestothePre-promptandInput/Outputtokens.
This reduction offers key benefits. It significantly decreases the model size,
as we can remove the majority of rows from the embedding layer. Further,
tokenmergesthattypicallyoccurduringtexttokenizationarenolongerpos-
sible. This ensures that the model can focus precisely on the data without
| the interference | of  | digit | separators. |     |     |     |     |     |
| ---------------- | --- | ----- | ----------- | --- | --- | --- | --- | --- |
AsillustratedinFigure3, weaddasmallnumberofextratokenstothestart
of a task. Surprisingly, this addition improves the model’s performance. We
believethatduringfine-tuning(wheretheembeddinglayersarealsotrained),
the model learns to use these extra tokens as a form of computational buffer,
which influences every subsequent token, thereby enhancing overall perfor-
mance.
We also experimented with adding extra tokens between input and output
7

Figure 4: Examples of augmented data. We generally allow rotations and transposition
of the examples,as well as permutations of color and shuffling of the order of examples.
of each problem instance. The idea here is to give the model time to in-
ternally process the input before having to generate the first token of the
output. However, we could not clearly determine if this improved model
performance.
3.3 Augmentation
Data augmentation has been a common approach in previous ARC-AGI
competitions. However, ourmethodextendsbeyondtraditionaldatasetaug-
mentation, applying transformations throughout our pipeline, both during
training and during inference. To formalize our approach, we first need to
| define | the structure | of ARC-AGI | puzzles. |     |     |     |
| ------ | ------------- | ---------- | -------- | --- | --- | --- |
A task, denoted as C (X ,Y ,X ,Y ,...,Xc,Yc), consists input grids X
|     |     | =   | 1 1 2 2 |     |     | i   |
| --- | --- | --- | ------- | --- | --- | --- |
paired with their corresponding output grids Y . The final pair, Xc and Yc,
i
represent the task grid input and the correct solution Yc. An augmen-
Xc
tation is a function that maps to another task while preserving
|     |     | T   | C   |     | T(C) |     |
| --- | --- | --- | --- | --- | ---- | --- |
the task’s essential structure. While formally defining “essential structure”
is difficult as it requires deep understanding of the task patterns, we can still
| identify | symmetries | of interest. |     |     |     |     |
| -------- | ---------- | ------------ | --- | --- | --- | --- |
Our transformations fall into three categories, as illustrated by an example
| in Figure | 4:  |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- |
• D symmetry operations (rotations and reflections) applied consis-
8
| tently | across | all grids | X ,Y |     |     |     |
| ------ | ------ | --------- | ---- | --- | --- | --- |
i i
• Color permutations applied uniformly throughout the task, with spe-
| cial | consideration | for | the background | color, | given by the token | 0 . |
| ---- | ------------- | --- | -------------- | ------ | ------------------ | --- |
8

• Reordering of input-output example pairs, which may improve out-of-
|     | distribution |     | performance, |     | as evidenced | in the literature | [7] |     |
| --- | ------------ | --- | ------------ | --- | ------------ | ----------------- | --- | --- |
While we use the same class of augmentations at each part of the pipeline,
the role and requirements of these augmentations vary across our approach:
During training, augmentations generate additional valid examples which
helps preventing the model from overfitting and provides a broader set of
tasks. For inference, we use augmentations to generate a more varied set
of solution candidates as outlined in Section 3.5. This requires T to be
reversible,enablingustomapthemodeloutputtotheoriginalun-augmented
task. In Section 3.6, we use augmentations to evaluate candidates from
different perspectives and pick the best two guesses to be submitted. Here
the only requirement is that the transformed tasks and solution candidates
| can | be meaningfully |     | evaluated  |     | by our model. |     |     |     |
| --- | --------------- | --- | ---------- | --- | ------------- | --- | --- | --- |
| 3.4 | Training        |     | the models |     |               |     |     |     |
Choosing the right large language model (LLM) was essential for achiev-
ing strong performance. We tested a variety of different models and found
that Mistral-NeMo-Minitron-8B-Base [5] performed the best in our ex-
periments. The model is rather large, so efficient fine-tuning methods were
| necessary |     | to use | it effectively. |     |     |     |     |     |
| --------- | --- | ------ | --------------- | --- | --- | --- | --- | --- |
As a result, we used Low-Rank Adaptation (LoRA)[18], 4-bit quantization
andgradientcheckpointing, allsupportedbytheunslothlibrary. Weapplied
the LoRA adaptations to all layers of the network, including the input and
output embeddings. We used a learning rate of for all layers, except
1e−4
| for | embedding | which | had | a reduced | learning | rate of 1e−5. |     |     |
| --- | --------- | ----- | --- | --------- | -------- | ------------- | --- | --- |
For each task C (X ,Y ,X ,Y ,...,Xc,Yc), we computed gradients only
|     |     | =   | 1   | 1 2 | 2   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
on the outputs Y for i > 1 and Yc, so the model never has to predict an
i
input grid, and as it is impossible to correctly predict the first output grid
without having seen at least on example. During our second finetuning we
| compute |     | the gradient | on  | all available | outputs. |     |     |     |
| ------- | --- | ------------ | --- | ------------- | -------- | --- | --- | --- |
Preliminary training: The Preliminary training used a LoRA rank of 256
andwasdoneonasingleH100GPU(seeTable3)forvariousdatasets. These
includedRe-ARC[11],theARC-AGIpublicevaluationset[1],Concept-ARC
[12], and ARC-Heavy [13]. The best-performing model on the private evalu-
ation set (Nemo-heavy) was trained on 531.318 training examples: 257.600
fromRe-Arc, 51.200fromARC-AGI-pub-eval, 22.528fromConcept-Arcand
|     | from | ARC-Heavy. |     | Due | to the data | augmentations | we applied, | there |
| --- | ---- | ---------- | --- | --- | ----------- | ------------- | ----------- | ----- |
200.000
| were | no exact | repetitions |     | in the | examples. |     |     |     |
| ---- | -------- | ----------- | --- | ------ | --------- | --- | --- | --- |
Secondary training: Secondarytrainingwastime-constrainedandfocused
solely on the hidden test set, using a LoRA rank of 64 and running for
9

| Model       | name | Re-Arc |      | public |     | Concept |     | ARC-Heavy |      |
| ----------- | ---- | ------ | ---- | ------ | --- | ------- | --- | --------- | ---- |
|             |      |        | [11] | eval   | [1] | ARC[12] |     |           | [13] |
| Llama-rearc | [19] | 368    | 400  |        | -   |         | -   |           | -    |
×
| Llama-mix  |      | 368 | × 400 | 64  | × 300 | 64  | × 176 |     | -      |
| ---------- | ---- | --- | ----- | --- | ----- | --- | ----- | --- | ------ |
| Nemo-mix   |      | 368 | × 400 | 64  | × 300 | 64  | × 176 |     | -      |
| Nemo-full  | [20] | 736 | 400   | 128 | 400   | 128 | 176   |     | -      |
|            |      |     | ×     |     | ×     |     | ×     |     |        |
| Nemo-heavy |      | 644 | 400   | 128 | 400   | 128 | 176   |     | 1 200k |
|            |      |     | ×     |     | ×     |     | ×     |     | ×      |
Table 3: Number of training examples (epochs dataset tasks) from each datasets in
×
| different | training runs. |     |     |     |     |     |     |     |     |
| --------- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
4 epochs. We also tried an additional separate fine-tuning for each task,
which increased the score slightly. However, it did not do enough to be
considered runtime efficient in our approach and was discarded. Instead, we
chose to make use of both GPUs available on Kaggle by splitting the test
T4
dataset in two parts, and running the full pipeline in parallel on the halves.
Using this approach we estimate the training took around 5:20 in the Kaggle
notebook.
For acomparisonof ourtrainingparametersused, seeTable 4. Ingeneralwe
found that modern architectures and larger models performed substantially
better. In the following sections we will present results for three different
models, each trained on different sections of the datasets (see 3). Llama-mix
andNemo-mixaredirectlycomparablemodels,thatshowcasethesubstantial
increase in performance from running the 8B Nemo model, compared to
the 3B Llama model. Llama-rearc provides us with a better evaluation set
performance baseline, as it has not seen any evaluation examples during
training. And finally, the table includes Nemo-full and Nemo-heavy, our
highest scoring models on the private evaluation set. However, as their
training includes the full evaluation set, we cannot provide any evaluation
performance except for our final Kaggle score (53.5 points Nemo-full and
56.5 for Nemo-heavy).
| 3.5 Solution |     | Inference |     |     |     |     |     |     |     |
| ------------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
Given a trained decoder-only network, standard solution generation is easy.
When provided with tokens x ,...,x , a model M will calculate the proba-
|                      |     |     | 1   |                                | n   |     |     |     |         |
| -------------------- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- | ------- |
| bilitydistributionsp |     |     |     | forsubsequenttokenpredictions. |     |     |     |     | Thenext |
,...,p
x2 xn+1
token, p , can then be selected either greedily (using argmax) or stochas-
xn+1
tically (using multinomial sampling). To generate a solution candidate we
repeat this procedure until we sample an ⟨eos⟩ token, indicating that the
example is done. The output is parsed into an array, with some checks to
| make sure | it is a | valid grid. |     |     |     |     |     |     |     |
| --------- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
10

|          |              |           | Initial |         | Fine-Tune | Secondary |     | Fine-Tune |     |
| -------- | ------------ | --------- | ------- | ------- | --------- | --------- | --- | --------- | --- |
|          |              |           |         | Locally |           | Locally   | On  | Kaggle    |     |
| Batch    | size         |           |         |         | 4         | 2         |     | 2         |     |
| Gradient | acc.         | steps     |         |         | 2         | 2         |     | 2         |     |
| LoRA     | rank         |           |         |         | 256       | 64        |     | 64        |     |
|          |              |           |         |         | 24        | 16        |     | 16        |     |
| LoRA     | α            |           |         |         |           |           |     |           |     |
| LR       | (LoRA        | adapters) |         |         |           |           |     |           |     |
|          |              |           |         |         | 1e−4      | 1e−4      |     | 1e−4      |     |
| LR       | (embeddings) |           |         |         | 1e−5      | 1e−5      |     | 1e−5      |     |
| Number   | of           | Epochs    |         | see     | Table 3   | 48        |     | 32        |     |
| LR       | schedule     |           |         | cosine  |           | cosine    |     | cosine    |     |
| LR       | warmup       | phase     |         |         | 25%       | 25%       |     | 100 steps |     |
| Weight   | decay        |           |         |         | off       | off       |     | 0.01      |     |
| Model    | quantization |           |         |         | 4 bit     | 4 bit     |     | 4 bit     |     |
| Train    | on 1st       | output    |         |         | no        | yes       |     | yes       |     |
Table 4: Training parameters for initial as well as secondary fine-tuining. Parameters for
| Kaggle were | chosen | differently | due | to time | and memory | constraints. |     |     |     |
| ----------- | ------ | ----------- | --- | ------- | ---------- | ------------ | --- | --- | --- |
While this can achieve good results, we found both sampling strategies to be
sub-optimal for generating correct solutions to a given task. We attribute
| these shortcomings |           | to  | several | reasons: |                    |     |     |          |     |
| ------------------ | --------- | --- | ------- | -------- | ------------------ | --- | --- | -------- | --- |
|                    |           |     | Greedy  | sampling | is straightforward |     | but | presents | two |
| Greedy             | Sampling: |     |         |          |                    |     |     |          |     |
majorissues. First, itoffersnoguaranteesregardingthefinalsamplingprob-
abilityofthegeneratedsolution. Itisentirelypossibleforgreedysamplingto
produceasolutionwithverylowoverallprobability: asinglehigh-confidence
mistakebytheLLMcanleadtoanundesirablesequence, therebypreventing
a successful continuation. Second, greedy sampling inherently lacks variabil-
| ity, yielding | only | a single | result | for | a given input. |     |     |     |     |
| ------------- | ---- | -------- | ------ | --- | -------------- | --- | --- | --- | --- |
Stochasic Sampling: Standard stochastic sampling techniques introduce
morevariabilitybutcanalsorepeatedly return thesamesolution. Moreover,
we have no guarantee that the best solution is sampled. If the optimal solu-
tion has, for example, a 10% sampling probability, we may need to sample
excessively to generate this specific solution. Given that inference is partic-
ularly slow – especially for longer tasks – this approach can quickly become
computationally prohibitive, even more so within the limitations of a Kaggle
notebook.
While there are several alternative sampling schemes, such as sampling with
temperature or beam search, we found that a custom sampling algorithm
| produced | the | best results | for | the | benchmark: |     |     |     |     |
| -------- | --- | ------------ | --- | --- | ---------- | --- | --- | --- | --- |
Weemployadepth-firstsearch(DFS)toexploreallpossiblepathsthrough
11

Figure 5: Illustration of the DFS scheme. In this example, only paths with a cumulative
probabilitygreaterthan5%arekept. Sincemanypathsarecut,wewilloftenendupwith
far less than 20 possible solutions.
the solution tree, as long as the path has a cumulative sampling probabil-
ity greater than a specified threshold p. As soon as a branch falls below
this probability, the path is discarded. By leveraging inference caches, this
algorithm is able to quickly and efficiently identify all potential candidates
that have a sampling probability greater than p. This also means that we
are guaranteed to extract the solution with the best possible score, provided
that it has a probability greater than p – something we cannot guarantee
with greedy or multinomial sampling.
We apply this DFS sampling approach across multiple augmented versions
of each task. The motivation here is that certain solutions are easier for the
model to "see" from alternative perspectives. Since we are using a decoder-
only text model, it lacks an intrinsic understanding of the 2D structure of
these tasks. This limitation becomes evident when solutions that involve
critical vertical lines receive a lower probability, while the same task, once
transposed, is much easier for the model to solve. (see Figure 6)
Consequently, depending on the specific run, we generate DFS candidates
across8to16augmentationsforeachtask. Thisresultsinasetofcandidates
that is guaranteed to include the correct solution if and only if the model
could have sampled that solution with a probability greater than p from at
least one augmented perspective.
This strategy is highly effective. Not only do we increase our scores by
several points, but the algorithm is also substantially faster then baseline
(see Table 5). The algorithm is also very memory efficient, as we only need
totrackasinglebranchatatimeandcanavoidduplicatingcaches–instark
contrast to beam search, which scales linearly with the number of beams
considered. We provide pseudocode for DFS sampling in Algorithm 1.
12

| Sampling   |     | Llama-rearc |       |       | Llama-mix |           |      | Nemo-mix  |      |
| ---------- | --- | ----------- | ----- | ----- | --------- | --------- | ---- | --------- | ---- |
|            |     | all         | 400   | tasks |           | 100 tasks |      | 100 tasks |      |
|            |     | A           | B     | R     | A         | B         | R    | A B       | R    |
| Greedy     |     | 51.00       | 60.00 | 10:51 | 51.5      | 65.5      | 2:36 | 59.0 75.0 | 3:49 |
| Stochastic |     | 50.50       | 59.50 | 10:58 | 50.5      | 65.5      | 2:39 | 58.5 72.0 | 3:55 |
| DFS 17%    |     | 51.25       | 61.50 | 7:21  | 51.5      | 66.5      | 1:51 | 60.5 76.5 | 2:35 |
| DFS 10%    |     | 51.00       | 63.50 | 9:54  | 51.5      | 73.0      | 2:34 |           |      |
|            |     |             |       |       |           |           |      | 60.5 80.0 | 3:40 |
Table 5: Percentage of correct solutions sampled for 100 randomly split-off tasks of the
ARC-AGI public evaluation set (full public evaluation set is used for Llama-rearc) using
different sampling strategies on 16 augmented versions (transposition and rotations, as
well as randomly permuted colors) of the task. The first column (A) denominates the
number of tasks correctly solved by the two best-scoring guesses of the model, while the
second column (B) represents the number of tasks for which the correct solution was
present among the candidates obtained during the inference run. Runtimes (column R),
hh:mm for inference on an Nvidia H100 GPU are given in parentheses. Note that there
mightbesomeconceptualleakagefromthe300evaluationtasksusedintrainingofLlama-
| Mix and | Nemo-Mix, | possibly | inflating | their | scores | a little. |     |     |     |
| ------- | --------- | -------- | --------- | ----- | ------ | --------- | --- | --- | --- |
Using this algorithm we are able to generate the correct solution pretty well
– in up to 80% of the cases on eval! However, we see a big drop in score
when selecting two of these candidates, as we have no way to reliably choose
| the correct   | solution |            | from our | large | number | of  | candidates | (yet). |     |
| ------------- | -------- | ---------- | -------- | ----- | ------ | --- | ---------- | ------ | --- |
| 3.6 Selection |          | Strategies |          |       |        |     |            |        |     |
Figure 6: Illustration of the idea behind our candidate scoring. Depending on the aug-
mentation used, the model is able to "see" places of high uncertainty more clear. In this
example, the wrong line is barely visible in the transposed version of the task, but when
rotating by 180° the model can clearly see that something is wrong. As a result, aggre-
| gating these | scores | provides | a   | highly effective | way | to filter | wrong | candidates. |     |
| ------------ | ------ | -------- | --- | ---------------- | --- | --------- | ----- | ----------- | --- |
Once we have generated a set of solution candidates, the next step is to
determine which ones to submit. Our pipeline up to this point is capable of
generating candidates that have a good chance of including the correct solu-
tion,buttoconsideratasksolved,wemustidentifyitamongthecandidates,
13

| Model       |        | No Aug. | Scoring    |     | Scoring |       | with 8 augmentations |              |
| ----------- | ------ | ------- | ---------- | --- | ------- | ----- | -------------------- | ------------ |
|             | Best   | 2       | Similarity |     | Best    | Worst | Sum of Sum           | of Time      |
|             | Scores |         | Selection  |     | Prob    | Prob  | Probs                | logp [mm:ss] |
| Llama-rearc |        | 51.00   | 51.25      |     | 48.38   | 51.63 | 48.75                | 54.5 37:31   |
| Llama-mix   |        | 51.5    | 57.0       |     | 57.0    | 62.0  | 57.5                 | 63.5 9:50    |
| Nemo-mix    |        | 60.5    | 66.0       |     | 66.0    | 65.5  | 65.0                 | 72.5 17:05   |
Table 6: Comparison of different selection strategies using 8 augmentations for each can-
didate (full evaluation set for Llama-rearc, 100 split-off tasks otherwise). Selecting the
candidate with the highest summed log-prob (or alternatively product of probabilities)
results in an increase of around over baseline, while only requiring and additional
25%
10-17 minutes on an H100 per 100 tasks. We also compare some alternative aggregation
methods. The aggregations compared are: ’best prob’ (max(P)), worst prob (min(P)),
| sum of probs | ((cid:80) P) | and      | sum of log | prob | ((cid:81) P) | which | we use. |     |
| ------------ | ------------ | -------- | ---------- | ---- | ------------ | ----- | ------- | --- |
| using no     | more         | than two | guesses.   |      |              |       |         |     |
Recallthatany(decoder-only)largelanguagemodelM alsoprovidesuswith
probabilities for each input token. Given a task and a solution candidate
C
S , the model can calculate a probability P (S |C), which represents how
| k   |     |     |     |     |     | M   | k   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
likely it is that the model would generate S when provided with C us-
k
ing standard sampling. At first glance, using this probability for selection
might seem redundant, as it appears that it would just result in selecting
the solution with the highest sampling probability – something that could
| be achieved | more | directly | by  | simply | sampling. |     |     |     |
| ----------- | ---- | -------- | --- | ------ | --------- | --- | --- | --- |
Instead, we leverage our augmentation techniques. As the network was
trained using augmentations, it stands to reason that a correct solution
should demonstrate greater stability in its sampling probability under aug-
| mented       | conditions | compared |               | to an | incorrect      | one. |     |     |
| ------------ | ---------- | -------- | ------------- | ----- | -------------- | ---- | --- | --- |
| We therefore | calculate  |          | the augmented |       | probabilities: |      |     |     |
Paug
|     |     |     | (S  | ) = | P (T | (S )|T | (C)) |     |
| --- | --- | --- | --- | --- | ---- | ------ | ---- | --- |
|     |     |     | i   | k   | M i  | k      | i    |     |
for augmentations T ,...,T . Our results always use D symmetry aug-
|     |     | 1   |     | n   |     |     | 8   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
mentations and optionally color permutations and example shuffling. Using
these augmentations for scoring, we observed that taking the product of
8
the probabilities (or alternatively, summing the logsoftmax values) led to a
highly stable and effective selection strategy. Our selection strategy can be
| summarized | as: |     |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:89)
Paug
|     |     |     | argmax |     |     | (S k | )   |     |
| --- | --- | --- | ------ | --- | --- | ---- | --- | --- |
i
k
i
14

A comparison of this selection procedure can be seen in Table 6. Adding
candidate scoring with augmentations improves our score by roughly 25%
over baseline, and shows a significant increase compared to our next best
solution. Using this technique we are able to select the correct candidate in
72.5 of 80 possible cases, thereby solving 72.5 out of the 100 randomly split
off tasks from the evaluation set.
Even the smaller Llama-rearc model, where preliminary training exclusively
used Re-ARC, was able to solve 218 of all 400 evaluation tasks after sec-
ondary finetuning on those tasks.
4 Discussion
Using the above pipeline we were able to score 53.5 points during the Kaggle
ARCPrize2024and56.5pointsshortlyafterthesubmissiondeadline(there-
fore not being reflected on the leaderboard), securing us the second place
in the competition. Our approach demonstrates that LLMs can effectively
tackle many of the complex tasks provided by the ARC-AGI benchmark.
The pipeline presented is highly interconnected and provides results greater
than the sum of its parts. The success of our scoring algorithm relies on the
custom DFS algorithm’s ability to provide candidates with arbitrary cutoff
values. This in turn is enabled by optimizing model training in both train-
ing phases and by simplifying the problem space. Each part of the pipeline
requires the augmentations to work as well as it does.
Finally, we believe that this approach has not yet completely saturated and
might scale further, using additional augmentation strategies, further opti-
mizing training hyperparameters or by using larger models.
Acknowledgement
We would like to express our sincere gratitude to Lambda, for providing
computational resources essential for optimizing our preliminary training
phase within our overall training pipeline. Specifically, they supplied us
with a server equipped with 8xH100 GPUs, enabling rapid iteration on our
ideas. Their support was instrumental for reaching our final score.
15

References
[1] FrançoisChollet.“OntheMeasureofIntelligence”.In:CoRRabs/1911.01547
| (2019). arXiv: |     | 1911.01547. |     | url: |     |     |     |     |
| -------------- | --- | ----------- | --- | ---- | --- | --- | --- | --- |
http://arxiv.org/abs/1911.
01547.
[2] Icecuber / top-quarks. Code for 1st place solution to Kaggle’s Ab-
|           |               |     | Challenge. |     | Accessed: | 2024-11-11. |     | 2024. url: |
| --------- | ------------- | --- | ---------- | --- | --------- | ----------- | --- | ---------- |
| straction | and Reasoning |     |            |     |           |             |     |            |
https://github.com/top-quarks/ARC-solution.
[3] Wenhao Li et al. Tackling the Abstraction and Reasoning Corpus with
Vision Transformers: the Importance of 2D Representation, Positions,
| Objects. | 2024. | arXiv: |     |            | [cs.CV]. | url: |                |     |
| -------- | ----- | ------ | --- | ---------- | -------- | ---- | -------------- | --- |
| and      |       |        |     | 2410.06405 |          |      | https://arxiv. |     |
org/abs/2410.06405.
[4] Abhimanyu Dubey et al. “The Llama 3 Herd of Models”. In: CoRR
abs/2407.21783 (2024). doi: 10.48550/ARXIV.2407.21783. arXiv:
| 2407.21783. | url: | https://doi.org/10.48550/arXiv.2407.21783. |     |     |     |     |     |     |
| ----------- | ---- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
[5] Sharath Turuvekere Sreenivas et al. LLM Pruning and Distillation in
|           |              |     | Approach. |     | 2024. | arXiv:     |     | [cs.CL]. |
| --------- | ------------ | --- | --------- | --- | ----- | ---------- | --- | -------- |
| Practice: | The Minitron |     |           |     |       | 2408.11796 |     |          |
url: https://arxiv.org/abs/2408.11796.
[6] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part
3.2, Knowledge Manipulation. 2024. arXiv: 2309.14402 [cs.CL]. url:
https://arxiv.org/abs/2309.14402.
[7] Zeyuan Allen-Zhu and Yuanzhi Li. “Physics of Language Models: Part
3.1,KnowledgeStorageandExtraction”.In:ArXive-prints abs/2309.14316
| (Sept. 2023). | Full | version |     | available | at  |     |     |     |
| ------------- | ---- | ------- | --- | --------- | --- | --- | --- | --- |
http://arxiv.org/abs/2309.
14316.
[8] Aaditya K. Singh and DJ Strouse. Tokenization counts: the impact of
|              |      |                                   |     |             | LLMs. | 2024. | arXiv: |            |
| ------------ | ---- | --------------------------------- | --- | ----------- | ----- | ----- | ------ | ---------- |
| tokenization | on   | arithmetic                        |     | in frontier |       |       |        | 2402.14903 |
| [cs.CL].     | url: | https://arxiv.org/abs/2402.14903. |     |             |       |       |        |            |
[9] Kaj Bostrom and Greg Durrett. Byte Pair Encoding is Suboptimal for
Language Model Pretraining. 2020. arXiv: 2004.03720 [cs.CL]. url:
https://arxiv.org/abs/2004.03720.
[10] Kaiser Sun et al. Tokenization Consistency Matters for Generative
Models on Extractive NLP Tasks. 2023. arXiv: 2212.09912 [cs.CL].
url: https://arxiv.org/abs/2212.09912.
| [11] Michael | Hodel. |            |     |                 |     |               |     |            |
| ------------ | ------ | ---------- | --- | --------------- | --- | ------------- | --- | ---------- |
|              |        | Addressing |     | the Abstraction |     | and Reasoning |     | Corpus via |
Procedural Example Generation. 2024. arXiv: 2404.07353 [cs.LG].
url: https://arxiv.org/abs/2404.07353.
[12] Arsenii Moskvichev, Victor Vikram Odouard, and Melanie Mitchell.
| “The ConceptARC |     |     | Benchmark: |     | Evaluating | Understanding |     | and Gen- |
| --------------- | --- | --- | ---------- | --- | ---------- | ------------- | --- | -------- |
eralization in the ARC Domain”. In: Trans. Mach. Learn. Res. 2023
| (2023). url: | https://openreview.net/forum?id=8ykyGbtt2q. |     |     |     |     |     |     |     |
| ------------ | ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
[13] Wen-DingLietal.Combining
|     |     |     |     | Induction |     | and Transduction |     | for Abstract |
| --- | --- | --- | --- | --------- | --- | ---------------- | --- | ------------ |
Reasoning. 2024. arXiv: 2411.02272 [cs.LG]. url: https://arxiv.
org/abs/2411.02272.
16

[14] ChuanLi.Llama-3.2-3B-Instruct-uncensored.Accessed:2024-11-11.2024.
url:https://huggingface.co/chuanli11/Llama-3.2-3B-Instruct-
uncensored.
[15] Yuren Mao et al. “A Survey on LoRA of Large Language Models”. In:
CoRR abs/2407.11046 (2024). doi: 10.48550/ARXIV.2407.11046.
arXiv: 2407.11046. url: https://doi.org/10.48550/arXiv.2407.
11046.
[16] Nelson F. Liu et al. “Lost in the Middle: How Language Models Use
LongContexts”.In:Trans.Assoc.Comput.Linguistics 12(2024),pp.157–
173. doi: 10.1162/TACL\_A\_00638. url: https://doi.org/10.
1162/tacl%5C_a%5C_00638.
[17] Aaditya K. Singh and DJ Strouse. “Tokenization counts: the impact of
tokenizationonarithmeticinfrontierLLMs”.In:CoRRabs/2402.14903
(2024). doi: 10.48550/ARXIV.2402.14903. arXiv: 2402.14903. url:
https://doi.org/10.48550/arXiv.2402.14903.
[18] Edward J. Hu et al. LoRA: Low-Rank Adaptation of Large Language
Models. 2021. arXiv: 2106.09685 [cs.CL]. url: https://arxiv.org/
abs/2106.09685.
[19] Daniel Franzen and Jan Disselhoff. Llama-3.2-3B-ARChitects-ReArc-
bnb-4bit. 2024. url: https://huggingface.co/da-fr/Llama-3.2-
3B-ARChitects-ReArc-bnb-4bit.
[20] DanielFranzenandJanDisselhoff.Mistral-NeMo-Minitron-8B-ARChitects-
Full-bnb-4bit. 2024. url: https://huggingface.co/da-fr/Mistral-
NeMo-Minitron-8B-ARChitects-Full-bnb-4bit.
17

A Miscellaneous
There were several optimizations and ideas that slightly boosted our score
but did not fit in the main text.
• We found that we could use the logsoftmax probabilities obtained in
the DFS Candidate generation in scoring, providing a slight boost at
no additional compute cost.
• Before using scores from various augmentations, we used a selection
algorithm based on the similarity of generated outputs, choosing the
candidate that had the most pixelwise similarity to all other candi-
dates.
• Due to the nature of the contest we had a very limited time window
for generating our solutions. In the end we were able to completely
parallelize our solution on both T4 GPUs provided.
• There is a trade-off between Generation and Selection, that we were
unable to optimize for the hidden test set. Reducing the DFS cutoff
value provides more potentially correct solutions, but makes it harder
toselectthem. Whilethebestvaluesforevaluationwereratherlow,we
found higher cutoffs of up to 20% to work better on Kaggle, which had
the added benefit of reducing run-time. But even for low percentage
valuestheDFSreturnssurprisinglyfewresultsonaverage,seeFigure7.
• As can be seen in our results, increasing the size of the base model
substantially increases the score at every part of the pipeline. We
invested a lot of time to make sure that we are able train and use the
8B Nemo Minitron model in a reasonable timeframe on Kaggle. We
believe that even larger models could elevate performance again.
Figure 7: Number of potential candidates returned by the DFS using a 5% cutoff value
on the evaluation dataset. In most cases the number of candidates is far lower than the
possible theoretical maximum of 20.
18

0
|     |     | 10.11 | noticeleS | ygetartS |            | erocSguAx4 | erocSguAx4 | erocSguAx4 erocSguAx8 | erocSguAx8 | erocSguAx8 erocSguAx8 |
| --- | --- | ----- | --------- | -------- | ---------- | ---------- | ---------- | --------------------- | ---------- | --------------------- |
|     |     |       |           |          | ytiralimis | ytiralimis |            |                       |            |                       |
09.11
- -
01
06.11
05.11
02
|     |     |     | gnilpmaS | ygetartS citsahcots | citsahcots citsahcots |     | citsahcots | %01 %71 | %71 | %41 %02 |
| --- | --- | --- | -------- | ------------------- | --------------------- | --- | ---------- | ------- | --- | ------- |
ydeerg ydeerg
|     |     |     | 03                  |     |       |         |     | SFD SFD | SFD | SFD SFD |
| --- | --- | --- | ------------------- | --- | ----- | ------- | --- | ------- | --- | ------- |
|     |     |     | 04 )noissimbuS .guA | x2  | x2 x4 | x61 x61 | x61 | x61 x8  | x8  | x61 x61 |
24.10
selpmaxE
05
|     |     |     | niarT | k62 | k62 k47 | k47 k47 | k661 | k661 k871 | k863 | k863 k135 |
| --- | --- | --- | ----- | --- | ------- | ------- | ---- | --------- | ---- | --------- |
rewen
|     |     | 18.10 | 06  |                  |                  |     |     |                  |                  |                                   |
| --- | --- | ----- | --- | ---------------- | ---------------- | --- | --- | ---------------- | ---------------- | --------------------------------- |
|     |     |       | ot  | B21-omeN-lartsiM | B21-omeN-lartsiM |     |     | B8-nortiniM-oMeN | B8-nortiniM-oMeN | B8-nortiniM-oMeN B8-nortiniM-oMeN |
sdnopserroc
|     |     |     |          |     | B3-2.3-amalL | B3-2.3-amalL B3-2.3-amalL | B3-2.3-amalL              | B3-2.3-amalL |             |                          |
| --- | --- | --- | -------- | --- | ------------ | ------------------------- | ------------------------- | ------------ | ----------- | ------------------------ |
|     |     |     |          |     | )derosnecnu( | )derosnecnu(              | )derosnecnu( )derosnecnu( | )derosnecnu( | "lluf-omeN" | "lluf-omeN" "yvaeh-omeN" |
|     |     |     | 07 ledoM |     |              |                           |                           |              |             |                          |
11.10
08 noisreV
06.10
09
rewol(
04.10
segnahC
|     |     |     |             |     | gniniarter tceles_ytiralimis+crAeR+ |     |              |     |     | yvaeH-CRA |
| --- | --- | --- | ----------- | --- | ----------------------------------- | --- | ------------ | --- | --- | --------- |
|     |     |     | 001 noisreV |     |                                     |     | gniniart-erp |     |     |           |
serocs
|     |                |     |     |                    |     | ecnerefni |     |     | niart-erp           | ecnerefni |
| --- | -------------- | --- | --- | ------------------ | --- | --------- | --- | --- | ------------------- | --------- |
|     | )5.65(erocSxaM |     |     | )esopsnart&lamron( |     |           |     |     | teslavellufedulcni+ |           |
snoitatnemguahtiw ecnerefni
|     |       |     |             | ylno | hctiws |             |     | hctiws | crAtpecnoCdda+ |     |
| --- | ----- | --- | ----------- | ---- | ------ | ----------- | --- | ------ | -------------- | --- |
|     | erocS |     | tnacfiingiS |      |        |             |     |        |                |     |
|     |       |     | 011         |      |        | gnitaluclaC |     |        |                |     |
)niart-erprof(
ecnerefnI emit-tseT
gniddA
|     |     |     |     |     | ledoM | dlof-61 | regnoL | ledoM | regnoL | dlof-61 |
| --- | --- | --- | --- | --- | ----- | ------- | ------ | ----- | ------ | ------- |
SFD
021
|     |     |     | erocS |     |           |           |      |           |      | *5.65 |
| --- | --- | --- | ----- | --- | --------- | --------- | ---- | --------- | ---- | ----- |
|     |     |     |       | 0.7 | 0.02 0.32 | 0.03 0.73 | 0.14 | 0.44 0.74 | 5.84 | 5.35  |
031
08.09
|     |     |     |     | 4202.80.82 | 4202.90.80 4202.01.40 | 4202.01.60 4202.01.11 | 4202.01.81 | 4202.01.42 4202.11.50 | 4202.11.60 | 4202.11.90 4202.11.01 |
| --- | --- | --- | --- | ---------- | --------------------- | --------------------- | ---------- | --------------------- | ---------- | --------------------- |
041
etaD
28.08.
| 06  | 04  | 02  | 0   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
erocS
Table 7: Timeline of scores in the kaggle competition, along with the changes we deem
most significant for the improved performance. The final submission marked with (*)
| finished | scoring | after | the deadline | had already | ended. |     |     |     |     |     |
| -------- | ------- | ----- | ------------ | ----------- | ------ | --- | --- | --- | --- | --- |
19

| Algorithm | 1   | Depth-First |     | Probability-Guided |     |     | Sampling | for LLMs |
| --------- | --- | ----------- | --- | ------------------ | --- | --- | -------- | -------- |
The algorithm presented here assumes that the model supports internal
cachingforalreadyseensequencesandonlyneedstoprocessthenewlyadded
tokens.
Our actual implementation differs from this simple variant, as we are using
unsloth, which does not support dynamic caching and requires us to prune
| the key-value-cache |     |     | of the | transformer |     | ourselves. |     |     |
| ------------------- | --- | --- | ------ | ----------- | --- | ---------- | --- | --- |
Furthermore, we use various performance optimizations, like a simultane-
ous initial forward pass of the best known sequence including prompt and
prediction (which is much faster than token-by-token generation) as well
as aggregating the sequences during backtracking to avoid the unnecessary
| processing | of  | sequences | that | would | be  | discarded | later. |     |
| ---------- | --- | --------- | ---- | ----- | --- | --------- | ------ | --- |
1: procedureDFS_sample(model,prompt,threshold,max_len,eos_id)
| 2:  | Input: | model  | is the | language   |     | model       |              |     |
| --- | ------ | ------ | ------ | ---------- | --- | ----------- | ------------ | --- |
|     |        |        | is     | the prompt |     | that should | be completed |     |
| 3:  | Input: | prompt |        |            |     |             |              |     |
4: Input: threshold is the maximum negative log probability allowed
5: Input: max_len is the maximum length (including the prompt)
|     |     | eos_id | is  | the index | of  | the end | of sentence | token |
| --- | --- | ------ | --- | --------- | --- | ------- | ----------- | ----- |
6: Input:
7:
| 8:  | function      | Explore(tokens,score) |                |          |     |          |           |      |
| --- | ------------- | --------------------- | -------------- | -------- | --- | -------- | --------- | ---- |
| 9:  | if tokens[−1] |                       |                | = eos_id | or  | |tokens| | ≥ max_len | then |
| 10: |               | return                | (score,tokens) |          |     |          |           |      |
| 11: | end           | if                    |                |          |     |          |           |      |
12:
|     | next_token_logits   |     |     |     | model.predict_logits(tokens)[−1] |     |     |     |
| --- | ------------------- | --- | --- | --- | -------------------------------- | --- | --- | --- |
| 13: |                     |     |     | ←   |                                  |     |     |     |
|     | next_token_log_prob |     |     |     | −log_softmax(logits)             |     |     |     |
| 14: |                     |     |     |     | ←                                |     |     |     |
| 15: | valid_sequences     |     |     | ←   | ∅                                |     |     |     |
16:
|     |              | each            | possible | next                           | token                           |                               |     |     |
| --- | ------------ | --------------- | -------- | ------------------------------ | ------------------------------- | ----------------------------- | --- | --- |
| 17: | for          |                 |          |                                |                                 | t do                          |     |     |
| 18: |              | next_score      |          | ← score+next_token_log_prob[t] |                                 |                               |     |     |
| 19: |              | if next_score   |          | ≤                              | threshold                       | then                          |     |     |
|     |              | next_tokens     |          |                                | current_tokens+[t]              |                               |     |     |
| 20: |              |                 |          |                                | ←                               |                               |     |     |
| 21: |              |                 |          |                                | Explore(next_tokens,next_score) |                               |     |     |
|     |              | continuations   |          |                                | ←                               |                               |     |     |
| 22: |              | valid_sequences |          |                                | ←                               | valid_sequences∪continuations |     |     |
| 23: |              | end if          |          |                                |                                 |                               |     |     |
| 24: | end          | for             |          |                                |                                 |                               |     |     |
| 25: | end function |                 |          |                                |                                 |                               |     |     |
26:
Explore(prompt,0.0)
27: return
| 28: end | procedure |     |     |     |     |     |     |     |
| ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
20