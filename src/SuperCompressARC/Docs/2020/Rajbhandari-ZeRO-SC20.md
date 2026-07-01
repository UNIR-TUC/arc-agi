ZeRO: Memory Optimizations Toward Training
Trillion Parameter Models
Samyam Rajbhandari∗, Jeff Rasley∗, Olatunji Ruwase, Yuxiong He
{samyamr, jerasley, olruwase, yuxhe}@microsoft.com
ABSTRACT common settings like mixed precision and ADAM optimizer
[6].OtherexistingsolutionssuchasPipelineParallelism(PP),
Largedeeplearningmodelsoffersignificantaccuracygains,
Model Parallelism (MP), CPU-Offloading, etc, make trade-
but training billions to trillions of parameters is challenging.
offs between functionality, usability, as well as memory and
Existingsolutionssuchasdataandmodelparallelismsexhibit
compute/communication efficiency, all of which are crucial to
fundamentallimitationstofitthesemodelsintolimiteddevice
training with speed and scale.
memory, while obtaining computation, communication and
Amongdifferentexistingsolutionfortraininglargemodels,
development efficiency. We develop a novel solution, Zero
MP is perhaps the most promising one. The largest models in
Redundancy Optimizer (ZeRO), to optimize memory, vastly
the current literature, the 11B T5 model [5], and Megatron-
improving training speed while increasing the model size that
LM 8.3B [4], were both powered by model parallelism,
can be efficiently trained. ZeRO eliminates memory redun-
implemented in Mesh-Tensorflow [7] and Megatron-LM[4],
dancies in data- and model-parallel training while retaining
respectively. However, MP cannot scale much further beyond
low communication volume and high computational granu-
thesemodelssizes.MPsplitsthemodelvertically,partitioning
larity, allowing us to scale the model size proportional to
the computation and parameters in each layer across multiple
the number of devices with sustained high efficiency. Our
devices, requiring significant communication between each
analysisonmemoryrequirementsandcommunicationvolume
layer. As a result, they work well within a single node where
demonstrates:ZeROhasthepotentialtoscalebeyond1Trillion
the inter-GPU communication bandwidth is high, but the
parameters using today’s hardware.
efficiency degrades quickly beyond a single node [4]. We
We implement and evaluate ZeRO: it trains large models
testeda40BparametermodelusingMegatron-LMacrosstwo
of over 100B parameter with super-linear speedup on 400
DGX-2 nodes and observe about 5Tflops per V100 GPU
GPUs, achieving throughput of 15 Petaflops. This represents
(less than 5% of hardware peak) compared to near 30Tflops
an 8x increase in model size and 10x increase in achievable
for models that can be trained within a node.
performance over state-of-the-art. In terms of usability, ZeRO
So, how can we overcome the limitations of existing so-
can train large models of up to 13B parameters (e.g., larger
lutions and train large models more efficiently? To answer
than Megatron GPT 8.3B and T5 11B) without requiring
this question, we first analyze the full spectrum of memory
model parallelism which is harder for scientists to apply.
consumption of the existing systems on model training and
Last but not the least, researchers have used the system
classify it into two parts: 1) For large models, the majority
breakthroughs of ZeRO to create Turing-NLG, the world’s
of the memory is occupied by model states which include
largest language model at the time (17B parameters) with
the optimizer states (such as momentum and variances in
record breaking accuracy.
Adam [6]), gradients, and parameters. 2) The remaining
I. EXTENDEDINTRODUCTION memory is consumed by activation, temporary buffers and
unusable fragmented memory, which we refer to collectively
Deep Learning (DL) models are becoming larger, and the
as residual states. We develop ZeRO— Zero Redundancy
increase in model size offers significant accuracy gain. In the
Optimizer — to optimize memory efficiency on both while
areaofNaturalLanguageProcessing(NLP),Transformers[1]
obtaining high compute and communication efficiency. As
have paved way for large models like Bert-large (0.3B) [2],
these two parts face different challenges, we develop and
GPT-2 (1.5B) [3], Megatron-LM (8.3B) [4], T5 (11B) [5]. To
discuss their solutions correspondingly.
enable the continuation of model size growth from 10s of
Optimizing Model State Memory Model states often
billionstotrillionsofparameters,weexperiencethechallenges
consume the largest amount of memory during training,
oftrainingthem-theyclearlydonotfitwithinthememoryof
but existing approaches such as DP and MP do not offer
a single device, e.g., GPU or TPU, and simply adding more
a satisfying solution. DP has good compute/communication
devices will not help scale the training.
efficiency but poor memory efficiency while MP can have
Basicdataparallelism(DP)doesnotreducememoryperde-
poor compute/communication efficiency. More specifically,
vice,andrunsoutofmemoryformodelswithmorethan1.4B
DP replicates the entire model states across all data parallel
parameters on GPUs with 32GB memory when trained using
process resulting in redundant memory consumption; while
∗EqualContribution MP partition these states to obtain high memory efficiency,
SC20,November9-19,2020,IsEverywhereWeAre
978-1-7281-9998-6/20/$31.00(cid:13)c2020IEEE

|     |     |     |     |     |     |     |     | memory          | consumed |            | by activations, |         | temporary     | buffers,  | and      |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | -------- | ---------- | --------------- | ------- | ------------- | --------- | -------- |
|     |     |     |     |     |     |     |     | unusable        | memory   | fragments  | could           | become  | a             | secondary | mem-     |
|     |     |     |     |     |     |     |     | ory bottleneck. |          | We develop | ZeRO-R          |         | to optimize   | the       | residual |
|     |     |     |     |     |     |     |     | memory          | consumed | by         | these three     | factors | respectively. |           |          |
1)Foractivations(storedduringtheforwardpassinorderto
|     |     |     |     |     |     |     |     | perform             | backward    | pass), | we noticed    |     | checkpointing | [8]        | helps |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | ----------- | ------ | ------------- | --- | ------------- | ---------- | ----- |
|     |     |     |     |     |     |     |     | but is insufficient |             | for    | large models. |     | Thus ZeRO-R   | optimizes  |       |
|     |     |     |     |     |     |     |     | activation          | memory      | by     | identifying   | and | removing      | activation |       |
|     |     |     |     |     |     |     |     | replication         | in existing |        | MP approaches |     | through       | activation | par-  |
titioning.ItalsooffloadsactivationstoCPUwhenappropriate.
2)ZeRO-Rdefinesappropriatesizefortemporarybuffersto
| Fig. 1: | Comparing | the | per-device | memory |     | consumption | of  |     |     |     |     |     |     |     |     |
| ------- | --------- | --- | ---------- | ------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
model states, with three stages of ZeRO-DP optimizations. Ψ strike for a balance of memory and computation efficiency.
denotes model size (number of parameters), K denotes the 3) We observe fragmented memory during training due
|        |            |     |           |         |     |           |     | to variations | in  | the | lifetime | of different | tensors. |     | Lack of |
| ------ | ---------- | --- | --------- | ------- | --- | --------- | --- | ------------- | --- | --- | -------- | ------------ | -------- | --- | ------- |
| memory | multiplier | of  | optimizer | states, | and | N denotes | DP  |               |     |     |          |              |          |     |         |
d
degree.Intheexample,weassumeamodelsizeofΨ=7.5B contiguous memory due to fragmentation can cause memory
allocationfailure,evenwhenenoughfreememoryisavailable.
| and DP | of N d | = 64 with | K   | = 12 based | on  | mixed-precision |     |     |     |     |     |     |     |     |     |
| ------ | ------ | --------- | --- | ---------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
training with Adam optimizer. ZeRO-R proactively manages memory based on the different
|     |     |     |     |     |     |     |     | lifetime | of tensors, | preventing |     | memory | fragmentation. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ---------- | --- | ------ | -------------- | --- | --- |
ZeRO-DPandZeRO-Rcombinedtogetherformsapowerful
butoftenresultintoofine-grainedcomputationandexpensive
|     |     |     |     |     |     |     |     | system | of memory |     | optimizations | for | DL training |     | that we |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --------- | --- | ------------- | --- | ----------- | --- | ------- |
communication that is less scaling efficient. Furthermore, all collectively refer to as ZeRO.
oftheseapproachesmaintainallthemodelstatesrequiredover
|     |     |     |     |     |     |     |     | ZeRO | and | MP: Since | ZeRO | eliminates |     | the memory | in- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --------- | ---- | ---------- | --- | ---------- | --- |
theentiretrainingprocessstatically,eventhoughnotallmodel
|     |     |     |     |     |     |     |     | efficiency | in DP, | it is | natural | to ask: | Do we | still need | MP, |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | ----- | ------- | ------- | ----- | ---------- | --- |
states are required all the time during the training. Based and when? How does ZeRO work with MP? With ZeRO, MP
| on these | observations, |     | we develop | ZeRO-DP, |     | ZeRO-powered |     |         |        |            |        |     |             |     |         |
| -------- | ------------- | --- | ---------- | -------- | --- | ------------ | --- | ------- | ------ | ---------- | ------ | --- | ----------- | --- | ------- |
|          |               |     |            |          |     |              |     | becomes | a less | attractive | option | for | the purpose | of  | fitting |
dataparallelism,thatachievesthecomputation/communication large models alone. ZeRO-DP is at least as effective on
| efficiency | of DP | while | achieving | memory | efficiency | of  | MP. |     |     |     |     |     |     |     |     |
| ---------- | ----- | ----- | --------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
reducingper-devicememoryfootprintasMP,ormoreeffective
ZeRO-DPremovesthememorystateredundanciesacrossdata-
|     |     |     |     |     |     |     |     | sometimes | when | MP  | cannot divide | the | model | evenly. | It also |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ---- | --- | ------------- | --- | ----- | ------- | ------- |
parallel processes by partitioning the model states instead of has comparable or better scaling efficiency. Furthermore, data
| replicating | them, | and | it retains | the compute/communication |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | ----- | --- | ---------- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
parallelismissoeasytousethatitiswidelyapplicableacross
efficiencybyretainingthecomputationalgranularityandcom- different workloads, while MP approaches today often need
| munication | volume | of        | DP using | a dynamic |     | communication |     |           |            |       |            |                 |           |            |        |
| ---------- | ------ | --------- | -------- | --------- | --- | ------------- | --- | --------- | ---------- | ----- | ---------- | --------------- | --------- | ---------- | ------ |
|            |        |           |          |           |     |               |     | some work | from       | model | developers |                 | to revise | their      | model, |
| schedule   | during | training. |          |           |     |               |     |           |            |       |            |                 |           |            |        |
|            |        |           |          |           |     |               |     | system    | developers | to    | work       | out distributed |           | operators, | and    |
ZeRO-DP has three main optimization stages (as depicted existing work like Megatron-LM only supports a limited set
inFigure1),whichcorrespondtothepartitioningofoptimizer
|     |     |     |     |     |     |     |     | of operators | and | models. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------- | --- | --- | --- | --- | --- |
states, gradients, and parameters (Sec. V). When enabled That being said, there are still cases where we want to
cumulatively:
|                                |     |     |     |     |                      |     |     | leverage | MP: | i) When | used | with ZeRO-R, |     | MP can | reduce |
| ------------------------------ | --- | --- | --- | --- | -------------------- | --- | --- | -------- | --- | ------- | ---- | ------------ | --- | ------ | ------ |
| 1)OptimizerStatePartitioning(P |     |     |     |     | ):4xmemoryreduction, |     |     |          |     |         |      |              |     |        |        |
os activation memory footprint for very large models. ii) For
same communication volume as DP; smaller models where activation memory is not an issue, MP
| 2)AddGradientPartitioning(P |     |     |     |     | ):8xmemoryreduction, |     |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
os+g can also have benefits when aggregated batch size using DP
same communication volume as DP; alone is too big to have good convergence.1 In those case,
| 3) Add | Parameter | Partitioning |     | (P     | ):  | Memory | reduc- |         |         | ZeRO |      |       |         |       |         |
| ------ | --------- | ------------ | --- | ------ | --- | ------ | ------ | ------- | ------- | ---- | ---- | ----- | ------- | ----- | ------- |
|        |           |              |     | os+g+p |     |        |        | one can | combine |      | with | MP to | fit the | model | with an |
tionislinearwithDPdegreeN
|     |     |     |     | d .Forexample,splittingacross |     |     |     | acceptable | aggregated |     | batch size. |     |     |     |     |
| --- | --- | --- | --- | ----------------------------- | --- | --- | --- | ---------- | ---------- | --- | ----------- | --- | --- | --- | --- |
64GPUs(N =64)willyielda64xmemoryreduction.There We show that ZeRO can be combined with MP, resulting in
d
| is a modest | 50% | increase | in  | communication |     | volume. |     |                                   |     |     |     |     |      |               |     |
| ----------- | --- | -------- | --- | ------------- | --- | ------- | --- | --------------------------------- | --- | --- | --- | --- | ---- | ------------- | --- |
|             |     |          |     |               |     |         |     | amaxtheoreticalmemoryreductionofN |     |     |     |     | d ×N | m timesoneach |     |
ZeRO-DP eliminates memory redundancies and makes the device with a DP degree of N and MP degree of N . This
|                |     |        |          |      |         |            |      |     |     |     |     | d   |     |     | m   |
| -------------- | --- | ------ | -------- | ---- | ------- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
| full aggregate |     | memory | capacity | of a | cluster | available. | With |     |     |     |     |     |     |     |     |
couldallowustofitatrillionparametermodelon1024GPUs
all three stages enabled, ZeRO can train a trillion-parameter with16-waymodelparallelism(withineachDGX2node)and
model on just 1024 NVIDIA GPUs. A trillion-parameter 64-way data parallelism across nodes, and run it efficiently
| model with | an  | optimizer | like | Adam | [6] in | 16-bit precision |     |         |        |       |       |     |     |     |     |
| ---------- | --- | --------- | ---- | ---- | ------ | ---------------- | --- | ------- | ------ | ----- | ----- | --- | --- | --- | --- |
|            |     |           |      |      |        |                  |     | using a | modest | batch | size. |     |     |     |     |
requires approximately 16 terabytes (TB) of memory to hold Implementation & Evaluation The complete set of opti-
the optimizer states, gradients, and parameters. 16TB divided mizations in ZeRO could allow us to run models with trillion
| by 1024 | is 16GB, | which | is  | well within | a reasonable |     | bound |     |     |     |     |     |     |     |     |
| ------- | -------- | ----- | --- | ----------- | ------------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
1Prior
for a GPU (e.g., with 32GB of on-device memory). work [9] shows, very large batch size could slow down conver-
|            |     |          |       |        |     |               |     | gence. For       | given | model and  | data, there | is a       | measure of   | critical-batch | size,    |
| ---------- | --- | -------- | ----- | ------ | --- | ------------- | --- | ---------------- | ----- | ---------- | ----------- | ---------- | ------------ | -------------- | -------- |
| Optimizing |     | Residual | State | Memory |     | After ZeRO-DP |     |                  |       |            |             |            |              |                |          |
|            |     |          |       |        |     |               |     | where increasing |       | batch size | further     | slows down | convergence. | The            | detailed |
boosts memory efficiency for model states, the rest of the discussionofthistopicisbeyondthescopeofthepaper.

Fig.2:ZeROtrainingthroughputandspeedupw.r.tSOTAbaselinefor Fig.3:SuperlinearscalabilityandperGPUtraining
| varying | model | sizes. | For ZeRO, | the | MP  | always | fit in a node, | while |     |     |     |     |     |     |     |
| ------- | ----- | ------ | --------- | --- | --- | ------ | -------------- | ----- | --- | --- | --- | --- | --- | --- | --- |
throughputofa60BparametermodelusingZeRO-
| for baseline, |     | models | larger | than 20B | require | MP  | across nodes. |     | 100B. |     |     |     |     |     |     |
| ------------- | --- | ------ | ------ | -------- | ------- | --- | ------------- | --- | ----- | --- | --- | --- | --- | --- | --- |
parameters on the high-end hardware cluster today (e.g., with optimization library — DeepSpeed2. We have released all
1024 V100 GPUs), however, the hardware compute capacity implementations described in this paper and we will extend
is still too limited and training time can be impractically long it further to support 1 trillion parameters by enabling ZeRO-
(>1 year). Therefore, our focus for this implementation is DPstage3partitioningparameters(P ).WemakeZeRO
os+g+p
to efficiently support models with 10x parameters (∼100B fullyaccessibletotheDLcommunitytocatalyzetheevolution
parameters) than state-of-the-art (SOTA) while still being and democratization of large model training at scale. While
within reach of the compute capabilities of current hardware. we evaluate ZeRO in the context of NLP in this paper, our
WeimplementandevaluateasubsetofoptimizationsinZeRO approach is model agnostic and is applicable across different
calledZeRO-100B—P ofZeRO-DPplusZeRO-R—that tasks and DL domains.
os+g
| allow us | to achieve | this | goal. | The results |     | show: |     |     |     |     |     |     |     |     |     |
| -------- | ---------- | ---- | ----- | ----------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
II. RELATEDWORK
| Model | Size | Combined | with | MP, | ZeRO-100B |     | runs 170B |     |     |     |     |     |     |     |     |
| ----- | ---- | -------- | ---- | --- | --------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
parameter models efficiently, while the existing system like A. Data, Model and Pipeline Parallelism
| using Megatron |      | alone    | cannot    | scale   | efficiently | beyond  | 20B         |                 |      |             |      |          |             |                   |        |
| -------------- | ---- | -------- | --------- | ------- | ----------- | ------- | ----------- | --------------- | ---- | ----------- | ---- | -------- | ----------- | ----------------- | ------ |
|                |      |          |           |         |             |         |             | Parallelization |      | is a        | key  | strategy | on training | large             | models |
| parameters,    | as   | shown    | in Figure | 2. This | is          | an over | 8x increase |                 |      |             |      |          |             |                   |        |
|                |      |          |           |         |             |         |             | at scale.       | For  | a model     | that | fits     | in the      | device memory     | for    |
| in model       | size | compared | to SOTA.  |         |             |         |             |                 |      |             |      |          |             |                   |        |
|                |      |          |           |         |             |         |             | training,       | data | parallelism | (DP) | is       | used        | to scale training | to     |
SpeedImprovedmemoryefficiencypowershigherthrough-
|     |     |     |     |     |     |     |     | multiple | devices. | In  | DP, model | parameters |     | are replicated | on  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | --- | --------- | ---------- | --- | -------------- | --- |
putandfastertraining.AsshowninFigure2,ZeROruns100B
|           |        |          |       |           |             |     |              | each     | device. | At each       | step,        | a mini-batch |             | is divided | evenly    |
| --------- | ------ | -------- | ----- | --------- | ----------- | --- | ------------ | -------- | ------- | ------------- | ------------ | ------------ | ----------- | ---------- | --------- |
| parameter | models | on       | a 400 | Nvidia    | V100        | GPU | cluster with |          |         |               |              |              |             |            |           |
|           |        |          |       |           |             |     |              | across   | all the | data parallel |              | processes,   | such        | that each  | process   |
| over 38   | TFlops | per GPU, | and   | aggregate | performance |     | over 15      |          |         |               |              |              |             |            |           |
|           |        |          |       |           |             |     |              | executes | the     | forward       | and backward |              | propagation | on a       | different |
Petaflops.Thisismorethan10ximprovementintrainingspeed
|          |         |     |          |       |       |     |     | subset    | of data | samples,   | and   | uses     | averaged | gradients | across |
| -------- | ------- | --- | -------- | ----- | ----- | --- | --- | --------- | ------- | ---------- | ----- | -------- | -------- | --------- | ------ |
| compared | to SOTA | for | the same | model | size. |     |     |           |         |            |       |          |          |           |        |
|          |         |     |          |       |       |     |     | processes | to      | update the | model | locally. |          |           |        |
ScalabilityWeobservesuperlinearspeedupintheregimeof
|     |     |     |     |     |     |     |     | When | a   | model does | not | fit in | the device | memory, | model |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | ---------- | --- | ------ | ---------- | ------- | ----- |
64-400GPUs,wheretheperformancemorethandoubleswhen
|           |         |        |          |           |      |              |          | parallelism |            | (MP) [7], | [4], [11] | and   | pipeline   | parallelism | (PP) |
| --------- | ------- | ------ | -------- | --------- | ---- | ------------ | -------- | ----------- | ---------- | --------- | --------- | ----- | ---------- | ----------- | ---- |
| we double | the     | number | of GPUs. | This      | is a | property     | of ZeRO- |             |            |           |           |       |            |             |      |
|           |         |        |          |           |      |              |          | [12],       | [13] split | the model |           | among | processes, | in vertical | and  |
| DP which  | reduces | the    | memory   | footprint |      | of the model | states   |             |            |           |           |       |            |             |      |
horizontalwayrespectively.Sec.IdiscussedhowZeROrelates
| as we increase  |             | the DP       | degree    | allowing         | us       | to fit   | larger batch |               |                 |                   |                         |             |        |                     |          |
| --------------- | ----------- | ------------ | --------- | ---------------- | -------- | -------- | ------------ | ------------- | --------------- | ----------------- | ----------------------- | ----------- | ------ | ------------------- | -------- |
|                 |             |              |           |                  |          |          |              | to DP         | and             | MP. We            | now                     | discuss     | PP and | how it relates      | to       |
| sizes per       | GPU         | resulting    | in better | performance.     |          | We       | expect this  |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | reducing      | memory          | consumption.      |                         |             |        |                     |          |
| behaviour       | to continue |              | further   | as we            | increase | the      | number of    |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | PP            | splits          | a model           | horizontally            |             | across | layers running      | each     |
| GPUs beyond     |             | 400.         |           |                  |          |          |              |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | partition     | on              | a different       | device                  | and         | use    | micro-batching      | to       |
| Democratization |             | of           | Large     | Model            | Training |          | ZeRO-100B    |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | hide          | the pipeline    | bubble            |                         | [12], [13]. | Model  | functionalities     |          |
| powers          | data        | scientist    | to train  | models           |          | with     | up to 13B    |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | such          | as tied-weights |                   | and batch-normalization |             |        | are difficult       | to       |
| parameters      | without     | any          | MP        | or               | PP that  | requires | model        |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | implement     |                 | due to horizontal |                         | splitting   |        | and micro-batching, |          |
| refactoring,    | where       | 13B          | is more   | parameters       |          | than     | the largest  |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | respectively. |                 | Popular           | PP implementation       |             |        | such as G-pipe      | [12]     |
| model in        | literature  | (T5          | with      | 11B parameters). |          | Data     | scientists   |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | partitions    | both            | model             | parameters              |             | and    | total activations   | but      |
| can thus        | experiment  |              | freely    | with             | large    | models   | without      |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | requires      | a               | batch size        | proportional            |             | to     | number of           | pipeline |
| worrying        | about       | parallelism. |           | In comparison,   |          | existing | systems      |               |                 |                   |                         |             |        |                     |          |
|                 |             |              |           |                  |          |          |              | partitions    | to              | hide the          | pipeline                | bubble.     | The    | large batch         | size     |
| (e.g., PyTorch  |             | Distributed  | Data      | Parallel)        | runs     | out      | of memory    |               |                 |                   |                         |             |        |                     |          |
canaffecttheconvergencerate,whilealsorequiringsignificant
| with 1.4B | parameter | models. |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------- | --------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
memorytostoreactivations.AdifferentimplementationofPP
| New              | SOTA | Model     | ZeRO   | powers | Turing-NLG |            | [10], the |              |     |            |          |     |        |                     |     |
| ---------------- | ---- | --------- | ------ | ------ | ---------- | ---------- | --------- | ------------ | --- | ---------- | -------- | --- | ------ | ------------------- | --- |
|                  |      |           |        |        |            |            |           | in PipeDream |     | [14] keeps | multiple |     | copies | of stale parameters |     |
| largest language |      | model     | at the | time   | with 17B   | parameters | and       |              |     |            |          |     |        |                     |     |
| record-breaking  |      | accuracy. |        |        |            |            |           |              |     |            |          |     |        |                     |     |
2GitHubRepository:https://github.com/microsoft/deepspeed,
We share ZeRO as a part of our open source DL training DeepSpeedWebsite:https://www.deepspeed.ai/

to hide the pipeline bubble without increasing the batch size singleGPUwith32GBmemoryusingTensorfloworPyTorch.
significantly,makingitlessmemoryefficient.Additionally,the One may wonder where all the memory goes. During model
implementation is not equivalent to the standard DL training training, most of the memory is consumed by model states,
and has implications on training convergence. In contrast, i.e., tensors comprising of optimizer states, gradients, and
ZeRO obtains the same or better memory efficiency than PP parameters.Besidesthesemodelstates,therestofthememory
without incurring functionality, performance and convergence isconsumedbyactivations,temporarybuffersandfragmented
related restrictions of PP. memorywhichwecallresidualstates.Welookatthememory
consumption from both in details.
B. Non-parallelism based approach to reduce memory
In addition to MP and PP, there are multiple lines of work
A. Model States: Optimizer States, Gradients and Parameters
that target reducing memory overheads of DL training.
1) Reducing Activation Memory: Multiple efforts have fo-
The majority of the device memory is consumed by model
cusedonreducingthememoryfootprintofactivationsthrough
statesduringtraining.Considerforinstance,Adam[6],oneof
compression [15], activation checkpointing [8], [16], or live
the most popular optimizers for DL training. Adam requires
analysis [17]. These efforts are complimentary and can work
storing two optimizer states, i) the time averaged momentum
together with ZeRO. In fact, activation memory reduction in
and ii) variance of the gradients to compute the updates.
ZeRO-R works in parallel with activation checkpointing.
Therefore, to train a model with ADAM, there has to be
2) CPUOffload: [18],[19]exploitheterogeneousnatureof
enough memory to hold a copy of both the momentum and
today’s compute nodes, offloading model states to CPU mem-
varianceofthegradients.Inaddition,thereneedstobeenough
ory through algorithmic design, virtualized memory, respec-
memory to store the gradients and the weights themselves.
tively.Upto50%oftrainingtimecanbespentonGPU-CPU-
Of these three types of the parameter-related tensors, the
GPUtransfers[18].ZeROdiffersinthatitreducesthememory
optimizer states usually consume the most memory, specially
consumption significantly without storing the model states to
when mixed-precision training is applied.
CPUmemorywhosebandwidthisseverelyconstraineddueto
Mixed-Precision Training The state-of-the-art approach
PCI-E. On rare cases, ZeRO-R may offload just the activation
to train large models on the current generation of NVIDIA
checkpointsforverylargemodelstoimproveperformance(see
GPUs is via mixed precision (fp16/32) training [26], where
Sec. VI-A for details). [20] uses graph rewriting to offload
parameters and activations are stored as fp16, enabling the
activations to CPU memory, but does not reduce memory
use of the high throughput tensor core units [27] on these
required by model states.
GPUs. During mixed-precision training, both the forward and
3) Memory Efficient Optimizer: [21], [22] focus on reduc-
backward propagation are performed using fp16 weights and
ing memory consumption of adaptive optimization methods
activations. However, to effectively compute and apply the
by maintaining coarser-grained statistics of model parameters
updates at the end of the backward propagation, the mixed-
and gradients, with potential impact on model convergence
precision optimizer keeps an fp32 copy of the parameters as
guarantees. ZeRO is orthogonal to these efforts, and its op-
well as an fp32 copy of all the other optimizer states.
timizations do not change the model optimization method
Let’s take Adam as a concrete example. Mixed precision
or affect model convergence, but effectively reduce memory
training of a model with Ψ parameters using Adam requires
footprint of optimizer states and gradients per device.
enough memory to hold an fp16 copy of the parameters and
C. Training Optimizers the gradients, with memory requirements of 2Ψ and 2Ψ bytes
respectively. In addition, it needs to hold the optimizer states:
Adaptive optimization methods [23], [6], [24], [25] are
anfp32copyoftheparameters,momentumandvariance,with
crucial to achieving SOTA performance and accuracy for
memory requirements of 4Ψ, 4Ψ, and 4Ψ bytes, respectively.
effective model training of large models. Compared to SGD,
Let’s use K to denote the memory multiplier of the optimizer
bymaintainingfine-grainedfirst-orderandsecond-orderstatis-
states, i.e., the additional memory required to store them is
tics for each model parameter and gradient at the cost of
KΨ bytes. Mixed-precision Adam has K = 12. In total, this
significant memory footprint. ZeRO can reduce the memory
resultsin2Ψ+2Ψ+KΨ=16Ψbytesofmemoryrequirement.
footprint of these optimizers by orders of magnitude, making
For a model such as GPT-2 with 1.5 Billion parameters, this
thesesophisticatedoptimizationmethodspracticalfortraining
leads to a memory requirement of at least 24GB, which is
largemodelsonhardwarewithmodestdevicememory.Italso
significantlyhigherthanthemeager3GB ofmemoryrequired
makes it possible to develop and use even more complex and
to hold the fp16 parameters alone.
memory hungry optimizers that may have better convergence.
III. WHEREDIDALLTHEMEMORYGO?
B. Residual Memory Consumption
Let’s take a step back to examine the memory consumption
of the current training system. For example, a 1.5B parameter Activationscantakeupasignificantamountofmemory[8]
GPT-2 model requires 3GB of memory for its weights (or during training. As a concrete example, the 1.5B parameter
parameters) in 16-bit precision, yet, it cannot be trained on a GPT-2 model trained with sequence length of 1K and batch

size of 32 requires about 60GB of memory3. Activation the scalability across GPUs, especially when crossing node
checkpointing (or activation recomputation) is a common boundaries.Onthecontrary,DPhasbothhighercomputational
approach to reduce the activation memory by approximately granularity and lower communication volume, allowing for
the square root of the total activations at the expense of 33% much higher efficiency.
re-computationoverhead[8].Thiswouldreducetheactivation b) DP is memory inefficient as model states are stored
memory consumption of this model to about 8GB. redundantlyacrossalldata-parallelprocesses.Onthecontrary,
Despite the significant reduction, the activation memory MP partitions the model states to obtain memory efficiency.
can grow quite large for bigger models even with activation c) Both DP and MP keep all the model states needed over
checkpointing. For example, a GPT-like model with 100 the entire training process, but not everything is required all
billionparametersrequiresaround60GBofmemoryforbatch thetime.Forexample,parameterscorrespondingtoeachlayer
size 32, even when using activation checkpointing. is only needed during the forward propagation and backward
Temporary buffers used for storing intermediate results propagation of the layer.
consumes non-trivial amount of memory for large models. Based on these insights, ZeRO-DP retains the training
Operations such as gradient all-reduce, or gradient norm efficiency of DP while achieving the memory efficiency of
computationtendtofuseallthegradientsintoasingleflattened MP.ZeRO-DPpartitionsthemodelstatesinsteadofreplicating
buffer before applying the operation in an effort to improve them(SectionV)andusesadynamiccommunicationschedule
throughput. For example, the bandwidth of all-reduce across that exploits the intrinsically temporal nature of the model
devicesimproveswithlargemessagesizes.Whilethegradient states while minimizing the communication volume (Section
themselvesareusuallystoredasfp16tensors,thefusedbuffer VII). By doing so, ZeRO-DP reduces per-device memory
can be an fp32 tensor depending on the operation. When the footprint of a model linearly with the increased DP degree
size of the model is large, these temporary buffer sizes are while maintaining the communication volume close to that of
non-trivial. For example, for a model with 1.5B parameters, a the default DP, retaining the efficiency.
| flattened     | fp32 buffer would | required | 6GB       | of memory.     |     |             |            |           |         |     |              |     |      |
| ------------- | ----------------- | -------- | --------- | -------------- | --- | ----------- | ---------- | --------- | ------- | --- | ------------ | --- | ---- |
|               |                   |          |           |                |     | B. Insights | and        | Overview: | ZeRO-R  |     |              |     |      |
| Memory        | Fragmentation:    | So       | far we    | have discussed | the |             |            |           |         |     |              |     |      |
|               |                   |          |           |                |     | 1) Reducing | Activation |           | Memory: | Two | key insights |     | are: |
| actual memory | consumption       | during   | training. | Additionally,  |     | it          |            |           |         |     |              |     |      |
is possible to run out of usable memory even when there is a) MP partitions the model states but often requires repli-
plenty of available memory. This can happen with memory cation of the activation memory. For example, if we split
fragmentation. A request for a memory will fail if there the parameters of a linear layer vertically and compute them
isn’t enough contiguous memory to satisfy it, even if the in parallel across two GPUs, each GPU requires the entire
|                 |        |           |                 |     |         | activation | to compute | its | partition |     |     |     |     |
| --------------- | ------ | --------- | --------------- | --- | ------- | ---------- | ---------- | --- | --------- | --- | --- | --- | --- |
| total available | memory | is larger | than requested. | We  | observe |            |            |     |           |     |     |     |     |
significant memory fragmentation when training very large b) For models such as GPT-2 or larger, the arithmetic
models, resulting in out of memory issue with over 30% of intensity (ratio of the amount of computation per iteration to
memory still available in some extreme cases. amount of activation checkpoints per iteration) is very large
(≥10K)andincreaseslinearlywithhiddendimensionmaking
IV. ZeRO:INSIGHTSANDOVERVIEW it possible to hide the data-movement cost for the activation
ZeRO has two sets of optimizations: i) ZeRO-DP aimed checkpoints, even when the bandwidth is low.
at reducing the memory footprint of the model states, and ZeRO removes the memory redundancies in MP by parti-
ii) ZeRO-R targeted towards reducing the residual memory tioning the activations checkpoints across GPUs, and uses all-
consumption. We present an overview of the optimizations gathertoreconstructthemondemand.Theactivationmemory
|     |     |     |     |     |     | footprint | is reduced | proportional |     | to the | MP degree. | For | very |
| --- | --- | --- | --- | --- | --- | --------- | ---------- | ------------ | --- | ------ | ---------- | --- | ---- |
andtheinsightsbehind,whichallowsZeROtoreducememory
footprint while remaining efficient. Please note efficiency is a large models, ZeRO can even choose to move the activation
key here: without this constraint, trivial solutions like moving partitions to the CPU memory, while still achieving good
all the parameter states to the CPU memory, or increasing the efficiency due to large arithmetic intensity in these models.
MP degree arbitrarily can reduce memory footprint. 2) Managing Temporary buffers: ZeRO-R uses constant
|             |               |          |       |               |     | size buffers      | to avoid        | temporary |       | buffers | from blowing |        | up as |
| ----------- | ------------- | -------- | ----- | ------------- | --- | ----------------- | --------------- | --------- | ----- | ------- | ------------ | ------ | ----- |
| A. Insights | and Overview: | ZeRO-DP  |       |               |     |                   |                 |           |       |         |              |        |       |
|             |               |          |       |               |     | the model         | size increases, |           | while | making  | them large   | enough | to    |
| ZeRO        | powered DP is | based on | three | key insights: |     | remain efficient. |                 |           |       |         |              |        |       |
a) DP has better scaling efficiency than MP because MP 3) Managing fragmented Memory: Memory fragmentation
reduces the granularity of the computation while also in- is a result of interleaving between short lived and long lived
|     |     |     |     |     |     | memory | objects. | During | the forward | propagation |     | activation |     |
| --- | --- | --- | --- | --- | --- | ------ | -------- | ------ | ----------- | ----------- | --- | ---------- | --- |
creasingthecommunicationoverhead.Beyondacertainpoint,
lower computational granularity reduces the efficiency per checkpoints are long lived but the activations that recomputed
GPU, while the increased communication overhead, hiders are short lived. Similarly, the backward computation, the acti-
|     |     |     |     |     |     | vation gradients |     | are short | lived while | the | parameter | gradients |     |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | --------- | ----------- | --- | --------- | --------- | --- |
3Theactivationmemoryofatransformer-basedmodelisproportionalto
|            |                       |          |            |            |        | are long | lived. Based | on  | this insight, | ZeRO | performs | on-the- |     |
| ---------- | --------------------- | -------- | ---------- | ---------- | ------ | -------- | ------------ | --- | ------------- | ---- | -------- | ------- | --- |
| the number | of transformer layers | × hidden | dimensions | × sequence | length |          |              |     |               |      |          |         |     |
flymemorydefragmentationbymovingactivationcheckpoints
| × batch | size. For a GPT-2 | like architecture | the | total activations | is about |     |     |     |     |     |     |     |     |
| ------- | ----------------- | ----------------- | --- | ----------------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
12×hidden dim×batch×seq length×transformer layers. and gradients to pre-allocated contiguous memory buffers.

|     |     |     |     |     |     |     |     | 7.5BModel(GB) |     |     | 128BModel(GB) |     |     | 1TModel(GB) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | ------------- | --- | --- | ----------- | --- |
DP
|     |     |     |     |     |     |       | Pos     | Pos+g      | Pos+g+p | Pos    | Pos+g       | Pos+g+p | Pos   | Pos+g     | Pos+g+p |
| --- | --- | --- | --- | --- | --- | ----- | ------- | ---------- | ------- | ------ | ----------- | ------- | ----- | --------- | ------- |
|     |     |     |     |     |     |       | 1 120   | 120        | 120     | 2048   | 2048        | 2048    | 16000 | 16000     | 16000   |
|     |     |     |     |     |     |       | 4 52.5  | 41.3       | 30      | 896    | 704         | 512     | 7000  | 5500      | 4000    |
|     |     |     |     |     |     |       | 16 35.6 | 21.6       | 7.5     | 608    | 368         | 128     | 4750  | 2875      | 1000    |
|     |     |     |     |     |     |       | 64 31.4 | 16.6       | 1.88    | 536    | 284         | 32      | 4187  | 2218      | 250     |
|     |     |     |     |     |     | 256   | 30.4    | 15.4       | 0.47    | 518    | 263         | 8       | 4046  | 2054      | 62.5    |
|     |     |     |     |     |     | 1024  | 30.1    | 15.1       | 0.12    | 513    | 257         | 2       | 4011  | 2013      | 15.6    |
|     |     |     |     |     |     | TABLE | I:      | Per-device |         | memory | consumption |         | of    | different | opti-   |
mizationsinZeRO-DPasafunctionofDPdegree.Bold-faced
|     |     |     |     |     |     | text    | are the | combinations |      | for   | which | the | model | can fit | into a |
| --- | --- | --- | --- | --- | --- | ------- | ------- | ------------ | ---- | ----- | ----- | --- | ----- | ------- | ------ |
|     |     |     |     |     |     | cluster | of      | 32GB         | V100 | GPUs. |       |     |       |         |        |
processattheendofeachtrainingsteptogetthefullyupdated
|     |     |     |     |     |     | parameters  |        | across   | all        | data parallel |              | process.  |         |          |        |
| --- | --- | --- | --- | --- | --- | ----------- | ------ | -------- | ---------- | ------------- | ------------ | --------- | ------- | -------- | ------ |
|     |     |     |     |     |     |             | Memory | Savings: |            | As            | shown        | in Figure | 1,      | the      | memory |
|     |     |     |     |     |     | consumption |        | after    | optimizing |               | state        | partition |         | reduces  | from   |
|     |     |     |     |     |     | 4Ψ+KΨ       |        | to 4Ψ+   | KΨ.        | As            | the concrete |           | example | depicted | in     |
Nd
Figure1,a7.5Bparametermodelrequires31.4GBofmemory
|     |     |     |     |     |     | usingP                           |     | with64-wayDP(N |         |        | =64),whilerequiring120GB |      |                   |            |       |
| --- | --- | --- | --- | --- | --- | -------------------------------- | --- | -------------- | ------- | ------ | ------------------------ | ---- | ----------------- | ---------- | ----- |
|     |     |     |     |     |     |                                  | os  |                |         |        | d                        |      |                   |            |       |
|     |     |     |     |     |     | withstandardDP.Furthermore,whenN |     |                |         |        |                          | d    | islarge,thememory |            |       |
|     |     |     |     |     |     | requirement                      |     | on             | model   | states | reduces                  | from | 4Ψ+12Ψ            |            | = 16Ψ |
|     |     |     |     |     |     | bytes                            | to  | 4Ψ+            | 12Ψ ≈4Ψ | bytes, | leading                  |      | to a 4x           | reduction. |       |
Nd
|     |     |     |     |     |     | B.  | P : Gradient |     | Partitioning |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ | --- | --- | --- | --- | --- | --- |
g
Aseachdataparallelprocessonlyupdatesitscorresponding
parameterpartition,itonlyneedsthereducedgradientsforthe
correspondingparameters.Therefore,aseachgradientofeach
|                     |              |     |         |      |           | layer     | becomes   | available         |     | during     | the         | backward  | propagation, |               | we      |
| ------------------- | ------------ | --- | ------- | ---- | --------- | --------- | --------- | ----------------- | --- | ---------- | ----------- | --------- | ------------ | ------------- | ------- |
|                     |              |     |         |      |           | only      | reduce    | them              | on  | the data   | parallel    | process   |              | responsible   | for     |
|                     |              |     |         |      |           | updating  |           | the corresponding |     |            | parameters. |           | After        | the reduction |         |
|                     |              |     |         |      |           | we        | no longer | need              | the | gradients  |             | and their | memory       |               | can be  |
|                     |              |     |         |      |           | released. |           | This reduces      |     | the memory |             | footprint | required     |               | to hold |
|                     |              |     |         |      |           | the       | gradients | from              | 2Ψ  | bytes      | to 2Ψ       | .         |              |               |         |
| Fig. 4: Pseudo-code | representing |     | ZeRO-DP | with | all three |           |           |                   |     |            | N           |           |              |               |         |
d
phases, P , enabled. Effectively this is a Reduce-Scatter operation, where gra-
os+g+p
|     |     |     |     |     |     | dients    | corresponding |          |     | to different |      | parameters | are       | reduced | to        |
| --- | --- | --- | --- | --- | --- | --------- | ------------- | -------- | --- | ------------ | ---- | ---------- | --------- | ------- | --------- |
|     |     |     |     |     |     | different |               | process. | To  | make         | this | more       | efficient | in      | practice, |
Thisnotonlyincreasesmemoryavailabilitybutalsoimproves
|     |     |     |     |     |     | we  | use a | bucketization |     | strategy, |     | where | we bucketize |     | all the |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------------- | --- | --------- | --- | ----- | ------------ | --- | ------- |
efficiency by reducing the time it takes for the memory gradients corresponding to a particular partition, and perform
allocator to find free contiguous memory. reduction on the entire bucket at once. This is similar in spirit
|     |     |     |     |     |     | to  | how | NVIDIA’s | AMP | [28] | optimizer |     | bucketizes |     | the all- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | ---- | --------- | --- | ---------- | --- | -------- |
V. DEEPDIVEINTOZeRO-DP
|           |                      |            |     |           |        | reduce       | gradient |     | computation |     | to overlap |     | communication |         | and |
| --------- | -------------------- | ---------- | --- | --------- | ------ | ------------ | -------- | --- | ----------- | --- | ---------- | --- | ------------- | ------- | --- |
| While the | existing DP approach | replicates |     | the model | states |              |          |     |             |     |            |     |               |         |     |
|           |                      |            |     |           |        | computation. |          | In  | our case    | we  | perform    | a   | reduction     | instead | of  |
at each device and introduces significant memory overhead, an all-reduce at the partition boundaries to reduce memory
ZeRO-DP eliminates this memory redundancy by partitioning footprint and overlap computation and communication.
| them — optimizer | states, gradients |     | and parameters |     | — across |     |     |     |     |     |     |     |     |     |     |
| ---------------- | ----------------- | --- | -------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MemorySavings:Byremovingbothgradientandoptimizer
data parallel processes. Figure 1 quantifies and visualizes the state redundancy, we reduce the memory footprint further
| memory requirement | with and | without | ZeRO-DP. |     | The figure |      |     |      | 14Ψ |       |        |         |     |           |      |
| ------------------ | -------- | ------- | -------- | --- | ---------- | ---- | --- | ---- | --- | ----- | ------ | ------- | --- | --------- | ---- |
|                    |          |         |          |     |            | down | to  | 2Ψ + |     | ≈ 2Ψ. | As the | example |     | in Figure | 1, a |
shows the memory footprint after partitioning (1) optimizer N d
|     |     |     |     |     |     | 7.5 | B parameter |     | m odel | requires |     | only | 16.6 GB | of  | memory |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | -------- | --- | ---- | ------- | --- | ------ |
state, (2) gradient and (3) parameter redundancies accumula- using P with 64-way DP (N = 64), while requiring
|                  |            |           |              |     |           |     | os+g |               |     |     |      | d   |           |     |        |
| ---------------- | ---------- | --------- | ------------ | --- | --------- | --- | ---- | ------------- | --- | --- | ---- | --- | --------- | --- | ------ |
| tively. We refer | to them as | the three | optimization |     | phases of |     |      |               |     |     |      |     |           |     |        |
|                  |            |           |              |     |           | 120 | GB   | with standard |     | DP. | When | N d | is large, | the | memory |
ZeRO-DP: P os , P g , and P p , which we elaborate below. requirement of model states reduces from 2Ψ+14Ψ = 16Ψ
|                     |                    |       |               |     |             |       |               | 2Ψ+ | 14Ψ ≈2Ψ      |        |         |     |         |            |     |
| ------------------- | ------------------ | ----- | ------------- | --- | ----------- | ----- | ------------- | --- | ------------ | ------ | ------- | --- | ------- | ---------- | --- |
|                     |                    |       |               |     |             | bytes | to            |     |              | bytes, | leading |     | to a 8x | reduction. |     |
| A. P os : Optimizer | State Partitioning |       |               |     |             |       |               |     | Nd           |        |         |     |         |            |     |
|                     |                    |       |               |     |             | C.    | P : Parameter |     | Partitioning |        |         |     |         |            |     |
| For a DP            | degree of N , we   | group | the optimizer |     | states into |       | p             |     |              |        |         |     |         |            |     |
d
N equalpartitions,suchthattheith dataparallelprocessonly Just as with the optimizer states, and the gradients, each
d
updatestheoptimizerstatescorrespondingtotheith
|     |     |     |     |     | partition. | process | only | stores | the | parameters |     | corresponding |     | to its | parti- |
| --- | --- | --- | --- | --- | ---------- | ------- | ---- | ------ | --- | ---------- | --- | ------------- | --- | ------ | ------ |
Thus,eachdataparallelprocessonlyneedstostoreandupdate tion. When the parameters outside of its partition are required
1 ofthetotaloptimizerstatesandthenonlyupdate 1 ofthe forforwardandbackwardpropagation,theyarereceivedfrom
| Nd  |     |     |     |     | Nd  |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
parameters. We perform an all-gather across the data parallel the appropriate data parallel process through broadcast. While

thismayseemtoincursignificantcommunicationoverheadat Table IV with a batch size of 32, sequence length of 1024
firstglance,weshowthatthisapproachonlyincreasesthetotal and a MP degree of 16. If we checkpoint a single activation
communicationvolumeofabaselineDPsystemto1.5x,while for each transformer layer, it would require about 33 GB of
enablingmemoryreductionproportionaltoN .Therefore,the memory per GPU just to store the activation checkpoints. But
d
memory savings using P is equivalent to the memory with P in ZeRO, it can be reduced to about 2 GB per GPU.
|     |     |     | os+p+g |     |     |     |     | a   |     |     |     |     |     |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
savings from MP, however at a fraction of the communication Furthermore, this 2GB can be offloaded to the CPU reducing
volume as elaborated in Sec. VII-C. the memory footprint for activations to nearly zero.
| Memory | Savings: |     | With parameter |     | partitioning, |     | we reduce |     |     |     |     |     |     |     |     |
| ------ | -------- | --- | -------------- | --- | ------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
thememoryconsumptionofanΨparametermodelfrom16Ψ B. C : Constant Size Buffers
B
16Ψ.
| to       | As the | example        | in Figure | 1,     | a 7.5 B | parameter | model |                                                       |        |     |         |             |     |        |           |
| -------- | ------ | -------------- | --------- | ------ | ------- | --------- | ----- | ----------------------------------------------------- | ------ | --- | ------- | ----------- | --- | ------ | --------- |
| Nd       |        |                |           |        |         |           |       | ZeROcarefullyselectsthesizesofthetemporal-databuffers |        |     |         |             |     |        |           |
| requires | 1.9 GB | of model-state |           | memory | using   | P         | with  |                                                       |        |     |         |             |     |        |           |
|          |        |                |           |        |         | os+p+g    |       | to balance                                            | memory | and | compute | efficiency. |     | During | training, |
(N =64),
| 64-way | DP d |     | while | requiring | 120 | GB with | standard |     |     |     |     |     |     |     |     |
| ------ | ---- | --- | ----- | --------- | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
thecomputationalefficiencyofsomeoperationscanbehighly
| DP. This    | has a          | profound | implication: |           | ZeRO     | powers | DP to      | fit                |      |           |           |         |            |        |           |
| ----------- | -------------- | -------- | ------------ | --------- | -------- | ------ | ---------- | ------------------ | ---- | --------- | --------- | ------- | ---------- | ------ | --------- |
|             |                |          |              |           |          |        |            | dependent          | on   | the input | size,     | with    | larger     | inputs | achieving |
| models      | with arbitrary |          | size—        | as long   | as there | are    | sufficient |                    |      |           |           |         |            |        |           |
|             |                |          |              |           |          |        |            | higher efficiency. |      | For       | example,  | a large | all-reduce |        | operation |
| number      | of devices     | to       | share        | the model | states.  | We     | show the   |                    |      |           |           |         |            |        |           |
|             |                |          |              |           |          |        |            | achieves           | much | higher    | bandwidth | than    | a smaller  | one.   | Hence,    |
| pseudo-code | for            | training | a model      | with      | P        | in     | Figure. 4. |                    |      |           |           |         |            |        |           |
os+p+g to get better efficiency, high performance libraries such as
D. Implication on Model Size NVIDIA Apex or Megatron fuses all the parameters into a
|           |        |     |              |     |       |       |     | single buffer | before |     | applying | these operations. |     | However, | the |
| --------- | ------ | --- | ------------ | --- | ----- | ----- | --- | ------------- | ------ | --- | -------- | ----------------- | --- | -------- | --- |
| The three | phases | of  | partitioning |     | P , P | , and | P   |               |        |     |          |                   |     |          |     |
os os+g os+g+p memory overhead of the fused buffers is proportional to the
reducesthememoryconsumptionofeachdataparallelprocess
|          |             |       |        |             |                   |          |         | model size, | and    | can become |       | inhibiting. | For  | example, | for a 3B |
| -------- | ----------- | ----- | ------ | ----------- | ----------------- | -------- | ------- | ----------- | ------ | ---------- | ----- | ----------- | ---- | -------- | -------- |
| on model | states      | by up | to 4x, | 8x, and     | N d respectively. |          | Table   | I           |        |            |       |             |      |          |          |
|          |             |       |        |             |                   |          |         | parameter   | model, | a 32-bit   | fused | buffer      | will | require  | 12 GB of |
| analyzes | model-state |       | memory | consumption |                   | of a few | example |             |        |            |       |             |      |          |          |
models under the 3 stages of ZeRO-DP optimizations for memory.Toaddressthisissue,weuseaperformance-efficient
|          |            |         |        |            |            |             |           | constant-size | fused    | buffer   | when | the model  | becomes |          | too large, |
| -------- | ---------- | ------- | ------ | ---------- | ---------- | ----------- | --------- | ------------- | -------- | -------- | ---- | ---------- | ------- | -------- | ---------- |
| varying  | DP degree. | Without |        | ZeRO,      | the memory | consumption |           |               |          |          |      |            |         |          |            |
|          |            |         |        |            |            |             |           | similar       | to [29]. | By doing | so,  | the buffer | size    | does not | depend     |
| is equal | to the     | first   | row in | the table, | regardless |             | of the DP |               |          |          |      |            |         |          |            |
onthemodelsize,andbykeepingthebuffersizelargeenough,
| degree.           | Note that,      | with    | N d =64,        | ZeRO | can     | train        | models with |        |               |                 |             |     |     |     |     |
| ----------------- | --------------- | ------- | --------------- | ---- | ------- | ------------ | ----------- | ------ | ------------- | --------------- | ----------- | --- | --- | --- | --- |
|                   |                 |         |                 |      |         |              |             | we can | still achieve | good            | efficiency. |     |     |     |     |
| up to 7.5B,       | 14B,            | and     | 128B parameters |      | using   | P ,          | P , and     |        |               |                 |             |     |     |     |     |
|                   |                 |         |                 |      |         | os           | os+g        |        |               |                 |             |     |     |     |     |
| P                 | , respectively. |         | When            | N =  | 1024,   | ZeRO         | with all of |        |               |                 |             |     |     |     |     |
| os+g+p            |                 |         |                 | d    |         |              |             |        |               |                 |             |     |     |     |     |
|                   |                 |         |                 |      |         |              |             | C. M : | Memory        | Defragmentation |             |     |     |     |     |
| its optimizations |                 | enabled | (P              |      | ) could | train models | with        | D      |               |                 |             |     |     |     |     |
os+g+p
1 trillion parameters. Without ZeRO, the largest model DP Memory fragmentation in model training occurs as a result
alone can run has less than 1.5 Billion parameters. of activation checkpointing and gradient computation. During
|     |     |     |     |     |     |     |     | the forward | propagation |     | with | activation | checkpointing, |     | only |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | --- | ---- | ---------- | -------------- | --- | ---- |
VI. DEEPDIVEINTOZeRO-R
selectedactivationsarestoredforbackpropagationwhilemost
| A. P : | Partitioned | Activation |     | Checkpointing |     |     |     |                                                       |     |     |     |     |     |     |     |
| ------ | ----------- | ---------- | --- | ------------- | --- | --- | --- | ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| a      |             |            |     |               |     |     |     | activationsarediscardedastheycanberecomputedagaindur- |     |     |     |     |     |     |     |
ingthebackpropagation.Thiscreatesaninterleavingofshort
AsdiscussedinIV-B,MPbydesignrequiresareplicationof
theactivations,resultinginredundantcopiesoftheactivations lived memory (discarded activations) and long lived memory
acrossmodelparallelGPUs.ZeROeliminatesthisredundancy (checkpointed activation), leading to memory fragmentation.
|     |     |     |     |     |     |     |     | Similarly, | during | the | backward | propagation, |     | the | parameter |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | --- | -------- | ------------ | --- | --- | --------- |
bypartitioningtheactivations,andonlymaterializesthemina
replicated form one activation layer at a time, right before the gradients are long lived, while activation gradients and any
activation is used in computation. More specifically, once the other buffers required to compute the parameter gradients are
forward propagation for a layer of a model is computed, the short lived. Once again, this interleaving of short term and
input activations are partitioned across all the model parallel long term memory causes memory fragmentation.
process, until it is needed again during the backprogation. At Limited memory fragmentation is generally not an issue,
thispoint,ZeROusesanall-gatheroperationtore-materializea when there is plenty of memory to spare, but for large model
replicatedcopyoftheactivations.Werefertothisoptimization training running with limited memory, memory fragmentation
as P . It works in conjunction with activation checkpointing leads to two issues, i) OOM due to lack of contiguous
a
[8], storing partitioned activation checkpoints only instead memory even when there is enough available memory, ii)
|               |         |     |              |     |        |         |            | poor efficiency |     | as a | result | of the memory | allocator |     | spending |
| ------------- | ------- | --- | ------------ | --- | ------ | ------- | ---------- | --------------- | --- | ---- | ------ | ------------- | --------- | --- | -------- |
| of replicated | copies. |     | Furthermore, |     | in the | case of | very large |                 |     |      |        |               |           |     |          |
models and very limited device memory, these partitioned significant time to search for a contiguous piece of memory
activation checkpoints can also be offloaded to the CPU to satisfy a memory request.
reducing the activation memory overhead to nearly zero at ZeRO does memory defragmentation on-the-fly by pre-
an additional communication cost, which we will discuss in allocating contiguous memory chunks for activation check-
VII. We refer to this as P a+cpu . points and gradients, and copying them over to the pre-
Memory Saving With partitioned activation checkpointing, allocated memory as they are produced. M not only enables
D
ZeRO reduces the activation footprint by a factor proportional ZeRO to train larger models with larger batch sizes, but also
to the MP degree. Consider training a 100B model shown in improves efficiency when training with limited memory.

VII. COMMUNICATIONANALYSISOFZeRO-DP toallthedataparallelprocesses.Oncetheforwardpropagation
As ZeRO boosts model size by removing memory redun- for that partition is done, the parameters can be discarded.
dancy, it is only natural to ask if we are trading communi- The total communication volume is thus Ψ× N N d d =Ψ. In other
words, we reschedule the parameter all-gather by spreading
cation volume for memory efficiency. In other words, what
is the communication volume of ZeRO-powered DP approach it across the entire forward propagation, and discarding the
parameters once they have been used. Note however that
compared to a baseline DP approach? The answer is in two
parts: i) ZeRO-DP incurs no additional communication using this all-gather needs to happen once again for the backward
P and P , while enabling up to 8x memory reduction, ii) propagation in the reverse order.
os g
ZeRO-DP incurs a maximum of 1.5x communication when The total communication volume is therefore the sum of
using P in addition to P and P , while further reducing the communication volumes incurred by these all-gathers in
p os g
the memory footprint by N times. We present the analysis in additiontothecommunicationvolumeincurredbythereduce-
d
this section. We begin by first presenting a brief overview of scatter of the gradients. The total volume is therefore 3Ψ
the communication volume for standard DP. which is 1.5x compared to the baseline. Both gradient and
parameterpartitioningleveragetheinsightthat—notallstates
A. Data Parallel Communication Volume
of gradients and parameters are needed all the time — to
During data parallel training, gradients across all data optimize memory by communicating the states judiciously.
parallel processes are averaged at the end of the backward
propagation before computing the updates for the next step. C. Comparison with MP
The averaging is performed using an all-reduce communica-
MP incurs significantly higher communication compared
tion collective. For a large model size, the all-reduce com-
to ZeRO-DP. The communication volume of MP per mini-
munication is entirely communication bandwidth bound, and
batch is proportional to the number of samples in the batch
therefore, we limit our analysis to the total communication
while communication for ZeRO-DP is a constant w.r.t mini-
volume send to and from each data parallel process.
batch. We show that beyond a very small batch size, MP
State-of-art implementation of all-reduce uses a two-step
incurs significantly higher communication volume compared
approach, where the first step is a reduce-scatter operation,
to ZeRO-DP.
which reduces different part of the data on different process.
MPsplitsthecomputationofeachdatasampleacrossGPUs
The next step is an all-gather operation where each process
at an operator level by partitioning both the computation and
gathersthereduceddataonalltheprocess.Theresultofthese
the parameters corresponding to each forward and backward
two steps is an all-reduce.
operatorinthemodel.Forexample,MPmaypartitionalinear
Both reduce-scatter and all-gather are implemented in a
layer across multiple GPUs each producing partial outputs
pipelined fashion, that results in a total data movement of
f Ψ or × ea N c N d h − d s 1 tep el . e T m h e e n r t e s fo ( r f e o , r th N e d st p a r n o d c a e r s d se D s P an in d c Ψ urs d 2 at × a Ψ el × em N e d n − ts 1 ) o to n ge e t a h c e h rf G or PU ea . ch A s s a a mp re le su ,i l n t, c t u h r e rin o g ut a p c u o ts m n m e u ed nic to ati b o e n c o o v m er b h i e n a e d d .
Nd The exact communication volume depends on model size,
data movement in each training step. For large value of N ,
d
model architecture, activation checkpointing and MP strategy.
this quantity is approximately equal to 2×Ψ.
To share concrete insights, we perform the communication
B. ZeRO-DP Communication Volume volume analysis in the context of transformer based models
1) Communication Volume with P : With gradient par- implemented using SOTA MP approach, Megatron-LM.
os+g
titioning,eachprocessonlystorestheportionofthegradients, In Megatron-LM with activation checkpointing, each trans-
thatisrequiredtoupdateitscorrespondingparameterpartition. former block performs two all-reduce operations of size
Assuch,insteadofanall-reduce,ZeROonlyrequiresascatter- batch×seq length×hidden dimintheforwardpropagation,
reduce operation on the gradients, incurring communication twoall-reduceforforwardre-computationandtwomoreinthe
volume of Ψ. After each process updates the partition of the backward propagation. The total communication per batch is
parametersthatitisresponsiblefor,anall-gatherisperformed 12×batch×seq length×hidden dim×num layerssince
to collect all the updated parameters from all the data parallel communicationvolumeofanall-reduceis2×message size.
process.ThisalsoincursacommunicationvolumeofΨ.Sothe In comparison, the max communication volume for ZeRO-
totalcommunicationvolumepertrainingstepisΨ+Ψ=2Ψ, DP is 3×model parameters, as described in Sec. VII-B2,
exactly the same as the baseline DP. which for a transformer based model is approximately 3 ×
2) CommunicationVolumewithP : Afterparameter 12×hidden dim×hidden dim×num layers, where the
os+g+p
partitioning,eachdataparallelprocessonlystorestheparame- factorof12isdueto4xforQKVandself-outputlinearlayers
ters that it updates. Therefore, during the forward propagation and 8x for intermediate linear layers within each transformer
it needs to receives the parameters for all the other partitions. layer. Therefore, the ratio between communication volume
However,thiscanbepipelinedtoavoidthememoryoverhead. of MP w.r.t. ZeRO-DP is batch×seq length. For language
3×hidden dim
Before computing the forward propagation on the part of the models,seq lengthisgenerallygreaterthan1024,thelargest
modelcorrespondingtoaparticularpartition,thedataparallel hidden dim used in literature at the time of submission is
processresponsibleforthatpartitioncanbroadcasttheweights less than 5K [4]. Therefore, even for models with massive

MaxTheoreticalModelSize MeasuredModelSize
hidden dim, MP incurs higher communication volume be- MP GPUs
Baseline Pos Pos+g Pos+g+p Baseline ZeRO-DP(Pos)
yond a small batch size of a few dozen. LM models are 1 64 2B 7.6B 14.4B 128B 1.3B 6.2B
2 128 4B 15.2B 28.8B 256B 2.5B 12.5B
generally trained with batch size over 512, and at those 4 256 8B 30.4B 57.6B 0.5T 5B 25B
batch size MP would incur over an order-of-magnitude in 8 512 16B 60.8B 115.2B 1T 10B 50B
16 1024 32B 121.6B 230.4B 2T 20B 100B
communication volume overhead compared to ZeRO-DP.
TABLE II: Maximum model size through memory analysis
VIII. COMMUNICATIONANALYSISOFZeRO-R (left) and the measured model size when running with ZeRO-
OS (right). The measured model size with P matches the
We compare the communication volume of partitioned ac- os
theoreticalmaximum,demonstratingthatourmemoryanalysis
tivation checkpointing (P ) in ZeRO-R with baseline MP, and
a
provides realistic upper bounds on model sizes.
show that P incurs a communication volume increase that is
a
ingenerallessthanonetenthofthebaselineMP.Furthermore,
movement to and from CPU memory compared to P . In
we analyze the communication overhead of P in relation a
a
extreme cases where DP communication volume is the major
to DP communication volume to identify scenarios when P
a
bottleneckduetoasmallbatchsizeevenwithP ,P can
improves efficiency by allowing for a larger batch size and a a+cpu
improve efficiency by increasing the batch size as long as the
reducing DP communication. We leverage such analysis to
CPUdatatransferoverheadislessthantheDPcommunication
decide if and when to apply P as well as P .
a a+cpu
volumeoverhead,whichisgenerallytrueforsmallbatchsizes.
Communication volume trade-off of partitioning activation
Given model and hardware characteristics, we leverage the
checkpointsdependsonthemodelsize,checkpointingstrategy
aboveanalysistodecideifandwhentoapplyP andP .
andtheMPstrategy.Onceagain,toprovideconcreteinsights, a a+cpu
we perform the analysis in the context of transformer based
IX. STEPTOWARDS1TRILLIONPARAMETERS
models implemented using SOTA MP approach, Megatron-
LM. The largest published models today are in the range of
In Megatron-LM with activation checkpointing, each trans- 10 billion parameters, which are already challenging to train.
former block performs two all-reduce operations of size Getting to a trillion parameters, 3-orders of magnitude larger,
batch×seq length×hidden dimintheforwardpropagation, will inevitably happen, but the road will be full of hurdles,
twoall-reduceforforwardre-computationandtwomoreinthe surprises and innovations. While we do not claim knowing
backward propagation. The total communication per block is or addressing all of them, ZeRO addresses one of the most
12×batch×seq length×hidden dimsincecommunication fundamental challenges from a system perspective: the ability
volume of an all-reduce is 2×message size. tofitamodelofthisscaleoncurrenthardwarewhileallowing
When ZeRO-R partitions activation checkpoints, it requires it to train with good system scalability.
an additional all-gather operation before the forward recom- A Leap from State-of-Art The largest model that the
putation of the back-propagation on each activation check- state-of-art framework, Megatron, can train with acceptable
point. In general, we checkpoint the input activation for throughputisa16-20BparametermodelinaDGX-2system.
each transformer block, requiring one all-gather per trans- Scaling further by having model parallelism across multiple
former block. The communication overhead P is therefore DGXnodesresultsinsignificantefficiencydropduetolimited
a
batch×seq length×hidden dim, since the communication internode bandwidth.
volume of an all-gather is message size. Therefore, the total ZeRO vastly increase the efficiently-runnable model size. It
communicationoverheadofP islessthan10%oftheoriginal enablesthecurrentgenerationofhardwaretorunsignificantly
a
communication volume for model parallelism. largermodelswithoutrequiringfine-grainedmodelparallelism
When MP is used in conjunction with DP, P can be used to go across the node boundaries. As demonstrated in Table
a
to increase the batch size and reduce the DP communication I, ZeRO, with all optimizations turned on (P ), could
os+g+p
volume by an order of magnitude at the expense of a 10% fit more than 1 Trillion parameters on 1024 GPUs using DP
increase in MP communication volume, This can significantly only. Alternatively, when combined with model parallelism
boost efficiency when DP communication is a performance (as shown in Table II), ZeRO could fit more than 1 Trillion
bottleneck.Moreconcretely,P reducestheactivationmemory parameters on 1024 GPUs with 16-way model parallelism
a
consumption by the MP degree allowing for a proportional (within each DGX2 node) and 64-way data parallelism across
increaseinbatchsize.Forlargemodels,MPcanbeaslargeas nodes. Running a model with a trillion parameters efficiently
16(#GPUsonaDGX-2node),allowingforupto16xincrease is no longer impossible!
in the batch size. The communication volume during the end- Compute Power Gap Training a trillion parameter model
to-end DP training for a fixed number of training samples is end-to-end within an acceptable time range, however, could
inverselyproportionaltothebatchsize.Therefore,anorderof still require significant amount of compute power, which is
magnitude increase in batch size due to P could result in an lacking in today’s AI clusters.
a
order-of-magnitude decrease in DP communication volume. To understand the resource requirement, we present a brief
Finally if P is applied, partitioned activation check- comparison with Bert-Large. Bert-Large can be trained in 67
a+cpu
points are offloaded to CPU, reducing the activation memory minutes on a 1024 GPU DGX-2H cluster [30]. A 1 Trillion
requirement to nearly zero at the expense of 2x added data Parameter model can easily contain 3000x (1 trillion / 330

Fig. 8: Max model throughput Fig. 9: Turing-NLG 17B
Fig. 5: Max model size with ZeRO-DP. enabled by ZeRO.
.
Fig.6:Maxcacheallocated.
Figure 2 Figures 3,8
Layers HD Layers HD
ZeRO-DP ZeRO-R
1.5B 48 1600 1.16B–2.5B 24,34,54 1920
1 Pos CB+MD
8B 72 3072 4B 64 2304
2 Pos CB+MD+Pa
20B 98 4096 6B-8B 52,72 3072
3 Pos+g CB+MD
40B–60B 88,132 6144 10B–13B 50,54,58,62 4096
4 Pos+g CB+MD+Pa
100,125,150,
5 Pos+g CB+MD+Pa+cpu 80B–170B
175,212
8192 60B 75 8192
TABLE III: ZeRO configu-
TABLE IV: Configurations for different model sizes, number
Fig. 7: Throughput per GPU. rations
of layers, and hidden dimensions (HD) across Figures 2, 3, 8.
million)morecomputationthanaBert-Largemodelforadata b) Hardware: We conducted all of our experiments on
sample. Even if we assume the same sequence length and the a cluster of 25 NVIDIA DGX-2 nodes. Each node consisted
totalnumberofsamplesrequiredtotrainthemodel,traininga of 16 NVIDIA Tesla V100-SXM3-32GB GPUs, 8x 100Gbps
1T model would take 140 days, assuming the same hardware MellanoxInfiniBand(ConnectX-5)cards,anddualIntelXeon
and similar computational efficiency. In practice, both data Platinum 8168 2.7 GHz CPUs (24-cores). All 25 nodes are
samples and sequence length are likely to increase with the connectedtogetherwithInfiniBandusinga648-portMellanox
increased model size requiring over a year to train. It would MLNX-OS CS7500 switch.
require an exa-flop system to train a 1T parameter model in c) Baseline: ForexperimentswithoutMP,weusetorch’s
a reasonable time. But when such compute capacity becomes distributed data parallel (DDP) as baseline. For experiments
available, we hope ZeRO will provide the system technology with MP, we use Megatron-LM because it is, to our knowl-
to run the 1T models efficiently. edge, the state-of-art. The most recent Megatron-LM results
report the ability to scale up to 16B parameter models using
X. IMPLEMENTATIONANDEVALUATION 32 DGX-2 nodes (total of 512 32GB V100 GPUs) [4].
d) ZeRO: Experiments without MP use ZeRO-100B
We focus our implementation on supporting efficient train-
only, while experiments with MP combine ZeRO-100B with
ing of models with ∼100B parameters, which are an order-
MP of Megatron-LM.
of-magnitude larger than the largest published models today
e) Model Configurations: The models presented in this
(e.g., T5-11B [5]) while trainable within a reasonable time
section are GPT-2 [3] like transformer based models. We vary
frame on current hardware (e.g., with 1K V100 GPUs). We
the hidden dimension and the number of layers to obtain
implement and evaluate a subset of optimizations in ZeRO—
models with different number of parameters. Table IV shows
P in ZeRO-DP plus ZeRO-R — that allows us to achieve
os+g the configuration parameters used in our experiments with
thisgoal.WewillrefertothisimplementationasZeRO-100B.
additional details in AE Appendix.
Our results show that ZeRO-100B can efficiently train models
with up to 170B parameters, 8x bigger than SOTA, up to 10x B. Speed and Model Size
fasterandwithimprovedusability.ZeRO-100BpowersTuring-
ZeRO-100B efficiently run models with up to 170B pa-
NLG, thelargest publishedmodel of thetime withnew SOTA
rameters on 400 GPUs, more than 8x bigger than Megatron-
accuracy.
LM. Figure 2 shows throughput per GPU for varying model
sizes using ZeRO-100B with MP versus using Megatron MP
A. Implementation and Methodology
alone. ZeRO-100B achieves a sustained throughput of 15
a) Implementation: We implemented ZeRO-100B in Py- PetaFlops (over 30% of the peak) on average for models
Torch including the full set of optimizations in P and with 8B to 100B parameters. In comparison, the baseline
os+g
ZeRO-R. Its interface is compatible with any model imple- MP performance degrades quickly with the increase in model
mentedastorch.nn.module.Userscansimplywraptheir size: MP incurs high communication volume between GPUs,
models using this interface and leverage ZeRO-powered DP and going beyond a single node to fit larger models causes
as they use the classic DP. Users do not need to modify their a communication bandwidth drop from 300GB/sec per link
model. ZeRO-powered DP can be combined with any form of (NVSwitch) to 12.5 GB/sec per link (Infiniband EDR), re-
MP including Megatron-LM. sulting in a significant performance drop as demonstrated by

the performance difference between 20B and 40B baseline E. Memory and Performance Analysis
models.ZeRO-100Bachievesupto10xspeedupoverbaseline,
significantly outperforming on large models. We look into the benefits and impact of different opti-
mizationsonmaximummodelsize,memoryconsumptionand
ForZeRO-100B,theslightreductioninperformancebeyond
performance. These optimizations are referred to as Config 1
100B is due to lack of enough memory to run larger batch
to 5 (C1-C5) in Table. III.
sizes. We expect the performance to improve as we increase
the number of GPUs due to super-linear speedup of ZeRO- a) Maximum Model Size: Figure 5 shows the largest
100B as we discuss next. trainable model by enabling different ZeRO optimizations for
afixedbatchsizeandMPof16.Themodelsizeincreasefrom
40B to 60B when trained with C1 vs C2 due to a 16x (MP
C. Super-Linear Scalability
degree) reduction in activation memory from using P , while
a
ZeRO-100B demonstrates super-linear scalability for very the jump to 140B using C4 is from enabling P which
os+g
largemodelsizes.Figure3showsscalabilityresultsfora60B halvesthememoryrequirementbythemodelstatescompared
parameter model going from 64 to 400 GPUs and we expect to P in C2. The increase to 150B using C5 is solely due
os
this trend to continue further for more GPUs. P reduces to further reduction in activation memory from offloading the
os+g
per GPU memory consumption of ZeRO-100B with increase partitioned activation checkpoints to the CPU memory.
in DP degree, allowing ZeRO-100B to fit larger batch sizes b) Max Cached Memory: Figure 6 shows the maximum
per GPU, which in turn improves throughput as a result of
memorycachedbyPyTorchduringeachtrainingiterationfora
increasing arithmetic intensity.
40Banda100Bparametermodel.Thedecreaseofthecached
Please note that the throughput scalability shown here do memory size is as expected from C1 to C2. The difference in
not represent strong or weak scaling in the traditional sense, memoryconsumptionbetweenC2andC3dependsonthesize
however, it is a proxy for the end-to-end training time, which of the model states in comparison to the activation memory,
is more meaningful in the context of DL training. In DL andcanincreasewhenactivationmemoryislarger,ordecrease
training,theamountofcomputationperiterationchangeswith when the model states are larger. It is note worthy that the
different batch sizes, while the total end-to-end computation cached memory does not decrease from C4 to C5 for 40B but
for a fixed total training samples is the same. Therefore, itdoesfor100B.Thisissimplybecausetheactivationmemory
despite the change in amount of computation per iteration for100Bismuchlargerforthedecreasetobenoticeable.This
from an increased batch size, the end-to-end training time makes P a valuable tool to fit a larger batch size when
a+cpu
is inversely proportional to the absolute throughput number we get to very large models. In Figure 7, P is needed
a+cpu
shown in Figure 3. for 170B model to execute without running out of memory.
It is also worth pointing out that a significant batch size
c) Max Achievable Performance: Figure 7 shows the
increasewithoutincreasingthetotaltrainingsamplescanlead
bestachievableperformancefordifferentsetofoptimizations.
to poor convergence. However, this is not the case for our
Noticethatperformanceimprovementcorrespondstodecrease
results.Thelargestbatchsizeusedinthisexperimentis1.6K,
in memory consumption between the optimizations. As men-
which is well within range of batch sizes used to train LM
tioned earlier, lower memory consumption allows for larger
models(512-2K)[3],[31].Infact,agrowingbodyofliterature
batch size which improves performance. The only caveat is
show that convergence rate for LM is resilient to batch sizes
the performance drop between C4 and C5 for 60B parameter
that are much larger than the ones used in this experiments
model. Despitelower memoryconsumption, C5 incursactiva-
[32], [25].
tion movement to and from the CPU, this will result in worse
performance in most cases, except for a few where the model
D. Democratizing Large Model Training is so large that the model simply cannot run without C5 or
the batch size that can run without C5 is very small (such
Using MP and PP is challenging for many data scientists,
as model with 170B parameters in Figure 7). During training,
which is a well-known hurdle to train large models. ZeRO
P is turned on only when it is beneficial.
a+cpu
does not require any changes to the model itself and it can be
used as simple as baseline DP while delivering significantly
boosted model size and speed. Fig. 8 shows that ZeRO-100B F. Turing-NLG, SOTA language model with 17B parameters
can train models with up to 13B parameters without MP on
128 GPUs, achieving throughput over 40TFlops per GPU on As of April 22, 2020, Turing-NLG [10] was the largest
average. In comparison, without ZeRO, the largest trainable model in the world with over 17B parameters. It achieved the
model with DP alone has 1.4B parameters with throughput new SOTA for language models with Webtext-103 perplexity
less than 20TFlops per GPU. Furthermore, in the absence of of 10.21. Turing-NLG was trained end-to-end using ZeRO-
the communication overhead from MP, these models can be 100B and Fig. 9 shows the validation perplexity over 300K
trained with lower-end compute nodes without very fast intra- iterations compared to previous SOTA, Megatron-LM 8.3B
node interconnect such as NVLINK or NVSwitch, which is parametermodel.ZeRO-100Bachievesasustainedthroughput
required to achieve good efficiency with MP. of 41.4TFlops/GPU for this model.

XI. CONCLUDINGREMARKS
|     |     |     |     |     |     |     |     | [10] Microsoft. | Turing-nlg: |                                                | A 17-billion-parameter |     | language | model |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | ----------- | ---------------------------------------------- | ---------------------- | --- | -------- | ----- |
|     |     |     |     |     |     |     |     | by microsoft.   |             | https://www.microsoft.com/en-us/research/blog/ |                        |     |          |       |
From a HPC and system perspective, we believe that ZeRO turing-nlg-a-17-billion-parameter-language-model-by-microsoft/,
| represents | a revolutionary |       | transformation |                 | in  | the large  | model | 2020.                                          |     |     |     |     |                     |     |
| ---------- | --------------- | ----- | -------------- | --------------- | --- | ---------- | ----- | ---------------------------------------------- | --- | --- | --- | --- | ------------------- | --- |
|            |                 |       |                |                 |     |            |       | [11] MinjieWang,Chien-chinHuang,andJinyangLi.  |     |     |     |     | Supportingverylarge |     |
| training   | landscape.      | While | our            | implementation, |     | ZeRO-100B, |       |                                                |     |     |     |     |                     |     |
|            |                 |       |                |                 |     |            |       | modelsusingautomaticdataflowgraphpartitioning. |     |     |     |     | InProceedingsof     |     |
enables 8x increase in model sizes, over 10x in throughput theFourteenthEuroSysConference2019,EuroSys’19,NewYork,NY,
USA,2019.AssociationforComputingMachinery.
improvement,achievessuper-linearspeedupsonmodernGPU
|           |            |     |         |       |        |        |             | [12] Yanping | Huang, | Yonglong | Cheng, Dehao | Chen, | HyoukJoong | Lee, Ji- |
| --------- | ---------- | --- | ------- | ----- | ------ | ------ | ----------- | ------------ | ------ | -------- | ------------ | ----- | ---------- | -------- |
| clusters, | and trains | the | largest | model | in the | world, | it is still |              |        |          |              |       |            |          |
quanNgiam,QuocV.Le,andZhifengChen.Gpipe:Efficienttrainingof
| just a tip | of the | iceberg. | ZeRO | in its | entirety | has the | potential |     |     |     |     |     |     |     |
| ---------- | ------ | -------- | ---- | ------ | -------- | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
giantneuralnetworksusingpipelineparallelism.ArXiv,abs/1811.06965,
2018.
| to increase | the | model size | by  | yet another | order | of  | magnitude, |                                                                 |     |     |     |     |     |     |
| ----------- | --- | ---------- | --- | ----------- | ----- | --- | ---------- | --------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|             |     |            |     |             |       |     |            | [13] AaronHarlap,DeepakNarayanan,AmarPhanishayee,VivekSeshadri, |     |     |     |     |     |     |
enablingthetrainingoftrillionparametermodelsofthefuture.
|     |     |     |     |     |     |     |     | Nikhil | R. Devanur, | Gregory | R. Ganger, | and | Phillip | B. Gibbons. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ----------- | ------- | ---------- | --- | ------- | ----------- |
Perhaps, what we feel most optimistic about ZeRO is that Pipedream: Fast and efficient pipeline parallel DNN training. CoRR,
it imposes no hurdles on the data scientists. Unlike existing abs/1806.03377,2018.
|     |     |     |     |     |     |     |     | [14] DeepakNarayanan,AaronHarlap,AmarPhanishayee,VivekSeshadri, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
approaches such as MP and PP, no model refactoring is Nikhil Devanur, Greg Granger, Phil Gibbons, and Matei Zaharia.
necessary,anditisaseasytouseasstandardDP,makingZeRO Pipedream: Generalized pipeline parallelism for dnn training. In ACM
a prime candidate for future investigations on large model Symposium on Operating Systems Principles (SOSP 2019), October
2019.
training.Throughopensourcingandcommunityfeedback,we
|         |      |            |            |     |        |              |     | [15] AnimeshJain,AmarPhanishayee,JasonMars,LingjiaTang,andGen- |     |     |     |     |     |     |
| ------- | ---- | ---------- | ---------- | --- | ------ | ------------ | --- | -------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| plan to | make | ZeRO fully | accessible |     | to the | DL community |     |                                                                |     |     |     |     |     |     |
nadyPekhimenko.Gist:Efficientdataencodingfordeepneuralnetwork
to catalyze the evolution and democratization of large model training. In International Symposium on Computer Architecture (ISCA
2018),2018.
| training | at scale. |     |     |     |     |     |     |                                                   |            |                 |           |     |               |            |
| -------- | --------- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | ---------- | --------------- | --------- | --- | ------------- | ---------- |
|          |           |     |     |     |     |     |     | [16] Paras                                        | Jain, Ajay | Jain, Aniruddha | Nrusimha, |     | Amir Gholami, | Pieter     |
|          |           |     |     |     |     |     |     | Abbeel,KurtKeutzer,IonStoica,andJosephE.Gonzalez. |            |                 |           |     |               | Checkmate: |
ACKNOWLEDGEMENT Breakingthememorywallwithoptimaltensorrematerialization.ArXiv,
abs/1910.02653,2019.
|     |     |     |     |     |     |     |     | [17] Linnan | Wang, Jinmian | Ye, | Yiyang Zhao, | Wei | Wu, Ang | Li, Shuai- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------------- | --- | ------------ | --- | ------- | ---------- |
WethankJunhuaWangforhisvaluablesupportandadvice.
|          |            |        |         |        |          |        |        | wenLeonSong,ZenglinXu,andTimKraska. |            |     |              |             | Superneurons:Dynamic |       |
| -------- | ---------- | ------ | ------- | ------ | -------- | ------ | ------ | ----------------------------------- | ---------- | --- | ------------ | ----------- | -------------------- | ----- |
| We thank | Minjia     | Zhang, | Elton   | Zheng, | Shaden   | Smith, | Reza   |                                     |            |     |              |             |                      |       |
|          |            |        |         |        |          |        |        | GPU memory                          | management |     | for training | deep neural | networks.            | CoRR, |
| Yazdani  | Aminabadi, | Arash  | Ashari, | and    | Niranjan | Uma    | Naresh |                                     |            |     |              |             |                      |       |
abs/1801.04380,2018.
for their great feedback and help on evaluating the work. [18] BharadwajPudipeddi,MaralMesmakhosroshahi,JinwenXi,andSujeeth
|          |         |         |       |          |      |        |      | Bharadwaj.              | Traininglargeneuralnetworkswithconstantmemoryusing |     |                            |     |     |     |
| -------- | ------- | ------- | ----- | -------- | ---- | ------ | ---- | ----------------------- | -------------------------------------------------- | --- | -------------------------- | --- | --- | --- |
| We thank | Brandon | Norick, | Corby | Rossett, | Gopi | Kumar, | Jack |                         |                                                    |     |                            |     |     |     |
|          |         |         |       |          |      |        |      | anewexecutionalgorithm. |                                                    |     | ArXiv,abs/2002.05645,2020. |     |     |     |
Zhang, Jing Zhao, Payal Bajaj, Rangan Majumder, Saksham [19] M. Rhu, N. Gimelshein, J. Clemons, A. Zulfiqar, and S. W. Keckler.
Singhal, Saurabh Tiwary, and Xia Song for many helpful vdnn: Virtualized deep neural networks for scalable, memory-efficient
discussions and suggestions. neural network design. In 2016 49th Annual IEEE/ACM International
SymposiumonMicroarchitecture(MICRO),pages1–13,2016.
|     |     |     |     |     |     |     |     | [20] Tung D. | Le, Haruki | Imai, | Yasushi Negishi, | and | Kiyokuni | Kawachiya. |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | ----- | ---------------- | --- | -------- | ---------- |
REFERENCES TFLMS:largemodelsupportintensorflowbygraphrewriting. CoRR,
abs/1807.02037,2018.
[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion [21] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates
Jones,AidanN.Gomez,LukaszKaiser,andIlliaPolosukhin. Attention withsublinearmemorycost. CoRR,abs/1804.04235,2018.
isallyouneed. CoRR,abs/1706.03762,2017. [22] RohanAnil,VineetGupta,TomerKoren,andYoramSinger. Memory-
|           |         |          |        |        |          |          |            | efficient | adaptive | optimization | for | large-scale | learning. | ArXiv, |
| --------- | ------- | -------- | ------ | ------ | -------- | -------- | ---------- | --------- | -------- | ------------ | --- | ----------- | --------- | ------ |
| [2] Jacob | Devlin, | Ming-Wei | Chang, | Kenton | Lee, and | Kristina | Toutanova. |           |          |              |     |             |           |        |
abs/1901.11150,2019.
| BERT: | pre-training | of  | deep bidirectional |     | transformers |     | for language |     |     |     |     |     |     |     |
| ----- | ------------ | --- | ------------------ | --- | ------------ | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
understanding. CoRR,abs/1810.04805,2018. [23] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient
[3] AlecRadford,JeffWu,RewonChild,DavidLuan,DarioAmodei,and methodsforonlinelearningandstochasticoptimization.J.Mach.Learn.
Ilya Sutskever. Language models are unsupervised multitask learners. Res.,12(null):2121–2159,July2011.
2019. [24] Yang You, Igor Gitman, and Boris Ginsburg. Scaling SGD batch size
|              |     |          |         |          |            |         |            | to32kforimagenettraining.                                    |     |     | CoRR,abs/1708.03888,2017. |     |     |     |
| ------------ | --- | -------- | ------- | -------- | ---------- | ------- | ---------- | ------------------------------------------------------------ | --- | --- | ------------------------- | --- | --- | --- |
| [4] Mohammad |     | Shoeybi, | Mostofa | Patwary, | Raul Puri, | Patrick | LeGresley, |                                                              |     |     |                           |     |     |     |
|              |     |          |         |          |            |         |            | [25] YangYou,JingLi,JonathanHseu,XiaodanSong,JamesDemmel,and |     |     |                           |     |     |     |
JaredCasper,andBryanCatanzaro.Megatron-lm:Trainingmulti-billion
parameterlanguagemodelsusingmodelparallelism,2019. Cho-Jui Hsieh. Reducing BERT pre-training time from 3 days to 76
[5] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan minutes. CoRR,abs/1904.00962,2019.
Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. [26] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos,
Exploring the limits of transfer learning with a unified text-to-text Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii
|     |     |     |     |     |     |     |     | Kuchaiev, | Ganesh | Venkatesh, | and Hao | Wu. Mixed | precision | training, |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ---------- | ------- | --------- | --------- | --------- |
transformer,2019.
| [6] Diederik | P. Kingma | and | Jimmy | Ba. Adam: | A method |     | for stochastic | 2017. |     |     |     |     |     |     |
| ------------ | --------- | --- | ----- | --------- | -------- | --- | -------------- | ----- | --- | --- | --- | --- | --- | --- |
optimization. InYoshuaBengioandYannLeCun,editors,3rdInterna- [27] NVIDIA Tesla V100 GPU architecture. http://images.nvidia.com/
tionalConferenceonLearningRepresentations,ICLR2015,SanDiego, content/volta-architecture/pdf/volta-architecture-whitepaper.pdf, 2017.
CA,USA,May7-9,2015,ConferenceTrackProceedings,2015. [Online,accessed22-April-2020].
[7] Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish [28] NVIDIA. Automatic mixed-precision. https://developer.nvidia.com/
automatic-mixed-precision,2019.
| Vaswani, | Penporn | Koanantakool, |     | Peter | Hawkins, | HyoukJoong | Lee, |     |     |     |     |     |     |     |
| -------- | ------- | ------------- | --- | ----- | -------- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- |
Mingsheng Hong, Cliff Young, Ryan Sepassi, and Blake A. Hecht- [29] Alexander Sergeev and Mike Del Balso. Horovod: fast and easy dis-
man. Mesh-tensorflow: Deep learning for supercomputers. CoRR, tributeddeeplearningintensorflow. arXivpreprintarXiv:1802.05799,
| abs/1811.02084,2018. |     |     |     |     |     |     |     | 2018. |     |     |     |     |     |     |
| -------------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
[8] TianqiChen,BingXu,ChiyuanZhang,andCarlosGuestrin. Training [30] SharNarasimhan.NVIDIAClocksWorld’sFastestBERTTrainingTime
deepnetswithsublinearmemorycost. CoRR,abs/1604.06174,2016. ... https://devblogs.nvidia.com/training-bert-with-gpus/, 2019. [Online;
[9] SamMcCandlish,JaredKaplan,DarioAmodei,andOpenAIDotaTeam. accessed25-September-2019].
An empirical model of large-batch training. CoRR, abs/1812.06162, [31] YinhanLiu,MyleOtt,NamanGoyal,JingfeiDu,MandarJoshi,Danqi
2018. Chen,OmerLevy,MikeLewis,LukeZettlemoyer,andVeselinStoyanov.

CoRR,
| Roberta: | A robustly optimized | BERT pretraining | approach. |     |
| -------- | -------------------- | ---------------- | --------- | --- |
abs/1907.11692,2019.
| [32] Jared Kaplan, | Sam McCandlish, | Tom Henighan, | Tom B. Brown, | Ben- |
| ------------------ | --------------- | ------------- | ------------- | ---- |
jaminChess,RewonChild,ScottGray,AlecRadford,JeffreyWu,and
| DarioAmodei. | Scalinglawsforneurallanguagemodels,2020. |     |     |     |
| ------------ | ---------------------------------------- | --- | --- | --- |

|     | Appendix: | Artifact |     | Description/Artifact | Evaluation |
| --- | --------- | -------- | --- | -------------------- | ---------- |
SUMMARYOFTHEEXPERIMENTSREPORTED
Keyalgorithms: ZeRO,ADAM
| Our implementation | of ZeRO | was written | in Python | on top of |     |
| ------------------ | ------- | ----------- | --------- | --------- | --- |
Inputdatasetsandversions: OpenWebText
PyTorch1.2.Weusedtorch.distributedandNVIDIANCCL2.4.8for
inter-GPUcommunication.WeusedNVIDIAMegatron-LMasour
baseline,andintegrateditwithourDeepSpeedLibraryandZeRO
implementationforallofourexperiments.Allofourcodeisopen
sourceandavailablewithdetailedtutorialsonhowtouseit,see
ARTIFACTAVAILABILITYfordetails.
Toofferfullreproducibilityofourresults,weprovidethemodel
configurations(numberoflayers,hiddensize,andattentionheads),
batchsizes,andnumberofGPUsweusedforeveryexperimentin
Tables1-6below.
ARTIFACTAVAILABILITY
| SoftwareArtifactAvailability: |     | Allauthor-createdsoftwarearti- |     |     |     |
| ----------------------------- | --- | ------------------------------ | --- | --- | --- |
factsaremaintainedinapublicrepositoryunderanOSI-approved
license.
| HardwareArtifactAvailability: |     | Therearenoauthor-createdhard- |     |     |     |
| ----------------------------- | --- | ----------------------------- | --- | --- | --- |
wareartifacts.
| Data Artifact | Availability: | There are | no author-created | data |     |
| ------------- | ------------- | --------- | ----------------- | ---- | --- |
artifacts.
| ProprietaryArtifacts: | Noauthor-createdartifactsareproprietary. |     |     |     |     |
| --------------------- | ---------------------------------------- | --- | --- | --- | --- |
Author-CreatedorModifiedArtifacts:
| Persistent | ID: https://github.com/microsoft/DeepSpeed |     |     |     |     |
| ---------- | ------------------------------------------ | --- | --- | --- | --- |
| Artifact   | name: DeepSpeed                            |     |     |     |     |
BASELINEEXPERIMENTALSETUP,AND
MODIFICATIONSMADEFORTHEPAPER
| Relevanthardwaredetails: |     | Allexperimentswereperformedon |     |     |     |
| ------------------------ | --- | ----------------------------- | --- | --- | --- |
NVIDIADGX-2hardware.Eachnodecontains:16xNVIDIATesla
V100-SXM3-32GB;DualIntelXeonPlatinum8168,2.7GHz(24-
cores);24x64GBDDR4Micron72ASS8G72LZ-2G6D2,2666MT/s
| 288-UDIMM; | 8x 100Gb/s | InfiniBand Mellanox | MT27800 | Family |     |
| ---------- | ---------- | ------------------- | ------- | ------ | --- |
(ConnectX-5);2xNVMESSDSamsung960GB(MZ1LW960HMJP);
MellanoxMLNX-OSCS7500648-ports
| Operatingsystemsandversions: |     | a.Ubuntu18.04.2LTS;Linux |     |     |     |
| ---------------------------- | --- | ------------------------ | --- | --- | --- |
ff422fc538c049ed804224fc761b952b-master-04.15.0-96-generic#97-
UbuntuSMPWedApr103:25:46UTC2020x86_64x86_64x86_64
GNU/Linux
| Compilersandversions: |     | g++(Ubuntu7.4.0-1ubuntu118.04)7.4.0; |     |     |     |
| --------------------- | --- | ------------------------------------ | --- | --- | --- |
nvccrelease10.0,V10.0.130
| Applicationsandversions: |     | NVIDIAMegatron-LMfromSeptem- |     |     |     |
| ------------------------ | --- | ---------------------------- | --- | --- | --- |
ber2019
| Librariesandversions: | Python3.6.7;DeepSpeed0.2;PyTorch1.2; |     |     |     |     |
| --------------------- | ------------------------------------ | --- | --- | --- | --- |
NVIDIANCCL2.4.8;NVIDIAApexfromFebruary2020

Rajbhandari,etal.
Figure2
Modelsize ZeRO/Baseline NumberofGPUs MP Layers Hiddensize Attentionhead Batchsize Totalbatchsize
| 1.5B ZeRO     | 400 | 1 48    | 1600 | 16  | 24 9600 |
| ------------- | --- | ------- | ---- | --- | ------- |
| 1.5B Baseline | 400 | 2 48    | 1600 | 16  | 16 3200 |
| 8B ZeRO       | 400 | 4 72    | 3072 | 24  | 64 6400 |
| 8B Baseline   | 400 | 8 72    | 3072 | 24  | 8 400   |
| 20B ZeRO      | 400 | 4 98    | 4096 | 32  | 32 3200 |
| 20B Baseline  | 400 | 16 98   | 4096 | 32  | 4 100   |
| 40B ZeRO      | 400 | 4 88    | 6144 | 32  | 12 1200 |
| 40B Baseline  | 384 | 32 88   | 6144 | 64  | 4 48    |
| 60B ZeRO      | 400 | 16 132  | 6144 | 32  | 64 1600 |
| 60B Baseline  | 384 | 64 132  | 6144 | 64  | 4 24    |
| 80B ZeRO      | 400 | 16 100  | 8192 | 64  | 32 800  |
| 80B Baseline  | 384 | 128 100 | 8192 | 128 | 4 12    |
| 100B ZeRO     | 400 | 16 125  | 8192 | 64  | 32 800  |
| 100B Baseline | 384 | 128 125 | 8192 | 128 | 2 6     |
| 120B ZeRO     | 400 | 16 150  | 8192 | 64  | 24 600  |
| 120B Baseline | 384 | 128 150 | 8192 | 128 | 2 6     |
| 140B ZeRO     | 400 | 16 175  | 8192 | 64  | 16 400  |
| 140B Baseline | 384 | 128 175 | 8192 | 128 | 2 6     |
| 170B ZeRO     | 400 | 16 212  | 8192 | 64  | 12 300  |
| 170B Baseline | 256 | 256 212 | 8192 | 256 | 2 2     |
Table1:ModelparametersandbatchsizestoreproduceresultsinFigure2relatedtoZeROthroughputcomparedwithbaseline.
Figure3
Modelsize ZeRO/Baseline NumberofGPUs MP Layers Hiddensize Attentionhead Batchsize Totalbatchsize
60B ZeRO 64,128,256,400 16 75 8192 32 16,48,48,64 64,384,768,1600
Table2:ModelparametersandbatchsizestoreproduceresultsinFigure3relatedtosuperlinearscalability.
Figure4
Modelsize ZeRO/Baseline NumberofGPUs MP Layers Hiddensize Attentionhead Batchsize Totalbatchsize
| 40B ZeRO  | 400 | 16 50  | 8192 | 32  | 16 400 |
| --------- | --- | ------ | ---- | --- | ------ |
| 60B ZeRO  | 400 | 16 132 | 6144 | 64  | 16 400 |
| 140B ZeRO | 400 | 16 175 | 8192 | 64  | 16 400 |
| 150B ZeRO | 400 | 16 187 | 8192 | 64  | 16 400 |
| 50B ZeRO  | 400 | 16 62  | 8192 | 32  | 16 400 |
Table 3: Model parameters and batch sizes to reproduce results in Figure 4 related to max model size with different ZeRO
configurations.
Figure5
Modelsize ZeRO/Baseline NumberofGPUs MP Layers Hiddensize Attentionhead Batchsize Totalbatchsize
| 40B ZeRO  | 400 | 16 50  | 8192 | 32  | 16 400 |
| --------- | --- | ------ | ---- | --- | ------ |
| 100B ZeRO | 400 | 16 125 | 8192 | 64  | 32 800 |
Table4:ModelparametersandbatchsizestoreproduceresultsinFigure5relatedtomemoryallocatedwithdifferentZeRO
configurations.

ZeRO:MemoryOptimizationsTowardTrainingTrillionParameterModels
Figure6
Modelsize ZeRO/Baseline NumberofGPUs MP Layers Hiddensize Attentionhead Batchsize Totalbatchsize
| 60B ZeRO  | 128 | 16 75  | 8192 | 64  | 2,4,32,32,8 16,32,256,256,64 |
| --------- | --- | ------ | ---- | --- | ---------------------------- |
| 170B ZeRO | 400 | 16 212 | 8192 | 64  | 12 300                       |
Table5:ModelparametersandbatchsizestoreproduceresultsinFigure6relatedtothroughputwithdifferentZeROconfigu-
rations.
Figure7
Modelsize ZeRO/Baseline NumberofGPUs MP Layers Hiddensize Attentionhead Batchsize Totalbatchsize
| 1.5B ZeRO      | 128 | 1 34 | 1920 | 16  | 24 3072 |
| -------------- | --- | ---- | ---- | --- | ------- |
| 2.5B ZeRO      | 128 | 1 54 | 1920 | 16  | 24 3072 |
| 4B ZeRO        | 128 | 1 64 | 2304 | 24  | 16 2048 |
| 6B ZeRO        | 128 | 1 52 | 3072 | 24  | 12 1536 |
| 8B ZeRO        | 128 | 1 72 | 3072 | 24  | 8 1024  |
| 10B ZeRO       | 128 | 1 50 | 4096 | 32  | 6 768   |
| 11B ZeRO       | 128 | 1 54 | 4096 | 32  | 4 512   |
| 12B ZeRO       | 128 | 1 58 | 4096 | 32  | 4 512   |
| 13B ZeRO       | 128 | 1 62 | 4096 | 32  | 2 256   |
| 1p16B Baseline | 128 | 1 24 | 1920 | 16  | 8 1024  |
| 1p38B Baseline | 128 | 1 40 | 1536 | 16  | 1 128   |
Table 6: Model parameters and batch sizes to reproduce results in Figure 7 related to evaluating maximum model sizes vs
throughputwhileusingonlydata-parallelism.