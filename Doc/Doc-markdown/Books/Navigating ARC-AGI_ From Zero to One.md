Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
An interactive guide to the theory, implementation, and state-of-
the-art strategies for the ARC�AGI Challenge.
"Easy for Humans, Hard for AI" — The defining characteristic that
makes ARC�AGI a powerful benchmark for General Intelligence.
Fluid Intelligence Program Synthesis Test-Time Adaptation
Induction & Transduction Domain-Specific Languages
Competition Ready
Hello! Like many others, I was fascinated by the ARC�AGI challenge but found the
learning curve a bit steep. I created this guide as a personal project to connect
the dots for myself.
My goal is simple: to provide a single, clear starting point for newcomers by
synthesizing the core concepts into one easy-to-follow narrative. If you're just
starting your ARC journey, I hope this resource helps. Happy Journey!
1 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
The Abstraction and Reasoning Corpus for Artificial General
Intelligence �ARC�AGI) is a benchmark designed to measure fluid
intelligence in AI systems. Unlike traditional benchmarks that test
accumulated knowledge, ARC�AGI evaluates the ability to acquire new
skills when faced with novel problems.
Each task is unique and designed to resist memorization.
Tests the efficiency of learning new skills, not just performance.
Uses only universal cognitive primitives for fair comparison.
2 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
→
→
→
The Abstraction and Reasoning Corpus for Artificial General Intelligence �ARC�AGI) is
more than a benchmark; it is the manifestation of a specific, rigorous philosophy about
the nature of intelligence itself. Introduced by François Chollet, it was designed to
address a fundamental flaw in how the AI community measured progress.
The central tenet of ARC�AGI is that true, general intelligence is not demonstrated by
the of a specific skill, but by the when
faced with novel problems. This stands in stark contrast to many traditional AI
benchmarks which measure performance on tasks that can be mastered through
extensive training on massive datasets.
In such cases, high performance can be "bought" with sufficient data and compute,
3 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
masking the system's underlying ability to generalize and adapt. An AI that achieves
superhuman performance at Go has mastered Go; it has not necessarily become more
intelligent in a general sense.
Chollet formalizes this concept by defining intelligence as a measure of a system's skill-
acquisition efficiency over a given scope of tasks, taking into account its prior
knowledge, experience, and the difficulty of generalization. ARC�AGI is the concrete
application of this definition. Each task is unique and designed to be unsolvable through
mere memorization or pattern matching against a training set.
To measure this skill acquisition in a controlled way, every puzzle adheres to a
consistent structure. This structure, the , presents a small number of
examples to learn from.
To create a fair and meaningful comparison between human and artificial intelligence,
ARC�AGI is meticulously designed to test —the ability to reason, adapt,
and solve novel problems—rather than , which relies on
accumulated, domain-specific knowledge and cultural learning.
This distinction is critical. A benchmark that required knowledge of historical facts or
the English language would unfairly favor systems (and humans) with specific pre-
training, turning the test into a measure of prior exposure rather than innate reasoning
ability.
ARC�AGI circumvents this by designing tasks that are solvable using only a minimal set
of . These are fundamental, universally shared cognitive
building blocks that are either innate or acquired very early in development.
The ability to perceive a scene in terms of discrete objects with properties like cohesion
(objects move as wholes) and persistence (objects don't randomly appear or disappear).
4 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Intuitive understanding of connectivity, symmetry, inside/outside relationships, and distance.
Simple counting and basic integer arithmetic.
The notion that actions are taken to achieve goals.
By restricting the required knowledge to these universally accessible primitives, ARC�
AGI isolates the capacity for generalization and ensures that success reflects a
system's intrinsic ability to learn, reason, and adapt. The public training set is explicitly
curated to expose a test-taker to all the Core Knowledge priors needed to solve the
evaluation tasks, effectively serving as a "tutorial" for the conceptual language of the
ARC universe.
The introduction of ARC�AGI�2 in 2025 marks a critical evolution, driven by the progress
and observed failure modes of AI systems on the original dataset. While powerful AI
systems could achieve high scores on ARC�AGI�1, often through brute-force search,
ARC�AGI�2 was designed to be less susceptible to these methods. Furthermore, it was
specifically created to probe known weaknesses in modern AI reasoning systems.
Based on the failures of frontier AI models, ARC�AGI�2 introduces tasks that test
for new, more complex reasoning abilities:
Tasks where visual symbols must be interpreted as
having semantic meaning beyond their shape, such as a shape representing
an action.
Tasks that require discovering and applying
multiple, interacting rules simultaneously.
5 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
 Tasks where the correct rule to apply depends
on the specific context within the grid, moving beyond superficial global
patterns.
To address
limitations and
| 2019 | 2025 |     |
| ---- | ---- | --- |
challenge modern
AI systems.
To stay ahead of AI
|     | Frontier AI | progress and |
| --- | ----------- | ------------ |
Deep Learning
|     | Reasoning | target new, |
| --- | --------- | ----------- |
�Memorization)
|     | Systems | complex reasoning |
| --- | ------- | ----------------- |
failures.
To ensure scores
reflect intelligent
| High | Low (by design) | adaptation, not just |
| ---- | --------------- | -------------------- |
computational
power.
Symbolic
To probe specific,
Interpretation,
| Generalization, |     | observed |
| --------------- | --- | -------- |
Compositional
| basic |     | weaknesses in |
| ----- | --- | ------------- |
Reasoning,
| abstraction. |     | state-of-the-art |
| ------------ | --- | ---------------- |
Contextual Rule
reasoning systems.
Application.
To create a wider
| High (e.g., | Very Low (e.g., |     |
| ----------- | --------------- | --- |
"signal bandwidth"
| �75% for o3- | �5% for o3- |     |
| ------------ | ----------- | --- |
to differentiate AI
| preview) | preview) |     |
| -------- | -------- | --- |
capabilities.
6 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Successfully navigating the ARC�AGI challenge requires a firm grasp of its practical
ecosystem, which includes a structured set of datasets, specific evaluation protocols,
and a vibrant community with essential resources.
The ARC�AGI data is partitioned into several distinct sets, each with a specific purpose.
Using them correctly is crucial for both development and fair evaluation.
Training
Contains easier,
algorithms,
"curriculum-style"
�1,000 learning Core Public
tasks. Use freely for
Knowledge
development.
priors.
Final local
120 evaluation of an Public Treat
algorithm. as a one-shot
evaluation.
Powers the
Used to test both
public Private
120 open and closed-
leaderboard on �Kaggle�
source models.
arcprize.org.
7 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
| Official ranking |         | The ultimate test of |
| ---------------- | ------- | -------------------- |
| for the Kaggle   | Private | generalization. No   |
120
| prize        | �Kaggle� | internet access |
| ------------ | -------- | --------------- |
| competition. |          | allowed.        |
 The official scoring metric is  , which
pass@k
measures the percentage of tasks solved within  k  attempts. For the ARC
Prize,  k=2 .
 All prize-eligible submissions must run within a
standardized Kaggle Notebook environment with no internet access and strict
runtime/hardware limits.
 To be eligible for prize money, teams must open-
source their complete, reproducible solution under a permissive license.
8 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Infer explicit rules from examples
→
Guide search with learned intuition
→
Adapt dynamically to each task
One of the most natural and historically significant approaches to ARC is program
synthesis. This paradigm directly tackles the core of the challenge: inferring a general
rule from examples. Program Synthesis, also known as Inductive Programming, is the
task of automatically generating a computer program that meets a given high-level
specification. In the context of ARC, the specification is the set of demonstration pairs.
The goal is to find a program, P , that correctly transforms each training input grid into
9 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
its corresponding output grid. If such a program is found, it is assumed to represent the
underlying rule of the task and can then be applied to the test input grid to generate a
solution.
This approach is fundamentally : it first infers a general, abstract rule (the
program) from specific examples, and only then executes that rule to produce a specific
prediction. This contrasts with methods, which predict the output directly
from the examples without necessarily forming an explicit, reusable program. Program
synthesis was the dominant strategy in the early days of ARC, with the 2020 Kaggle
competition winner employing these techniques.
The primary challenge in program synthesis is the vastness of the search space. A
is essential. A DSL is a small, specialized
programming language designed for ARC, consisting of a curated set of functions, or
, that perform common grid operations. A good DSL must be expressive
enough to solve tasks while simple enough to keep the search space manageable.
rotate_grid find_objects
mirror_object count_colors
draw_line shift_object
solve_5521c0d9.py
# Simplified representation of the solver for task 5521c0d9 from Hodel's arc-dsl
def solve_5521c0d9(I):
# 1. Extract all non-background objects from the input grid 'I'.
objs = dsl.objects(I, univalued=True, diagonal=False, without_background=True)
# 2. Merge all extracted objects into a single 'foreground' object.
foreground = dsl.merge(objs)
# 3. Create a new grid by removing the foreground, leaving only the
10 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
background.
empty_grid = dsl.cover(I, foreground)
# 4. Create a function 'offset_getter' that calculates an upward shift vector
# equal to an object's height. This is done by composing three functions:
# height -> invert -> toivec (get height, negate it, convert to vector).
offset_getter = dsl.chain(dsl.toivec, dsl.invert, dsl.height)
# 5. Create a function 'shifter' that takes an object and moves it.
# The 'fork' primitive applies the 'shift' operation, using the object
# itself as the first argument and the result of 'offset_getter(object)'
# as the second argument.
shifter = dsl.fork(dsl.shift, dsl.identity, offset_getter)
# 6. Apply the 'shifter' function to every object in the 'objs' list
# and merge the results into a single object of shifted shapes.
shifted = dsl.mapply(shifter, objs)
# 7. Paint the final 'shifted' object onto the 'empty_grid'.
O = dsl.paint(empty_grid, shifted)
return O
Combined primitives in action
This example beautifully illustrates the paradigm. The solution is not a monolithic neural
network but an interpretable, multi-step program. Each line applies a well-defined
primitive from the DSL. The program first deconstructs the input grid into objects
( dsl.objects ), then computes a transformation for them (the shifter function), applies
this transformation ( dsl.mapply ), and finally reconstructs the output grid
( dsl.paint ). A program synthesis system would need to find this specific sequence of
seven function calls out of a vast number of possibilities.
Once a DSL is defined, the core problem becomes one of search. The system must find
the correct sequence of DSL primitives that solves the task.
11 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Even with a constrained DSL of 100 primitives, the number of possible programs
grows exponentially:
Length 1 Length 2
Length 3 Length 4
Modern ARC solvers employ sophisticated search algorithms to navigate the vast
program space efficiently:
Balances exploration and exploitation to find
promising program paths.
An advanced variant from Sakana AI that
adaptively decides whether to search deeper (refine) or wider (explore).
Methods that use rules of thumb or maintain multiple
candidate programs to guide search toward likely solutions.
12 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Uses a Transformer to predict the most likely sequence of DSL primitives,
guiding the search probabilistically.
A neural network learns a distance metric between grids to
evaluate which intermediate step is "closest" to the goal.
The main GridCoder approach where a model predicts
the final program directly.
The most significant advance is using neural networks to guide the search process.
This moves from a model of "search as enumeration" to "search as learned intuition,"
which is far more efficient and mirrors human problem-solving.
While program synthesis was dominant early on, the most significant breakthroughs in
2024 came from . This strategy, where a model
dynamically adapts itself at the moment of inference using the task's own
demonstration examples, was a necessary component of every top-performing
solution.
TTA is a broad category that includes two main sub-strategies: Test-Time Scaling �TTS�,
which allocates more compute without changing model weights, and Test-Time Training
�TTT�, which temporarily fine-tunes the model.
TTS refers to improving performance by allocating more computational resources at
inference time, without changing the model's weights. This can range from simple
techniques like repeated sampling to more complex search procedures like chain-of-
thought synthesis or Sakana AI's advanced
.
13 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
TTT is a powerful technique where a model's parameters are temporarily updated via
gradient descent at inference time. The model is briefly fine-tuned on the few
demonstration pairs of the specific task it is trying to solve. This was pioneered for ARC
by researchers at MIT and became the basis for several top-scoring 2024 solutions.
Receive a novel ARC task with a few train/test examples.
Expand the small demo set using symmetries (rotations, flips, color swaps) to create a
temporary training set.
Rapidly fine-tune a small LoRA adapter on the augmented data, leaving the base model
frozen for efficiency.
Predict the output. Often done under multiple augmentations and combined via majority
vote for robustness.
14 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
A fundamental duality in problem-solving strategies has become apparent in ARC
research, formalized in the prize-winning paper by Li et al. This duality mirrors the
concepts of , a popular model in cognitive science used
to describe the two paths of human reasoning. Understanding this is key to building a
top-tier solver, as the best solutions are ensembles that combine both approaches.
The , deliberate reasoning path. The goal is to first infer a latent, explicit
program or function f that fully explains the transformation in the training
examples. This program f is then applied to the test input x_test to get the
prediction y_test = f(x_test) . The classic DSL-based search methods
described earlier are inductive.
Tasks requiring precision, multi-step logic, compositionality, and explicit
computation.
The , intuitive path. The goal is to directly predict the test output
y_test by considering the training examples (x_train, y_train) and the test
input x_test all at once, without necessarily creating an explicit, intermediate
program. The LLM-based Test-Time Training approaches described earlier are
primarily transductive.
Tasks relying on "fuzzy" perception, pattern completion, and holistic
transformations.
15 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
This section provides a hands-on tutorial for building a simple, yet complete,
ARC solver in Python. We'll use a minimal DSL and simple brute-force search to
demonstrate the core logic of the inductive programming paradigm.
Python DSL Search Visualization
16 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
A minimal Domain-Specific Language �DSL�
A brute-force search algorithm
Grid visualization tools
A complete end-to-end solver
Understand the program synthesis approach
Learn how DSLs constrain search space
Implement search and verification logic
Create submission-ready output
This tutorial demonstrates the core concepts. Real competitive solvers use
much larger DSLs, smarter search algorithms, and neural guidance!
Before writing a single line of solver code, the most important step for any serious
competitor is to build a robust local validation pipeline. The goal is to create a setup that
allows you to reliably estimate your performance on the hidden private test set without
deceiving yourself.
Your pipeline should strictly separate the public training and public
17 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
evaluation datasets. Your algorithm should never see the evaluation tasks during its
development phase.
Repeatedly modifying your algorithm based on its score on the
evaluation set, or manually inspecting those tasks to guide development, constitutes
data leakage. This will lead to an inflated, unreliable local score that will not translate
to the private leaderboard.
To get your best estimate of true performance, run your final,
trained solver on the entire public evaluation set in a single execution. If your
solution is computationally expensive, you can build confidence by testing on a
random sample of tasks, holding out the rest for a final validation run before a full
execution.
To move from a simple solver to a competitive one, it is essential to study the strategies
of those who have reached the top of the leaderboards. The open-source nature of the
ARC Prize provides an unprecedented opportunity to deconstruct the winning solutions
from the 2024 competition.
Custom DFS
the LLM-based TTT sampling,
53.5%
ARChitects �Transduction) "Product of
Experts" scoring.
18 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Hybrid solver
Ensemble
| G.         |       |              | combining DSL  |
| ---------- | ----- | ------------ | -------------- |
|            | 40.0% | �Induction + |                |
| Barbadillo |       |              | search and LLM |
Transduction)
prediction.
Foundational
| Akyürek et | 61.9%    | LLM-based TTT  | method for TTT |
| ---------- | -------- | -------------- | -------------- |
| al.        | (public) | �Transduction) | on ARC using   |
LoRA.
|       |     | Neurally-Guided | GridCoder: using |
| ----- | --- | --------------- | ---------------- |
| Simon |     | Program         | a Transformer to |
N/A
| Ouellette |     | Synthesis   | guide DSL |
| --------- | --- | ----------- | --------- |
|           |     | �Induction) | search.   |
→
Advanced TTT with custom sampling and "Product of Experts" scoring
→
Foundational TTT methodology with LoRA and data augmentation
19 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
→
Specialized Transformer guiding DSL search for efficient induction
Armed with an understanding of ARC's philosophy, methodologies, and winning
strategies, you can now chart a course for tackling the ARC Prize 2025. Success will
require a combination of solid engineering, strategic thinking, and novel ideas.
The results from 2024 send a clear message: no single approach currently solves all
ARC tasks. The state-of-the-art is a hybrid. A highly effective strategy for a new
competitor would be:
Start by implementing a robust Test-Time Training �TTT) solver based on the work of the
ARChitects and Akyürek et al.
20 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Concurrently, build or adapt a DSL-based program synthesis solver. This could be based on
Michael Hodel's DSL or a neurally-guided approach like GridCoder.
Design a meta-solver that runs both your transductive and inductive systems on each task
and develops heuristics to choose which solution to submit.
Simply re-implementing 2024 solutions is unlikely to win the Grand Prize on ARC�AGI�2.
Your strategic focus should be on creating novel solutions for the known weaknesses of
today's systems—the very challenges ARC�AGI�2 was built to test:
Understanding that pixels can represent an action or concept, rather than just being a pattern
to transform.
Discovering and applying multiple, interacting rules simultaneously, especially when those
rules interact with each other.
21 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
Recognizing the context that determines which rule to apply, moving beyond superficial
global patterns.
Everything you need to begin competing in the ARC Prize 2025 and
contributing to the future of artificial general intelligence.
Understand the pass@2 metric
Build robust validation pipeline
Study winning solutions
Prepare for open-source requirement
22 de 23 24/5/26, 15:21

Navigating ARC-AGI: From Zero to One https://arahim3.github.io/arc-agi-guide/
The Abstraction and Reasoning Corpus is far more than a conventional
AI benchmark. It is a challenge, a philosophy, and a compass for the
field of AGI research. It posits that true intelligence lies not in
accumulated skill but in the efficient acquisition of new skills in the
face of novelty.
By engaging with the open-source code of past champions, participating in the
vibrant research community, and focusing on the unsolved frontiers, a
dedicated researcher has all the tools necessary to not only compete in—and
perhaps even claim the Grand Prize for solving—the ARC Prize, but to
contribute meaningfully to the collective, open pursuit of Artificial General
Intelligence.
23 de 23 24/5/26, 15:21