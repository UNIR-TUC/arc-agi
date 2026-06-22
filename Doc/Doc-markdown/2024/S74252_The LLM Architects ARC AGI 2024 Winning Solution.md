The (LLM) Architects
ARC-AGI 2024 Winning Solution
Jan Disselhoff, Daniel Franzen, David Hartmann
GTC 25 Winning Solution of the ARC AGI Challenge

Overview 2
| What is ARC | Key Progresses | Optimizations | Our Pipeline |
| ----------- | -------------- | ------------- | ------------ |
Reasoning as Benchmark DSL for ARC Problems Challenge Constraints Efficient Fine-Tuning
| Task Examples | Additional Datasets | Tokenization | DFS Inference |
| ------------- | ------------------- | ------------ | ------------- |
ARC is hard! Test-Time Training VRAM Reduction Tricks Candidate Ranking
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

| What is  | ARC-AGI |  ?  |
| -------- | ------- | --- |

What is ARC-AGI ? 4
“It’s easy for humans,
but hard for AI.”
— arcprize.org
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 5
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 6
ARC-AGI = Reasoning As A Benchmark
A series of cognitive challenges on small grids
Rules must be learned from only a few (2 – 6) examples
Simple for humans, very hard for AI
+ "explicitly designed to compare artificial intelligence with human
intelligence"
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 7
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 8
Only Pixel-Perfect Predictions Count!
(... and LLMs don’t “see” 2D data)
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 9
How well do frontier LLMs perform?
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 10
Claim:
"Progress has stalled"
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 11
How well do Open-Source LLMs perform?
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Community Progresses

Community Progresses 13
Community Identified:
Test-Time Training,
DSL of Tasks,
Data Augmentation
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Community Progresses 14
Test-Time Training
● Arc has small
conceptual Tasks
● Each task includes some
examples
● … even the test examples
● Idea: Use the examples
as a small Dataset
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Community Progresses 15
Test-Time Training
● Arc has small
conceptual Tasks
● Each task includes some
examples
● … even the test examples
● Idea: Use the examples
as a small Dataset
Free Additional
Training Data
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Community Progresses 16
Domain-Specific Language of ARC Tasks
Introduced by github.com/michaelhodel/re-arc
● Functions designed explicitly to
manipulate ARC-like tasks
● In previous challenges:
combination with program
search to solve arc tasks
● In the 2024 challenge:
program generation using LLMs
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Community Progresses 17
Domain-Specific Language of ARC Tasks
Introduced by github.com/michaelhodel/re-arc
Datasets generated with DSLs
● Re-ARC
○ Reimplements original ARC
challenges
○ (Only 400 of the 800 public
available tasks)
● ARC-Heavy
○ LLM-based generators
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 18
Original
Problem:
Rotation by 90°
Data Augmentation
• Modifications under which
the problem is still solvable
• "Asking the same question
from different Transposition and subsequent rotation
perspectives"
• Can be used in every step
o Training: increase
amount of training
data
o Inference: increase
chance of sampling
the correct solution Random permutation of pixel colors
0123456789 2943076158
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Optimizations & Tricks

Optimizations & Tricks 20
Efficient Use of
Limited Resources
(2 x NVIDIA T4 GPU)
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 21
Why Are Resources Limited
Kaggle challenge limitations
● “Code challenge” - code
Two Steps:
must be uploaded and
● Offline and extensive
is executed by kaggle
● 2x NVIDIA T4 GPUs Fine-Tune on NVIDIA H100 GPU
● Second “Test-Time” Fine-Tune
(with 16GB VRAM each)
● 4 CPU cores on challenge examples on
● max. 12 hours runtime 2 NVIDIA T4 GPUs
(to utilize both GPUs, we split
the dataset in two parts)
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 22
Tokenizer Changes
• Text is split into chunks
• Makes inference cheaper
• But number problems
become harder
• This is bad for us!
We remove 99.95% of all • Model is smaller ✅
Takeaway
Tokens and merges from • Inference is simpler ✅
When we have very
Model!
• No hallucinations ✅ specific use cases it can
be useful to reduce LLM
• No filler tokens ✅
capabilities
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 23
Our Tokenization approach
● Grids are tokenized
line-by-line
● End-of-line indicates
start of new line
● I and O mark beginning
of input and output
● Colors are represented
by number tokens
● 2D positional encoding only 64 tokens required
not necessary
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 24
VRAM Reduction Techniques
● Shrinking embedding layer
from 128k to only 64 tokens
● 4bit quantized base model
● QLoRA finetuning
● Gradient Checkpointing
● using the unsloth library
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Our Pipeline

Our Pipeline 26
No prompt tuning
No chain-of-thought
No large model
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Our Pipeline 27
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 28
Finetune on Augmented Dataset
● Create Augmented Dataset from
different sources
○ ReARC, ARC-Heavy, ConceptArc…
● Finetune base LLM using QLoRA and
unsloth
● “Small” Language Model:
NVIDIA Mistral-NeMo-Minitron-8B-Base
● Trained for ~100h on NVIDIA H100 GPU
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 29
DFS based Inference
● Trained model is good but how to
sample?
● Standard Multinomial Sampling has
issues:
○ Repetition of same solution
○ No guarantees of sampling
probability
○ Increasing chance of errors for
longer problems
● Beam search slightly better, but at cost
of additional VRAM
● Is there an alternative?
Yes! Write your own
sampling algorithm!
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 30
DFS based Inference
● DFS based sampling with threshold
● Search through partial solutions in
Tree, but discard all paths below some
threshold
● Can use inference caches to make this
fast and cheap!
○ Same VRAM as standard
sampling
● NO Repetitions!
● NO low probability solutions!
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 31
Augmented Scoring
● LLMs have causal inference
● But ARC-AGI tasks are 2D!
● Some problems are easier to solve
from certain perspectives
● Can we use this?
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 32
Augmented Scoring
● Take solution Candidates
● Calculate sampling probability from
each perspective
● Aggregate using mean of log-probs
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 34
Claim:
"Progress has stalled"
Our Solution
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Key Progresses 35
In the meantime …
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 36
How well do frontier LLMs perform?
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

What is ARC-AGI ? 37
How well do frontier LLMs perform?
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Takeaways 38
• LLMs can be used for cognitive tasks
o IF they are finetuned well
• Restricting the problem space can
o3 price per task: 17$
be very effective
Ours per task: 0.14$*
o IF the problem allows it
• Sampling correctly is key! * assuming 2.5$ per hour for a NVIDIA H100 GPU
• Useful trick: Augment queries to
double check results
www.lambdalabs.com The (LLM) Architects — ARC-AGI 2024 Winning Solution

Tutorial Use Our Code

Frequently Asked
Questions

Thank You For
Listening!
Questions