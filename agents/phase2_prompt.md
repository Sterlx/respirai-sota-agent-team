\# Phase 2 Prompt for RespirAI Agent Team



We finished Milestone 1.



New project constraint:

We will not use the cluster. We will use two computers:



1\. Laptop:

\- VS Code

\- DeepSeek extension

\- AgentScope agents

\- Git push



2\. Training computer:

\- Git pull

\- ICBHI2017 dataset storage

\- GPU training

\- experiment summaries



Goal of Phase 2:

Create the implementation skeleton for the RespirAI ICBHI2017 official 60/40 pipeline.



Do not build the SOTA model yet.



Create these files in this order:

1\. literature/sota\_summary.md

2\. src/data/official\_split.py

3\. src/data/icbhi\_dataset.py

4\. src/audio/preprocessing.py

5\. src/evaluation/metrics.py

6\. src/models/cnn\_baseline.py

7\. src/training/train.py

8\. src/evaluation/evaluate.py



Rules:

\- Use Git.

\- Do not commit datasets.

\- Do not commit checkpoints.

\- Do not commit .env files.

\- Code must be beginner-readable.

\- Every Python file should have comments.

\- First version can be simple, but it must be correct and extensible.



Output:

1\. File name

2\. Purpose

3\. Exact code or markdown content

4\. How to test it

5\. Git commit message

