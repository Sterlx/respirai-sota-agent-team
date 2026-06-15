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



Create these files in this order (each file is assigned to a specialist agent):

1. literature/sota_summary.md          → literature_review_agent
2. src/data/official_split.py          → dataset_split_agent
3. src/data/icbhi_dataset.py           → dataset_split_agent
4. src/audio/preprocessing.py          → audio_preprocessing_agent
5. src/audio/augmentation.py           → augmentation_agent
6. src/evaluation/metrics.py           → evaluation_agent
7. src/models/cnn_baseline.py          → model_architect_agent
8. src/training/train.py               → training_engineer_agent
9. src/evaluation/evaluate.py          → evaluation_agent
10. src/training/hpo.py                → hpo_agent
11. src/evaluation/experiment_tracker.py → experiment_planner_agent



Rules:

\- Use Git.

\- Do not commit datasets.

\- Do not commit checkpoints.

\- Do not commit .env files.

\- Code must be beginner-readable.

\- Every Python file should have comments.

\- First version can be simple, but it must be correct and extensible.

\- Each file is created by its assigned specialist agent (see file→agent mapping above).

\- Agents load external skills from AI-research-SKILLs and scientific-agent-skills.

Output:

1\. File name

2\. Purpose

3\. Exact code or markdown content

4\. How to test it

5\. Git commit message

How to run the multi-agent team:

    python agents/run_multi_agent_team.py

This creates all agents, loads their skills, and produces every file.

