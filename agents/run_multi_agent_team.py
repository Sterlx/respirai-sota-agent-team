"""
Run the full multi-agent team to create source files.

Each specialist agent gets role-specific external skills and produces
actual source code that is saved to disk. The PI agent orchestrates
and reviews all outputs.

Usage:
    python agents/run_multi_agent_team.py
"""

import os
import sys
import asyncio
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.credential import OpenAICredential

from skill_system.skill_loader import find_relevant_skills

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TEAM_SPEC_PATH = ROOT / "agents" / "team_spec.yaml"
PHASE2_PROMPT_PATH = ROOT / "agents" / "phase2_prompt.md"

# ---------------------------------------------------------------------------
# File → Agent assignments
# ---------------------------------------------------------------------------
# Each entry: (relative_file_path, agent_name, skill_search_query)
# The agent_name must match a key in AGENT_PROMPTS below.

FILE_ASSIGNMENTS: list[tuple[str, str, str]] = [
    (
        "literature/sota_summary.md",
        "literature_review_agent",
        "ICBHI2017 literature review lung sound classification "
        "respiratory sound detection papers survey systematic review "
        "PubMed arXiv PapersWithCode GitHub",
    ),
    (
        "src/data/official_split.py",
        "dataset_split_agent",
        "ICBHI2017 dataset preparation data splitting patient-wise split "
        "train test validation medical dataset file structure",
    ),
    (
        "src/data/icbhi_dataset.py",
        "dataset_split_agent",
        "PyTorch Dataset DataLoader ICBHI2017 audio dataset loading "
        "batching label encoding class distribution",
    ),
    (
        "src/audio/preprocessing.py",
        "audio_preprocessing_agent",
        "audio preprocessing lung sound respiratory sound signal processing "
        "mel spectrogram log-mel feature extraction resampling filtering "
        "respiratory cycle segmentation",
    ),
    (
        "src/audio/augmentation.py",
        "augmentation_agent",
        "audio augmentation SpecAugment mixup CutMix time stretching "
        "pitch shifting background noise injection medical audio data augmentation",
    ),
    (
        "src/evaluation/metrics.py",
        "evaluation_agent",
        "evaluation metrics sensitivity specificity ICBHI score confusion matrix "
        "F1 ROC AUC medical AI evaluation classification metrics per-class",
    ),
    (
        "src/models/cnn_baseline.py",
        "model_architect_agent",
        "CNN convolutional neural network PyTorch model architecture "
        "audio classification spectrogram classifier baseline ResNet EfficientNet",
    ),
    (
        "src/training/train.py",
        "training_engineer_agent",
        "PyTorch training loop mixed precision AMP gradient accumulation "
        "checkpointing configurable YAML config training script",
    ),
    (
        "src/evaluation/evaluate.py",
        "evaluation_agent",
        "model evaluation inference test evaluation metrics computation "
        "ICBHI score calculation per-class breakdown",
    ),
    (
        "src/training/hpo.py",
        "hpo_agent",
        "hyperparameter optimization Optuna Ray Tune learning rate tuning "
        "sweep search pruning early stopping",
    ),
    (
        "src/evaluation/experiment_tracker.py",
        "experiment_planner_agent",
        "experiment tracking ablation studies experiment planning "
        "results logging hypothesis tracking",
    ),
]

# ---------------------------------------------------------------------------
# Agent system prompts (derived from team_spec.yaml + phase2_prompt.md)
# ---------------------------------------------------------------------------

AGENT_PROMPTS: dict[str, dict[str, str]] = {
    "literature_review_agent": {
        "role": "Literature Review Agent — finds and summarizes ICBHI2017 SOTA methods.",
        "instructions": """
You are the Literature Review Agent for RespirAI.

Your job: produce a structured literature summary document (markdown) covering:
- Why the ICBHI2017 official 60/40 patient-wise split matters
- Model families to investigate (CNNs, CRNNs, transformers, ensembles)
- Metrics we must report (per-class Se/Sp, ICBHI score, confusion matrix)
- Why we do NOT claim SOTA yet
- Repositories and papers to investigate later
- External skill sources used

Rules:
- Do NOT invent benchmark numbers or scores.
- Flag papers that used random split (not official patient-wise).
- Be beginner-readable.
""",
    },
    "dataset_split_agent": {
        "role": "Dataset Split Agent — verifies ICBHI2017 official 60/40 split and creates PyTorch datasets.",
        "instructions": """
You are the Dataset Split Agent for RespirAI.

Your job: write CORRECT, RUNNABLE Python code for:
1. src/data/official_split.py — verify patient-wise split, detect leakage, print class distribution
2. src/data/icbhi_dataset.py — PyTorch Dataset that loads .wav files, applies labels, respects the split

THE ACTUAL DATASET is at data/audio/*.wav + data/audio/*.txt + data/official_split.txt.
Refer to the DATASET CONTEXT section below for exact file formats.

For official_split.py:
- Read data/official_split.txt (tab-separated: filename, train|test).
- Extract patient_id from filename (first underscore-delimited field).
- Group by patient: verify no patient appears in BOTH train and test.
- Print counts: total files, train files, test files, unique patients in each.
- Print class distribution from annotation files.

For icbhi_dataset.py:
- torch.utils.data.Dataset subclass.
- __init__ takes: data_dir, split_file, split ("train" or "test"), transform.
- __getitem__ returns: (waveform, labels_dict) where labels = {crackles, wheezes, both, normal}.
- Determine label from annotation .txt file: if any row has crackles=1 AND wheezes=1 → "both";
  elif any row has crackles=1 → "crackles"; elif any row has wheezes=1 → "wheeze"; else "normal".
- Use torchaudio.load() for .wav files.
- Use pathlib.Path, not hardcoded strings.

Rules:
- Use pathlib, not hardcoded paths. Read dataset root from ICBHI2017_ROOT env var
  with fallback to data/ relative to project root.
- Include docstrings and comments.
- Code must run without errors when dataset is present.
""",
    },
    "audio_preprocessing_agent": {
        "role": "Audio Preprocessing Agent — researches and designs the optimal lung sound preprocessing pipeline.",
        "instructions": """
You are the Audio Preprocessing Agent for RespirAI.

Your job: write src/audio/preprocessing.py — a preprocessing module for ICBHI2017 lung sounds.

Your module should:
- Research what preprocessing choices top ICBHI2017 papers use
- Provide resampling (target rate configurable)
- Provide bandpass filtering (default 50–2000 Hz for lung sounds)
- Compute log-mel spectrograms (configurable n_mels, n_fft, hop_length)
- Handle variable-length recordings with a segmentation strategy
- Document WHY each choice was made (cite papers where possible)

Rules:
- Use torchaudio for audio I/O and transformations.
- Every function should have a docstring.
- Parameters should be configurable.
""",
    },
    "augmentation_agent": {
        "role": "Augmentation Agent — designs audio augmentations for robust training.",
        "instructions": """
You are the Augmentation Agent for RespirAI.

Your job: write src/audio/augmentation.py — an augmentation module for lung sound training.

Implement:
- SpecAugment (time masking, frequency masking) for spectrograms
- Time stretching and pitch shifting for waveforms
- Background noise injection (hospital/ambient noise mix)
- A composable augmentation pipeline (torch.nn.Sequential or similar)
- Per-class augmentation configuration (some classes need more augmentation)

Rules:
- Augmentations must NOT distort the clinical signal beyond recognition.
- Use torchaudio transforms where possible.
- Every transform should have a docstring explaining its clinical safety.
""",
    },
    "model_architect_agent": {
        "role": "Model Architect Agent — designs candidate models.",
        "instructions": """
You are the Model Architect Agent for RespirAI.

Your job: write src/models/cnn_baseline.py — a baseline CNN for 4-class lung sound classification.

The model should:
- Take log-mel spectrograms as input (shape: [B, 1, n_mels, time])
- Use a simple but effective CNN architecture (3–4 conv blocks)
- Output 4-class logits: normal, crackle, wheeze, both
- Be compatible with torch.nn.CrossEntropyLoss
- Include a forward() method and a factory function

Rules:
- Keep it simple — this is a baseline, not the final SOTA model.
- Use nn.Sequential or explicit layers (your choice, just be clear).
- Include a comment showing expected input/output shapes.
- Do NOT hardcode spectrogram dimensions — accept them as constructor args.
""",
    },
    "training_engineer_agent": {
        "role": "Training Engineer Agent — writes training code for the two-computer workflow.",
        "instructions": """
You are the Training Engineer Agent for RespirAI.

Your job: write src/training/train.py — a PyTorch training script.

Implement:
- YAML config loading (read from configs/baseline.yaml)
- Training loop with:
  - Mixed precision (torch.cuda.amp)
  - Gradient accumulation for effective large batch sizes
  - Checkpointing (best model by validation score, periodic, last)
  - Progress bar (tqdm)
  - Logging to console and file
- Validation loop after each epoch
- Early stopping (configurable patience)

Rules:
- Use argparse or read config path from command line.
- Save checkpoints to the path from config.
- Do NOT hardcode dataset paths — read from config/env.
- Print per-class metrics at the end of each epoch.
""",
    },
    "evaluation_agent": {
        "role": "Evaluation Agent — computes all required metrics.",
        "instructions": """
You are the Evaluation Agent for RespirAI.

Your job: write TWO files:
1. src/evaluation/metrics.py — metric computation functions
2. src/evaluation/evaluate.py — evaluation script that loads a model and runs inference

For metrics.py, implement:
- Per-class sensitivity and specificity
- ICBHI official score: average of (Se + Sp)/2 per class
- Confusion matrix (4×4)
- Macro F1, Micro F1
- ROC-AUC per class (optional, nice to have)

For evaluate.py, implement:
- Load a trained model from checkpoint
- Run inference on the test set
- Print all metrics in a formatted table
- Save results to a JSON file

Rules:
- Use sklearn.metrics where possible (but compute ICBHI score manually — sklearn doesn't have it).
- Do NOT compute accuracy alone — it's misleading on imbalanced data.
""",
    },
    "hpo_agent": {
        "role": "HPO Agent — designs and runs hyperparameter optimization.",
        "instructions": """
You are the HPO Agent for RespirAI.

Your job: write src/training/hpo.py — a hyperparameter optimization script.

Implement:
- Optuna study setup (or Ray Tune — your choice, document why)
- Search space: learning rate, weight decay, batch size, n_mels, dropout rate
- Objective function that trains a model and returns the ICBHI validation score
- Pruning with MedianPruner or Hyperband
- Save best hyperparameters to a YAML file

Rules:
- Keep it modular — the objective function should call train.py logic.
- Include a README-style docstring at the top.
- Use reasonable search ranges (not infinite).
""",
    },
    "experiment_planner_agent": {
        "role": "Experiment Planner Agent — tracks experiments and proposes next steps.",
        "instructions": """
You are the Experiment Planner Agent for RespirAI.

Your job: write src/evaluation/experiment_tracker.py — a lightweight experiment tracking module.

Implement:
- A function to log experiment config + results to a JSON/CSV file
- A function to list all past experiments with key metrics
- A function to compare two experiments and highlight deltas
- A function to suggest the next ablation based on current results
- No external dependencies beyond stdlib + PyYAML

Rules:
- Keep it simple — we are not rebuilding MLflow.
- Use a single JSON-lines file as the experiment log.
- Each experiment record includes: timestamp, git commit, config snapshot, metrics dict.
""",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_deepseek_model() -> OpenAIChatModel:
    """Create a DeepSeek model instance for AgentScope agents."""
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        stream=True,
    )


async def ask(agent: Agent, message: str, label: str = "") -> str:
    """Send a message to an agent and collect the streaming response."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}\n", flush=True)
    else:
        print("Sending message to agent...\n", flush=True)

    chunks: list[str] = []

    async for evt in agent.reply_stream(UserMsg("user", message)):
        text = (
            getattr(evt, "text", None)
            or getattr(evt, "content", None)
            or getattr(evt, "delta", None)
            or getattr(evt, "message", None)
        )
        if text:
            text = str(text)
            chunks.append(text)
            print(text, end="", flush=True)

    print("\n", flush=True)
    return "".join(chunks)


def extract_code_block(response: str, file_path: str) -> str:
    """
    Extract the code or markdown content from an agent's response.
    Returns the content of the first fenced code block, or the raw
    response if no code block is found.
    """
    # Try to find a ```python or ```markdown or ``` block
    pattern = r"```(?:python|markdown|yaml|json)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        # Return the longest match (usually the actual code, not examples)
        return max(matches, key=len).strip()

    # Fallback: no code block found, return raw response
    print(f"  ⚠ No code block found in response for {file_path}, using raw text.")
    return response.strip()


def save_file(rel_path: str, content: str) -> Path:
    """Save content to a file relative to project root. Creates parent dirs."""
    full_path = ROOT / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content + "\n", encoding="utf-8")
    return full_path


# ---------------------------------------------------------------------------
# Dataset context — injected into agents that need to know the real data
# ---------------------------------------------------------------------------

DATASET_CONTEXT = """
=== ACTUAL DATASET ON DISK ===

The ICBHI2017 dataset is at: data/
  data/official_split.txt   — train/test split (tab-separated)
  data/audio/               — .wav + .txt annotation files

Split file format (data/official_split.txt):
  Each line:  {filename}\t{train|test}
  Filename is the recording base name (no extension).
  Example:  101_1b1_Al_sc_Meditron\ttest

Filename structure:
  {patient}_{recording_index}_{chest_location}_{acquisition_mode}_{equipment}
  - patient: 101–226 (first number before underscore)
  - chest_location: Tc (Trachea), Al/Ar (Anterior left/right),
    Pl/Pr (Posterior left/right), Ll/Lr (Lateral left/right)
  - acquisition_mode: sc (sequential), mc (multichannel)
  - equipment: Meditron, Litt3200, LittC2SE, AKGC417L

Annotation file format (data/audio/{filename}.txt):
  Tab-separated, 4 columns per row:
    start_time  end_time  crackles(0/1)  wheezes(0/1)
  Example: 0.364  3.25  0  1  (wheeze present, no crackles)
  Multiple rows per file = multiple respiratory cycles.

Patient extraction:
  patient_id = filename.split("_")[0]  (e.g., "101")
  The official split groups patients: same patient → entirely train OR test.
"""


# ---------------------------------------------------------------------------
# Cross-file compatibility rules (injected into every agent's system prompt)
# ---------------------------------------------------------------------------

CROSS_FILE_RULES = """
=== CROSS-FILE COMPATIBILITY RULES ===

Before writing any code, you MUST ensure compatibility with ALL other files
being created by other agents. Check these rules:

1. IMPORTS: If you import from another src/ file, the imported name
   MUST actually exist in that file. Double-check function/class names.

2. FUNCTION SIGNATURES: If you call a function or class from another file,
   the arguments you pass MUST match its signature exactly.
   No extra kwargs, no missing required args.

3. LABEL FORMAT: The dataset, model, and metrics MUST agree on labels.
   - ICBHIDataset returns: {"normal": 0/1, "crackles": 0/1, "wheezes": 0/1, "both": 0/1}
   - CNNBaseline output logit order: [normal, crackles, wheezes, both] → indices 0-3
   - Metrics expect: integer class indices 0-3
   - Use LABEL_TO_INDEX = {"normal": 0, "crackles": 1, "wheezes": 2, "both": 3}

4. DATA FLOW: Dataset → DataLoader → Model → Loss.
   - Dataset.__getitem__ returns (waveform, labels_dict)
   - train.py converts labels_dict → integer class index via a collate_fn
   - Preprocessing (waveform → spectrogram) is applied as a transform
   - Model outputs (batch, 4) logits for CrossEntropyLoss

5. CONFIG KEYS: If you access config["some_key"], ensure it exists
   in configs/baseline.yaml. Add it to the config if missing.

6. ONLY IMPORT FROM FILES IN THE PHASE 2 FILE LIST.
   Do NOT import from modules that don't exist yet.
"""


def build_file_creation_task(file_path: str, relevant_skills: str) -> str:
    """Build the task message asking an agent to create a specific file,
    with per-file skill injection and mandatory skill citation."""
    ext = Path(file_path).suffix

    if ext == ".md":
        format_instruction = (
            "Output the COMPLETE markdown document. "
            "Wrap your entire output in a ```markdown code block. "
            "The code block will be automatically extracted and saved to disk."
        )
    else:
        format_instruction = (
            "Output ONLY the Python source code. "
            "Wrap your entire output in a single ```python code block. "
            "The code block will be automatically extracted and saved to disk. "
            "Do NOT include explanatory text outside the code block."
        )

    skills_section = ""
    if relevant_skills and "No relevant external skills found" not in relevant_skills:
        skills_section = f"""
=== SKILLS FOR THIS FILE ===
Use the following external skills to guide your implementation.
You MUST cite which skill(s) you used by including a comment like:
  # Skill source: external/.../skill_file.md
at the top of your code / document.

{relevant_skills}
"""

    return f"""
CREATE FILE: {file_path}

{format_instruction}

{skills_section}
Before the code block, you may briefly explain your design decisions
(max 3 sentences). Then output the code block with the complete,
production-ready source code.

Requirements:
- The code must be complete and runnable (no placeholders, no "TODO" comments).
- Include all necessary imports.
- Include docstrings and inline comments.
- Follow the coding standards from the project rules.
- You MUST cite which external skill file(s) influenced your code.
  Use a comment at the top: # Skill source: external/.../skill_name.md
"""


def build_agent_system_prompt(
    agent_name: str,
    team_spec: str,
    phase2_prompt: str,
    relevant_skills: str,
) -> str:
    """Build a full system prompt for a specialist agent."""
    agent_info = AGENT_PROMPTS.get(agent_name)
    if not agent_info:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Inject dataset context for data-aware agents
    dataset_section = ""
    data_aware_agents = {
        "dataset_split_agent",
        "audio_preprocessing_agent",
        "augmentation_agent",
        "evaluation_agent",
        "training_engineer_agent",
    }
    if agent_name in data_aware_agents:
        dataset_section = DATASET_CONTEXT

    return f"""
You are the {agent_info['role']}

You are part of the RespirAI AI research team building a state-of-the-art
lung sound classification model using the ICBHI2017 official 60/40 split.

{agent_info['instructions']}

{dataset_section}
{CROSS_FILE_RULES}

=== TEAM SPECIFICATION ===
{team_spec}

=== PHASE 2 PROMPT ===
{phase2_prompt}

=== RELEVANT EXTERNAL SKILLS ===
{relevant_skills}

=== FILE OUTPUT RULES ===
When asked to create a file, output the complete source code in a fenced
code block (```python or ```markdown). The code block will be automatically
saved to the target file path. Do NOT use placeholders or "pass" statements —
write real, working code.

General project rules:
- Use Git for all code. Do NOT commit datasets, checkpoints, API keys, or .env files.
- Respect ICBHI2017 official 60/40 patient-wise split.
- Always ask for evidence before claiming SOTA.
- Code must be beginner-readable with comments.
- Refer to the external skills when they apply to your task.
- If dataset context is provided above, use the EXACT file paths and formats described.
"""


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def main():
    # --- Validate prerequisites ---
    if not TEAM_SPEC_PATH.exists():
        raise FileNotFoundError(f"Missing team spec: {TEAM_SPEC_PATH}")
    if not PHASE2_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 2 prompt: {PHASE2_PROMPT_PATH}")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Check your .env file.")

    print("=" * 60)
    print("  RespirAI Multi-Agent Team Runner")
    print("=" * 60)
    print(f"  Project root: {ROOT}")
    print(f"  Model: {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")
    print(f"  Files to create: {len(FILE_ASSIGNMENTS)}")
    print()

    team_spec = TEAM_SPEC_PATH.read_text(encoding="utf-8")
    phase2_prompt = PHASE2_PROMPT_PATH.read_text(encoding="utf-8")

    model = make_deepseek_model()

    # --- Collect all unique agents needed ---
    agent_names_needed = sorted({name for _, name, _ in FILE_ASSIGNMENTS})
    print(f"Agents needed: {', '.join(agent_names_needed)}\n")

    # --- Preload skills for each agent ---
    print("Loading external skills for each agent...\n")
    agent_skills: dict[str, str] = {}

    for agent_name in agent_names_needed:
        # Collect all skill queries for this agent's files
        queries = [q for _, name, q in FILE_ASSIGNMENTS if name == agent_name]
        combined_query = " ".join(queries)

        skills = find_relevant_skills(
            query=combined_query,
            max_results=5,
            excerpt_chars=3000,
        )
        agent_skills[agent_name] = skills

        # Count how many skills were found
        skill_count = skills.count("SKILL_SOURCE_REPO:")
        print(f"  {agent_name}: {skill_count} relevant skill(s) found")

    print()

    # --- Create all agents ---
    print("Creating agents...\n")
    agents: dict[str, Agent] = {}

    for agent_name in agent_names_needed:
        system_prompt = build_agent_system_prompt(
            agent_name=agent_name,
            team_spec=team_spec,
            phase2_prompt=phase2_prompt,
            relevant_skills=agent_skills[agent_name],
        )

        agent = Agent(
            name=f"RespirAI_{agent_name}",
            system_prompt=system_prompt,
            model=model,
        )
        agents[agent_name] = agent
        print(f"  ✓ {agent_name} created")

    # --- Create files: dispatch each file to its agent ---
    results: list[dict] = []

    for idx, (file_path, agent_name, skill_query) in enumerate(FILE_ASSIGNMENTS, 1):
        # Create __init__.py files alongside Python packages
        if file_path.endswith(".py"):
            pkg_dir = Path(file_path).parent
            init_path = pkg_dir / "__init__.py"
            init_full = ROOT / init_path
            if not init_full.exists():
                init_full.parent.mkdir(parents=True, exist_ok=True)
                init_full.write_text(
                    f"# RespirAI — {pkg_dir.as_posix().replace('/', ' › ')}\n",
                    encoding="utf-8",
                )

        # --- Per-file skill loading (stronger integration) ---
        per_file_skills = find_relevant_skills(
            query=f"{skill_query} {file_path}",
            max_results=3,
            excerpt_chars=2500,
        )

        agent = agents[agent_name]
        task = build_file_creation_task(file_path, per_file_skills)

        label = f"File {idx}/{len(FILE_ASSIGNMENTS)}: {file_path}  (agent: {agent_name})"
        response = await ask(agent, task, label=label)

        code = extract_code_block(response, file_path)
        saved_path = save_file(file_path, code)

        # Check if the agent cited any skills
        has_citation = "Skill source:" in code or "skill source:" in code

        results.append({
            "file": file_path,
            "agent": agent_name,
            "saved_to": str(saved_path),
            "lines": code.count("\n") + 1,
            "cited_skills": has_citation,
        })
        citation_note = "✓ cited skills" if has_citation else "⚠ no skill citation"
        print(f"  ✓ Saved {code.count(chr(10)) + 1} lines to {saved_path}  ({citation_note})\n")

    # --- PI Review pass ---
    print(f"\n{'='*60}")
    print("  PI REVIEW — Principal Investigator checks all outputs")
    print(f"{'='*60}\n")

    file_list = "\n".join(
        f"- {r['file']} (by {r['agent']}, {r['lines']} lines, "
        f"skills: {'cited' if r['cited_skills'] else 'NOT cited'})"
        for r in results
    )

    pi_prompt = f"""
You are the Principal Investigator for RespirAI.
You just coordinated the agent team to create the following files:

{file_list}

Your task: review the output. Answer these questions:
1. Are there any gaps or missing files?
2. Do the files follow the project rules (official split, Git-friendly, beginner-readable)?
3. What should the team do next?
4. Are we ready to move to Phase 3 (training)?

Be honest. If something is wrong, say so.
"""

    pi_agent = Agent(
        name="RespirAI_PI_Reviewer",
        system_prompt=f"""
You are the Principal Investigator Agent for RespirAI.
You review all agent outputs for quality, consistency, and completeness.

Team spec:
{team_spec}

Phase 2 prompt:
{phase2_prompt}

Rules:
- Be critical but constructive.
- Check that all files follow the official ICBHI2017 split rules.
- Check that no API keys, hardcoded user paths, or dataset files are referenced incorrectly.
- Recommend concrete next steps.
""",
        model=model,
    )

    review = await ask(pi_agent, pi_prompt, label="PI Final Review")

    # --- Save report ---
    report_dir = ROOT / "outputs" / "agent_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"multi_agent_run_{timestamp}.md"

    report_content = f"""# Multi-Agent Team Run Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}

## Files Created

| # | File | Agent | Lines | Skills Cited |
|---|------|-------|-------|:-------------:|
"""
    for r in results:
        cited = "✓" if r["cited_skills"] else "✗"
        report_content += (
            f"| {results.index(r)+1} | `{r['file']}` "
            f"| {r['agent']} | {r['lines']} | {cited} |\n"
        )

    report_content += f"""
## PI Review

{review}
"""
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\n📄 Report saved to: {report_path}")

    # --- Summary ---
    total_lines = sum(r["lines"] for r in results)
    cited_count = sum(1 for r in results if r["cited_skills"])
    print(f"\n{'='*60}")
    print(f"  DONE — {len(results)} files, {total_lines} total lines")
    print(f"  Agents used: {len(agents)}")
    print(f"  Skills cited: {cited_count}/{len(results)} files")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
