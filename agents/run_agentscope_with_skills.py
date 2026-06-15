import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.credential import OpenAICredential

from skill_system.skill_loader import find_relevant_skills


load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TEAM_SPEC_PATH = ROOT / "agents" / "team_spec.yaml"
PHASE2_PROMPT_PATH = ROOT / "agents" / "phase2_prompt.md"


def make_deepseek_model():
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        stream=True,
    )


async def ask(agent: Agent, message: str) -> str:
    print("Sending message to skill-enabled AgentScope agent...\n", flush=True)

    chunks = []

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

    print("\n\nDone.", flush=True)
    return "".join(chunks)


async def main():
    print("Starting RespirAI skill-enabled AgentScope runner...", flush=True)

    if not TEAM_SPEC_PATH.exists():
        raise FileNotFoundError(f"Missing team spec: {TEAM_SPEC_PATH}")

    if not PHASE2_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 2 prompt: {PHASE2_PROMPT_PATH}")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Check your .env file.")

    team_spec = TEAM_SPEC_PATH.read_text(encoding="utf-8")
    phase2_prompt = PHASE2_PROMPT_PATH.read_text(encoding="utf-8")

    task = """
Begin Phase 2H.5.

We are now using external skills with AgentScope.

Create the implementation plan for File 1 only:

literature/sota_summary.md

The file should explain:
1. Why ICBHI2017 official 60/40 split matters
2. What model families we should investigate later
3. What metrics we must report
4. Why we do not claim SOTA yet
5. What repositories or papers should be investigated later
6. Which external skill sources influenced the plan

Output the full markdown content for literature/sota_summary.md.
"""

    skill_query = f"""
    {phase2_prompt}

    {task}

    Relevant concepts:
    AI research workflow, literature review, model training, evaluation,
    medical AI, clinical safety, reproducibility, respiratory sound classification,
    ICBHI2017 official split, lung sound detection
    """

    relevant_skills = find_relevant_skills(
        query=skill_query,
        max_results=6,
        excerpt_chars=3000,
    )

    model = make_deepseek_model()

    pi_agent = Agent(
        name="RespirAI_Skill_Enabled_PI",
        system_prompt=f"""
You are the Principal Investigator Agent for RespirAI.

You coordinate an AI research team building a lung sound classification model
for the ICBHI2017 official 60/40 split.

TEAM SPEC:
{team_spec}

PHASE 2 PROMPT:
{phase2_prompt}

EXTERNAL SKILLS LOADED:
{relevant_skills}

Rules:
- Use the external skills when relevant.
- Include a section named "External skill sources used".
- Mention source files that influenced your answer.
- Be practical and beginner-readable.
- Do not claim SOTA without evidence.
- Do not invent benchmark numbers.
- Do not put datasets, checkpoints, API keys, or .env files into Git.
""",
        model=model,
    )

    result = await ask(pi_agent, task)

    out_dir = ROOT / "outputs" / "agent_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "phase2_file1_sota_summary_with_skills.md"
    out_file.write_text(result, encoding="utf-8")

    print(f"Saved output to: {out_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())