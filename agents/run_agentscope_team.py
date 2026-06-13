import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.credential import OpenAICredential

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TEAM_SPEC_PATH = ROOT / "agents" / "team_spec.yaml"


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
    print("Sending message to agent...", flush=True)

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

    print("\nDone.", flush=True)
    return "".join(chunks)


async def main():
    print("Starting RespirAI AgentScope test...", flush=True)

    if not TEAM_SPEC_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEAM_SPEC_PATH}")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Check your .env file.")

    print(f"Project root: {ROOT}", flush=True)
    print(f"Model: {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}", flush=True)

    team_spec = TEAM_SPEC_PATH.read_text(encoding="utf-8")

    model = make_deepseek_model()

    pi_agent = Agent(
        name="RespirAI_Principal_Investigator",
        system_prompt=f"""
You are the Principal Investigator Agent for RespirAI.
You coordinate the AI research team.

Team spec:
{team_spec}

Rules:
- Be practical.
- Use Git for all code.
- Do not put datasets, keys, or checkpoints in Git.
- Respect ICBHI2017 official 60/40 split.
- Always ask for evidence before claiming SOTA.
""",
        model=model,
    )

    task = """
Create the first milestone plan for the RespirAI agent team.

Output:
1. Agent roles
2. Git workflow
3. Files to create next
4. What the training computer should do
5. What the laptop should do
6. Definition of done
"""

    result = await ask(pi_agent, task)

    out_dir = ROOT / "outputs" / "agent_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "agentscope_first_milestone.md"
    out_file.write_text(result, encoding="utf-8")

    print(f"Saved output to: {out_file}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())