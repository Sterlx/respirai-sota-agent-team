import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TEAM_SPEC = ROOT / "agents" / "team_spec.yaml"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


def ask_agent(agent_name: str, role_prompt: str, task: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": role_prompt,
            },
            {
                "role": "user",
                "content": task,
            },
        ],
        stream=False,
    )
    return response.choices[0].message.content


def main() -> None:
    team_spec = TEAM_SPEC.read_text(encoding="utf-8")

    task = f"""
You are part of the RespirAI AI research team.

Project:
- Smart stethoscope lung sound classification
- Dataset: ICBHI2017
- Target split: official 60/40
- Goal: create a SOTA model, but first create the research/build plan.

Team specification:
{team_spec}

Your task:
Create a 2-week plan for the agent team.
The plan must include:
1. Git workflow
2. Dataset preparation
3. Baseline model
4. SOTA model search
5. Training computer workflow
6. Evaluation metrics
7. Safety checks
"""

    pi_prompt = """
You are the Principal Investigator Agent for RespirAI.
You coordinate all agents and produce clear, practical research plans.
You are strict about reproducibility, correct ICBHI2017 split usage, and clinical safety.
"""

    answer = ask_agent("principal_investigator", pi_prompt, task)

    out_dir = ROOT / "outputs" / "agent_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "week_1_plan.md"
    out_file.write_text(answer, encoding="utf-8")

    print(answer)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()