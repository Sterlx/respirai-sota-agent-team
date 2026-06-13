\# RespirAI Two-Computer Workflow



\## Laptop



The laptop is used for:

\- Writing code in VS Code

\- Running the DeepSeek coding extension

\- Running the AgentScope AI agent team

\- Creating plans, prompts, configs, and documentation

\- Committing and pushing code to GitHub



The laptop does not run full model training.



\## Training Computer



The training computer is used for:

\- Pulling code from GitHub

\- Storing the ICBHI2017 dataset locally

\- Running PyTorch training

\- Saving checkpoints locally

\- Writing small experiment summaries

\- Pushing only code/config/report files back to GitHub



The training computer does not run the agent team at first.



\## GitHub



GitHub is used to sync code between the laptop and training computer.



Do not commit:

\- dataset files

\- WAV files

\- checkpoints

\- API keys

\- .env files

\- private keys

