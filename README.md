# GSM-NoOp Robustness

Evaluating whether prompting strategies (CoT, Self-Consistency) can recover 
LLM accuracy degradation caused by irrelevant context in GSM-Symbolic NoOp 
math problems.

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to a .env file

## Project Structure
- `data/` — sampled problem pairs (original + NoOp)
- `prompts/` — prompt templates
- `models/` — inference code for each model
- `evaluation/` — answer extraction and metrics
- `results/` — raw model outputs
- `analysis/` — RQ3 trace annotation
- `notebooks/` — exploratory analysis
