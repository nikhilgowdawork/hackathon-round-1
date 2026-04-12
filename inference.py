import asyncio
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from tasks.task_easy import create_easy_task
from tasks.task_medium import create_medium_task
from tasks.task_hard import create_hard_task

# ---------------- ENV ----------------
load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not API_KEY:
    raise ValueError("HF_TOKEN or API_KEY required")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------- SCORE FIX ----------------
def adjust_score(score: float) -> float:
    """
    Force score into (0,1) range strictly
    """
    if score <= 0.0:
        return 0.05
    if score >= 1.0:
        return 0.95
    return score

# ---------------- LOGGING ----------------
def log_start(task: str):
    print(f"[START] task={task} env=crisis_response_env model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ---------------- LLM ----------------
def get_llm_output(observation: Dict):
    task_type = observation.get("task")

    if task_type == "classify_urgency":
        format_hint = "Return ONLY one word: low, medium, or high"
    elif task_type == "allocate_resources":
        format_hint = 'Return ONLY JSON list like ["ambulance","fire_truck"]'
    else:
        format_hint = 'Return ONLY JSON like {"plan":[{"incident_id":1,"resources":["fire_truck"]}]}'

    prompt = f"""
You are an expert crisis response system.

{format_hint}

Situation:
{observation}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    output = response.choices[0].message.content.strip()

    try:
        return json.loads(output), None
    except:
        return output, f"invalid_output={output}"

# ---------------- RUN TASK ----------------
def run_task(name, task):
    log_start(name)

    observation = task.get_observation()

    output, error = get_llm_output(observation)

    raw_score = task.grade(output)
    score = adjust_score(raw_score)   # 🔥 FIX HERE

    action_str = str(output).replace("\n", "")

    log_step(
        step=1,
        action=action_str,
        reward=score,
        done=True,
        error=error
    )

    log_end(success=score > 0, steps=1, rewards=[score])

    return score

# ---------------- MAIN ----------------
async def main():
    tasks = [
        ("easy", create_easy_task()),
        ("medium", create_medium_task()),
        ("hard", create_hard_task())
    ]

    total = 0

    for name, task in tasks:
        score = run_task(name, task)
        total += score

    final_score = total / len(tasks)
    print(f"\nFINAL SCORE: {final_score:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
    