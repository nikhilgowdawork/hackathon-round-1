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

# ---------------- SAFE FORMAT (ONLY FOR PRINT) ----------------
def safe_format(value: float) -> str:
    value = float(value)

    if value >= 1.0:
        value = 0.999
    elif value <= 0.0:
        value = 0.001

    return f"{value:.4f}"

# ---------------- LOGGING ----------------
def log_start(task: str):
    print(f"[START] task={task} env=crisis_response_env model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"

    print(
        f"[STEP] step={step} action={action} reward={safe_format(reward)} "
        f"done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(safe_format(r) for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )

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

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )

        output = response.choices[0].message.content.strip()

        
        if task_type == "classify_urgency":
            return output.strip().lower(), None

        try:
            return json.loads(output), None
        except:
            return output, f"invalid_output={output}"

    except Exception as e:
        return None, str(e)

# ---------------- RUN TASK ----------------
def run_task(name, task):
    log_start(name)

    rewards = []

    try:
        observation = task.get_observation()
        output, error = get_llm_output(observation)

        if output is None:
            log_step(1, "error", 0.001, True, error)
            log_end(False, 1, [0.001])
            return 0.001

        raw_score = task.grade(output)

      
        if raw_score >= 1.0:
            raw_score = 0.999
        elif raw_score <= 0.0:
            raw_score = 0.001

        action_str = str(output).replace("\n", "")

        log_step(
            step=1,
            action=action_str,
            reward=raw_score,
            done=True,
            error=error
        )

        rewards.append(raw_score)

        log_end(
            success=raw_score > 0.1,
            steps=1,
            rewards=rewards
        )

        return raw_score

    except Exception as e:
        log_step(1, "error", 0.001, True, str(e))
        log_end(False, 1, [0.001])
        return 0.001

# ---------------- MAIN ----------------
async def main():
    tasks = [
        ("easy", create_easy_task()),
        ("medium", create_medium_task()),
        ("hard", create_hard_task())
    ]

    total = 0.0

    for name, task in tasks:
        score = run_task(name, task)
        total += score

    final_score = max(0.001, min(0.999, total / len(tasks)))
    print(f"\nFINAL SCORE: {final_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())