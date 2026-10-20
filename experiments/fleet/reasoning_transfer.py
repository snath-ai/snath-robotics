"""
Reasoning Transfer -- language for the REASONING step, real numbers for
direction and magnitude.
=============================================================================
Session correction to the "ship language, not raw numbers" idea: the LLM's
job here is not to invent a correction -- lgsl_routing_gate.py already
proved why that's the wrong ask ("asking a small, non-fine-tuned model to
emit precise floats reliably is unrealistic"). The LLM's job is to REASON
over extraction.py's real measured data and produce an interpretable,
auditable instruction, the same "LLM suggests, a real mechanism decides"
split already proven in this repo -- except here the real mechanism is
extract_correction()'s actual measurement, not a human eyeballing numbers.

This file does exactly one thing, honestly: give the real Qwen2.5-0.5B
model (same checkpoint as lgsl_routing_gate.py) the REAL per-joint
extraction output from extraction.py, constrained to a small per-joint
menu (INCREASE / DECREASE / NO_CHANGE -- not raw numbers, the same LGSL
discipline), and CHECK whether its answer matches the real measured sign
pattern. This is not assumed to work. A 0.5B model reasoning correctly
over six numbers with a trustworthiness flag each is a real, testable
claim, not a given -- it is checked against ground truth below, not just
printed and trusted.

What this does NOT yet do: feed the result into PERSIST's increment search
for magnitude calibration on the weak policy. That is the next step, kept
separate so a bug in the magnitude-search code (already caught once
tonight) doesn't get conflated with whether the reasoning step itself is
trustworthy.

Run:
    .venv/bin/python experiments/fleet/reasoning_transfer.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import gymnasium
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stable_baselines3 import PPO

from experiments.fleet.fleet_n2_transfer import FRICTION_NORMAL, FRICTION_ICE, set_friction
from experiments.fleet.extraction import extract_correction

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
JOINT_NAMES = ["thigh_r", "leg_r", "foot_r", "thigh_l", "leg_l", "foot_l"]
CHOICES = {"INCREASE", "DECREASE", "NO_CHANGE"}


def build_prompt(extracted) -> str:
    """Give the LLM the real measured numbers, not a vague description --
    and constrain its answer to a small per-joint menu, LGSL-style."""
    lines = [
        "A walking robot's joint effort was measured on icy ground versus "
        "normal ground, 8 real trials, with statistics for each of 6 joints.",
        "",
    ]
    for i, name in enumerate(JOINT_NAMES):
        trust = "statistically reliable" if extracted.trustworthy_dims()[i] else "NOT reliable (ignore this one)"
        direction = "higher" if extracted.delta[i] > 0 else "lower"
        lines.append(
            f"  {name}: effort on ice is {direction} than normal by "
            f"{abs(extracted.delta[i]):.3f} (t={extracted.t_stats[i]:+.1f}, {trust})"
        )
    lines.append("")
    lines.append(
        "For each joint, answer with exactly one word: INCREASE, DECREASE, "
        "or NO_CHANGE -- INCREASE if a robot coping well with ice should use "
        "MORE of that joint's effort than it does on normal ground, DECREASE "
        "if less, NO_CHANGE if the data is not reliable enough to say. "
        "Answer as six lines, 'joint_name: WORD', nothing else."
    )
    return "\n".join(lines)


def parse_llm_answer(text: str) -> dict:
    """Conservative parsing, matching lgsl_routing_gate.py's own rule: an
    unparseable line is None, never guessed."""
    answers = {}
    for name in JOINT_NAMES:
        m = re.search(rf"{name}\s*:\s*(\w+)", text, re.IGNORECASE)
        if m and m.group(1).upper() in CHOICES:
            answers[name] = m.group(1).upper()
        else:
            answers[name] = None
    return answers


def main():
    print("=" * 78)
    print("REASONING TRANSFER -- a real LLM reasoning over real measured data")
    print("=" * 78)

    env = gymnasium.make("Walker2d-v5")
    model = PPO.load(PPO_CHECKPOINT)

    def ppo_action(o):
        a, _ = model.predict(o, deterministic=True)
        return np.asarray(a, dtype=np.float32)

    def ppo_reset():
        pass

    print("\nStep 1: extract_correction() -- the real measurement (not re-derived, "
          "same function verified earlier tonight).")
    extracted = extract_correction(
        ppo_action, ppo_reset, env,
        lambda e: set_friction(e, FRICTION_ICE),
        lambda e: set_friction(e, FRICTION_NORMAL),
        n_seeds=8, n_steps=200, seed_base=1,
    )
    print(extracted.summary())

    ground_truth = {}
    for i, name in enumerate(JOINT_NAMES):
        if not extracted.trustworthy_dims()[i]:
            ground_truth[name] = "NO_CHANGE"
        else:
            ground_truth[name] = "INCREASE" if extracted.delta[i] > 0 else "DECREASE"
    print(f"\nGround truth (from the real measurement, trustworthy dims only): {ground_truth}")

    print(f"\nStep 2: load the real Qwen2.5-0.5B-Instruct model (same checkpoint "
          f"lgsl_routing_gate.py uses) and ask it to reason over this data.")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    llm.eval()

    prompt = build_prompt(extracted)
    print(f"\n--- PROMPT ---\n{prompt}\n--------------")

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenized = tokenizer(rendered, return_tensors="pt")
    with torch.no_grad():
        out_ids = llm.generate(
            **tokenized, max_new_tokens=80, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_output = tokenizer.decode(
        out_ids[0][tokenized["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"\n--- RAW LLM OUTPUT ---\n{raw_output}\n----------------------")

    print("\nStep 3: parse conservatively and check against ground truth -- "
          "not assumed correct, checked.")
    llm_answers = parse_llm_answer(raw_output)
    correct, total_parsed, total_checked = 0, 0, 0
    for name in JOINT_NAMES:
        gt = ground_truth[name]
        got = llm_answers[name]
        total_checked += 1
        if got is not None:
            total_parsed += 1
            match = (got == gt)
            correct += int(match)
            print(f"  {name}: ground_truth={gt:9s} llm_said={got:9s} "
                  f"{'MATCH' if match else 'MISMATCH'}")
        else:
            print(f"  {name}: ground_truth={gt:9s} llm_said=UNPARSEABLE")

    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"Parsed {total_parsed}/{total_checked} joints into a valid answer.")
    if total_parsed > 0:
        print(f"Matched ground truth on {correct}/{total_parsed} parsed joints "
              f"({100*correct/total_parsed:.0f}%).")
    if total_parsed == total_checked and correct == total_parsed:
        print("\nPASS -- the LLM's reasoning, over real measured data, exactly "
              "matches the real sign pattern. Worth trusting as the direction "
              "input to a magnitude search.")
    elif total_parsed < total_checked:
        print("\nPARTIAL -- some joints were not parseable. Per lgsl_routing_"
              "gate.py's own rule, unparseable means 'cannot determine, do not "
              "commit' -- those joints should not be acted on, not guessed.")
    else:
        print("\nHONEST NEGATIVE -- the LLM's reasoning does not reliably match "
              "the real data. A 0.5B model should not be trusted to supply the "
              "direction here without this check; the real extraction's own "
              "sign pattern is the one to use, not the LLM's restatement of it.")

    env.close()


if __name__ == "__main__":
    main()
