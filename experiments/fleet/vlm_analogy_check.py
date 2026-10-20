"""
VLM Analogy Check -- can a real vision-language model look at what the
robot is actually doing and tell whether a NEW situation is the same
TYPE of problem as one it already has a stored fix for, or a genuinely
different type that needs a different fix?
=============================================================================
Not a retry of reasoning_transfer.py's failed test. That test asked a
small LLM to read RAW NUMBERS (per-joint deltas, t-stats) and output
INCREASE/DECREASE/NO_CHANGE -- precise numerical inference, exactly what
small models are bad at, and it scored at chance. This is a different
task: analogical, visual classification -- "is this the same kind of
problem as the one I already solved, or different" -- closer to what a
VLM is actually trained for, and the one explicitly proposed this session
(bicycle balance transferring to motorcycle balance; the same underlying
skill, a different instance).

Real ground truth, not assumed: foot_leg_ratio_check.py and
wind_confound_check.py already established, with real multi-seed
statistics, that ice and wind push the SAME diagnostic signal (foot/leg
effort ratio) in OPPOSITE directions -- ice drops it (t=-20.69), wind
raises it (t>0 at every tested level, all four magnitudes). That is not
"a bigger version of the same problem" -- it is a genuinely different
mechanism (losing foot traction vs. bracing against a push), the exact
distinction PERSIST's library is supposed to make before deciding whether
a stored adaptor applies or a new one is needed.

This script renders REAL MuJoCo frames (not text, not numbers) of PPO's
real gait under ice, wind, and normal friction, and asks a real, small,
edge-deployable VLM (SmolVLM-256M-Instruct) to look at a stored
"reference" scenario's frames and a "new" scenario's frames and judge:
same type of problem, or different. Checked against the real ground truth
above, not trusted on sight -- same discipline as every other check this
session.

"Suggest, don't decide" still holds: this only tests whether the VLM can
tell same-type apart from different-type. It is not asked to, and
shouldn't be trusted to, output a correction itself -- that is PERSIST's
job, downstream, same as the discovery mechanism's output always was.

Run:
    .venv/bin/python experiments/fleet/vlm_analogy_check.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from PIL import Image
from stable_baselines3 import PPO
from experiments.fleet.fleet_n2_transfer import set_friction, apply_wind, FRICTION_NORMAL, FRICTION_ICE

PPO_CHECKPOINT = "/Users/aadithya/Desktop/Lar_Main/lar_jepa/models/walker2d_ppo.zip"
WIND_FORCE = 20.0  # matches ood_guard.py's own sweep, a level already confirmed discriminative
FRAME_STEPS = [25, 45, 65]  # sample the gait mid-rollout, after initial settle
N_PAIRS = 6  # paired comparisons per condition, same discipline as everywhere else tonight

OUT_DIR = Path("/private/tmp/claude-501/-Users-aadithya-Desktop-Private-personal/"
                "ea57c1b6-e261-4622-acad-5a728be990e2/scratchpad/vlm_frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_scenario(model, seed, condition):
    """condition in {'normal','ice','wind'}. Returns list of PIL Images."""
    env = gymnasium.make("Walker2d-v5", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    if condition == "normal":
        set_friction(env, FRICTION_NORMAL)
    elif condition == "ice":
        set_friction(env, FRICTION_ICE)
    elif condition == "wind":
        set_friction(env, FRICTION_NORMAL)
    else:
        raise ValueError(condition)

    frames = []
    max_step = max(FRAME_STEPS)
    for i in range(max_step + 1):
        if condition == "wind":
            apply_wind(env, WIND_FORCE)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if i in FRAME_STEPS:
            frames.append(Image.fromarray(env.render()))
        if term or trunc:
            # pad remaining requested frames with the last available one --
            # a fall itself is real information, not something to hide
            while len(frames) < len(FRAME_STEPS):
                frames.append(frames[-1] if frames else Image.fromarray(env.render()))
            break
    env.close()
    return frames


def save_frames(frames, label):
    paths = []
    for i, f in enumerate(frames):
        p = OUT_DIR / f"{label}_{i}.png"
        f.save(p)
        paths.append(p)
    return paths


def build_prompt():
    return (
        "You are looking at two short sequences of frames from a walking robot simulation. "
        "The FIRST set of frames shows a REFERENCE problem a robot already has a fix for. "
        "The SECOND set of frames shows a NEW situation a different robot just encountered. "
        "Based on what you can see about how the robot is moving or struggling in each set, "
        "is the NEW situation the SAME TYPE of problem as the REFERENCE (just happening with "
        "different intensity), or a DIFFERENT TYPE of problem that would need a different kind "
        "of fix? Answer with exactly one word first: SAME or DIFFERENT. Then briefly explain "
        "what you see that led you to that answer."
    )


def parse_answer(text):
    m = re.search(r"\b(SAME|DIFFERENT)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def main():
    print("=" * 78)
    print("VLM ANALOGY CHECK -- same type of problem, or different?")
    print("=" * 78)

    model = PPO.load(PPO_CHECKPOINT)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
    print("\nLoading SmolVLM-256M-Instruct (a genuinely small, edge-deployable VLM).")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float32
    )
    vlm.eval()

    def ask_vlm(ref_frames, new_frames):
        images = ref_frames + new_frames
        content = (
            [{"type": "text", "text": "REFERENCE problem (already solved):"}]
            + [{"type": "image"} for _ in ref_frames]
            + [{"type": "text", "text": "NEW situation (just encountered):"}]
            + [{"type": "image"} for _ in new_frames]
            + [{"type": "text", "text": build_prompt()}]
        )
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        with torch.no_grad():
            out_ids = vlm.generate(**inputs, max_new_tokens=100, do_sample=False)
        raw = processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return raw

    seeds = [1013, 1026, 1039, 1052, 1065, 1078, 1091, 1104][:N_PAIRS]

    print(f"\nStep 1: render real PPO rollouts, {len(seeds)} seeds, ice/wind/normal, "
          f"frames at steps {FRAME_STEPS}.")
    ice_frames = {s: render_scenario(model, s, "ice") for s in seeds}
    wind_frames = {s: render_scenario(model, s, "wind") for s in seeds}

    print(f"\nStep 2: ask the VLM, {N_PAIRS} SAME-type pairs (ice vs ice, different seed) "
          f"+ {N_PAIRS} DIFFERENT-type pairs (ice vs wind). Ground truth from "
          f"foot_leg_ratio_check.py / wind_confound_check.py: ice and wind push the SAME "
          f"diagnostic signal in OPPOSITE directions -- a real, different mechanism, not "
          f"a scaled-up version of one problem.")

    results = []
    print("\n--- SAME-type pairs (ice vs ice, different seed) ---")
    for i in range(N_PAIRS):
        ref_seed, new_seed = seeds[i], seeds[(i + 1) % N_PAIRS]
        raw = ask_vlm(ice_frames[ref_seed], ice_frames[new_seed])
        answer = parse_answer(raw)
        correct = (answer == "SAME")
        results.append(("SAME", answer, correct))
        print(f"  ref=ice(seed{ref_seed}) new=ice(seed{new_seed}): "
              f"vlm_said={answer}  {'CORRECT' if correct else 'WRONG' if answer else 'UNPARSEABLE'}")
        print(f"    raw: {raw[:150]!r}")

    print("\n--- DIFFERENT-type pairs (ice reference vs wind new) ---")
    for i in range(N_PAIRS):
        ref_seed, new_seed = seeds[i], seeds[i]
        raw = ask_vlm(ice_frames[ref_seed], wind_frames[new_seed])
        answer = parse_answer(raw)
        correct = (answer == "DIFFERENT")
        results.append(("DIFFERENT", answer, correct))
        print(f"  ref=ice(seed{ref_seed}) new=wind(seed{new_seed}): "
              f"vlm_said={answer}  {'CORRECT' if correct else 'WRONG' if answer else 'UNPARSEABLE'}")
        print(f"    raw: {raw[:150]!r}")

    parsed = [r for r in results if r[1] is not None]
    correct = [r for r in parsed if r[2]]
    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"Parsed {len(parsed)}/{len(results)} into a valid SAME/DIFFERENT answer.")
    if parsed:
        print(f"Correct against real ground truth: {len(correct)}/{len(parsed)} "
              f"({100*len(correct)/len(parsed):.0f}%). Chance level on a binary task is 50%.")
    same_results = [r for r in results if r[0] == "SAME"]
    diff_results = [r for r in results if r[0] == "DIFFERENT"]
    same_acc = sum(1 for r in same_results if r[2]) / len(same_results) if same_results else 0
    diff_acc = sum(1 for r in diff_results if r[2]) / len(diff_results) if diff_results else 0
    print(f"  SAME-type pairs: {sum(1 for r in same_results if r[2])}/{len(same_results)} correct")
    print(f"  DIFFERENT-type pairs: {sum(1 for r in diff_results if r[2])}/{len(diff_results)} correct")
    if same_acc > 0.5 and diff_acc > 0.5:
        print("\nReal signal above chance on BOTH directions -- not just always answering "
              "one word. Worth taking seriously as a retrieval-time signal, still only to "
              "suggest which stored adaptor is relevant, never to decide the correction itself.")
    else:
        print("\nAt or below chance on at least one direction -- consistent with this VLM "
              "not reliably telling these specific visual scenarios apart from rendered "
              "frames at this resolution/size, not evidence the general approach is wrong.")


if __name__ == "__main__":
    main()
