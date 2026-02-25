#!/usr/bin/env python3
"""
Compare three LogitMax generation scenarios on Llama3 with dummy adapters.

Scenarios:
1) Naive separate case: three adapter-specific cache trajectories.
   Combine per-step logits with elementwise max, then choose next token.
2) Composed adapters in order [1, 2, 3] with FuseKit LogitMax composition.
3) Composed adapters in order [3, 2, 1] with FuseKit LogitMax composition.

For each scenario, prints:
- Generated text for each sample
- Top-5 next-token candidates at each generation step for each sample
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from fusekit import Modeling


DEFAULT_PROMPTS = [
    "Which option is most plausible?\nA. Red\nB. Blue\nC. Green\nD. Yellow\nAnswer:",
    "Complete the pattern: 2, 4, 8, 16, ?\nA. 18\nB. 24\nC. 32\nD. 64\nAnswer:",
]


def build_prompt_batches(num_batches: int) -> list[list[str]]:
    if num_batches == 1:
        return [list(DEFAULT_PROMPTS)]

    # Deliberately vary batch size across consecutive calls so stale KV-cache
    # bugs become visible when the same model instance is reused.
    prompt_batches = [list(DEFAULT_PROMPTS[:1])]
    for _ in range(num_batches - 1):
        prompt_batches.append(list(DEFAULT_PROMPTS))
    return prompt_batches


def normalize_scenario(raw: str) -> str:
    key = raw.strip().lower()
    aliases = {
        "all": "all",
        "1": "naive",
        "2": "compose_123",
        "3": "compose_321",
        "naive": "naive",
        "compose_123": "compose_123",
        "compose123": "compose_123",
        "compose_321": "compose_321",
        "compose321": "compose_321",
    }
    if key not in aliases:
        raise ValueError(
            f"Invalid scenario '{raw}'. Use one of: all, 1, 2, 3, "
            "naive, compose_123, compose_321"
        )
    return aliases[key]


def parse_devices(devices: str) -> list[int] | None:
    s = devices.strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_llama3_8b(
    device_list: list[int] | None,
    memory_limit: int,
    precision: torch.dtype,
):
    model = Modeling.Llama3_8b(
        device=device_list,
        memory_limit=memory_limit,
        precision=precision,
        force_sharding=True,
    )
    model.eval()
    model.dispatch_model()
    return model


def unload_model(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    clear_cuda()


def create_dummy_adapters(
    adapter_root: Path,
    device_list: list[int] | None,
    memory_limit: int,
    precision: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    force_recreate: bool,
) -> list[Path]:
    adapter_paths = [adapter_root / f"adapter_{i}" for i in (1, 2, 3)]
    if not force_recreate and all((p / "adapter_config.json").exists() for p in adapter_paths):
        return adapter_paths

    adapter_root.mkdir(parents=True, exist_ok=True)
    base = build_llama3_8b(device_list, memory_limit, precision)
    base = base.init_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    scales = [0.005, 0.010, 0.015]
    seeds = [101, 202, 303]

    for path, scale, seed in zip(adapter_paths, scales, seeds):
        if path.exists():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    import shutil
                    shutil.rmtree(child)
        path.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for name, param in base.model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    noise = torch.randn_like(param)
                    param.copy_(noise.mul_(scale))
        base.model.save_pretrained(path)

    unload_model(base)
    return adapter_paths


def encode_prompts(tokenizer, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in batch.items()}


def token_str(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def format_top5_for_sample(top_ids: torch.Tensor, top_vals: torch.Tensor, tokenizer) -> list[dict[str, Any]]:
    out = []
    for token_id, val in zip(top_ids.tolist(), top_vals.tolist()):
        out.append(
            {
                "id": int(token_id),
                "token": token_str(tokenizer, int(token_id)),
                "logit": float(val),
            }
        )
    return out


def append_next_token(batch: dict[str, torch.Tensor], next_token: torch.Tensor) -> None:
    input_ids = batch["input_ids"]
    attn_mask = batch["attention_mask"]

    # With model sharding, logits can be produced on a different CUDA device than
    # the original token tensors. Force alignment before concatenation.
    next_token = next_token.to(device=input_ids.device, dtype=input_ids.dtype)

    batch["input_ids"] = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    ones = torch.ones(
        (next_token.shape[0], 1),
        dtype=attn_mask.dtype,
        device=attn_mask.device,
    )
    batch["attention_mask"] = torch.cat([attn_mask, ones], dim=-1)


def run_autoregressive_loop(
    tokenizer,
    batch: dict[str, torch.Tensor],
    num_steps: int,
    get_next_logits,
) -> dict[str, Any]:
    bsz = batch["input_ids"].shape[0]
    generated_ids = [[] for _ in range(bsz)]
    per_step_top5 = []

    for _step in range(num_steps):
        with torch.inference_mode():
            logits = get_next_logits(batch).detach()
            top_vals, top_ids = torch.topk(logits, k=5, dim=-1)
            next_token = top_ids[:, 0]

            step_rows = []
            for row in range(bsz):
                generated_ids[row].append(int(next_token[row].item()))
                step_rows.append(
                    format_top5_for_sample(
                        top_ids[row],
                        top_vals[row],
                        tokenizer,
                    )
                )
            per_step_top5.append(step_rows)
            append_next_token(batch, next_token)

    generated_text = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in generated_ids
    ]
    return {
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "top5_per_step": per_step_top5,
    }


def unwrap_hf_model(model):
    return model.model.module if isinstance(model.model, torch.nn.DataParallel) else model.model


def run_generate_with_model(
    model,
    prompts: list[str],
    num_steps: int,
) -> dict[str, Any]:
    device = torch.device(
        f"cuda:{model.device_list[0]}" if model.device_list else "cpu"
    )
    tokenizer = model.tokenizer
    hf_model = unwrap_hf_model(model)
    batch = encode_prompts(tokenizer, prompts, device)

    with torch.inference_mode():
        out = hf_model.generate(
            **batch,
            max_new_tokens=num_steps,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )

    sequences = out.sequences
    step_raw_logits = list(getattr(out, "logits", ()) or ())
    if not step_raw_logits:
        step_raw_logits = list(out.scores)
    bsz = sequences.shape[0]

    generated_ids = []
    generated_text = []
    prompt_width = batch["input_ids"].shape[1]
    for row in range(bsz):
        start = int(prompt_width)
        row_ids = sequences[row, start:start + len(step_raw_logits)].tolist()
        generated_ids.append([int(token_id) for token_id in row_ids])
        generated_text.append(tokenizer.decode(row_ids, skip_special_tokens=True))

    per_step_top5 = []
    for logits in step_raw_logits:
        top_vals, top_ids = torch.topk(logits, k=5, dim=-1)
        step_rows = []
        for row in range(bsz):
            step_rows.append(format_top5_for_sample(top_ids[row], top_vals[row], tokenizer))
        per_step_top5.append(step_rows)

    return {
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "top5_per_step": per_step_top5,
    }


def run_naive_generate_with_model(
    model,
    prompts: list[str],
    num_steps: int,
) -> dict[str, Any]:
    device = torch.device(
        f"cuda:{model.device_list[0]}" if model.device_list else "cpu"
    )
    tokenizer = model.tokenizer
    hf_model = unwrap_hf_model(model)
    adapter_names = list(hf_model.active_adapters())
    state = encode_prompts(tokenizer, prompts, device)

    bsz = state["input_ids"].shape[0]
    eos_token_id = tokenizer.eos_token_id
    finished = torch.zeros(bsz, dtype=torch.bool, device=state["input_ids"].device)

    generated_ids = [[] for _ in range(bsz)]
    per_step_top5 = []

    with torch.inference_mode():
        for _ in range(num_steps):
            logits_list = []
            for adapter_name in adapter_names:
                hf_model.set_adapter(adapter_name)
                out = hf_model.generate(
                    **state,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    return_dict_in_generate=True,
                    output_logits=True,
                )
                step_logits = list(getattr(out, "logits", ()) or ())[0]
                logits_list.append(step_logits.detach())

            hf_model.set_adapter(adapter_names)
            merged = logits_list[0].clone()
            for logits in logits_list[1:]:
                torch.maximum(merged, logits, out=merged)

            top_vals, top_ids = torch.topk(merged, k=5, dim=-1)
            next_token = top_ids[:, 0]
            if eos_token_id is not None:
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished, eos_fill, next_token)

            step_rows = []
            for row in range(bsz):
                generated_ids[row].append(int(next_token[row].item()))
                step_rows.append(format_top5_for_sample(top_ids[row], top_vals[row], tokenizer))
            per_step_top5.append(step_rows)

            append_next_token(state, next_token)
            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)

    generated_text = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in generated_ids
    ]
    return {
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "top5_per_step": per_step_top5,
    }


def make_scenario_naive_separate_runner(
    adapter_paths: list[Path],
    device_list: list[int] | None,
    memory_limit: int,
    precision: torch.dtype,
    num_steps: int,
):
    # Shared model across all batches in this scenario.
    model = build_llama3_8b(device_list, memory_limit, precision)
    model.load_adapters(
        [str(adapter_path) for adapter_path in adapter_paths],
        composition=Modeling.SumOfDeltas(),
    )
    hf_model = unwrap_hf_model(model)

    def run_once(prompts: list[str]) -> dict[str, Any]:
        return run_naive_generate_with_model(model, prompts, num_steps)

    def cleanup():
        unload_model(model)

    return run_once, cleanup


def run_scenario_naive_separate(
    prompts: list[str],
    adapter_paths: list[Path],
    device_list: list[int] | None,
    memory_limit: int,
    precision: torch.dtype,
    num_steps: int,
) -> dict[str, Any]:
    run_once, cleanup = make_scenario_naive_separate_runner(
        adapter_paths=adapter_paths,
        device_list=device_list,
        memory_limit=memory_limit,
        precision=precision,
        num_steps=num_steps,
    )
    try:
        return run_once(prompts)
    finally:
        cleanup()


def make_scenario_composed_runner(
    adapter_paths: list[Path],
    device_list: list[int] | None,
    memory_limit: int,
    precision: torch.dtype,
    num_steps: int,
):
    # Shared model across all batches in this scenario.
    model = build_llama3_8b(device_list, memory_limit, precision)
    model.load_adapters([str(p) for p in adapter_paths], composition=Modeling.LogitMax())

    def run_once(prompts: list[str]) -> dict[str, Any]:
        return run_generate_with_model(model, prompts, num_steps)

    def cleanup():
        unload_model(model)

    return run_once, cleanup


def run_scenario_composed(
    prompts: list[str],
    adapter_paths: list[Path],
    device_list: list[int] | None,
    memory_limit: int,
    precision: torch.dtype,
    num_steps: int,
) -> dict[str, Any]:
    run_once, cleanup = make_scenario_composed_runner(
        adapter_paths=adapter_paths,
        device_list=device_list,
        memory_limit=memory_limit,
        precision=precision,
        num_steps=num_steps,
    )
    try:
        return run_once(prompts)
    finally:
        cleanup()


def print_scenario(name: str, prompts: list[str], result: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    for idx, prompt in enumerate(prompts):
        print(f"\nSample {idx}")
        print(f"Prompt: {prompt}")
        print(f"Generated: {result['generated_text'][idx]}")
        for step_idx, step_rows in enumerate(result["top5_per_step"], start=1):
            top5 = step_rows[idx]
            rendered = " | ".join(
                f"{tok['id']}:{tok['token']} ({tok['logit']:.4f})"
                for tok in top5
            )
            print(f"Step {step_idx} top5: {rendered}")


def run_scenario_for_batches(
    scenario_name: str,
    prompt_batches: list[list[str]],
    run_once,
    cleanup=None,
) -> list[dict[str, Any]]:
    total_batches = len(prompt_batches)
    outputs = []
    try:
        for batch_idx, prompts in enumerate(prompt_batches, start=1):
            result = run_once(prompts)
            outputs.append(
                {
                    "batch_index": batch_idx,
                    "prompts": prompts,
                    "result": result,
                }
            )
            title = scenario_name
            if total_batches > 1:
                title = f"{scenario_name} (Batch {batch_idx}/{total_batches})"
            print_scenario(title, prompts, result)
    finally:
        if cleanup is not None:
            cleanup()
    return outputs


def serialize_scenario_outputs(scenario_outputs: list[dict[str, Any]] | None):
    if scenario_outputs is None:
        return None
    if len(scenario_outputs) == 1:
        return scenario_outputs[0]["result"]
    return scenario_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama3 LogitMax scenario comparison")
    parser.add_argument("--devices", type=str, default="", help='Comma-separated GPU ids, e.g. "0,1"')
    parser.add_argument("--memory_limit", type=int, default=18000, help="Per-GPU memory limit in MB")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of next tokens to generate")
    parser.add_argument(
        "--num_batches",
        type=int,
        default=2,
        help="Number of sequential batches per scenario; for num_batches>1 the first batch is size 1 to expose cache-size bugs",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Which scenario to run: all | 1 | 2 | 3 "
             "(or naive | compose_123 | compose_321)",
    )
    parser.add_argument(
        "--adapter_root",
        type=Path,
        default=Path("tests/fusekit_data/logitmax_llama3_dummy_adapters"),
        help="Directory where dummy adapters are created",
    )
    parser.add_argument("--force_recreate_adapters", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Model precision",
    )
    parser.add_argument(
        "--save_json",
        type=Path,
        default=None,
        help="Optional path to save all scenario outputs as JSON",
    )
    args = parser.parse_args()

    if args.num_batches < 1:
        parser.error("--num_batches must be >= 1")

    precision = torch.bfloat16 if args.precision == "bf16" else torch.float16
    device_list = parse_devices(args.devices)
    prompt_batches = build_prompt_batches(args.num_batches)
    scenario = normalize_scenario(args.scenario)

    adapter_paths = create_dummy_adapters(
        adapter_root=args.adapter_root,
        device_list=device_list,
        memory_limit=args.memory_limit,
        precision=precision,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        force_recreate=args.force_recreate_adapters,
    )

    scenario_1 = None
    scenario_2 = None
    scenario_3 = None

    if scenario in ("all", "naive"):
        run_once, cleanup = make_scenario_naive_separate_runner(
            adapter_paths=adapter_paths,
            device_list=device_list,
            memory_limit=args.memory_limit,
            precision=precision,
            num_steps=args.num_steps,
        )
        scenario_1 = run_scenario_for_batches(
            scenario_name="Scenario #1 Naive Separate",
            prompt_batches=prompt_batches,
            run_once=run_once,
            cleanup=cleanup,
        )

    if scenario in ("all", "compose_123"):
        run_once, cleanup = make_scenario_composed_runner(
            adapter_paths=adapter_paths,
            device_list=device_list,
            memory_limit=args.memory_limit,
            precision=precision,
            num_steps=args.num_steps,
        )
        scenario_2 = run_scenario_for_batches(
            scenario_name="Scenario #2 Compose 1,2,3",
            prompt_batches=prompt_batches,
            run_once=run_once,
            cleanup=cleanup,
        )

    if scenario in ("all", "compose_321"):
        run_once, cleanup = make_scenario_composed_runner(
            adapter_paths=[adapter_paths[2], adapter_paths[1], adapter_paths[0]],
            device_list=device_list,
            memory_limit=args.memory_limit,
            precision=precision,
            num_steps=args.num_steps,
        )
        scenario_3 = run_scenario_for_batches(
            scenario_name="Scenario #3 Compose 3,2,1",
            prompt_batches=prompt_batches,
            run_once=run_once,
            cleanup=cleanup,
        )

    same_12 = None
    same_13 = None
    same_23 = None
    per_batch_equalities = None
    if scenario == "all":
        per_batch_equalities = []
        for idx in range(args.num_batches):
            s1_ids = scenario_1[idx]["result"]["generated_token_ids"]
            s2_ids = scenario_2[idx]["result"]["generated_token_ids"]
            s3_ids = scenario_3[idx]["result"]["generated_token_ids"]
            per_batch_equalities.append(
                {
                    "batch_index": idx + 1,
                    "s1_eq_s2": s1_ids == s2_ids,
                    "s1_eq_s3": s1_ids == s3_ids,
                    "s2_eq_s3": s2_ids == s3_ids,
                }
            )
        same_12 = all(row["s1_eq_s2"] for row in per_batch_equalities)
        same_13 = all(row["s1_eq_s3"] for row in per_batch_equalities)
        same_23 = all(row["s2_eq_s3"] for row in per_batch_equalities)
        print("\n=== Token Equality Checks ===")
        print(f"Scenario1 == Scenario2 (all batches): {same_12}")
        print(f"Scenario1 == Scenario3 (all batches): {same_13}")
        print(f"Scenario2 == Scenario3 (all batches): {same_23}")
        if args.num_batches > 1:
            for row in per_batch_equalities:
                print(
                    f"Batch {row['batch_index']}: "
                    f"s1==s2 {row['s1_eq_s2']} | "
                    f"s1==s3 {row['s1_eq_s3']} | "
                    f"s2==s3 {row['s2_eq_s3']}"
                )

    if args.save_json is not None:
        payload = {
            "num_batches": args.num_batches,
            "prompt_batches": prompt_batches,
            "prompts": prompt_batches[0],
            "adapter_paths": [str(p) for p in adapter_paths],
            "selected_scenario": scenario,
            "scenario_1_naive": serialize_scenario_outputs(scenario_1),
            "scenario_2_compose_123": serialize_scenario_outputs(scenario_2),
            "scenario_3_compose_321": serialize_scenario_outputs(scenario_3),
            "equalities": {
                "s1_eq_s2": same_12,
                "s1_eq_s3": same_13,
                "s2_eq_s3": same_23,
                "per_batch": per_batch_equalities,
            },
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.save_json}")


if __name__ == "__main__":
    main()
