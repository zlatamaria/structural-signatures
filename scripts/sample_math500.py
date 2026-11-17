"""Sample and grade MATH500 completions for a fresh structural-selector run."""

import argparse
import csv
import gzip
import json
from importlib.metadata import version
from itertools import islice
from pathlib import Path

from structural_signatures.cli import fraction, positive_int
from structural_signatures.io import run_directory, write_metadata

GENERATOR = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM = "You are a helpful and expert mathematical assistant. Solve the given math problem step-by-step."
INSTRUCTIONS = (
    "Reason step-by-step to solve the math problem.\n"
    "At the end of your solution, strictly follow this format to present your final answer:\n"
    "Final Answer: \\boxed{answer}\n"
    "Ensure the answer within \\boxed{} is concise and contains only the final value or expression."
)


def boxed_answer(text):
    """Read the last boxed expression using the historical brace-matching rule."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""
    start += len(marker)
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                value = text[start:pos].strip()
                # The expanded test CSV represents a literal boxed None as missing.
                return "" if value == "None" else value
    return ""


def sample(args, directory):
    import torch
    from math_verify import parse, verify
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.generator_model, revision=args.generator_revision
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.generator_model, revision=args.generator_revision, torch_dtype="auto"
        )
        .to(args.device)
        .eval()
    )
    n_questions = 0
    with (
        args.questions.open(encoding="utf-8") as questions,
        gzip.open(
            directory / "traces.csv.gz", "wt", encoding="utf-8", newline=""
        ) as stream,
        torch.inference_mode(),
    ):
        writer = csv.writer(stream)
        writer.writerow(["id", "completion", "final_answer_new", "row_hit"])
        for qid, line in enumerate(islice(questions, args.limit)):
            item = json.loads(line)
            gold = parse("$" + item["answer"] + "$")
            chat = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": INSTRUCTIONS + "\n\nProblem: " + item["problem"],
                },
            ]
            inputs = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_length = inputs["input_ids"].shape[1]
            for start in range(0, args.samples, args.batch_size):
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=0,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=min(args.batch_size, args.samples - start),
                    pad_token_id=tokenizer.eos_token_id,
                )
                for text in tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                ):
                    answer = boxed_answer(text)
                    hit = bool(answer) and bool(verify(gold, parse("$" + answer + "$")))
                    writer.writerow([qid, text, answer, hit])
            n_questions += 1
    if n_questions == 0:
        raise ValueError("The question JSONL is empty")
    write_metadata(
        directory,
        {"questions": args.questions},
        command="sample",
        n_ids=n_questions,
        generated_rows=n_questions * args.samples,
        generator_model=args.generator_model,
        generator_revision=args.generator_revision,
        resolved_revision=getattr(model.config, "_commit_hash", None),
        samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=str(model.device),
        dtype=str(model.dtype),
        generation_packages={
            name: version(name) for name in ("torch", "transformers", "math-verify")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="MATH500 JSONL with problem and gold answer, in question order",
    )
    parser.add_argument("--out", type=Path, required=True, help="new run directory")
    parser.add_argument(
        "--generator-model",
        default=GENERATOR,
        help=f"causal chat model ID or local snapshot (default {GENERATOR})",
    )
    parser.add_argument(
        "--generator-revision", help="commit, tag or branch of the generator"
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=99,
        help="draws per problem (default 99)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="sampling temperature (default 0.9)",
    )
    parser.add_argument(
        "--top-p",
        type=fraction,
        default=1.0,
        help="nucleus sampling mass (default 1.0)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=2048,
        help="completion length cap (default 2048 tokens)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="sampling RNG seed (default 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=4,
        help="draws generated together per question (default 4)",
    )
    parser.add_argument(
        "--device", default="cuda", help="generator device (default cuda)"
    )
    parser.add_argument(
        "--limit", type=positive_int, help="process the first N questions"
    )
    args = parser.parse_args()
    if not 0 < args.temperature < float("inf") or args.top_p == 0:
        parser.error(
            "--temperature must be positive and finite and --top-p must be in (0, 1]"
        )
    with run_directory(args.out) as directory:
        sample(args, directory)
    print(args.out)


if __name__ == "__main__":
    main()
