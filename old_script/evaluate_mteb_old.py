#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from old_script.openelm_model import LLMEmbeddingWrapper
import mteb
from mteb import MTEB

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on MTEB benchmark tasks.")
    
    parser.add_argument('--model_name', type=str, required=True, default=None, help='Model name for tokenizer logic (default: same as --model)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for encoding')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--device', type=str, default="auto", help='Device for model (e.g., "cuda", "cpu", "auto")')
    args = parser.parse_args()

    model_name = args.model_name if args.model_name else args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name in ["openelm", "OpenELM"]:
        from old_script.models import OpenELMEncoder as opl
        model = opl(device=args.device,cache_dir="/data/shared/")
    elif args.model_name in ["olmo", "OLMo"]:
        from old_script.models import OLMoEncoder as opl
        model = opl(device=args.device,cache_dir="/data/shared/")

    desired_task_types = [
        "Clustering",
        "PairClassification",
        "Reranking",
        "Retrieval",
        "STS",
        "Summarization"
    ]
    print(f"\nFetching MTEB tasks: {desired_task_types} (English only)")
    fetch_tasks = mteb.get_tasks(task_types=desired_task_types[:1], languages=["eng"])

    evaluation = MTEB(tasks=fetch_tasks)
    encode_args = {"batch_size": args.batch_size}
    print("\nRunning MTEB evaluation...")
    results = evaluation.run(
        model,
        verbosity=3,
        encode_kwargs=encode_args,
        output_folder=str(output_dir / model_name.replace('/', '_'))
    )

    # Print and save results
    print("\n--- MTEB Evaluation Results ---")
    for task_name, task_results in results.items():
        print(f"\nTask: {task_name}")
        for metric_type, metric_value in task_results.items():
            if isinstance(metric_value, dict):
                print(f"  {metric_type}:")
                for sub_metric, sub_value in metric_value.items():
                    print(f"    {sub_metric}: {sub_value:.4f}")
            else:
                print(f"  {metric_type}: {metric_value:.4f}")

    # Save to JSON
    result_path = output_dir / f"mteb_results_{model_name.replace('/', '_')}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

if __name__ == "__main__":
    main() 