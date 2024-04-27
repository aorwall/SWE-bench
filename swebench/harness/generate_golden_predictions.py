import argparse
import json
import os

from swebench import get_eval_refs


def main(predictions_path: str, swe_bench_tasks: str):
    tasks = list(get_eval_refs(swe_bench_tasks).values())

    bench_name = swe_bench_tasks.split("/")[-1]
    model_name = f"{bench_name}_golden"

    prediction_file = os.path.join(predictions_path, f"{model_name}_predictions.jsonl")
    if os.path.exists(prediction_file):
        os.remove(prediction_file)

    for task in tasks:
        prediction = {
            "model_name_or_path": model_name,
            "instance_id": task["instance_id"],
            "model_patch": task["patch"],
        }

        with open(prediction_file, "a") as file:
            json_string = json.dumps(prediction)
            file.write(json_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file (must be .json)", required=True)
    parser.add_argument("--swe_bench_tasks", type=str, help="Path to dataset file or HF datasets name", required=True)

    args = parser.parse_args()
    main(**vars(args))
