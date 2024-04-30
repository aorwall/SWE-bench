#!/usr/bin/env python3

"""Run evaluation"""
import argparse
import hashlib
import json
import logging
import os

from tqdm import tqdm

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION, PatchType,
)
from swebench.harness.docker_context_manager import DockerTaskEnvContextManager
from swebench.harness.utils import get_instances, DotDict, extract_minimal_patch
from swebench.metrics.getters import get_eval_refs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")

def main(
    instance_id: str,
    testbed_name: str,
    predictions_path: str,
    swe_bench_tasks: str,
    log_dir: str,
    timeout: int,
    verbose: bool,
    log_suffix: str = None
):

    tasks = get_eval_refs(swe_bench_tasks)
    task_instance = tasks[instance_id]
    task_instance["test_cmd"] = "pytest --no-header -rA --tb=no -p no:cacheprovider lib/matplotlib/tests/test_artist.py"

    predictions = get_instances(predictions_path)

    for prediction in predictions:
        if prediction[KEY_INSTANCE_ID] == instance_id:
            task_instance[KEY_PREDICTION] = prediction[KEY_PREDICTION]
            task_instance[KEY_MODEL] = prediction[KEY_MODEL]
            break

    with DockerTaskEnvContextManager(
            task_instance,
            testbed_name,
            log_dir,
            verbose=verbose,
            timeout=timeout,
            is_eval=True,
            log_suffix=log_suffix,
    ) as tcm:
        # Attempt to set up environment with task instance

        # Attempt to apply prediction
        patch_type = PatchType.PATCH_PRED_TRY.value

        # If prediction patch doesn't apply, try to do some minor patch refactoring and try again
        if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type) \
                and task_instance[KEY_PREDICTION] is not None \
                and task_instance[KEY_PREDICTION] != "":
            task_instance[KEY_PREDICTION] = extract_minimal_patch(task_instance[KEY_PREDICTION])
            patch_type = PatchType.PATCH_PRED_MINIMAL_TRY.value
            if not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type):
                # Continue if edited patch still doesn't apply
                return
        tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type, revert=True)

        # Set prediction patch label based on whether patch was edited
        if patch_type == PatchType.PATCH_PRED_MINIMAL_TRY.value:
            patch_type = PatchType.PATCH_PRED_MINIMAL.value
        else:
            patch_type = PatchType.PATCH_PRED.value

        # Run installation + testing script
        if (
                not tcm.apply_patch(task_instance[KEY_PREDICTION], patch_type=patch_type)
                or not tcm.apply_patch(task_instance["test_patch"], patch_type=PatchType.PATCH_TEST.value)
                or not tcm.run_tests_task(task_instance)
        ):
            return



if __name__ == "__main__":
    main(
        instance_id="matplotlib__matplotlib-22835",
        testbed_name="matplotlib__matplotlib__3.5",
        predictions_path="/home/albert/repos/albert/swebench/swe-bench/SWE-bench_Lite_golden_predictions.jsonl",
        swe_bench_tasks="princeton-nlp/SWE-bench_Lite",
        log_dir="/home/albert/repos/albert/swebench/swe-bench/testlogs",
        timeout=900,
        verbose=True
    )