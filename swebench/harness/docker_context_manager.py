import logging, os, platform, subprocess, json

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DEFAULT_CONDA_LINK,
    INSTALL_FAIL,
    INSTALL_PASS,
    INSTALL_TIMEOUT,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    MAP_REPO_TO_INSTALL,
    MAP_REPO_TO_TEST_FRAMEWORK,
    MAP_REPO_VERSION_TO_CONDA_LINK,
    MAP_VERSION_TO_INSTALL,
    RESET_FAILED,
    TESTS_FAILED,
    TESTS_PASSED,
    TESTS_TIMEOUT,
    TESTS_ERROR,
    PatchType,
)
from swebench.harness.context_manager import ExecWrapper
from swebench.harness.utils import (
    clone_repo,
    get_conda_env_names,
    get_environment_yml,
    get_requirements,
    get_test_directives,
)
from tempfile import TemporaryDirectory
from traceback import format_exc

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger_testbed = logging.getLogger("testbed")


class LogWrapper:
    def __init__(
        self,
        log_file: str,
        logger: logging.Logger = None,
        prefix: str = None,
    ):
        self.log_file = log_file
        self.logger = logger
        self.prefix = prefix

    def write(
            self,
            message: str,
            mode: str = "a",
            level: int = INFO):
        with open(self.log_file, mode) as f:
            log = f"{self.prefix} {message} \n" if self.prefix \
                is not None else f"{message} \n"
            f.write(log)
        if self.logger is not None:
            self.logger.log(level, message)



class DockerExecWrapper(ExecWrapper):

    def __init__(self, container_name, subprocess_args=None, logger=None):
        super().__init__(subprocess_args, logger)
        self.container_name = container_name

    def __call__(self, cmd, raise_error=True, **kwargs):
        self.logger.write(f"Command: {cmd}", level=DEBUG)
        if isinstance(cmd, list):
            cmd = ["docker", "exec", self.container_name] + cmd
        else:
            cmd = f"docker exec {self.container_name} {cmd}"

        return super().__call__(cmd, raise_error, **kwargs)


logger_taskenv = logging.getLogger("taskenv")


class DockerTaskEnvContextManager:
    def __init__(
        self,
        instance: dict,
        testbed_name: str,
        log_dir: str,
        verbose: bool = False,
        timeout: int = None,
        is_eval: bool = False,
        log_suffix: str = None,
    ):
        """
        Sets up execution context for a single task instance

        Args:
            instance (dict): Task instance
            testbed_name (str): Name of testbed environment
            log_dir (str): Path to log directory
            verbose (bool): Whether to show logs
            timeout (int): Timeout for actions
            is_eval (bool): Whether this is for evaluating a model on SWE Bench
                (Mainly for logging purposes)
        """
        if verbose:
            logger_taskenv.setLevel(logging.INFO)
        self.instance = instance
        self.is_eval = is_eval
        self.testbed_name = testbed_name

        logger_taskenv.info(f"Setting up task environment for {instance[KEY_INSTANCE_ID]}")

        # Log file naming
        log_file_name = (
            f"{instance[KEY_INSTANCE_ID]}.{instance[KEY_MODEL]}.eval.log"
            if self.is_eval
            else f"{instance[KEY_INSTANCE_ID]}.log"
        )
        if log_suffix is not None:
            log_file_name = (
                f"{instance[KEY_INSTANCE_ID]}.{instance[KEY_MODEL]}.{log_suffix}.eval.log"
                if self.is_eval
                else f"{instance[KEY_INSTANCE_ID]}.{log_suffix}.log"
            )
        self.log_file = os.path.join(log_dir, log_file_name)
        self.log = LogWrapper(
            self.log_file, logger=logger_taskenv,
            prefix=f"[{self.testbed_name}] [{self.instance[KEY_INSTANCE_ID]}]")

        self.timeout = timeout

        self.exec = ExecWrapper(
            subprocess_args={
                "check": True,
                "shell": False,
                "capture_output": False,
                "text": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            },
            logger=self.log,
        )

    def __enter__(self):
        """
        Enter task environment, set up log file
        """

        enter_msg = (
            f"Task Metadata:\n\t- "
            f"Instance ID: {self.instance[KEY_INSTANCE_ID]}\n\t- "
            f"Testbed: {self.testbed_name}\n\t- "
        )
        if self.is_eval:
            enter_msg += f"\n\t- Evaluation Model: {self.instance[KEY_MODEL]}"
        self.log.write(enter_msg, mode="w")

        self._reset_docker()

        run_cmd = f"docker run -d -t --name {self.testbed_name} {self.testbed_name}".split()
        output = self.exec(run_cmd)
        self.log.write(f"Container started: {output.stdout}")

        container_id = output.stdout.strip()

        self.docker_exec = DockerExecWrapper(
            container_name=container_id,
            subprocess_args={
                "check": False,
                "shell": False,
                "capture_output": False,
                "text": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            },
            logger=self.log,
        )

        return self

    def apply_patch(
        self, patch: str, patch_type: PatchType = "", revert: bool = False
    ) -> bool:
        """
        Apply patch to task environment

        Args:
            patch (str): Plaintext of patch to apply
            patch_type (str): Type of patch (e.g. "eval", "test")
        Returns:
            bool: True if patch applied successfully, False otherwise
        """
        # If patch is `None`, indicate in log and skip
        if patch is None:
            self.log.write(f"Patch is `None` ({patch_type})")
            with open(self.log_file, "a") as f:
                f.write(f"{APPLY_PATCH_FAIL}; Prediction patch is `None`")
            return False

        # Write patch to temporary patch file
        patch_path = f"temp_{self.instance[KEY_INSTANCE_ID]}_{patch_type}.patch"

        command = f"sh -c 'echo {json.dumps(patch)} > {patch_path}'"
        try:
            self.docker_exec(command, shell=True)
        except Exception as e:
            self.log.write(f"Failed to write patch: {e}")
            return False

        # Restore test files before applying if patch_type is 'test'
        #if patch_type == PatchType.PATCH_TEST.value:
        #    for test in self.instance["test_directives"]:
        #        restore_command = ["git", "restore", test]
        #        try:
        #            self.exec(restore_command)
        #        except Exception as e:
        #            self.log.write(f"Failed to restore file {test}: {e}")

        # Apply patch to testbed directory
        apply_cmd = (
            f"git apply -v -R {patch_path}" if revert else f"git apply -v {patch_path}"
        )
        out_patch = self.docker_exec(apply_cmd.split(" "), raise_error=False, check=False)

        remove_cmd = f"rm {patch_path}"
        self.docker_exec(remove_cmd.split(" "), raise_error=False, check=False)

        log_cmd = "Revert" if revert else "Apply"
        if out_patch.returncode != 0:
            # Patch apply failed
            self.log.write(f"{log_cmd} patch failed ({patch_type})", level=ERROR)
            with open(self.log_file, "a") as f:
                f.write(f"{APPLY_PATCH_FAIL}; ({patch_type})\nOutput:\n")
                f.write(out_patch.stdout)
                if out_patch.stderr:
                    f.write(out_patch.stderr)
            return False

        # Patch apply succeeded
        self.log.write(f"{log_cmd} patch successful ({patch_type})")
        with open(self.log_file, "a") as f:
            f.write(f"{APPLY_PATCH_PASS} ({patch_type})\n")
        return True

    def run_tests_task(self, instance: dict):
        """
        Run tests for task instance

        Args:
            instance (dict): Task instance
        Returns:
            bool: True if test script ran successfully, False otherwise
        """
        try:
            # Run test command for task instance
            test_cmd = instance['test_cmd']
            with open(self.log_file, "a") as f:
                f.write(f"Test Script: {test_cmd};\n")

            # Set environment variables if provided
            specifications = MAP_VERSION_TO_INSTALL[instance["repo"]][instance["version"]]
            if "env_vars_test" in specifications:
                self.docker_exec.subprocess_args["env"].update(specifications["env_vars_test"])

            out_test = self.docker_exec(
                test_cmd, shell=True, timeout=self.timeout, check=False
            )

            # Unset environment variables if provided
            if "env_vars_test" in specifications:
                for key in specifications["env_vars_test"]:
                    del self.docker_exec.subprocess_args["env"][key]

            # Write pass/fail status to log file
            with open(self.log_file, "a") as f:
                if out_test.returncode != 0:
                    f.write(f"\n{TESTS_FAILED}\n")
                else:
                    f.write(f"\n{TESTS_PASSED}\n")

            self.log.write(f"Test script run successful")
            return True
        except subprocess.TimeoutExpired:
            # Test command run timed out
            self.log.write("Test script run timed out", level=ERROR)
            with open(self.log_file, "a") as f:
                f.write(f"{TESTS_TIMEOUT} after {self.timeout} seconds\n")
            return False
        except Exception as e:
            # Test command run failed
            self.log.write(f"Test script run failed", level=ERROR)
            with open(self.log_file, "a") as f:
                f.write(f"{TESTS_ERROR}: {e}")
            return False

    def _reset_docker(self):
        kill_cmd = f"docker kill {self.testbed_name}".split()
        self.exec(kill_cmd, raise_error=False)
        rm_cmd = f"docker rm {self.testbed_name}".split()
        self.exec(rm_cmd, raise_error=False)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._reset_docker()

