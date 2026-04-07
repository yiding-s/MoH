import os
import copy
import time
import logging
import subprocess
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from utils.llm_client.base import BaseClient

from utils.utils import (
    read_file_as_str, extract_code,
    extract_idea, clean_code,
    find_txt_block, match_number,
)
from utils.population import Pop
from utils.run_logger import RunLogger

logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class MoH:
    """
    Meta-optimizer of Heuristics.
    """

    def __init__(
        self,
        cfg: DictConfig,           # Hydra config 
        root_dir: str,             
        heu_llm: BaseClient,       # base LLM client 
        meta_llm: Optional[BaseClient] = None,  # separate LLM for meta-level
    ) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.heu_llm = heu_llm
        self.meta_llm = meta_llm or heu_llm

        self.iteration = cfg.n_iterations
        self.best_improver = None
        self.meta_utility_val = None

        self.problem_name = cfg.problem.problem_name
        self.subtask_list = [f"{self.problem_name}-{s}" for s in cfg.problem.problem_size]
        self.size_weights = np.array(cfg.problem.problem_size, dtype=float) # size list

        self.subtask_pop = Pop(self.subtask_list, cfg.pop_size)
        self.improver_pop = Pop(["meta-optimizer"], cfg.pop_size)

        # Eval call budget: per-subtask × N, null/-1 = unlimited
        max_calls = getattr(cfg, 'max_eval_calls', None)
        if max_calls and max_calls > 0:
            self.max_eval_calls = max_calls * len(self.subtask_list)
        else:
            self.max_eval_calls = None
        self._total_eval_calls = 0

        # Centralized I/O manager
        self.run_logger = RunLogger()

        self.init_prompts()
        self.init_meta_improver()

    # =========================================================================
    # Initialization 
    # =========================================================================

    def init_prompts(self):
        # meta-level prompts
        self.meta_prompts = read_file_as_str(
            os.path.join(self.root_dir, "prompts", "meta", "desc.txt")
        )

        # subtask prompts
        prompt_dir = os.path.join(self.root_dir, "prompts", self.problem_name)
        self.subtask_form = read_file_as_str(os.path.join(prompt_dir, "desc.txt"))
        self.task_description = read_file_as_str(os.path.join(prompt_dir, "task.txt"))
        self.subtask_direction = read_file_as_str(os.path.join(prompt_dir, "plan.txt"))

        # else
        self.system_check_prompt = read_file_as_str(
            os.path.join(self.root_dir, "prompts", "helper", "check_iter.txt")
        )

    def init_meta_improver(self):
        self.improver_str = read_file_as_str(
            os.path.join(self.root_dir, "problems", "meta", "seed_algorithm.py")
        )
        self.algorithm_to_improve = self.improver_str

    # =========================================================================
    # Task-level evaluation via subprocess
    # exec(algorithm_str, globals()) in utility()
    # =========================================================================

    def _run_code(self, algorithm_str: str, problem_type: str, mode: str, stdout_path: str) -> subprocess.Popen:
        # Step 1: write generated code to gpt.py
        gpt_path = os.path.join(self.root_dir, "problems", self.problem_name, "gpt.py")
        with open(gpt_path, 'w', encoding="utf-8") as f:
            f.write(algorithm_str + '\n')

        # Step 2: launch eval.py as subprocess
        eval_path = os.path.join(self.root_dir, "problems", self.problem_name, "eval.py")
        problem_size = int(problem_type.split("-")[-1])
        with open(stdout_path, 'w', encoding="utf-8") as f:
            f.write("# ===== Heuristic Code =====\n")
            f.write(algorithm_str)
            f.write("\n# ===== Eval Output =====\n")
            f.flush()
            process = subprocess.Popen(
                ['python', '-u', eval_path, str(problem_size), self.root_dir, mode, str(self.cfg.timeout)],
                stdout=f, stderr=f
            )
        return process


    def evaluate_heuristic(self, algorithm_str: str, problem_type: str, mode: str = "val") -> float:
        if not algorithm_str:
            logger.info(f"algorithm_str is {repr(algorithm_str)}, returning")
            return 1e6
        if "random" in algorithm_str:
            logger.info("random is used in algorithm")
            return 1e6
        if len(algorithm_str) < 30:
            logger.info("algorithm_str is too short")
            return 1e6

        self._total_eval_calls += 1
        if self.max_eval_calls:
            logger.info(f"Eval call: {self._total_eval_calls}/{self.max_eval_calls}")
        # two-level counter: per-subtask eval index within current iteration
        eval_key = (self._cur_iter, problem_type)
        if not hasattr(self, '_eval_counters'):
            self._eval_counters = {}
        self._eval_counters[eval_key] = self._eval_counters.get(eval_key, 0) + 1
        subtask_idx = self.subtask_list.index(problem_type) if problem_type in self.subtask_list else 0
        stdout_path = os.path.join(
            self.run_logger.dirs["logs"],
            f"eval_iter{self._cur_iter}_s{subtask_idx}_e{self._eval_counters[eval_key]}_{mode}.txt"
        )

        try:
            process = self._run_code(algorithm_str, problem_type, mode, stdout_path)
        except Exception as e:
            logger.warning(f"_run_code failed: {e}")
            return 1e6

        try:
            process.communicate(timeout=self.cfg.timeout * 10)  
        except subprocess.TimeoutExpired:
            logger.warning(f"Evaluation timed out for {problem_type} mode={mode}")
            process.kill()
            return 1e6

        try:
            with open(stdout_path, 'r', encoding="utf-8") as f:
                stdout_str = f.read()
            # Check for errors in output
            if "Traceback" in stdout_str or "Error" in stdout_str:
                logger.warning(f"Eval error: {stdout_str[-500:]}")
                return 1e6
            ave_gap = float(stdout_str.strip().split('\n')[-1])
            logger.info(f"Eval {problem_type} mode={mode}: ave_gap={ave_gap}")
            return ave_gap
        except Exception as e:
            logger.warning(f"Failed to parse eval result: {e}")
            return 1e6

    # =========================================================================
    # Seed generation  
    # =========================================================================

    def generate_directions(self, problem, size, cur_direction=None, seed=False, solution=None):
        """← from version_iclr/meta/secret_utility.py: generate_directions()"""
        import json
        directions = []
        task_description = self.task_description

        if seed:
            extra_info = "" if cur_direction is None else \
                f"Do not make it the same with previous generated directions: {cur_direction}."
            message = f'''The problem is {problem} with corresponding size {size}. \
                According to the task description:{task_description} Provide several high-level directions for generating the seed prompt, each aimed at minimizing the utility as a result. \
                {extra_info} Format your response as a JSON codeblock below:
                    {{
                    "direction": [
                        {{"content": "Your first piece of direction suggestion here."}},
                        {{"content": "Your second piece of direction suggestion here."}},
                        {{"content": "Your third piece of direction suggestion here."}},
                        ...
                        {{"content": "Your last direction suggestion here."}}
                    ]
                    }}'''
        else:
            solution_str = solution["best_sol"]
            solution_util = solution["utility"]
            subtask_direction = self.subtask_direction
            message = f"The task is: {task_description}. Given current solution {solution_str} with utility {solution_util}. {subtask_direction}"

        expertise = "You are an expert in the domain of optimization heuristics and combinatorial optimization problems. Your task is to design heuristics that can effectively solve optimization problems."
        new_directions = self.heu_llm.prompt(expertise, message, 0.7)
        json_file = extract_code(new_directions)
        try:
            data = json.loads(json_file)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}, retrying...")
            new_message = f"There is an exception {e} while parsing the json file {json_file}, please check the json file and return a correct one."
            expert = "You are an expert in finding coding errors and debugging."
            new_directions = self.heu_llm.prompt(expert, new_message, 0.7)
            json_file = extract_code(new_directions)
            data = json.loads(json_file)

        for i in range(len(data['direction'])):
            directions.append(data['direction'][i]["content"])
        return directions

    def generate_seed_algorithm(self, formula_str, prob_type, threshold, t=0.7):
        problem = prob_type.split("-")[0]
        size = prob_type.split("-")[1]
        cur_direction = []
        cur_pop_size = 0

        while cur_pop_size < 5:
            directions = self.generate_directions(problem, size, cur_direction, seed=True)
            for idx in range(len(directions)):
                role = "You are an expert in the domain of optimization heuristics and combinatorial optimization problems. Your task is to design heuristics that can effectively solve optimization problems."
                message = f"""Write a function that will implement a Python algorithm to solve a problem as well as possible.
            The optimization problem is {problem} and the size you should focus on is {size}.
            The output function is formatted as follows:
            ```python
            {formula_str}
            ```
            You should develop the algorithm that follows the direction: {directions[idx]},  \
            First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. \
            Then, provide a Python function that implements the algorithm. \
            """
                utility_score = 1e6
                response_str = self.heu_llm.prompt(role, message, temperature=t)
                try:
                    algorithm_str = extract_code(response_str)
                    idea_str = extract_idea(response_str)
                    algorithm_str = clean_code(algorithm_str)
                    utility_score = self.evaluate_heuristic(algorithm_str, prob_type, mode="val")
                    if utility_score < threshold:
                        self.subtask_pop.save_solution(prob_type, idea_str, algorithm_str, utility_score)
                        cur_pop_size += 1
                except Exception as e:
                    logger.warning(f"Utility failed: {e}")
                    logger.debug(traceback.format_exc())
            cur_direction += directions
            cur_pop_size = self.subtask_pop.get_subtask_size(prob_type)

        return self.subtask_pop.get_random_solution(prob_type)

    def _get_threshold(self, subtask):
        """Get threshold for a subtask from cfg."""
        size_key = int(subtask.split("-")[-1])
        return self.cfg.problem.threshold.get(
            size_key, self.cfg.problem.threshold.get(str(size_key), 6)
        )

    def read_base_algorithm(self, subtask):
        solution_path = os.path.join(self.root_dir, "problems", self.problem_name, f"solution_{subtask}.json")

        if os.path.exists(solution_path):
            self.subtask_pop.load_subtask_from_file(subtask, solution_path)
            base_algorithm = self.subtask_pop.get_random_solution(subtask)
        else:
            threshold = self._get_threshold(subtask)
            base_algorithm = self.generate_seed_algorithm(
                self.subtask_form, subtask, threshold
            )
            self.subtask_pop.save_subtask_to_file(subtask, solution_path)

        if self.subtask_pop.get_subtask_size(subtask) < 3:
            threshold = self._get_threshold(subtask)
            self.generate_seed_algorithm(self.subtask_form, subtask, threshold)
            self.subtask_pop.save_subtask_to_file(subtask, solution_path)

        return base_algorithm

    # =========================================================================
    # Meta-level evaluation 
    # =========================================================================

    def _make_subtask_utility(self, subtask):
        def utility(algorithm_str, idea=None, problem_type=None):
            problem_type = problem_type or subtask
            return self.evaluate_heuristic(algorithm_str, problem_type, mode="val")
        return utility

    def get_improver(self, improve_str, subtask_str, subtask):
        try:
            exec(improve_str, globals())
            subtask_pop_copy = copy.deepcopy(self.subtask_pop)
            utility = self._make_subtask_utility(subtask)
            # exec'd code defines improve_algorithm in globals
            new_idea, improved_algorithm_str, utility_val = improve_algorithm(
                subtask_pop_copy, utility, self.heu_llm, subtask_str, subtask
            )
            return new_idea, improved_algorithm_str, utility_val
        except Exception as e:
            logger.warning(f"get_improver failed: {e}")
            logger.debug(traceback.format_exc())
            return e

    def meta_utility(self, improve_str: str, idea: str, task: str = None):
        if not improve_str:
            logger.info(f"improve_str is {repr(improve_str)}, returning")
            return 1e6

        # Save candidate improver code
        self.run_logger.save_candidate_improver(improve_str, idea)

        if "language_model.prompt" not in improve_str and "language_model.prompt_batch" not in improve_str:
            logger.info("No language model prompt in improve_str, returning")
            self.run_logger.save_candidate_improver(improve_str, idea + " [rejected: no prompting step]")
            return 1e6

        result = self.meta_llm.prompt(
            "You will act as a professional coder and analyze a given function provided as a string.",
            self.system_check_prompt + improve_str
        )
        iterations = match_number(find_txt_block(result))
        if isinstance(iterations, int):
            if int(iterations) > 10:
                logger.info("Too many iterations, returning")
                self.run_logger.save_candidate_improver(improve_str, idea + f" [rejected: {iterations} iterations]")
                return 1e6

        n_tests = len(self.subtask_list)
        utility_vals = []
        improved_algorithm_strs = []
        idea_list = []

        for test_idx in tqdm(range(n_tests)):
            self.read_base_algorithm(self.subtask_list[test_idx])

            result = self.get_improver(improve_str, self.subtask_form, self.subtask_list[test_idx])
            if isinstance(result, Exception):
                return 1e6
            else:
                idea_r, improved_algorithm_str, new_utility_val = result
                if improved_algorithm_str == "":
                    return 1e6

            self.subtask_pop.save_solution(self.subtask_list[test_idx], idea_r, improved_algorithm_str, new_utility_val)
            improved_algorithm_strs.append(improved_algorithm_str)
            utility_vals.append(new_utility_val)
            idea_list.append(idea_r)

        # Save subtask populations to problem dir
        for i in range(n_tests):
            solution_path = os.path.join(
                self.root_dir, "problems", self.problem_name,
                f"solution_{self.subtask_list[i]}.json"
            )
            self.subtask_pop.save_subtask_to_file(self.subtask_list[i], solution_path)

        if not (len(improved_algorithm_strs) == len(utility_vals) == len(self.size_weights) == len(idea_list) == n_tests):
            logger.warning(f"Length mismatch: strs={len(improved_algorithm_strs)}, vals={len(utility_vals)}, weights={len(self.size_weights)}")
            return 1e6

        # Log validation results
        for test_idx, improved_algorithm_str in enumerate(improved_algorithm_strs):
            if not improved_algorithm_str:
                return 1e6
            logger.info(f"val utility on size {self.subtask_list[test_idx]}: {utility_vals[test_idx]}")
            self.run_logger.log_utility(
                self._cur_iter, self.subtask_list[test_idx], utility_vals[test_idx]
            )

        # Weighted utility computation
        if sum(self.size_weights) > 0:
            expected_utility_val = sum(w * v for w, v in zip(self.size_weights, utility_vals)) / sum(self.size_weights)
        else:
            expected_utility_val = 1e6

        logger.info(f"meta_utility: val={expected_utility_val}")

        # Save candidate with final utility annotation
        self.run_logger.save_candidate_improver(improve_str, idea, utility=expected_utility_val)

        # Update improver population
        self.improver_pop.save_solution("meta-optimizer", idea, improve_str, expected_utility_val)

        return expected_utility_val

    # =========================================================================
    # Core optimization loop 
    # =========================================================================

    def get_seed(self):
        self._cur_iter = -1  # used by meta_utility for logging
        idea = "seed optimizer"
        self.meta_utility_val = self.meta_utility(self.improver_str, idea)
        self.improver_pop.save_solution("meta-optimizer", idea, self.improver_str, self.meta_utility_val)
        self.run_logger.save_improver_pop(self.improver_pop, "pre")
        self.run_logger.save_subtask_pop(self.subtask_pop, "pre")

    def try_improvement(self, improve_algorithm_func, previous_algo):
        """← from version_iclr/meta_optimizer.py: try_improvement()"""
        improvement_successful = False
        new_algorithm_str = None
        try:
            improver_pop_copy = copy.deepcopy(self.improver_pop)
            new_idea, new_algorithm_str, new_utility = improve_algorithm_func(
                improver_pop_copy,
                self.meta_utility,
                self.meta_llm,     
                self.meta_prompts,
                "meta-optimizer",
            )
            if not new_utility:
                raise ValueError("Utility is invalid or zero")

            improvement_successful = True
            self.improver_pop.save_solution("meta-optimizer", new_idea, self.improver_str, self.meta_utility_val)

            if self.meta_utility_val > new_utility:
                self.meta_utility_val = new_utility
                self.improver_str = new_algorithm_str

        except Exception as e:
            logger.warning(f"Improvement failed: {e}")
            improve_algorithm_func = previous_algo

        return improvement_successful, new_algorithm_str, improve_algorithm_func

    def run_meta_optimizer(self):
        self.get_seed()

        from problems.meta.seed_algorithm_improved import improve_algorithm as first_improver
        improver = first_improver
        previous_algorithm = improver

        for cur_iter in range(self.iteration):
            self._cur_iter = cur_iter
            logger.info(f"Start iteration {cur_iter}")

            improvement_successful, new_algorithm_str, improver = self.try_improvement(improver, previous_algorithm)

            accepted = False
            if improvement_successful and new_algorithm_str:
                logger.info("Improvement successful")
                accepted = True
                self.improver_str = new_algorithm_str
                self.algorithm_to_improve = new_algorithm_str
                previous_algorithm = improver
                exec(self.improver_str, globals())
            else:
                logger.info("Failed to improve algorithm, reverting to previous version")
                best_solution = self.improver_pop.get_best_solution("meta-optimizer")["best_sol"]
                exec(best_solution, globals())
                improver = improve_algorithm

            self.run_logger.log_meta_utility(cur_iter, self.meta_utility_val, accepted=accepted)
            self.run_logger.save_iteration(
                cur_iter, self.improver_pop, self.subtask_pop,
                improver_code=self.improver_str if accepted else None,
                improver_idea="" if not accepted else "accepted",
            )
            logger.info(f"Finish iteration {cur_iter}")

            if self.max_eval_calls and self._total_eval_calls >= self.max_eval_calls:
                logger.info(f"Reached max utility calls ({self.max_eval_calls}), stopping.")
                break

        logger.info("Finish all iterations")
