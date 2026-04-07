"""
Population management with heap-based size limiting.
"""

import json
import heapq
import os
import logging

import numpy as np

from utils.utils import clean_code

logger = logging.getLogger(__name__)


class Pop:
    def __init__(self, name_list, size):
        self.size = size
        self.__population = {}
        for name in name_list:
            self.__population[name] = []

    def check_solution(self, task_name, solution):
        return any(entry["best_sol"] == solution for entry in self.__population[task_name])

    def save_solution(self, task_name, idea, solution, utility):
        solution = clean_code(solution)

        # Check if the solution already exists
        if any(entry["best_sol"] == solution for entry in self.__population[task_name]):
            logger.debug(f"Solution already exists in population for task {task_name}.")
            return False
        if solution is None or "def " not in solution or "return " not in solution:
            logger.debug(f"Solution is not valid code.")
            return False

        # For non-meta tasks: replace if same utility but shorter
        if task_name != "meta-optimizer":
            for entry in self.__population[task_name]:
                if entry["utility"] == utility:
                    if len(solution) < len(entry["best_sol"]):
                        entry["idea"] = idea
                        entry["best_sol"] = solution
                        return True
                    else:
                        return False

        initial_population = self.__population[task_name][:]
        self.__population[task_name].append({"idea": idea, "best_sol": solution, "utility": utility})

        # Maintain maximum size constraint via heap
        size = min(self.size, len(self.__population[task_name]))
        self.__population[task_name] = heapq.nsmallest(size, self.__population[task_name], key=lambda x: x["utility"])

        return initial_population != self.__population[task_name]

    def get_solution_by_index(self, task_name, index):
        return self.__population[task_name][index]

    def get_best_solution(self, task_name):
        return self.__population[task_name][0]

    def get_random_solution(self, task_name):
        if not self.__population[task_name]:
            raise ValueError("The population is empty.")
        ranks = np.arange(1, len(self.__population[task_name]) + 1)
        probs = 1 / (ranks + (self.size / 2))
        probabilities = probs / np.sum(probs)
        selected_index = np.random.choice(len(self.__population[task_name]), p=probabilities)
        return self.__population[task_name][selected_index]

    def get_population(self, task_name):
        return self.__population[task_name]

    def save_subtask_to_file(self, task_name, filename):
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.__population[task_name], f, indent=4)

    def save_all_data_to_file(self, filename):
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.__population, f, indent=4)

    def load_subtask_from_file(self, task_name, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        with open(filename, "r") as f:
            self.__population[task_name] = json.load(f)

    def load_all_data_from_file(self, filename):
        with open(filename, "r") as f:
            self.__population = json.load(f)

    def get_subtask_size(self, task_name):
        if not self.__population[task_name]:
            return 0
        return len(self.__population[task_name])
