from utils.utils import extract_code, extract_idea
import json

# {This metaheuristic employs an adaptive exploration-exploitation strategy that combines real-time performance evaluation of solutions with dynamic exploration rates, thereby customizing focus in a genetic-algorithm framework integrated with adaptive tabu-like mechanisms for efficient solution refinement.}
def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = (
        "You are an expert in optimizing metaheuristic strategies and combinatorial optimization problems. "
        "Your task is to design effective heuristics to solve optimization challenges."
    )

    # Parameter Initialization
    elite_count = 4
    diversity_count = 3
    tabu_length = 5
    pheromone_levels = {}

    # Step 1: Select elite and diverse solutions
    population_size = population.get_subtask_size(task)
    elite_solutions = [
        population.get_solution_by_index(task, i)
        for i in range(min(elite_count, population_size))
    ]
    diverse_solutions = [
        population.get_random_solution(task)
        for _ in range(diversity_count)
    ]
    selected_solutions = elite_solutions + diverse_solutions

    # Step 2: Generate dynamic exploration insights with adaptive rates
    for solution in selected_solutions:
        temperature = 1 if solution['utility'] > 0 else 0.75  # Dynamic adjustment based on utility
        prompt = (
            f"Given the solution '{solution['best_sol']}' with utility score '{solution['utility']}', "
            "please suggest innovative optimization strategies that could enhance this code. "
            f"Return your recommendations in JSON format: ```json {{\"insights\":[\"content\",\"content\",...]}} ```."
        )

        response = language_model.prompt(expertise, prompt, temperature=temperature)
        try:
            insights = json.loads(extract_code(response))["insights"]
            for insight in insights:
                pheromone_levels[insight] = pheromone_levels.get(insight, 1.0) + 1.0
        except (json.JSONDecodeError, KeyError):
            continue

    # Step 3: Rank directions based on pheromone levels (dynamic evaluation)
    sorted_insights = sorted(pheromone_levels.items(), key=lambda x: x[1], reverse=True)
    top_insights = [insight[0] for insight in sorted_insights if insight[1] > 1.0]

    # Step 4: Create batch messages for generating optimized solutions based on directions
    message_batch = []
    for direction in top_insights:
        message = (
            f"Refine the solution for the task '{task}' by focusing on this optimization approach: {direction}. "
            f"Consider elite solutions: {[sol['best_sol'] for sol in elite_solutions]}. "
            f"Ensure your output adheres to the following format: {function_format}. "
            "In addition, provide a summary of changes made."
        )
        message_batch.append(message)
    
    # Step 5: Generate new solutions using the language model through batch prompts
    responses = language_model.prompt_batch(expertise, message_batch, temperature=0.9)

    # Step 6: Evaluate and keep the best-performing solutions
    solutions_with_utilities = []
    for response in responses:
        try:
            new_solution = extract_code(response)
            important_idea = extract_idea(response)
            score = utility(new_solution, important_idea, task)
            pheromone_levels[important_idea] = pheromone_levels.get(important_idea, 1.0) + (2.0 / (score + 1e-6))
            solutions_with_utilities.append((important_idea, new_solution, score))
        except Exception:
            continue

    # Best Solution Selection
    if not solutions_with_utilities:
        best_existing = population.get_solution_by_index(task, 0)
        return best_existing['idea'], best_existing['best_sol'], best_existing['utility']

    best_idea, best_solution, best_utility = min(solutions_with_utilities, key=lambda x: x[2])

    # Adjustments for ongoing adaptability of pheromones
    for key in pheromone_levels:
        pheromone_levels[key] *= 0.9  # Mild evaporation to allow exploration

    return best_idea, best_solution, best_utility

