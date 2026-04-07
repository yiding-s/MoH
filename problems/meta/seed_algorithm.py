from utils.utils import extract_code, extract_idea
import json

def improve_algorithm(population, utility, language_model, function_format, task):
    expertise = "You are an expert in the domain of designing meta optimization strategy and combinatorial optimization problems. Your task is to design heuristics that can effectively solve optimization problems."
    
    # Step 1: Select a random solution from the population
    selected_solution = population.get_random_solution(task)
    
    # Step 2: Generate directions for improvement
    direction_prompt = (
        f"Given the following heuristic for subtask: {selected_solution['best_sol']} with its idea: {selected_solution['idea']} and utility score: {selected_solution['utility']}, "
        "Summarize the key idea from this heuristic, then provide several totally different ideas from the given one to design improved algorithms with lower utility score. "
        "Provide a single string as the answer, less than 50 words. Your response should be formatted as a json structure: "
        "```json\n{{\"insights\":[\"content\",\"content\",\"content\", ... ,\"content\"]}}\n```."
    )
    response = language_model.prompt(expertise, direction_prompt, temperature=1)
    directions = json.loads(extract_code(response))["insights"]

    # Step 3: Create messages based on generated directions
    message_batch = []
    for direction in directions:
        message = (
            f"Improve the following solution:\n"
            f"```python\n{selected_solution}\n```\n"
            f"You must return an improved solution. Formatted as follows:\n{function_format}\n"
            f"To better solve the problem, you are encouraged to develop new solutions based on the direction proposed: {direction}. "
            "You will be evaluated based on a score function. The lower the score, the better the solution.\n"
            "Your response must firstly provide a summary of the key idea inside a brace and marked as a comment, followed by the code implementation. "
            "Be as creative as you can under the constraints."
        )
        message_batch.append(message)

    # Step 4: Generate new solutions using the language model
    responses = language_model.prompt_batch(expertise, message_batch, temperature=1)
    new_solutions = extract_code(responses)
    new_ideas = extract_idea(responses)

    # Step 5: Evaluate new solutions
    solutions_with_utilities = [(idea, solution, utility(solution,idea,task)) for idea, solution in zip(new_ideas, new_solutions)]
    best_idea, best_solution, best_utility = min(solutions_with_utilities, key=lambda x: x[2])

    return best_idea, best_solution, best_utility