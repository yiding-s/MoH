from utils.utils import extract_code,extract_idea
import json

def improve_algorithm(population,utility, language_model,function_format,subtask):
    """Improves a solution according to a utility function."""
    expertise = "You are an expert in the domain of designing meta optimization strategy and designing combinatorial optimization problems. Your task is to design heuristics that can effectively solve optimization problems."
    n_messages = language_model.batch_size
    temperature_values = [0.7, 1.0]
    solutions_cache = set()
    new_solutions = []
    new_ideas=[]
    utility_cache = {}

    def evaluate_solution(solution,idea):
        if solution not in utility_cache:
            utility_cache[solution] = utility(solution,idea,subtask)
        return utility_cache[solution]
    
    for temp in temperature_values:
        message_batch=[]
        selected_solution=population.get_random_solution(subtask)
        direction_prompt=f"Given the following heuristic for subtask: {selected_solution['best_sol']} with its idea: {selected_solution['idea']}, and utility score: {selected_solution['utility']},\
        summarize the key idea from this heuristic, then provide several totally different ideas from the given one to design improved algorithms. \
        Propose some conventional optimization techniques, or some novel metaheuristic strategies that can be used to improve the current solution. For example, Population-Based, Heuristic (Partial) Search Methods, Local Search and Iterative Improvement, Bandit-Based (Exploration-Exploitation) Methods, etc.\
        Please provide a single string as the answer, less than 50 words. Your response should be formatted as a json structure: ```json\n{{\"insights\":[\"content\",\"content\",\"content\", ... ,\"content\"}}\n```."
        response = language_model.prompt(expertise,direction_prompt)
        directions=json.loads(extract_code(response))["insights"]
        for direction in directions:
            # select_solution=population.get_solution_by_index(subtask)['best_sol']
            message =  f"""Improve the following solution:
        ```python
        {selected_solution}
        ```
        You must return an improved solution. Formatted as follows:
        {function_format}
        To better solve the problem, you are encouraged to develop new solutions base on the direction proposed: {direction},\
        If you think the direction is a refinement to current solution, just improve the current solution.\
        If you think the direction is a new idea, you can develop a brand new solution with NO RELATION to the solution given.\
        You can add appropriate loops in the code for your needs, but not larger than 5.
        You will be evaluated based on a score function. The lower the score, the better the solution.
        Be as creative as you can under the constraints.
        Generate a solution with temperature={temp} that focuses on different aspects of optimization."""
            message_batch.append(message)

        responses = language_model.prompt_batch(expertise, message_batch, temperature=temp)
        generated_solutions = extract_code(responses)
        generated_ideas = extract_idea(responses)
        # Evaluate and sort the generated solutions by their utility score
        scored_solutions = [(idea, sol, evaluate_solution(sol,idea)) for sol, idea in zip(generated_solutions, generated_ideas) if sol not in solutions_cache]
        scored_solutions.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only the top n_messages solutions
        top_solutions = scored_solutions[:n_messages]
        
        for idea, sol, _ in top_solutions:
            new_ideas.append(idea)
            new_solutions.append(sol)
            solutions_cache.add(sol)

    solutions_and_scores = [(idea, solution, evaluate_solution(solution,idea)) for idea,solution in zip(new_ideas,new_solutions)]
    best_idea, best_solution, best_utility = min(solutions_and_scores, key=lambda x: x[2])
    return best_idea, best_solution, best_utility