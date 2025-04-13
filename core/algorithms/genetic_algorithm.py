"""
Genetic Algorithm optimization implementation.
"""

import numpy as np
import logging
import random
from deap import base, creator, tools, algorithms

def genetic_algorithm_optimization(objective_function, bounds, config):
    """
    Run genetic algorithm optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Genetic Algorithm optimization")
    
    # Get configuration parameters
    try:
        population_size = int(config["genetic_algorithm"].get("population_size", "50"))
    except (KeyError, ValueError):
        population_size = 50
        logging.warning(f"Using default population size: {population_size}")
    
    try:
        crossover_prob = float(config["genetic_algorithm"].get("crossover_prob", "0.8"))
    except (KeyError, ValueError):
        crossover_prob = 0.8
        logging.warning(f"Using default crossover probability: {crossover_prob}")
    
    try:
        mutation_prob = float(config["genetic_algorithm"].get("mutation_prob", "0.2"))
    except (KeyError, ValueError):
        mutation_prob = 0.2
        logging.warning(f"Using default mutation probability: {mutation_prob}")
    
    try:
        generations = int(config["genetic_algorithm"].get("generations", "50"))
    except (KeyError, ValueError):
        generations = 50
        logging.warning(f"Using default generations: {generations}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Setup DEAP
    if maximize:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Define attribute generator
    def create_attribute(i):
        return random.uniform(bounds[i][0], bounds[i][1])
    
    # Register attribute generator
    for i in range(len(bounds)):
        toolbox.register(f"attr_{i}", create_attribute, i)
    
    # Register individual and population creation
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [getattr(toolbox, f"attr_{i}") for i in range(len(bounds))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define evaluation function
    def evaluate(individual):
        return (objective_function(individual),)
    
    toolbox.register("evaluate", evaluate)
    
    # Define genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    # Define mutation operator with bounds
    def bounded_mutation(individual, mu, sigma, indpb):
        for i, (a, b) in enumerate(zip(individual, tools.mutGaussian(individual, mu, sigma, indpb)[0])):
            individual[i] = min(max(bounds[i][0], b), bounds[i][1])
        return (individual,)
    
    toolbox.register("mutate", bounded_mutation, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Track best solution
    hof = tools.HallOfFame(1)
    
    # Track statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run genetic algorithm
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, 
                                          mutpb=mutation_prob, ngen=generations, 
                                          stats=stats, halloffame=hof, verbose=True)
        
        # Get best solution
        best_ind = hof[0]
        best_fitness = best_ind.fitness.values[0]
        
        # Format results
        optimization_result = {
            'algorithm': 'genetic_algorithm',
            'success': True,
            'x': list(best_ind),
            'fun': best_fitness if maximize else -best_fitness,
            'nit': generations,
            'nfev': population_size * (generations + 1),  # Initial pop + generations
            'message': "Genetic algorithm completed successfully"
        }
        
        logging.info(f"Genetic Algorithm completed: {optimization_result['fun']}")
        
        return optimization_result
    
    except Exception as e:
        logging.error(f"Genetic Algorithm failed: {str(e)}")
        return {
            'algorithm': 'genetic_algorithm',
            'success': False,
            'x': None,
            'fun': None,
            'nit': 0,
            'nfev': 0,
            'message': str(e)
        }
