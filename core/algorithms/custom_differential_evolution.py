"""
Custom Differential Evolution optimization implementation.
"""

import numpy as np
import logging
import random

def custom_differential_evolution_optimization(objective_function, bounds, config):
    """
    Run custom differential evolution optimization with enhanced features.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Custom Differential Evolution optimization")
    
    # Get configuration parameters
    try:
        population_size = int(config["custom_de"].get("population_size", "30"))
    except (KeyError, ValueError):
        population_size = 30
        logging.warning(f"Using default population size: {population_size}")
    
    try:
        mutation_min = float(config["custom_de"].get("mutation_min", "0.5"))
    except (KeyError, ValueError):
        mutation_min = 0.5
        logging.warning(f"Using default minimum mutation: {mutation_min}")
    
    try:
        mutation_max = float(config["custom_de"].get("mutation_max", "1.0"))
    except (KeyError, ValueError):
        mutation_max = 1.0
        logging.warning(f"Using default maximum mutation: {mutation_max}")
    
    try:
        crossover = float(config["custom_de"].get("crossover", "0.7"))
    except (KeyError, ValueError):
        crossover = 0.7
        logging.warning(f"Using default crossover: {crossover}")
    
    try:
        max_iterations = int(config["custom_de"].get("max_iterations", "100"))
    except (KeyError, ValueError):
        max_iterations = 100
        logging.warning(f"Using default max iterations: {max_iterations}")
    
    try:
        adaptive = config["custom_de"].get("adaptive", "true").lower() == "true"
    except (KeyError, ValueError):
        adaptive = True
        logging.warning(f"Using default adaptive setting: {adaptive}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Problem dimensions
    dimensions = len(bounds)
    
    # Initialize population
    population = []
    for i in range(population_size):
        # Random position within bounds
        individual = np.array([random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimensions)])
        
        # Evaluate fitness
        fitness = objective_function(individual)
        
        population.append({
            'position': individual,
            'fitness': fitness
        })
    
    # Sort population
    if maximize:
        population.sort(key=lambda x: x['fitness'], reverse=True)
    else:
        population.sort(key=lambda x: x['fitness'])
    
    # Track best solution
    best_solution = population[0]['position'].copy()
    best_fitness = population[0]['fitness']
    
    # Track evaluations
    evaluations = population_size
    
    # Run optimization
    iterations = 0
    stagnation_counter = 0
    
    while iterations < max_iterations:
        # Track if population improved
        improved = False
        
        # For each individual
        for i in range(population_size):
            # Select three random individuals different from current
            candidates = list(range(population_size))
            candidates.remove(i)
            random.shuffle(candidates)
            a, b, c = candidates[:3]
            
            # Create trial vector
            if adaptive:
                # Adaptive mutation factor based on iteration progress
                progress = iterations / max_iterations
                mutation = mutation_min + (mutation_max - mutation_min) * (1 - progress)
            else:
                mutation = random.uniform(mutation_min, mutation_max)
            
            # Create mutant vector
            mutant = population[a]['position'] + mutation * (population[b]['position'] - population[c]['position'])
            
            # Ensure within bounds
            for j in range(dimensions):
                mutant[j] = max(bounds[j][0], min(bounds[j][1], mutant[j]))
            
            # Crossover
            trial = np.zeros(dimensions)
            for j in range(dimensions):
                if random.random() < crossover or j == random.randint(0, dimensions - 1):
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i]['position'][j]
            
            # Evaluate trial
            trial_fitness = objective_function(trial)
            evaluations += 1
            
            # Selection
            if (maximize and trial_fitness > population[i]['fitness']) or \
               (not maximize and trial_fitness < population[i]['fitness']):
                population[i]['position'] = trial
                population[i]['fitness'] = trial_fitness
                improved = True
                
                # Update best solution
                if (maximize and trial_fitness > best_fitness) or \
                   (not maximize and trial_fitness < best_fitness):
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
        
        # Update stagnation counter
        if improved:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Apply diversity mechanism if stagnation
        if stagnation_counter >= 10:
            # Reinitialize 20% of population
            reinit_count = max(1, int(population_size * 0.2))
            for i in range(reinit_count):
                idx = random.randint(1, population_size - 1)  # Keep best solution
                
                # Random position within bounds
                individual = np.array([random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimensions)])
                
                # Evaluate fitness
                fitness = objective_function(individual)
                evaluations += 1
                
                population[idx]['position'] = individual
                population[idx]['fitness'] = fitness
                
                # Update best solution
                if (maximize and fitness > best_fitness) or \
                   (not maximize and fitness < best_fitness):
                    best_solution = individual.copy()
                    best_fitness = fitness
            
            stagnation_counter = 0
        
        iterations += 1
    
    # Format results
    optimization_result = {
        'algorithm': 'custom_differential_evolution',
        'success': True,
        'x': best_solution.tolist(),
        'fun': best_fitness if maximize else -best_fitness,
        'nit': iterations,
        'nfev': evaluations,
        'message': "Custom differential evolution completed successfully"
    }
    
    logging.info(f"Custom Differential Evolution completed: {optimization_result['fun']}")
    
    return optimization_result
