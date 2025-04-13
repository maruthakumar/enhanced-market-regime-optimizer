"""
Ant Colony Optimization implementation.
"""

import numpy as np
import logging
import random

def ant_colony_optimization(objective_function, bounds, config):
    """
    Run ant colony optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Ant Colony Optimization")
    
    # Get configuration parameters
    try:
        num_ants = int(config["ant_colony"].get("ants", "20"))
    except (KeyError, ValueError):
        num_ants = 20
        logging.warning(f"Using default number of ants: {num_ants}")
    
    try:
        evaporation = float(config["ant_colony"].get("evaporation", "0.1"))
    except (KeyError, ValueError):
        evaporation = 0.1
        logging.warning(f"Using default evaporation rate: {evaporation}")
    
    try:
        alpha = float(config["ant_colony"].get("alpha", "1.0"))
    except (KeyError, ValueError):
        alpha = 1.0
        logging.warning(f"Using default alpha: {alpha}")
    
    try:
        beta = float(config["ant_colony"].get("beta", "2.0"))
    except (KeyError, ValueError):
        beta = 2.0
        logging.warning(f"Using default beta: {beta}")
    
    try:
        max_iterations = int(config["ant_colony"].get("max_iterations", "50"))
    except (KeyError, ValueError):
        max_iterations = 50
        logging.warning(f"Using default max iterations: {max_iterations}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Problem dimensions
    dimensions = len(bounds)
    
    # Discretize the search space
    grid_points = 20
    grid = []
    for i in range(dimensions):
        grid.append(np.linspace(bounds[i][0], bounds[i][1], grid_points))
    
    # Initialize pheromone matrix
    pheromone = np.ones((dimensions, grid_points))
    
    # Initialize best solution
    best_solution = None
    if maximize:
        best_fitness = float('-inf')
    else:
        best_fitness = float('inf')
    
    # Run optimization
    iterations = 0
    evaluations = 0
    
    while iterations < max_iterations:
        # Solutions for this iteration
        solutions = []
        fitnesses = []
        
        # For each ant
        for ant in range(num_ants):
            # Generate solution
            solution = np.zeros(dimensions)
            for i in range(dimensions):
                # Calculate probabilities
                probabilities = pheromone[i] ** alpha
                probabilities = probabilities / np.sum(probabilities)
                
                # Select grid point
                grid_idx = np.random.choice(grid_points, p=probabilities)
                solution[i] = grid[i][grid_idx]
            
            # Evaluate solution
            fitness = objective_function(solution)
            evaluations += 1
            
            # Store solution
            solutions.append(solution)
            fitnesses.append(fitness)
            
            # Update best solution
            if (maximize and fitness > best_fitness) or (not maximize and fitness < best_fitness):
                best_solution = solution.copy()
                best_fitness = fitness
        
        # Update pheromones
        pheromone = (1 - evaporation) * pheromone
        
        # Add new pheromones
        for ant in range(num_ants):
            solution = solutions[ant]
            fitness = fitnesses[ant]
            
            # Calculate pheromone deposit
            if maximize:
                deposit = fitness if fitness > 0 else 1e-10
            else:
                deposit = 1 / (fitness + 1e-10)
            
            # Update pheromone matrix
            for i in range(dimensions):
                # Find closest grid point
                grid_idx = np.abs(grid[i] - solution[i]).argmin()
                pheromone[i, grid_idx] += deposit
        
        iterations += 1
    
    # Format results
    optimization_result = {
        'algorithm': 'ant_colony',
        'success': True,
        'x': best_solution.tolist(),
        'fun': best_fitness if maximize else -best_fitness,
        'nit': iterations,
        'nfev': evaluations,
        'message': "Ant colony optimization completed successfully"
    }
    
    logging.info(f"Ant Colony Optimization completed: {optimization_result['fun']}")
    
    return optimization_result
