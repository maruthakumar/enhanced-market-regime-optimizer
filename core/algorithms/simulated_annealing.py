"""
Simulated Annealing optimization implementation.
"""

import numpy as np
import logging
import random
import math

def simulated_annealing_optimization(objective_function, bounds, config):
    """
    Run simulated annealing optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Simulated Annealing optimization")
    
    # Get configuration parameters
    try:
        initial_temp = float(config["simulated_annealing"].get("initial_temp", "100"))
    except (KeyError, ValueError):
        initial_temp = 100
        logging.warning(f"Using default initial temperature: {initial_temp}")
    
    try:
        cooling_rate = float(config["simulated_annealing"].get("cooling_rate", "0.95"))
    except (KeyError, ValueError):
        cooling_rate = 0.95
        logging.warning(f"Using default cooling rate: {cooling_rate}")
    
    try:
        max_iterations = int(config["simulated_annealing"].get("max_iterations", "100"))
    except (KeyError, ValueError):
        max_iterations = 100
        logging.warning(f"Using default max iterations: {max_iterations}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Problem dimensions
    dimensions = len(bounds)
    
    # Initialize current solution
    current_solution = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)])
    current_energy = objective_function(current_solution)
    
    # Initialize best solution
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    # Initialize temperature
    temperature = initial_temp
    
    # Run optimization
    iterations = 0
    evaluations = 1
    
    while iterations < max_iterations:
        # Generate neighbor
        neighbor = current_solution.copy()
        
        # Modify one random dimension
        dim = random.randint(0, dimensions - 1)
        step_size = (bounds[dim][1] - bounds[dim][0]) * 0.1 * (1 - iterations / max_iterations)
        neighbor[dim] += random.uniform(-step_size, step_size)
        
        # Ensure within bounds
        neighbor[dim] = max(bounds[dim][0], min(bounds[dim][1], neighbor[dim]))
        
        # Evaluate neighbor
        neighbor_energy = objective_function(neighbor)
        evaluations += 1
        
        # Determine if we should accept the neighbor
        if maximize:
            delta_e = neighbor_energy - current_energy
        else:
            delta_e = current_energy - neighbor_energy
        
        # Accept if better
        if delta_e > 0:
            current_solution = neighbor
            current_energy = neighbor_energy
            
            # Update best solution
            if (maximize and neighbor_energy > best_energy) or (not maximize and neighbor_energy < best_energy):
                best_solution = neighbor.copy()
                best_energy = neighbor_energy
        # Accept with probability based on temperature
        elif random.random() < math.exp(delta_e / temperature):
            current_solution = neighbor
            current_energy = neighbor_energy
        
        # Cool down
        temperature *= cooling_rate
        iterations += 1
    
    # Format results
    optimization_result = {
        'algorithm': 'simulated_annealing',
        'success': True,
        'x': best_solution.tolist(),
        'fun': best_energy if maximize else -best_energy,
        'nit': iterations,
        'nfev': evaluations,
        'message': "Simulated annealing completed successfully"
    }
    
    logging.info(f"Simulated Annealing completed: {optimization_result['fun']}")
    
    return optimization_result
