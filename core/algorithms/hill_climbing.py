"""
Hill Climbing optimization algorithm implementation.
"""

import numpy as np
import logging
import random

def hill_climbing_optimization(objective_function, bounds, config):
    """
    Run hill climbing optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Hill Climbing optimization")
    
    # Get configuration parameters
    try:
        step_size = float(config["hill_climbing"].get("step_size", "0.1"))
    except (KeyError, ValueError):
        step_size = 0.1
        logging.warning(f"Using default step size: {step_size}")
    
    try:
        max_iterations = int(config["hill_climbing"].get("max_iterations", "100"))
    except (KeyError, ValueError):
        max_iterations = 100
        logging.warning(f"Using default max iterations: {max_iterations}")
    
    try:
        restarts = int(config["hill_climbing"].get("restarts", "5"))
    except (KeyError, ValueError):
        restarts = 5
        logging.warning(f"Using default restarts: {restarts}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Initialize best solution
    best_x = None
    best_value = float('-inf') if maximize else float('inf')
    
    # Run optimization with multiple restarts
    for restart in range(restarts):
        logging.info(f"Hill Climbing restart {restart + 1}/{restarts}")
        
        # Initialize random solution within bounds
        current_x = np.array([random.uniform(b[0], b[1]) for b in bounds])
        current_value = objective_function(current_x)
        
        # Track iterations
        iterations = 0
        evaluations = 1
        
        # Run hill climbing
        while iterations < max_iterations:
            # Generate neighbor
            neighbor_x = current_x.copy()
            
            # Modify one random dimension
            dim = random.randint(0, len(bounds) - 1)
            delta = random.uniform(-step_size, step_size) * (bounds[dim][1] - bounds[dim][0])
            neighbor_x[dim] += delta
            
            # Ensure within bounds
            neighbor_x[dim] = max(bounds[dim][0], min(bounds[dim][1], neighbor_x[dim]))
            
            # Evaluate neighbor
            neighbor_value = objective_function(neighbor_x)
            evaluations += 1
            
            # Check if neighbor is better
            if (maximize and neighbor_value > current_value) or (not maximize and neighbor_value < current_value):
                current_x = neighbor_x
                current_value = neighbor_value
            
            iterations += 1
        
        # Update best solution
        if (maximize and current_value > best_value) or (not maximize and current_value < best_value):
            best_x = current_x
            best_value = current_value
    
    # Format results
    optimization_result = {
        'algorithm': 'hill_climbing',
        'success': True,
        'x': best_x.tolist(),
        'fun': best_value,
        'nit': max_iterations * restarts,
        'nfev': evaluations,
        'message': "Hill climbing completed successfully"
    }
    
    logging.info(f"Hill Climbing completed: {optimization_result['fun']}")
    
    return optimization_result
