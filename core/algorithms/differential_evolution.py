"""
Differential Evolution optimization algorithm implementation.
"""

import numpy as np
import logging
from scipy.optimize import differential_evolution

def differential_evolution_optimization(objective_function, bounds, config):
    """
    Run differential evolution optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Differential Evolution optimization")
    
    # Get configuration parameters
    try:
        population_size = int(config["differential_evolution"].get("population_size", "20"))
    except (KeyError, ValueError):
        population_size = 20
        logging.warning(f"Using default population size: {population_size}")
    
    try:
        mutation = float(config["differential_evolution"].get("mutation", "0.8"))
    except (KeyError, ValueError):
        mutation = 0.8
        logging.warning(f"Using default mutation: {mutation}")
    
    try:
        crossover = float(config["differential_evolution"].get("crossover", "0.7"))
    except (KeyError, ValueError):
        crossover = 0.7
        logging.warning(f"Using default crossover: {crossover}")
    
    try:
        max_iterations = int(config["differential_evolution"].get("max_iterations", "100"))
    except (KeyError, ValueError):
        max_iterations = 100
        logging.warning(f"Using default max iterations: {max_iterations}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    if direction == "maximize":
        # Negate objective function for maximization (scipy minimizes by default)
        def wrapped_objective(x):
            return -objective_function(x)
    else:
        wrapped_objective = objective_function
    
    # Run optimization
    try:
        result = differential_evolution(
            wrapped_objective,
            bounds,
            popsize=population_size,
            mutation=mutation,
            recombination=crossover,
            maxiter=max_iterations,
            disp=True
        )
        
        # Format results
        optimization_result = {
            'algorithm': 'differential_evolution',
            'success': result.success,
            'x': result.x.tolist(),
            'fun': -result.fun if direction == "maximize" else result.fun,
            'nit': result.nit,
            'nfev': result.nfev,
            'message': result.message
        }
        
        logging.info(f"Differential Evolution completed: {optimization_result['fun']}")
        
        return optimization_result
    
    except Exception as e:
        logging.error(f"Differential Evolution failed: {str(e)}")
        return {
            'algorithm': 'differential_evolution',
            'success': False,
            'x': None,
            'fun': None,
            'nit': 0,
            'nfev': 0,
            'message': str(e)
        }
