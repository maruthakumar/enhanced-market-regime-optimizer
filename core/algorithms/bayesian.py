"""
Bayesian Optimization implementation.
"""

import numpy as np
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

def bayesian_optimization(objective_function, bounds, config):
    """
    Run Bayesian optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Bayesian Optimization")
    
    # Get configuration parameters
    try:
        n_iter = int(config["bayesian"].get("n_iter", "30"))
    except (KeyError, ValueError):
        n_iter = 30
        logging.warning(f"Using default number of iterations: {n_iter}")
    
    try:
        n_initial = int(config["bayesian"].get("n_initial", "5"))
    except (KeyError, ValueError):
        n_initial = 5
        logging.warning(f"Using default number of initial points: {n_initial}")
    
    try:
        kappa = float(config["bayesian"].get("kappa", "2.5"))
    except (KeyError, ValueError):
        kappa = 2.5
        logging.warning(f"Using default kappa (exploration-exploitation trade-off): {kappa}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Problem dimensions
    dimensions = len(bounds)
    
    # Initialize Gaussian Process
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    
    # Generate initial random points
    X_sample = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_initial, dimensions)
    )
    
    # Evaluate initial points
    y_sample = np.array([objective_function(x) for x in X_sample])
    if not maximize:
        y_sample = -y_sample  # Convert to maximization problem
    
    # Track best solution
    best_idx = np.argmax(y_sample)
    best_x = X_sample[best_idx]
    best_y = y_sample[best_idx]
    
    # Track evaluations
    evaluations = n_initial
    
    # Run optimization
    for i in range(n_iter):
        # Fit GP model
        gp.fit(X_sample, y_sample)
        
        # Find next point to evaluate
        next_x = propose_location(acquisition_function, X_sample, y_sample, gp, bounds, kappa)
        
        # Evaluate next point
        next_y = objective_function(next_x)
        evaluations += 1
        
        if not maximize:
            next_y = -next_y  # Convert to maximization problem
        
        # Update samples
        X_sample = np.vstack((X_sample, next_x.reshape(1, -1)))
        y_sample = np.append(y_sample, next_y)
        
        # Update best solution
        if next_y > best_y:
            best_x = next_x
            best_y = next_y
    
    # Format results
    optimization_result = {
        'algorithm': 'bayesian',
        'success': True,
        'x': best_x.tolist(),
        'fun': best_y if maximize else -best_y,
        'nit': n_iter,
        'nfev': evaluations,
        'message': "Bayesian optimization completed successfully"
    }
    
    logging.info(f"Bayesian Optimization completed: {optimization_result['fun']}")
    
    return optimization_result

def acquisition_function(x, X_sample, y_sample, gp, kappa):
    """
    Expected Improvement acquisition function.
    
    Args:
        x (array): Point to evaluate
        X_sample (array): Sample points
        y_sample (array): Sample values
        gp (GaussianProcessRegressor): Gaussian Process model
        kappa (float): Exploration-exploitation trade-off parameter
        
    Returns:
        float: Expected improvement
    """
    # Reshape x for prediction
    x = x.reshape(1, -1)
    
    # Get mean and standard deviation
    mu, sigma = gp.predict(x, return_std=True)
    
    # Get current best
    y_max = np.max(y_sample)
    
    # Calculate improvement
    with np.errstate(divide='ignore'):
        imp = mu - y_max - kappa * sigma
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei[0]

def propose_location(acquisition, X_sample, y_sample, gp, bounds, kappa):
    """
    Propose next sampling location by optimizing the acquisition function.
    
    Args:
        acquisition (callable): Acquisition function
        X_sample (array): Sample points
        y_sample (array): Sample values
        gp (GaussianProcessRegressor): Gaussian Process model
        bounds (list): List of (min, max) pairs for each parameter
        kappa (float): Exploration-exploitation trade-off parameter
        
    Returns:
        array: Next point to evaluate
    """
    # Define negative acquisition function (for minimization)
    def min_obj(x):
        return -acquisition(x.reshape(1, -1), X_sample, y_sample, gp, kappa)
    
    # Find best starting point from previous samples
    x_tries = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(100, len(bounds))
    )
    
    # Add the best observed point
    x_tries = np.vstack((x_tries, X_sample[np.argmax(y_sample)]))
    
    # Evaluate acquisition function at all points
    ys = np.array([min_obj(x) for x in x_tries])
    
    # Find best point
    x_min = x_tries[np.argmin(ys)]
    
    # Run local optimization from best point
    result = minimize(min_obj, x_min, bounds=bounds, method='L-BFGS-B')
    
    # Return optimized point
    return result.x
