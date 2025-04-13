"""
Particle Swarm Optimization implementation.
"""

import numpy as np
import logging
import random

def particle_swarm_optimization(objective_function, bounds, config):
    """
    Run particle swarm optimization.
    
    Args:
        objective_function (callable): Function to optimize
        bounds (list): List of (min, max) pairs for each parameter
        config (dict): Configuration settings
        
    Returns:
        dict: Optimization results
    """
    logging.info("Running Particle Swarm Optimization")
    
    # Get configuration parameters
    try:
        num_particles = int(config["particle_swarm"].get("particles", "30"))
    except (KeyError, ValueError):
        num_particles = 30
        logging.warning(f"Using default number of particles: {num_particles}")
    
    try:
        cognitive = float(config["particle_swarm"].get("cognitive", "1.5"))
    except (KeyError, ValueError):
        cognitive = 1.5
        logging.warning(f"Using default cognitive parameter: {cognitive}")
    
    try:
        social = float(config["particle_swarm"].get("social", "1.5"))
    except (KeyError, ValueError):
        social = 1.5
        logging.warning(f"Using default social parameter: {social}")
    
    try:
        inertia = float(config["particle_swarm"].get("inertia", "0.7"))
    except (KeyError, ValueError):
        inertia = 0.7
        logging.warning(f"Using default inertia: {inertia}")
    
    try:
        max_iterations = int(config["particle_swarm"].get("max_iterations", "100"))
    except (KeyError, ValueError):
        max_iterations = 100
        logging.warning(f"Using default max iterations: {max_iterations}")
    
    # Determine optimization direction
    direction = config["optimization"].get("direction", "maximize").lower()
    maximize = direction == "maximize"
    
    # Problem dimensions
    dimensions = len(bounds)
    
    # Initialize particles
    particles = []
    for _ in range(num_particles):
        # Random position within bounds
        position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)])
        
        # Random velocity
        velocity = np.array([random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) / 10 for i in range(dimensions)])
        
        # Evaluate fitness
        fitness = objective_function(position)
        
        # Initialize personal best
        personal_best_position = position.copy()
        personal_best_fitness = fitness
        
        particles.append({
            'position': position,
            'velocity': velocity,
            'fitness': fitness,
            'personal_best_position': personal_best_position,
            'personal_best_fitness': personal_best_fitness
        })
    
    # Initialize global best
    if maximize:
        global_best_fitness = float('-inf')
        global_best_position = None
        for particle in particles:
            if particle['fitness'] > global_best_fitness:
                global_best_fitness = particle['fitness']
                global_best_position = particle['position'].copy()
    else:
        global_best_fitness = float('inf')
        global_best_position = None
        for particle in particles:
            if particle['fitness'] < global_best_fitness:
                global_best_fitness = particle['fitness']
                global_best_position = particle['position'].copy()
    
    # Run optimization
    iterations = 0
    evaluations = num_particles
    
    while iterations < max_iterations:
        # Update particles
        for particle in particles:
            # Update velocity
            r1 = random.random()
            r2 = random.random()
            
            cognitive_component = cognitive * r1 * (particle['personal_best_position'] - particle['position'])
            social_component = social * r2 * (global_best_position - particle['position'])
            
            particle['velocity'] = inertia * particle['velocity'] + cognitive_component + social_component
            
            # Update position
            particle['position'] = particle['position'] + particle['velocity']
            
            # Ensure within bounds
            for i in range(dimensions):
                particle['position'][i] = max(bounds[i][0], min(bounds[i][1], particle['position'][i]))
            
            # Evaluate fitness
            particle['fitness'] = objective_function(particle['position'])
            evaluations += 1
            
            # Update personal best
            if (maximize and particle['fitness'] > particle['personal_best_fitness']) or \
               (not maximize and particle['fitness'] < particle['personal_best_fitness']):
                particle['personal_best_position'] = particle['position'].copy()
                particle['personal_best_fitness'] = particle['fitness']
                
                # Update global best
                if (maximize and particle['fitness'] > global_best_fitness) or \
                   (not maximize and particle['fitness'] < global_best_fitness):
                    global_best_fitness = particle['fitness']
                    global_best_position = particle['position'].copy()
        
        iterations += 1
    
    # Format results
    optimization_result = {
        'algorithm': 'particle_swarm',
        'success': True,
        'x': global_best_position.tolist(),
        'fun': global_best_fitness,
        'nit': iterations,
        'nfev': evaluations,
        'message': "Particle swarm optimization completed successfully"
    }
    
    logging.info(f"Particle Swarm Optimization completed: {optimization_result['fun']}")
    
    return optimization_result
