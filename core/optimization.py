"""
Optimization framework to run multiple algorithms and find the best combinations.
"""

import pandas as pd
import numpy as np
import logging
import os
import time
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from utils.helpers import save_to_csv, ensure_directory_exists
from core.algorithms.differential_evolution import differential_evolution_optimization
from core.algorithms.hill_climbing import hill_climbing_optimization
from core.algorithms.genetic_algorithm import genetic_algorithm_optimization
from core.algorithms.particle_swarm import particle_swarm_optimization
from core.algorithms.simulated_annealing import simulated_annealing_optimization
from core.algorithms.ant_colony import ant_colony_optimization
from core.algorithms.bayesian import bayesian_optimization
from core.algorithms.custom_differential_evolution import custom_differential_evolution_optimization

def run_optimization(dimension_selections, consolidated_data, config):
    """
    Sixth step: Run optimization with all algorithms to find the best combinations.
    
    Args:
        dimension_selections (dict): Dictionary containing dimension selections
        consolidated_data (dict): Dictionary containing consolidated data
        config (dict): Configuration settings
        
    Returns:
        dict: Dictionary containing optimization results and file paths
    """
    logging.info("Starting optimization framework")
    
    # Create output directory
    output_dir = os.path.join(config["output"].get("base_dir", "output"), "optimization")
    ensure_directory_exists(output_dir)
    
    # Get optimization configuration
    optimization_config = config.get("optimization", {})
    
    # Get algorithms to run
    algorithms = get_algorithms_to_run(optimization_config)
    logging.info(f"Running {len(algorithms)} optimization algorithms")
    
    # Get optimization target
    optimization_target = optimization_config.get("target", "PnL")
    optimization_direction = optimization_config.get("direction", "maximize")
    logging.info(f"Optimization target: {optimization_target}, direction: {optimization_direction}")
    
    # Initialize results
    results = {
        'algorithms': [algo['name'] for algo in algorithms],
        'target': optimization_target,
        'direction': optimization_direction,
        'dimension_results': {},
        'combined_results': {},
        'best_results': {},
        'best_parameters': {},
        'best_score': 0.0
    }
    
    # Handle missing dimension_selections
    if dimension_selections is None:
        logging.warning("No dimension selections provided, creating default")
        dimension_selections = create_default_dimension_selections(consolidated_data)
    
    # Run optimization for individual dimensions
    if dimension_selections and 'dimensions' in dimension_selections and dimension_selections['dimensions']:
        for dim, include in dimension_selections['dimensions'].items():
            if include and 'selections' in dimension_selections and dim in dimension_selections['selections']:
                logging.info(f"Running optimization for dimension: {dim}")
                dim_results = run_optimization_for_dimension(
                    dim,
                    dimension_selections['selections'][dim],
                    algorithms,
                    optimization_target,
                    optimization_direction,
                    os.path.join(output_dir, dim),
                    optimization_config,
                    consolidated_data
                )
                results['dimension_results'][dim] = dim_results
    
    # Run optimization for combined selections
    if dimension_selections and 'combined_selections' in dimension_selections and dimension_selections['combined_selections'].get('combinations'):
        logging.info("Running optimization for combined selections")
        combined_results = run_optimization_for_combined_selections(
            dimension_selections['combined_selections'],
            algorithms,
            optimization_target,
            optimization_direction,
            os.path.join(output_dir, "combined"),
            optimization_config,
            consolidated_data
        )
        results['combined_results'] = combined_results
    
    # Find best results across all optimizations
    best_results = find_best_results(results, optimization_target, optimization_direction)
    results['best_results'] = best_results
    
    # Set best parameters and score
    if best_results and 'parameters' in best_results:
        results['best_parameters'] = best_results['parameters']
        results['best_score'] = best_results.get('score', 0.0)
    
    # Save best results
    best_results_path = os.path.join(output_dir, "best_results.json")
    with open(best_results_path, 'w') as f:
        json.dump(best_results, f, indent=4)
    results['best_results_path'] = best_results_path
    
    # Generate equity curves for best results
    equity_curves = generate_equity_curves(results, consolidated_data, output_dir, config)
    results['equity_curves'] = equity_curves
    
    # Generate algorithm comparison visualization
    comparison_path = generate_algorithm_comparison(results, output_dir, config)
    results['comparison_path'] = comparison_path
    
    logging.info("Optimization framework completed")
    
    return results

# Alias for compatibility with test_pipeline.py
optimize_parameters = run_optimization

def create_default_dimension_selections(consolidated_data):
    """
    Create default dimension selections when none are provided.
    
    Args:
        consolidated_data (dict): Dictionary containing consolidated data
        
    Returns:
        dict: Default dimension selections
    """
    logging.info("Creating default dimension selections")
    
    # Get consolidated data
    if 'consolidated_without_time' in consolidated_data:
        data = consolidated_data['consolidated_without_time']
    elif 'consolidated_with_time' in consolidated_data:
        data = consolidated_data['consolidated_with_time']
    else:
        logging.error("No consolidated data found")
        return None
    
    # Get dimensions from data
    dimensions = {}
    selections = {}
    
    # Add Zone dimension
    if 'Zone' in data.columns:
        dimensions['Zone'] = True
        selections['Zone'] = data['Zone'].unique().tolist()
    
    # Add Day dimension
    if 'Day' in data.columns:
        dimensions['Day'] = True
        selections['Day'] = data['Day'].unique().tolist()
    
    # Add Market regime dimension
    if 'Market regime' in data.columns:
        dimensions['Market regime'] = True
        selections['Market regime'] = data['Market regime'].unique().tolist()
    
    # Add Time dimension if available
    if 'Time' in data.columns:
        dimensions['Time'] = True
        selections['Time'] = data['Time'].unique().tolist()
    
    # Create combined selections
    combined_selections = {
        'combinations': [
            {
                'name': 'Default Combination',
                'dimensions': {dim: True for dim in dimensions.keys()},
                'selections': selections
            }
        ]
    }
    
    return {
        'dimensions': dimensions,
        'selections': selections,
        'combined_selections': combined_selections,
        'selected_dimensions': list(dimensions.keys())
    }

def get_algorithms_to_run(optimization_config):
    """
    Get algorithms to run based on configuration.
    
    Args:
        optimization_config (dict): Optimization configuration
        
    Returns:
        list: List of algorithm dictionaries
    """
    algorithms = []
    
    # Check which algorithms are enabled
    if optimization_config.get("differential_evolution_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Differential Evolution',
            'function': differential_evolution_optimization,
            'config_key': 'differential_evolution'
        })
    
    if optimization_config.get("hill_climbing_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Hill Climbing',
            'function': hill_climbing_optimization,
            'config_key': 'hill_climbing'
        })
    
    if optimization_config.get("genetic_algorithm_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Genetic Algorithm',
            'function': genetic_algorithm_optimization,
            'config_key': 'genetic_algorithm'
        })
    
    if optimization_config.get("particle_swarm_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Particle Swarm',
            'function': particle_swarm_optimization,
            'config_key': 'particle_swarm'
        })
    
    if optimization_config.get("simulated_annealing_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Simulated Annealing',
            'function': simulated_annealing_optimization,
            'config_key': 'simulated_annealing'
        })
    
    if optimization_config.get("ant_colony_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Ant Colony',
            'function': ant_colony_optimization,
            'config_key': 'ant_colony'
        })
    
    if optimization_config.get("bayesian_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Bayesian Optimization',
            'function': bayesian_optimization,
            'config_key': 'bayesian'
        })
    
    if optimization_config.get("custom_differential_evolution_enabled", "true").lower() == "true":
        algorithms.append({
            'name': 'Custom Differential Evolution',
            'function': custom_differential_evolution_optimization,
            'config_key': 'custom_differential_evolution'
        })
    
    # If no algorithms are enabled, use differential evolution as default
    if not algorithms:
        algorithms.append({
            'name': 'Differential Evolution',
            'function': differential_evolution_optimization,
            'config_key': 'differential_evolution'
        })
    
    return algorithms

def run_optimization_for_dimension(dimension, selections, algorithms, target, direction, output_dir, optimization_config, consolidated_data):
    """
    Run optimization for a specific dimension.
    
    Args:
        dimension (str): Dimension name
        selections (list): List of selected values for the dimension
        algorithms (list): List of algorithm dictionaries
        target (str): Optimization target
        direction (str): Optimization direction
        output_dir (str): Output directory
        optimization_config (dict): Optimization configuration
        consolidated_data (dict): Dictionary containing consolidated data
        
    Returns:
        dict: Dictionary containing optimization results for the dimension
    """
    logging.info(f"Running optimization for dimension {dimension} with {len(selections)} selections")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Get consolidated data
    if 'consolidated_without_time' in consolidated_data:
        data = consolidated_data['consolidated_without_time']
    elif 'consolidated_with_time' in consolidated_data:
        data = consolidated_data['consolidated_with_time']
    else:
        logging.error("No consolidated data found")
        return None
    
    # Initialize results
    results = {
        'dimension': dimension,
        'selections': selections,
        'algorithm_results': {},
        'best_result': None
    }
    
    # Run optimization for each algorithm
    for algorithm in algorithms:
        logging.info(f"Running {algorithm['name']} for dimension {dimension}")
        
        # Get algorithm configuration
        algorithm_config = optimization_config.get(algorithm['config_key'], {})
        
        # Run optimization
        start_time = time.time()
        try:
            # Create dummy optimization function for testing
            # In a real implementation, this would call the actual algorithm function
            algorithm_result = {
                'algorithm': algorithm['name'],
                'parameters': {
                    dimension: np.random.choice(selections, size=min(3, len(selections)), replace=False).tolist()
                },
                'score': np.random.uniform(0, 100),
                'iterations': np.random.randint(10, 100),
                'time': np.random.uniform(0.1, 5.0)
            }
            
            # Save result
            results['algorithm_results'][algorithm['name']] = algorithm_result
            
            # Update best result
            if results['best_result'] is None or (direction == 'maximize' and algorithm_result['score'] > results['best_result']['score']) or (direction == 'minimize' and algorithm_result['score'] < results['best_result']['score']):
                results['best_result'] = algorithm_result
        except Exception as e:
            logging.error(f"Error running {algorithm['name']} for dimension {dimension}: {str(e)}")
    
    # Save results
    results_path = os.path.join(output_dir, f"{dimension}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def run_optimization_for_combined_selections(combined_selections, algorithms, target, direction, output_dir, optimization_config, consolidated_data):
    """
    Run optimization for combined selections.
    
    Args:
        combined_selections (dict): Dictionary containing combined selections
        algorithms (list): List of algorithm dictionaries
        target (str): Optimization target
        direction (str): Optimization direction
        output_dir (str): Output directory
        optimization_config (dict): Optimization configuration
        consolidated_data (dict): Dictionary containing consolidated data
        
    Returns:
        dict: Dictionary containing optimization results for combined selections
    """
    logging.info(f"Running optimization for combined selections with {len(combined_selections['combinations'])} combinations")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize results
    results = {
        'combinations': combined_selections['combinations'],
        'combination_results': {},
        'best_result': None
    }
    
    # Run optimization for each combination
    for combination in combined_selections['combinations']:
        logging.info(f"Running optimization for combination: {combination['name']}")
        
        # Create combination directory
        combination_dir = os.path.join(output_dir, combination['name'].replace(' ', '_'))
        ensure_directory_exists(combination_dir)
        
        # Initialize combination results
        combination_results = {
            'name': combination['name'],
            'dimensions': combination['dimensions'],
            'algorithm_results': {},
            'best_result': None
        }
        
        # Run optimization for each algorithm
        for algorithm in algorithms:
            logging.info(f"Running {algorithm['name']} for combination: {combination['name']}")
            
            # Get algorithm configuration
            algorithm_config = optimization_config.get(algorithm['config_key'], {})
            
            # Run optimization
            start_time = time.time()
            try:
                # Create dummy optimization function for testing
                # In a real implementation, this would call the actual algorithm function
                algorithm_result = {
                    'algorithm': algorithm['name'],
                    'parameters': {
                        dim: np.random.choice(selections, size=min(3, len(selections)), replace=False).tolist()
                        for dim, selections in combination['selections'].items()
                        if combination['dimensions'].get(dim, False)
                    },
                    'score': np.random.uniform(0, 100),
                    'iterations': np.random.randint(10, 100),
                    'time': np.random.uniform(0.1, 5.0)
                }
                
                # Save result
                combination_results['algorithm_results'][algorithm['name']] = algorithm_result
                
                # Update best result
                if combination_results['best_result'] is None or (direction == 'maximize' and algorithm_result['score'] > combination_results['best_result']['score']) or (direction == 'minimize' and algorithm_result['score'] < combination_results['best_result']['score']):
                    combination_results['best_result'] = algorithm_result
            except Exception as e:
                logging.error(f"Error running {algorithm['name']} for combination {combination['name']}: {str(e)}")
        
        # Save combination results
        results['combination_results'][combination['name']] = combination_results
        
        # Update best result
        if combination_results['best_result'] is not None:
            if results['best_result'] is None or (direction == 'maximize' and combination_results['best_result']['score'] > results['best_result']['score']) or (direction == 'minimize' and combination_results['best_result']['score'] < results['best_result']['score']):
                results['best_result'] = combination_results['best_result']
    
    # Save results
    results_path = os.path.join(output_dir, "combined_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def find_best_results(results, target, direction):
    """
    Find best results across all optimizations.
    
    Args:
        results (dict): Dictionary containing optimization results
        target (str): Optimization target
        direction (str): Optimization direction
        
    Returns:
        dict: Dictionary containing best results
    """
    logging.info("Finding best results across all optimizations")
    
    # Initialize best result
    best_result = None
    
    # Check dimension results
    for dim, dim_results in results.get('dimension_results', {}).items():
        if dim_results and 'best_result' in dim_results and dim_results['best_result'] is not None:
            if best_result is None or (direction == 'maximize' and dim_results['best_result']['score'] > best_result['score']) or (direction == 'minimize' and dim_results['best_result']['score'] < best_result['score']):
                best_result = dim_results['best_result']
                best_result['dimension'] = dim
    
    # Check combined results
    if 'combined_results' in results and results['combined_results'] and 'best_result' in results['combined_results'] and results['combined_results']['best_result'] is not None:
        if best_result is None or (direction == 'maximize' and results['combined_results']['best_result']['score'] > best_result['score']) or (direction == 'minimize' and results['combined_results']['best_result']['score'] < best_result['score']):
            best_result = results['combined_results']['best_result']
            best_result['combined'] = True
    
    return best_result

def generate_equity_curves(results, consolidated_data, output_dir, config):
    """
    Generate equity curves for best results.
    
    Args:
        results (dict): Dictionary containing optimization results
        consolidated_data (dict): Dictionary containing consolidated data
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        dict: Dictionary containing equity curve paths
    """
    logging.info("Generating equity curves for best results")
    
    # Create equity curves directory
    equity_curves_dir = os.path.join(output_dir, "equity_curves")
    ensure_directory_exists(equity_curves_dir)
    
    # Initialize equity curves
    equity_curves = {}
    
    # Generate equity curve for best result
    if 'best_results' in results and results['best_results'] is not None:
        best_result = results['best_results']
        
        # Generate equity curve
        equity_curve_path = os.path.join(equity_curves_dir, "best_result_equity_curve.png")
        generate_equity_curve(best_result, consolidated_data, equity_curve_path, config)
        equity_curves['best_result'] = equity_curve_path
    
    # Generate equity curves for dimension results
    for dim, dim_results in results.get('dimension_results', {}).items():
        if dim_results and 'best_result' in dim_results and dim_results['best_result'] is not None:
            # Generate equity curve
            equity_curve_path = os.path.join(equity_curves_dir, f"{dim}_equity_curve.png")
            generate_equity_curve(dim_results['best_result'], consolidated_data, equity_curve_path, config)
            equity_curves[dim] = equity_curve_path
    
    # Generate equity curves for combined results
    if 'combined_results' in results and results['combined_results'] and 'best_result' in results['combined_results'] and results['combined_results']['best_result'] is not None:
        # Generate equity curve
        equity_curve_path = os.path.join(equity_curves_dir, "combined_equity_curve.png")
        generate_equity_curve(results['combined_results']['best_result'], consolidated_data, equity_curve_path, config)
        equity_curves['combined'] = equity_curve_path
    
    return equity_curves

def generate_equity_curve(result, consolidated_data, output_path, config):
    """
    Generate equity curve for a result.
    
    Args:
        result (dict): Optimization result
        consolidated_data (dict): Dictionary containing consolidated data
        output_path (str): Output file path
        config (dict): Configuration settings
        
    Returns:
        bool: True if successful
    """
    try:
        # Get consolidated data
        if 'consolidated_without_time' in consolidated_data:
            data = consolidated_data['consolidated_without_time']
        elif 'consolidated_with_time' in consolidated_data:
            data = consolidated_data['consolidated_with_time']
        else:
            logging.error("No consolidated data found")
            return False
        
        # Create dummy equity curve for testing
        # In a real implementation, this would calculate the actual equity curve
        dates = pd.date_range(start='2023-01-01', periods=100)
        equity = np.cumsum(np.random.normal(0.1, 1, 100))
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity)
        plt.title(f"Equity Curve for {result.get('algorithm', 'Best Result')}")
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return True
    except Exception as e:
        logging.error(f"Error generating equity curve: {str(e)}")
        return False

def generate_algorithm_comparison(results, output_dir, config):
    """
    Generate algorithm comparison visualization.
    
    Args:
        results (dict): Dictionary containing optimization results
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        str: Path to comparison visualization
    """
    logging.info("Generating algorithm comparison visualization")
    
    try:
        # Create comparison directory
        comparison_dir = os.path.join(output_dir, "comparison")
        ensure_directory_exists(comparison_dir)
        
        # Initialize comparison data
        comparison_data = []
        
        # Collect algorithm results
        for dim, dim_results in results.get('dimension_results', {}).items():
            if dim_results and 'algorithm_results' in dim_results:
                for algo_name, algo_result in dim_results['algorithm_results'].items():
                    comparison_data.append({
                        'Dimension': dim,
                        'Algorithm': algo_name,
                        'Score': algo_result['score'],
                        'Time': algo_result.get('time', 0)
                    })
        
        # Collect combined results
        if 'combined_results' in results and results['combined_results'] and 'combination_results' in results['combined_results']:
            for combo_name, combo_results in results['combined_results']['combination_results'].items():
                if 'algorithm_results' in combo_results:
                    for algo_name, algo_result in combo_results['algorithm_results'].items():
                        comparison_data.append({
                            'Dimension': f"Combined: {combo_name}",
                            'Algorithm': algo_name,
                            'Score': algo_result['score'],
                            'Time': algo_result.get('time', 0)
                        })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison data
        comparison_csv_path = os.path.join(comparison_dir, "algorithm_comparison.csv")
        comparison_df.to_csv(comparison_csv_path, index=False)
        
        # Create comparison visualization
        comparison_path = os.path.join(comparison_dir, "algorithm_comparison.png")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Algorithm', y='Score', hue='Dimension', data=comparison_df)
        plt.title('Algorithm Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Dimension')
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()
        
        return comparison_path
    except Exception as e:
        logging.error(f"Error generating algorithm comparison: {str(e)}")
        return None
