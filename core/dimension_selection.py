"""
Dimension selection module to identify the most important dimensions for optimization.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

from utils.helpers import save_to_csv, ensure_directory_exists

def select_dimensions(consolidated_data, config):
    """
    Fifth step: Select dimensions for optimization.
    
    Args:
        consolidated_data (dict): Dictionary containing consolidated data
        config (dict): Configuration settings
        
    Returns:
        dict: Dictionary containing dimension selections
    """
    logging.info("Starting dimension selection process")
    
    # Create output directory
    output_dir = os.path.join(config["output"].get("base_dir", "output"), "dimension_selection")
    ensure_directory_exists(output_dir)
    
    # Get dimension selection configuration
    dimension_config = config.get("dimension_selection", {})
    
    # Get method
    method = dimension_config.get("method", "correlation").lower()
    logging.info(f"Using dimension selection method: {method}")
    
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
        'method': method,
        'dimensions': {},
        'selections': {},
        'combined_selections': {
            'combinations': []
        },
        'selected_dimensions': []
    }
    
    # Identify available dimensions
    available_dimensions = identify_available_dimensions(data)
    logging.info(f"Available dimensions: {available_dimensions}")
    
    # Select dimensions based on method
    if method == "correlation":
        dimension_selections = select_dimensions_by_correlation(data, available_dimensions, dimension_config)
    elif method == "feature_importance":
        dimension_selections = select_dimensions_by_feature_importance(data, available_dimensions, dimension_config)
    elif method == "all":
        dimension_selections = select_all_dimensions(data, available_dimensions)
    else:
        logging.warning(f"Unknown dimension selection method: {method}, using correlation")
        dimension_selections = select_dimensions_by_correlation(data, available_dimensions, dimension_config)
    
    # Update results
    results.update(dimension_selections)
    
    # Create combined selections
    combined_selections = create_combined_selections(dimension_selections, dimension_config)
    results['combined_selections'] = combined_selections
    
    # Save results
    save_dimension_selection_results(results, output_dir)
    
    logging.info("Dimension selection process completed")
    
    return results

def identify_available_dimensions(data):
    """
    Identify available dimensions in the data.
    
    Args:
        data (DataFrame): Consolidated data
        
    Returns:
        dict: Dictionary of available dimensions
    """
    available_dimensions = {}
    
    # Check for Zone dimension
    if 'Zone' in data.columns:
        available_dimensions['Zone'] = data['Zone'].unique().tolist()
    
    # Check for Day dimension
    if 'Day' in data.columns:
        available_dimensions['Day'] = data['Day'].unique().tolist()
    
    # Check for Market regime dimension
    if 'Market regime' in data.columns:
        available_dimensions['Market regime'] = data['Market regime'].unique().tolist()
    
    # Check for Time dimension
    if 'Time' in data.columns:
        available_dimensions['Time'] = data['Time'].unique().tolist()
    elif 'Time_str' in data.columns:
        available_dimensions['Time'] = data['Time_str'].unique().tolist()
    
    # Check for DTE dimension
    if 'DTE' in data.columns:
        available_dimensions['DTE'] = data['DTE'].unique().tolist()
    
    return available_dimensions

def select_dimensions_by_correlation(data, available_dimensions, dimension_config):
    """
    Select dimensions by correlation with PnL.
    
    Args:
        data (DataFrame): Consolidated data
        available_dimensions (dict): Dictionary of available dimensions
        dimension_config (dict): Dimension selection configuration
        
    Returns:
        dict: Dictionary containing dimension selections
    """
    logging.info("Selecting dimensions by correlation")
    
    # Initialize results
    results = {
        'dimensions': {},
        'selections': {},
        'selected_dimensions': []
    }
    
    # Get strategy columns (ending with _PnL)
    strategy_columns = [col for col in data.columns if col.endswith('_PnL')]
    
    # Calculate correlation for each dimension
    dimension_correlations = {}
    
    for dim, values in available_dimensions.items():
        # Skip dimensions that can't be correlated (like Time)
        if dim in ['Time']:
            results['dimensions'][dim] = True
            results['selections'][dim] = values
            results['selected_dimensions'].append(dim)
            continue
        
        # Calculate correlation for this dimension
        dim_correlation = calculate_dimension_correlation(data, dim, strategy_columns)
        
        if dim_correlation is not None:
            dimension_correlations[dim] = dim_correlation
            
            # Determine if dimension should be included
            include_dimension = dim_correlation['abs_mean_correlation'] >= 0.1
            results['dimensions'][dim] = include_dimension
            
            if include_dimension:
                results['selections'][dim] = values
                results['selected_dimensions'].append(dim)
    
    # Sort selected dimensions by correlation
    results['selected_dimensions'] = sorted(
        results['selected_dimensions'],
        key=lambda dim: dimension_correlations.get(dim, {}).get('abs_mean_correlation', 0),
        reverse=True
    )
    
    # Limit to top dimensions if specified
    top_dimensions = int(dimension_config.get("top_dimensions", 5))
    if top_dimensions > 0 and len(results['selected_dimensions']) > top_dimensions:
        results['selected_dimensions'] = results['selected_dimensions'][:top_dimensions]
        
        # Update dimensions dictionary
        for dim in list(results['dimensions'].keys()):
            if dim not in results['selected_dimensions'] and dim != 'Time':
                results['dimensions'][dim] = False
    
    return results

def calculate_dimension_correlation(data, dimension, strategy_columns):
    """
    Calculate correlation between a dimension and strategy PnL.
    
    Args:
        data (DataFrame): Consolidated data
        dimension (str): Dimension name
        strategy_columns (list): List of strategy columns
        
    Returns:
        dict: Dictionary containing correlation results
    """
    try:
        # Create dummy variables for the dimension
        if dimension in data.columns:
            dummies = pd.get_dummies(data[dimension], prefix=dimension)
            
            # Calculate correlation with each strategy
            correlations = {}
            for strategy in strategy_columns:
                strategy_correlations = {}
                for dummy in dummies.columns:
                    corr = dummies[dummy].corr(data[strategy])
                    if not np.isnan(corr):
                        strategy_correlations[dummy] = corr
                
                if strategy_correlations:
                    correlations[strategy] = strategy_correlations
            
            # Calculate mean correlation
            all_correlations = []
            for strategy, strategy_correlations in correlations.items():
                all_correlations.extend(list(strategy_correlations.values()))
            
            if all_correlations:
                mean_correlation = np.mean(all_correlations)
                abs_mean_correlation = np.mean([abs(corr) for corr in all_correlations])
                
                return {
                    'correlations': correlations,
                    'mean_correlation': mean_correlation,
                    'abs_mean_correlation': abs_mean_correlation
                }
    
    except Exception as e:
        logging.error(f"Error calculating correlation for dimension {dimension}: {str(e)}")
    
    return None

def select_dimensions_by_feature_importance(data, available_dimensions, dimension_config):
    """
    Select dimensions by feature importance.
    
    Args:
        data (DataFrame): Consolidated data
        available_dimensions (dict): Dictionary of available dimensions
        dimension_config (dict): Dimension selection configuration
        
    Returns:
        dict: Dictionary containing dimension selections
    """
    logging.info("Selecting dimensions by feature importance")
    
    # Initialize results
    results = {
        'dimensions': {},
        'selections': {},
        'selected_dimensions': []
    }
    
    # Get strategy columns (ending with _PnL)
    strategy_columns = [col for col in data.columns if col.endswith('_PnL')]
    
    # Calculate feature importance for each dimension
    dimension_importance = {}
    
    for dim, values in available_dimensions.items():
        # Skip dimensions that can't be evaluated (like Time)
        if dim in ['Time']:
            results['dimensions'][dim] = True
            results['selections'][dim] = values
            results['selected_dimensions'].append(dim)
            continue
        
        # Calculate feature importance for this dimension
        dim_importance = calculate_dimension_importance(data, dim, strategy_columns)
        
        if dim_importance is not None:
            dimension_importance[dim] = dim_importance
            
            # Determine if dimension should be included
            include_dimension = dim_importance['mean_importance'] >= 0.1
            results['dimensions'][dim] = include_dimension
            
            if include_dimension:
                results['selections'][dim] = values
                results['selected_dimensions'].append(dim)
    
    # Sort selected dimensions by importance
    results['selected_dimensions'] = sorted(
        results['selected_dimensions'],
        key=lambda dim: dimension_importance.get(dim, {}).get('mean_importance', 0),
        reverse=True
    )
    
    # Limit to top dimensions if specified
    top_dimensions = int(dimension_config.get("top_dimensions", 5))
    if top_dimensions > 0 and len(results['selected_dimensions']) > top_dimensions:
        results['selected_dimensions'] = results['selected_dimensions'][:top_dimensions]
        
        # Update dimensions dictionary
        for dim in list(results['dimensions'].keys()):
            if dim not in results['selected_dimensions'] and dim != 'Time':
                results['dimensions'][dim] = False
    
    return results

def calculate_dimension_importance(data, dimension, strategy_columns):
    """
    Calculate feature importance for a dimension.
    
    Args:
        data (DataFrame): Consolidated data
        dimension (str): Dimension name
        strategy_columns (list): List of strategy columns
        
    Returns:
        dict: Dictionary containing importance results
    """
    try:
        # Create dummy variables for the dimension
        if dimension in data.columns:
            dummies = pd.get_dummies(data[dimension], prefix=dimension)
            
            # Calculate importance for each strategy
            importance = {}
            for strategy in strategy_columns:
                # Prepare data
                X = dummies.values
                y = data[strategy].values
                
                # Skip if not enough samples
                if len(y) < 10:
                    continue
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Select features
                selector = SelectKBest(score_func=f_regression, k='all')
                selector.fit(X_scaled, y)
                
                # Get scores
                scores = selector.scores_
                
                # Handle NaN scores
                scores = np.nan_to_num(scores)
                
                # Normalize scores
                if np.sum(scores) > 0:
                    scores = scores / np.sum(scores)
                
                # Store importance
                importance[strategy] = {
                    'scores': scores.tolist(),
                    'mean_score': np.mean(scores)
                }
            
            # Calculate mean importance
            if importance:
                mean_importance = np.mean([imp['mean_score'] for imp in importance.values()])
                
                return {
                    'importance': importance,
                    'mean_importance': mean_importance
                }
    
    except Exception as e:
        logging.error(f"Error calculating importance for dimension {dimension}: {str(e)}")
    
    return None

def select_all_dimensions(data, available_dimensions):
    """
    Select all available dimensions.
    
    Args:
        data (DataFrame): Consolidated data
        available_dimensions (dict): Dictionary of available dimensions
        
    Returns:
        dict: Dictionary containing dimension selections
    """
    logging.info("Selecting all dimensions")
    
    # Initialize results
    results = {
        'dimensions': {},
        'selections': {},
        'selected_dimensions': []
    }
    
    # Include all dimensions
    for dim, values in available_dimensions.items():
        results['dimensions'][dim] = True
        results['selections'][dim] = values
        results['selected_dimensions'].append(dim)
    
    return results

def create_combined_selections(dimension_selections, dimension_config):
    """
    Create combined selections for optimization.
    
    Args:
        dimension_selections (dict): Dictionary containing dimension selections
        dimension_config (dict): Dimension selection configuration
        
    Returns:
        dict: Dictionary containing combined selections
    """
    logging.info("Creating combined selections")
    
    # Initialize combined selections
    combined_selections = {
        'combinations': []
    }
    
    # Get selected dimensions
    selected_dimensions = dimension_selections.get('selected_dimensions', [])
    
    # Create default combination with all selected dimensions
    default_combination = {
        'name': 'All Selected Dimensions',
        'dimensions': {dim: True for dim in selected_dimensions},
        'selections': dimension_selections.get('selections', {})
    }
    combined_selections['combinations'].append(default_combination)
    
    # Create individual combinations for each dimension
    for dim in selected_dimensions:
        if dim != 'Time':  # Time is always included
            combination = {
                'name': f'Only {dim}',
                'dimensions': {d: (d == dim or d == 'Time') for d in selected_dimensions},
                'selections': dimension_selections.get('selections', {})
            }
            combined_selections['combinations'].append(combination)
    
    # Create combination without Time
    if 'Time' in selected_dimensions and len(selected_dimensions) > 1:
        combination = {
            'name': 'Without Time',
            'dimensions': {dim: (dim != 'Time') for dim in selected_dimensions},
            'selections': dimension_selections.get('selections', {})
        }
        combined_selections['combinations'].append(combination)
    
    # Create pairwise combinations
    if len(selected_dimensions) > 2:
        for i, dim1 in enumerate(selected_dimensions):
            if dim1 == 'Time':
                continue
            
            for dim2 in selected_dimensions[i+1:]:
                if dim2 == 'Time':
                    continue
                
                combination = {
                    'name': f'{dim1} and {dim2}',
                    'dimensions': {dim: (dim == dim1 or dim == dim2 or dim == 'Time') for dim in selected_dimensions},
                    'selections': dimension_selections.get('selections', {})
                }
                combined_selections['combinations'].append(combination)
    
    return combined_selections

def save_dimension_selection_results(results, output_dir):
    """
    Save dimension selection results.
    
    Args:
        results (dict): Dictionary containing dimension selections
        output_dir (str): Output directory
        
    Returns:
        bool: True if successful
    """
    try:
        # Create output directory
        ensure_directory_exists(output_dir)
        
        # Save selected dimensions
        selected_dimensions_path = os.path.join(output_dir, "selected_dimensions.txt")
        with open(selected_dimensions_path, 'w') as f:
            f.write("Selected Dimensions:\n")
            for dim in results['selected_dimensions']:
                f.write(f"- {dim}\n")
        
        # Save dimension selections
        for dim, values in results.get('selections', {}).items():
            dim_path = os.path.join(output_dir, f"{dim}_selections.txt")
            with open(dim_path, 'w') as f:
                f.write(f"{dim} Selections:\n")
                for value in values:
                    f.write(f"- {value}\n")
        
        # Save combined selections
        combined_path = os.path.join(output_dir, "combined_selections.txt")
        with open(combined_path, 'w') as f:
            f.write("Combined Selections:\n")
            for combination in results.get('combined_selections', {}).get('combinations', []):
                f.write(f"\n{combination['name']}:\n")
                for dim, include in combination.get('dimensions', {}).items():
                    f.write(f"- {dim}: {'Included' if include else 'Excluded'}\n")
        
        return True
    except Exception as e:
        logging.error(f"Error saving dimension selection results: {str(e)}")
        return False
