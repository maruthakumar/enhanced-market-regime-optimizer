# Dimension Selection

## Overview

The Dimension Selection module is a critical component of the Enhanced Market Regime Optimizer pipeline that identifies the most relevant dimensions for strategy optimization. This step analyzes the consolidated data to determine which dimensions (DTE, Zone, market regime, day) have the strongest correlation with strategy performance, allowing for more targeted optimization.

The dimension selection process is implemented in `core/dimension_selection.py` and serves as the bridge between consolidation and optimization. It ensures that optimization efforts are focused on the dimensions that have the most significant impact on strategy performance.

## Purpose and Importance

Dimension selection is crucial for efficient optimization because:

1. **Reduced Complexity**: By focusing on the most relevant dimensions, the optimization process becomes more manageable and computationally efficient.

2. **Improved Results**: Optimizing for the dimensions that have the strongest correlation with strategy performance leads to better optimization results.

3. **Targeted Strategies**: Different strategies may perform better in different market conditions. Dimension selection helps identify these relationships.

4. **Resource Efficiency**: By eliminating dimensions with little impact on strategy performance, computational resources can be used more efficiently.

## Available Dimensions

The dimension selection process considers the following dimensions:

### 1. DTE (Days to Expiry)

DTE is a critical dimension for options strategies. Different strategies may perform better at different points in the expiry cycle.

### 2. Zone

Trading zones represent different time periods during the trading day. Different strategies may perform better in different zones.

### 3. Market Regime

Market regimes represent different market conditions. Different strategies may perform better in different market regimes.

### 4. Day

Different days of the week may exhibit different market behavior. Different strategies may perform better on different days.

## Dimension Selection Methods

The dimension selection process supports several methods for identifying the most relevant dimensions:

### 1. Correlation-Based Selection

This method calculates the correlation between each dimension and strategy PnL, selecting dimensions with the highest correlation.

```python
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
```

### 2. Feature Importance-Based Selection

This method uses machine learning techniques to calculate the importance of each dimension for predicting strategy PnL, selecting dimensions with the highest importance.

```python
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
```

### 3. All Dimensions Selection

This method simply selects all available dimensions, which can be useful for exploratory analysis or when there is no clear preference for which dimensions to include.

```python
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
```

## Combined Selections

After selecting the most relevant dimensions, the dimension selection process creates combined selections for optimization. These combined selections represent different combinations of dimensions that can be used for optimization.

```python
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
```

## Process Flow

The dimension selection process follows these steps:

1. **Identify Available Dimensions**: The process begins by identifying the available dimensions in the consolidated data.

```python
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
```

2. **Select Dimensions**: The process then selects the most relevant dimensions using one of the available methods.

```python
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
```

3. **Create Combined Selections**: The process then creates combined selections for optimization.

4. **Save Results**: Finally, the process saves the dimension selection results to files.

```python
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
        
        return True
    except Exception as e:
        logging.error(f"Error saving dimension selection results: {str(e)}")
        return False
```

## Output Format

The dimension selection process produces several output files, but the main output is the dimension selection results, which has the following format:

```json
{
    "method": "correlation",
    "dimensions": {
        "Zone": true,
        "Day": true,
        "Market regime": true,
        "Time": true,
        "DTE": true
    },
    "selections": {
        "Zone": ["Opening", "Morning", "Lunch", "Afternoon", "Closing"],
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "Market regime": ["Strong_Bullish", "Bullish", "Mild_Bullish", "Neutral", "Mild_Bearish", "Bearish", "Strong_Bearish"],
        "Time": ["09:15:00", "09:30:00", "09:45:00", "10:00:00", "..."],
        "DTE": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    },
    "selected_dimensions": ["Market regime", "DTE", "Zone", "Day", "Time"],
    "combined_selections": {
        "combinations": [
            {
                "name": "All Selected Dimensions",
                "dimensions": {
                    "Market regime": true,
                    "DTE": true,
                    "Zone": true,
                    "Day": true,
                    "Time": true
                },
                "selections": {
                    "Zone": ["Opening", "Morning", "Lunch", "Afternoon", "Closing"],
                    "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "Market regime": ["Strong_Bullish", "Bullish", "Mild_Bullish", "Neutral", "Mild_Bearish", "Bearish", "Strong_Bearish"],
                    "Time": ["09:15:00", "09:30:00", "09:45:00", "10:00:00", "..."],
                    "DTE": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                }
            },
            {
                "name": "Only Market regime",
                "dimensions": {
                    "Market regime": true,
                    "DTE": false,
                    "Zone": false,
                    "Day": false,
                    "Time": true
                },
                "selections": {
                    "Zone": ["Opening", "Morning", "Lunch", "Afternoon", "Closing"],
                    "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "Market regime": ["Strong_Bullish", "Bullish", "Mild_Bullish", "Neutral", "Mild_Bearish", "Bearish", "Strong_Bearish"],
                    "Time": ["09:15:00", "09:30:00", "09:45:00", "10:00:00", "..."],
                    "DTE": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                }
            },
            ...
        ]
    }
}
```

## Configuration Options

The dimension selection process can be configured through several parameters in the configuration dictionary:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | 'output/dimension_selection' | Directory where output files will be saved |
| `dimension_selection.method` | string | 'correlation' | Dimension selection method ('correlation', 'feature_importance', or 'all') |
| `dimension_selection.top_dimensions` | int | 5 | Maximum number of dimensions to select |
| `dimension_selection.correlation_threshold` | float | 0.1 | Minimum correlation for a dimension to be selected |
| `dimension_selection.importance_threshold` | float | 0.1 | Minimum importance for a dimension to be selected |

Example configuration:

```python
config = {
    "output": {
        "base_dir": "output"
    },
    "dimension_selection": {
        "method": "correlation",
        "top_dimensions": 5,
        "correlation_threshold": 0.1,
        "importance_threshold": 0.1
    }
}
```

## Integration with the Unified Pipeline

The dimension selection process is integrated with the unified pipeline through the `select_dimensions` function, which is called by the unified pipeline after the consolidation process.

The dimension selection results are then used by the optimization step to find the optimal parameters for each strategy based on the selected dimensions.

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Missing or incomplete consolidated data

**Symptoms**: The dimension selection process fails to process data or produces incomplete results.

**Solutions**:
- Check that the consolidated data files exist and contain all required columns.
- Ensure that the data has the correct format and column names.
- Check for missing values in the data and handle them appropriately.

#### Issue: No dimensions selected

**Symptoms**: The dimension selection process completes but does not select any dimensions.

**Solutions**:
- Check that the correlation or importance thresholds are not set too high.
- Try a different dimension selection method.
- Check that the consolidated data contains enough data for meaningful correlation or importance calculations.

#### Issue: Too many dimensions selected

**Symptoms**: The dimension selection process selects too many dimensions, making optimization computationally expensive.

**Solutions**:
- Reduce the `top_dimensions` parameter to limit the number of selected dimensions.
- Increase the correlation or importance thresholds to be more selective.
- Use a different dimension selection method that is more selective.

#### Issue: Performance issues

**Symptoms**: The dimension selection process takes too long to process data or uses too much memory.

**Solutions**:
- Reduce the amount of data being processed by filtering or sampling.
- Optimize the correlation or importance calculations by using more efficient algorithms.
- Consider using a database for large datasets.

### Logging and Debugging

The dimension selection process includes comprehensive logging to help diagnose issues. By default, logs are written to the console and can be configured to write to a file.

To enable more detailed logging, you can adjust the logging level:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dimension_selection_debug.log"),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

### Optimizing for Speed

To optimize the dimension selection process for speed, consider the following:

1. **Reduce the amount of data**: Filter or sample the data to reduce the amount of data being processed.

2. **Optimize correlation calculations**: Use more efficient algorithms for correlation calculations, such as using NumPy's vectorized operations.

3. **Limit the number of dimensions**: Reduce the number of dimensions being considered to speed up the process.

4. **Use efficient data structures**: Use memory-efficient data structures like NumPy arrays instead of Python lists where appropriate.

### Optimizing for Memory Usage

To optimize the dimension selection process for memory usage, consider the following:

1. **Process data in chunks**: Instead of loading all data at once, process it in smaller chunks.

2. **Clean up temporary data**: Remove temporary data structures when they are no longer needed.

3. **Use generators**: Use generators instead of lists for large datasets to reduce memory usage.

4. **Use efficient data types**: Use more memory-efficient data types, such as categorical data types for columns with repeated values.

## Conclusion

The dimension selection process is a critical component of the Enhanced Market Regime Optimizer pipeline that identifies the most relevant dimensions for strategy optimization. By properly configuring and using the dimension selection process, you can focus optimization efforts on the dimensions that have the most significant impact on strategy performance, leading to more efficient and effective optimization.

For more information on other components of the pipeline, refer to the following documentation:

- [Unified Market Regime Pipeline](Unified_Market_Regime_Pipeline.md)
- [Market Regime Formation](Market_Regime_Formation.md)
- [Consolidation](Consolidation.md)
- [Results Visualization](Results_Visualization.md)
- [PostgreSQL Integration](PostgreSQL_Integration.md)
- [GDFL Live Data Feed](GDFL_Live_Data_Feed.md)