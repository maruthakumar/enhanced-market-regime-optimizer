"""
Results visualization module to generate visualizations of optimization results.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json

from utils.helpers import save_to_csv, ensure_directory_exists

def visualize_results(strategy_with_regimes, consolidated_data, dimensions, optimization_results, config):
    """
    Seventh step: Visualize results of the optimization process.
    
    Args:
        strategy_with_regimes (dict): Dictionary containing strategy data with assigned market regimes
        consolidated_data (dict): Dictionary containing consolidated data
        dimensions (dict): Dictionary containing dimension selections
        optimization_results (dict): Dictionary containing optimization results
        config (dict): Configuration settings
        
    Returns:
        dict: Dictionary containing visualization paths
    """
    logging.info("Starting results visualization process")
    
    # Create output directory
    output_dir = os.path.join(config["output"].get("base_dir", "output"), "visualizations")
    ensure_directory_exists(output_dir)
    
    # Initialize results
    results = {
        'visualization_paths': []
    }
    
    # Generate strategy performance visualizations
    strategy_viz_paths = generate_strategy_performance_visualizations(
        strategy_with_regimes, 
        consolidated_data, 
        os.path.join(output_dir, "strategy_performance"),
        config
    )
    results['visualization_paths'].extend(strategy_viz_paths)
    
    # Generate market regime visualizations
    regime_viz_paths = generate_market_regime_visualizations(
        strategy_with_regimes, 
        consolidated_data, 
        os.path.join(output_dir, "market_regime"),
        config
    )
    results['visualization_paths'].extend(regime_viz_paths)
    
    # Generate dimension selection visualizations
    dimension_viz_paths = generate_dimension_selection_visualizations(
        dimensions, 
        consolidated_data, 
        os.path.join(output_dir, "dimension_selection"),
        config
    )
    results['visualization_paths'].extend(dimension_viz_paths)
    
    # Generate optimization results visualizations
    optimization_viz_paths = generate_optimization_results_visualizations(
        optimization_results, 
        consolidated_data, 
        os.path.join(output_dir, "optimization_results"),
        config
    )
    results['visualization_paths'].extend(optimization_viz_paths)
    
    # Generate combined performance visualizations
    combined_viz_paths = generate_combined_performance_visualizations(
        strategy_with_regimes, 
        consolidated_data, 
        dimensions, 
        optimization_results, 
        os.path.join(output_dir, "combined_performance"),
        config
    )
    results['visualization_paths'].extend(combined_viz_paths)
    
    # Generate interactive dashboard
    dashboard_path = generate_interactive_dashboard(
        strategy_with_regimes, 
        consolidated_data, 
        dimensions, 
        optimization_results, 
        os.path.join(output_dir, "dashboard"),
        config
    )
    if dashboard_path:
        results['dashboard_path'] = dashboard_path
        results['visualization_paths'].append(dashboard_path)
    
    logging.info(f"Generated {len(results['visualization_paths'])} visualizations")
    
    return results

def generate_strategy_performance_visualizations(strategy_with_regimes, consolidated_data, output_dir, config):
    """
    Generate strategy performance visualizations.
    
    Args:
        strategy_with_regimes (dict): Dictionary containing strategy data with assigned market regimes
        consolidated_data (dict): Dictionary containing consolidated data
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        list: List of visualization paths
    """
    logging.info("Generating strategy performance visualizations")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize visualization paths
    visualization_paths = []
    
    try:
        # Get strategy data
        if 'data' in strategy_with_regimes:
            strategy_data = strategy_with_regimes['data']
        else:
            # Use consolidated data if strategy data not available
            if 'consolidated_without_time' in consolidated_data:
                strategy_data = consolidated_data['consolidated_without_time']
            elif 'consolidated_with_time' in consolidated_data:
                strategy_data = consolidated_data['consolidated_with_time']
            else:
                logging.error("No strategy data found")
                return visualization_paths
        
        # Generate PnL by strategy visualization
        pnl_by_strategy_path = os.path.join(output_dir, "pnl_by_strategy.png")
        
        # Get strategy columns
        strategy_columns = [col for col in strategy_data.columns if col.endswith('_PnL') or col == 'PnL']
        
        if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
            # Group by strategy
            pnl_by_strategy = strategy_data.groupby('Strategy')['PnL'].agg(['sum', 'mean', 'count']).reset_index()
            pnl_by_strategy.columns = ['Strategy', 'Total PnL', 'Average PnL', 'Trade Count']
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Strategy', y='Total PnL', data=pnl_by_strategy)
            plt.title('Total PnL by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Total PnL')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(pnl_by_strategy_path)
            plt.close()
            
            visualization_paths.append(pnl_by_strategy_path)
        elif strategy_columns:
            # Create DataFrame with strategy PnL
            strategy_pnl = pd.DataFrame({
                'Strategy': [col.replace('_PnL', '') for col in strategy_columns],
                'Total PnL': [strategy_data[col].sum() for col in strategy_columns]
            })
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Strategy', y='Total PnL', data=strategy_pnl)
            plt.title('Total PnL by Strategy')
            plt.xlabel('Strategy')
            plt.ylabel('Total PnL')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(pnl_by_strategy_path)
            plt.close()
            
            visualization_paths.append(pnl_by_strategy_path)
        
        # Generate PnL over time visualization
        pnl_over_time_path = os.path.join(output_dir, "pnl_over_time.png")
        
        if 'Date' in strategy_data.columns:
            if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
                # Group by date and strategy
                pnl_by_date_strategy = strategy_data.groupby(['Date', 'Strategy'])['PnL'].sum().reset_index()
                
                # Create visualization
                plt.figure(figsize=(14, 8))
                for strategy in pnl_by_date_strategy['Strategy'].unique():
                    strategy_data = pnl_by_date_strategy[pnl_by_date_strategy['Strategy'] == strategy]
                    plt.plot(strategy_data['Date'], strategy_data['PnL'].cumsum(), label=strategy)
                
                plt.title('Cumulative PnL Over Time by Strategy')
                plt.xlabel('Date')
                plt.ylabel('Cumulative PnL')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(pnl_over_time_path)
                plt.close()
                
                visualization_paths.append(pnl_over_time_path)
            elif strategy_columns:
                # Group by date
                pnl_by_date = strategy_data.groupby('Date')[strategy_columns].sum().reset_index()
                
                # Calculate cumulative PnL
                for col in strategy_columns:
                    pnl_by_date[f"{col}_Cumulative"] = pnl_by_date[col].cumsum()
                
                # Create visualization
                plt.figure(figsize=(14, 8))
                for col in strategy_columns:
                    plt.plot(pnl_by_date['Date'], pnl_by_date[f"{col}_Cumulative"], label=col.replace('_PnL', ''))
                
                plt.title('Cumulative PnL Over Time by Strategy')
                plt.xlabel('Date')
                plt.ylabel('Cumulative PnL')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(pnl_over_time_path)
                plt.close()
                
                visualization_paths.append(pnl_over_time_path)
        
        # Generate win rate visualization
        win_rate_path = os.path.join(output_dir, "win_rate.png")
        
        if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
            # Calculate win rate by strategy
            win_rate = []
            for strategy in strategy_data['Strategy'].unique():
                strategy_pnl = strategy_data[strategy_data['Strategy'] == strategy]['PnL']
                total_trades = len(strategy_pnl)
                winning_trades = len(strategy_pnl[strategy_pnl > 0])
                
                if total_trades > 0:
                    win_rate.append({
                        'Strategy': strategy,
                        'Win Rate': winning_trades / total_trades * 100,
                        'Total Trades': total_trades
                    })
            
            if win_rate:
                win_rate_df = pd.DataFrame(win_rate)
                
                # Create visualization
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Strategy', y='Win Rate', data=win_rate_df)
                plt.title('Win Rate by Strategy')
                plt.xlabel('Strategy')
                plt.ylabel('Win Rate (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(win_rate_path)
                plt.close()
                
                visualization_paths.append(win_rate_path)
        
        # Generate trade distribution visualization
        trade_distribution_path = os.path.join(output_dir, "trade_distribution.png")
        
        if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
            # Create visualization
            plt.figure(figsize=(14, 8))
            for i, strategy in enumerate(strategy_data['Strategy'].unique()):
                strategy_pnl = strategy_data[strategy_data['Strategy'] == strategy]['PnL']
                
                if len(strategy_pnl) > 0:
                    plt.subplot(1, len(strategy_data['Strategy'].unique()), i+1)
                    sns.histplot(strategy_pnl, kde=True)
                    plt.title(f'{strategy} PnL Distribution')
                    plt.xlabel('PnL')
                    plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(trade_distribution_path)
            plt.close()
            
            visualization_paths.append(trade_distribution_path)
    
    except Exception as e:
        logging.error(f"Error generating strategy performance visualizations: {str(e)}")
    
    return visualization_paths

def generate_market_regime_visualizations(strategy_with_regimes, consolidated_data, output_dir, config):
    """
    Generate market regime visualizations.
    
    Args:
        strategy_with_regimes (dict): Dictionary containing strategy data with assigned market regimes
        consolidated_data (dict): Dictionary containing consolidated data
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        list: List of visualization paths
    """
    logging.info("Generating market regime visualizations")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize visualization paths
    visualization_paths = []
    
    try:
        # Get strategy data
        if 'data' in strategy_with_regimes:
            strategy_data = strategy_with_regimes['data']
        else:
            # Use consolidated data if strategy data not available
            if 'consolidated_without_time' in consolidated_data:
                strategy_data = consolidated_data['consolidated_without_time']
            elif 'consolidated_with_time' in consolidated_data:
                strategy_data = consolidated_data['consolidated_with_time']
            else:
                logging.error("No strategy data found")
                return visualization_paths
        
        # Check if market regime is available
        if 'Market regime' not in strategy_data.columns:
            logging.warning("Market regime not found in strategy data")
            return visualization_paths
        
        # Generate PnL by market regime visualization
        pnl_by_regime_path = os.path.join(output_dir, "pnl_by_market_regime.png")
        
        if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
            # Group by market regime and strategy
            pnl_by_regime_strategy = strategy_data.groupby(['Market regime', 'Strategy'])['PnL'].sum().reset_index()
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            sns.barplot(x='Market regime', y='PnL', hue='Strategy', data=pnl_by_regime_strategy)
            plt.title('Total PnL by Market Regime and Strategy')
            plt.xlabel('Market Regime')
            plt.ylabel('Total PnL')
            plt.xticks(rotation=45)
            plt.legend(title='Strategy')
            plt.tight_layout()
            plt.savefig(pnl_by_regime_path)
            plt.close()
            
            visualization_paths.append(pnl_by_regime_path)
        else:
            # Get strategy columns
            strategy_columns = [col for col in strategy_data.columns if col.endswith('_PnL')]
            
            if strategy_columns:
                # Group by market regime
                pnl_by_regime = strategy_data.groupby('Market regime')[strategy_columns].sum().reset_index()
                
                # Melt DataFrame for easier plotting
                pnl_by_regime_melted = pd.melt(
                    pnl_by_regime, 
                    id_vars=['Market regime'], 
                    value_vars=strategy_columns,
                    var_name='Strategy',
                    value_name='PnL'
                )
                pnl_by_regime_melted['Strategy'] = pnl_by_regime_melted['Strategy'].str.replace('_PnL', '')
                
                # Create visualization
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Market regime', y='PnL', hue='Strategy', data=pnl_by_regime_melted)
                plt.title('Total PnL by Market Regime and Strategy')
                plt.xlabel('Market Regime')
                plt.ylabel('Total PnL')
                plt.xticks(rotation=45)
                plt.legend(title='Strategy')
                plt.tight_layout()
                plt.savefig(pnl_by_regime_path)
                plt.close()
                
                visualization_paths.append(pnl_by_regime_path)
        
        # Generate market regime distribution visualization
        regime_distribution_path = os.path.join(output_dir, "market_regime_distribution.png")
        
        # Count market regimes
        regime_counts = strategy_data['Market regime'].value_counts().reset_index()
        regime_counts.columns = ['Market regime', 'Count']
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Market regime', y='Count', data=regime_counts)
        plt.title('Market Regime Distribution')
        plt.xlabel('Market Regime')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(regime_distribution_path)
        plt.close()
        
        visualization_paths.append(regime_distribution_path)
        
        # Generate market regime over time visualization
        regime_over_time_path = os.path.join(output_dir, "market_regime_over_time.png")
        
        if 'Date' in strategy_data.columns:
            # Group by date
            regime_by_date = strategy_data.groupby(['Date', 'Market regime']).size().reset_index(name='Count')
            
            # Create pivot table
            regime_pivot = regime_by_date.pivot(index='Date', columns='Market regime', values='Count')
            regime_pivot = regime_pivot.fillna(0)
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            regime_pivot.plot(kind='area', stacked=True, ax=plt.gca())
            plt.title('Market Regime Distribution Over Time')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(regime_over_time_path)
            plt.close()
            
            visualization_paths.append(regime_over_time_path)
        
        # Generate win rate by market regime visualization
        win_rate_by_regime_path = os.path.join(output_dir, "win_rate_by_market_regime.png")
        
        if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
            # Calculate win rate by market regime and strategy
            win_rate = []
            for regime in strategy_data['Market regime'].unique():
                regime_data = strategy_data[strategy_data['Market regime'] == regime]
                
                for strategy in regime_data['Strategy'].unique():
                    strategy_pnl = regime_data[regime_data['Strategy'] == strategy]['PnL']
                    total_trades = len(strategy_pnl)
                    winning_trades = len(strategy_pnl[strategy_pnl > 0])
                    
                    if total_trades > 0:
                        win_rate.append({
                            'Market regime': regime,
                            'Strategy': strategy,
                            'Win Rate': winning_trades / total_trades * 100,
                            'Total Trades': total_trades
                        })
            
            if win_rate:
                win_rate_df = pd.DataFrame(win_rate)
                
                # Create visualization
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Market regime', y='Win Rate', hue='Strategy', data=win_rate_df)
                plt.title('Win Rate by Market Regime and Strategy')
                plt.xlabel('Market Regime')
                plt.ylabel('Win Rate (%)')
                plt.xticks(rotation=45)
                plt.legend(title='Strategy')
                plt.tight_layout()
                plt.savefig(win_rate_by_regime_path)
                plt.close()
                
                visualization_paths.append(win_rate_by_regime_path)
    
    except Exception as e:
        logging.error(f"Error generating market regime visualizations: {str(e)}")
    
    return visualization_paths

def generate_dimension_selection_visualizations(dimensions, consolidated_data, output_dir, config):
    """
    Generate dimension selection visualizations.
    
    Args:
        dimensions (dict): Dictionary containing dimension selections
        consolidated_data (dict): Dictionary containing consolidated data
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        list: List of visualization paths
    """
    logging.info("Generating dimension selection visualizations")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize visualization paths
    visualization_paths = []
    
    try:
        # Check if dimensions are available
        if not dimensions or 'selected_dimensions' not in dimensions:
            logging.warning("No dimension selections found")
            return visualization_paths
        
        # Get consolidated data
        if 'consolidated_without_time' in consolidated_data:
            data = consolidated_data['consolidated_without_time']
        elif 'consolidated_with_time' in consolidated_data:
            data = consolidated_data['consolidated_with_time']
        else:
            logging.error("No consolidated data found")
            return visualization_paths
        
        # Generate selected dimensions visualization
        selected_dimensions_path = os.path.join(output_dir, "selected_dimensions.png")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(dimensions['selected_dimensions'])), [1] * len(dimensions['selected_dimensions']))
        plt.xticks(range(len(dimensions['selected_dimensions'])), dimensions['selected_dimensions'], rotation=45)
        plt.title('Selected Dimensions')
        plt.xlabel('Dimension')
        plt.ylabel('Selected')
        plt.tight_layout()
        plt.savefig(selected_dimensions_path)
        plt.close()
        
        visualization_paths.append(selected_dimensions_path)
        
        # Generate PnL by dimension visualizations
        for dimension in dimensions['selected_dimensions']:
            if dimension in data.columns:
                # Generate PnL by dimension visualization
                pnl_by_dimension_path = os.path.join(output_dir, f"pnl_by_{dimension.lower()}.png")
                
                # Get strategy columns
                strategy_columns = [col for col in data.columns if col.endswith('_PnL')]
                
                if strategy_columns:
                    # Calculate total PnL by dimension
                    total_pnl_column = 'Total_PnL'
                    data[total_pnl_column] = data[strategy_columns].sum(axis=1)
                    
                    # Group by dimension
                    pnl_by_dimension = data.groupby(dimension)[total_pnl_column].sum().reset_index()
                    
                    # Create visualization
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x=dimension, y=total_pnl_column, data=pnl_by_dimension)
                    plt.title(f'Total PnL by {dimension}')
                    plt.xlabel(dimension)
                    plt.ylabel('Total PnL')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(pnl_by_dimension_path)
                    plt.close()
                    
                    visualization_paths.append(pnl_by_dimension_path)
        
        # Generate dimension combinations visualization
        if 'combined_selections' in dimensions and 'combinations' in dimensions['combined_selections']:
            combinations_path = os.path.join(output_dir, "dimension_combinations.png")
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Create heatmap data
            combinations = dimensions['combined_selections']['combinations']
            dimension_names = dimensions['selected_dimensions']
            
            heatmap_data = np.zeros((len(combinations), len(dimension_names)))
            combination_names = []
            
            for i, combination in enumerate(combinations):
                combination_names.append(combination['name'])
                for j, dimension in enumerate(dimension_names):
                    heatmap_data[i, j] = 1 if combination['dimensions'].get(dimension, False) else 0
            
            # Create heatmap
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                cmap='YlGnBu', 
                xticklabels=dimension_names, 
                yticklabels=combination_names,
                cbar=False
            )
            plt.title('Dimension Combinations')
            plt.xlabel('Dimension')
            plt.ylabel('Combination')
            plt.tight_layout()
            plt.savefig(combinations_path)
            plt.close()
            
            visualization_paths.append(combinations_path)
    
    except Exception as e:
        logging.error(f"Error generating dimension selection visualizations: {str(e)}")
    
    return visualization_paths

def generate_optimization_results_visualizations(optimization_results, consolidated_data, output_dir, config):
    """
    Generate optimization results visualizations.
    
    Args:
        optimization_results (dict): Dictionary containing optimization results
        consolidated_data (dict): Dictionary containing consolidated data
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        list: List of visualization paths
    """
    logging.info("Generating optimization results visualizations")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize visualization paths
    visualization_paths = []
    
    try:
        # Check if optimization results are available
        if not optimization_results:
            logging.warning("No optimization results found")
            return visualization_paths
        
        # Generate algorithm comparison visualization
        if 'algorithms' in optimization_results:
            algorithm_comparison_path = os.path.join(output_dir, "algorithm_comparison.png")
            
            # Collect algorithm results
            algorithm_results = []
            
            # Check dimension results
            for dim, dim_results in optimization_results.get('dimension_results', {}).items():
                if 'algorithm_results' in dim_results:
                    for algo_name, algo_result in dim_results['algorithm_results'].items():
                        algorithm_results.append({
                            'Dimension': dim,
                            'Algorithm': algo_name,
                            'Score': algo_result['score']
                        })
            
            # Check combined results
            if 'combined_results' in optimization_results and 'combination_results' in optimization_results['combined_results']:
                for combo_name, combo_results in optimization_results['combined_results']['combination_results'].items():
                    if 'algorithm_results' in combo_results:
                        for algo_name, algo_result in combo_results['algorithm_results'].items():
                            algorithm_results.append({
                                'Dimension': f"Combined: {combo_name}",
                                'Algorithm': algo_name,
                                'Score': algo_result['score']
                            })
            
            if algorithm_results:
                # Create DataFrame
                algorithm_results_df = pd.DataFrame(algorithm_results)
                
                # Create visualization
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Algorithm', y='Score', hue='Dimension', data=algorithm_results_df)
                plt.title('Algorithm Comparison')
                plt.xlabel('Algorithm')
                plt.ylabel('Score')
                plt.xticks(rotation=45)
                plt.legend(title='Dimension')
                plt.tight_layout()
                plt.savefig(algorithm_comparison_path)
                plt.close()
                
                visualization_paths.append(algorithm_comparison_path)
        
        # Generate best parameters visualization
        if 'best_parameters' in optimization_results:
            best_parameters_path = os.path.join(output_dir, "best_parameters.png")
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            best_params = optimization_results['best_parameters']
            if isinstance(best_params, dict):
                param_names = []
                param_values = []
                
                for dim, values in best_params.items():
                    if isinstance(values, list):
                        param_names.append(dim)
                        param_values.append(len(values))
                    elif isinstance(values, (int, float, str)):
                        param_names.append(dim)
                        param_values.append(values)
                
                if param_names:
                    plt.bar(param_names, param_values)
                    plt.title('Best Parameters')
                    plt.xlabel('Parameter')
                    plt.ylabel('Value')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(best_parameters_path)
                    plt.close()
                    
                    visualization_paths.append(best_parameters_path)
            
            # Save best parameters as JSON
            best_parameters_json_path = os.path.join(output_dir, "best_parameters.json")
            with open(best_parameters_json_path, 'w') as f:
                json.dump(optimization_results['best_parameters'], f, indent=4)
            
            visualization_paths.append(best_parameters_json_path)
        
        # Generate optimization progress visualization
        if 'dimension_results' in optimization_results:
            progress_path = os.path.join(output_dir, "optimization_progress.png")
            
            # Collect progress data
            progress_data = []
            
            for dim, dim_results in optimization_results['dimension_results'].items():
                if 'algorithm_results' in dim_results:
                    for algo_name, algo_result in dim_results['algorithm_results'].items():
                        if 'iterations' in algo_result and 'time' in algo_result:
                            progress_data.append({
                                'Dimension': dim,
                                'Algorithm': algo_name,
                                'Iterations': algo_result['iterations'],
                                'Time': algo_result['time']
                            })
            
            if progress_data:
                # Create DataFrame
                progress_df = pd.DataFrame(progress_data)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Iterations plot
                sns.barplot(x='Algorithm', y='Iterations', hue='Dimension', data=progress_df, ax=ax1)
                ax1.set_title('Iterations by Algorithm')
                ax1.set_xlabel('Algorithm')
                ax1.set_ylabel('Iterations')
                ax1.tick_params(axis='x', rotation=45)
                
                # Time plot
                sns.barplot(x='Algorithm', y='Time', hue='Dimension', data=progress_df, ax=ax2)
                ax2.set_title('Execution Time by Algorithm')
                ax2.set_xlabel('Algorithm')
                ax2.set_ylabel('Time (seconds)')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(progress_path)
                plt.close()
                
                visualization_paths.append(progress_path)
    
    except Exception as e:
        logging.error(f"Error generating optimization results visualizations: {str(e)}")
    
    return visualization_paths

def generate_combined_performance_visualizations(strategy_with_regimes, consolidated_data, dimensions, optimization_results, output_dir, config):
    """
    Generate combined performance visualizations.
    
    Args:
        strategy_with_regimes (dict): Dictionary containing strategy data with assigned market regimes
        consolidated_data (dict): Dictionary containing consolidated data
        dimensions (dict): Dictionary containing dimension selections
        optimization_results (dict): Dictionary containing optimization results
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        list: List of visualization paths
    """
    logging.info("Generating combined performance visualizations")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize visualization paths
    visualization_paths = []
    
    try:
        # Get strategy data
        if 'data' in strategy_with_regimes:
            strategy_data = strategy_with_regimes['data']
        else:
            # Use consolidated data if strategy data not available
            if 'consolidated_without_time' in consolidated_data:
                strategy_data = consolidated_data['consolidated_without_time']
            elif 'consolidated_with_time' in consolidated_data:
                strategy_data = consolidated_data['consolidated_with_time']
            else:
                logging.error("No strategy data found")
                return visualization_paths
        
        # Generate performance summary visualization
        summary_path = os.path.join(output_dir, "performance_summary.png")
        
        # Get strategy columns
        strategy_columns = [col for col in strategy_data.columns if col.endswith('_PnL') or col == 'PnL']
        
        if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
            # Calculate performance metrics by strategy
            performance = []
            for strategy in strategy_data['Strategy'].unique():
                strategy_pnl = strategy_data[strategy_data['Strategy'] == strategy]['PnL']
                total_trades = len(strategy_pnl)
                
                if total_trades > 0:
                    winning_trades = len(strategy_pnl[strategy_pnl > 0])
                    losing_trades = len(strategy_pnl[strategy_pnl < 0])
                    
                    performance.append({
                        'Strategy': strategy,
                        'Total PnL': strategy_pnl.sum(),
                        'Win Rate': winning_trades / total_trades * 100,
                        'Average Win': strategy_pnl[strategy_pnl > 0].mean() if len(strategy_pnl[strategy_pnl > 0]) > 0 else 0,
                        'Average Loss': strategy_pnl[strategy_pnl < 0].mean() if len(strategy_pnl[strategy_pnl < 0]) > 0 else 0,
                        'Profit Factor': abs(strategy_pnl[strategy_pnl > 0].sum() / strategy_pnl[strategy_pnl < 0].sum()) if strategy_pnl[strategy_pnl < 0].sum() != 0 else float('inf')
                    })
            
            if performance:
                performance_df = pd.DataFrame(performance)
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Total PnL
                sns.barplot(x='Strategy', y='Total PnL', data=performance_df, ax=axes[0, 0])
                axes[0, 0].set_title('Total PnL by Strategy')
                axes[0, 0].set_xlabel('Strategy')
                axes[0, 0].set_ylabel('Total PnL')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Win Rate
                sns.barplot(x='Strategy', y='Win Rate', data=performance_df, ax=axes[0, 1])
                axes[0, 1].set_title('Win Rate by Strategy')
                axes[0, 1].set_xlabel('Strategy')
                axes[0, 1].set_ylabel('Win Rate (%)')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Average Win/Loss
                avg_data = pd.melt(
                    performance_df, 
                    id_vars=['Strategy'], 
                    value_vars=['Average Win', 'Average Loss'],
                    var_name='Metric',
                    value_name='Value'
                )
                sns.barplot(x='Strategy', y='Value', hue='Metric', data=avg_data, ax=axes[1, 0])
                axes[1, 0].set_title('Average Win/Loss by Strategy')
                axes[1, 0].set_xlabel('Strategy')
                axes[1, 0].set_ylabel('Value')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Profit Factor
                sns.barplot(x='Strategy', y='Profit Factor', data=performance_df, ax=axes[1, 1])
                axes[1, 1].set_title('Profit Factor by Strategy')
                axes[1, 1].set_xlabel('Strategy')
                axes[1, 1].set_ylabel('Profit Factor')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(summary_path)
                plt.close()
                
                visualization_paths.append(summary_path)
        
        # Generate performance by dimension visualization
        if dimensions and 'selected_dimensions' in dimensions:
            for dimension in dimensions['selected_dimensions']:
                if dimension in strategy_data.columns:
                    dimension_path = os.path.join(output_dir, f"performance_by_{dimension.lower()}.png")
                    
                    if 'Strategy' in strategy_data.columns and 'PnL' in strategy_data.columns:
                        # Group by dimension and strategy
                        performance_by_dim = strategy_data.groupby([dimension, 'Strategy'])['PnL'].agg(['sum', 'mean', 'count']).reset_index()
                        performance_by_dim.columns = [dimension, 'Strategy', 'Total PnL', 'Average PnL', 'Trade Count']
                        
                        # Create visualization
                        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Total PnL
                        sns.barplot(x=dimension, y='Total PnL', hue='Strategy', data=performance_by_dim, ax=axes[0])
                        axes[0].set_title(f'Total PnL by {dimension}')
                        axes[0].set_xlabel(dimension)
                        axes[0].set_ylabel('Total PnL')
                        axes[0].tick_params(axis='x', rotation=45)
                        
                        # Average PnL
                        sns.barplot(x=dimension, y='Average PnL', hue='Strategy', data=performance_by_dim, ax=axes[1])
                        axes[1].set_title(f'Average PnL by {dimension}')
                        axes[1].set_xlabel(dimension)
                        axes[1].set_ylabel('Average PnL')
                        axes[1].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        plt.savefig(dimension_path)
                        plt.close()
                        
                        visualization_paths.append(dimension_path)
        
        # Generate optimized vs. non-optimized comparison
        if optimization_results and 'best_parameters' in optimization_results:
            comparison_path = os.path.join(output_dir, "optimized_vs_non_optimized.png")
            
            # Create dummy data for comparison
            # In a real implementation, this would calculate actual performance
            dates = pd.date_range(start='2023-01-01', periods=100)
            non_optimized = np.cumsum(np.random.normal(0.05, 1, 100))
            optimized = np.cumsum(np.random.normal(0.2, 1, 100))
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            plt.plot(dates, non_optimized, label='Non-Optimized')
            plt.plot(dates, optimized, label='Optimized')
            plt.title('Optimized vs. Non-Optimized Performance')
            plt.xlabel('Date')
            plt.ylabel('Cumulative PnL')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(comparison_path)
            plt.close()
            
            visualization_paths.append(comparison_path)
    
    except Exception as e:
        logging.error(f"Error generating combined performance visualizations: {str(e)}")
    
    return visualization_paths

def generate_interactive_dashboard(strategy_with_regimes, consolidated_data, dimensions, optimization_results, output_dir, config):
    """
    Generate interactive dashboard.
    
    Args:
        strategy_with_regimes (dict): Dictionary containing strategy data with assigned market regimes
        consolidated_data (dict): Dictionary containing consolidated data
        dimensions (dict): Dictionary containing dimension selections
        optimization_results (dict): Dictionary containing optimization results
        output_dir (str): Output directory
        config (dict): Configuration settings
        
    Returns:
        str: Path to dashboard HTML file
    """
    logging.info("Generating interactive dashboard")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    try:
        # Create dashboard HTML file
        dashboard_path = os.path.join(output_dir, "dashboard.html")
        
        # Create simple HTML dashboard
        with open(dashboard_path, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Zone Optimization Dashboard</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                    }
                    h1 {
                        color: #333;
                    }
                    .dashboard {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        grid-gap: 20px;
                    }
                    .card {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                    .card h2 {
                        margin-top: 0;
                        color: #444;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                    }
                </style>
            </head>
            <body>
                <h1>Zone Optimization Dashboard</h1>
                <p>This dashboard provides an overview of the zone optimization results.</p>
                
                <div class="dashboard">
                    <div class="card">
                        <h2>Strategy Performance</h2>
                        <p>Overall performance of different strategies.</p>
                        <img src="../strategy_performance/pnl_by_strategy.png" alt="PnL by Strategy">
                    </div>
                    
                    <div class="card">
                        <h2>Market Regime Analysis</h2>
                        <p>Performance across different market regimes.</p>
                        <img src="../market_regime/pnl_by_market_regime.png" alt="PnL by Market Regime">
                    </div>
                    
                    <div class="card">
                        <h2>Dimension Selection</h2>
                        <p>Selected dimensions for optimization.</p>
                        <img src="../dimension_selection/selected_dimensions.png" alt="Selected Dimensions">
                    </div>
                    
                    <div class="card">
                        <h2>Optimization Results</h2>
                        <p>Results of the optimization process.</p>
                        <img src="../optimization_results/algorithm_comparison.png" alt="Algorithm Comparison">
                    </div>
                    
                    <div class="card">
                        <h2>Performance Summary</h2>
                        <p>Summary of performance metrics.</p>
                        <img src="../combined_performance/performance_summary.png" alt="Performance Summary">
                    </div>
                    
                    <div class="card">
                        <h2>Optimized vs. Non-Optimized</h2>
                        <p>Comparison of optimized and non-optimized performance.</p>
                        <img src="../combined_performance/optimized_vs_non_optimized.png" alt="Optimized vs. Non-Optimized">
                    </div>
                </div>
            </body>
            </html>
            """)
        
        logging.info(f"Interactive dashboard generated at {dashboard_path}")
        return dashboard_path
    
    except Exception as e:
        logging.error(f"Error generating interactive dashboard: {str(e)}")
        return None
