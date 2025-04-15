"""
Performance Metrics and Visualizations Generator

This script generates comprehensive performance metrics and visualizations
for the market regime identification system, helping to evaluate its effectiveness.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
sys.path.append('/home/ubuntu/market_regime_testing')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMetricsGenerator:
    """
    Performance Metrics Generator.
    
    This class generates comprehensive performance metrics and visualizations
    for the market regime identification system.
    """
    
    def __init__(self, results_dir, output_dir=None):
        """
        Initialize Performance Metrics Generator.
        
        Args:
            results_dir (str): Directory containing market regime results
            output_dir (str, optional): Output directory for metrics and visualizations
        """
        self.results_dir = results_dir
        self.output_dir = output_dir or os.path.join(results_dir, 'performance_metrics')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Market regime categories
        self.market_regimes = [
            'Strong_Bullish',
            'Bullish',
            'Mild_Bullish',
            'Sideways_To_Bullish',
            'Neutral',
            'Sideways',
            'Sideways_To_Bearish',
            'Mild_Bearish',
            'Bearish',
            'Strong_Bearish',
            'Reversal_Imminent_Bullish',
            'Reversal_Imminent_Bearish',
            'Exhaustion_Bullish',
            'Exhaustion_Bearish',
            'Failed_Breakout_Bullish',
            'Failed_Breakout_Bearish',
            'Institutional_Accumulation',
            'Institutional_Distribution'
        ]
        
        # Regime categories for simplified analysis
        self.regime_categories = {
            'Bullish': ['Strong_Bullish', 'Bullish', 'Mild_Bullish', 'Sideways_To_Bullish'],
            'Neutral': ['Neutral', 'Sideways'],
            'Bearish': ['Strong_Bearish', 'Bearish', 'Mild_Bearish', 'Sideways_To_Bearish'],
            'Reversal': ['Reversal_Imminent_Bullish', 'Reversal_Imminent_Bearish', 
                         'Failed_Breakout_Bullish', 'Failed_Breakout_Bearish'],
            'Exhaustion': ['Exhaustion_Bullish', 'Exhaustion_Bearish'],
            'Institutional': ['Institutional_Accumulation', 'Institutional_Distribution']
        }
        
        # Color maps for visualizations
        self.regime_colors = {
            'Strong_Bullish': '#006400',  # Dark Green
            'Bullish': '#32CD32',  # Lime Green
            'Mild_Bullish': '#90EE90',  # Light Green
            'Sideways_To_Bullish': '#E0FFE0',  # Very Light Green
            'Neutral': '#F5F5F5',  # White Smoke
            'Sideways': '#D3D3D3',  # Light Gray
            'Sideways_To_Bearish': '#FFE0E0',  # Very Light Red
            'Mild_Bearish': '#FFA07A',  # Light Salmon
            'Bearish': '#FF4500',  # Orange Red
            'Strong_Bearish': '#8B0000',  # Dark Red
            'Reversal_Imminent_Bullish': '#00BFFF',  # Deep Sky Blue
            'Reversal_Imminent_Bearish': '#FF69B4',  # Hot Pink
            'Exhaustion_Bullish': '#9370DB',  # Medium Purple
            'Exhaustion_Bearish': '#BA55D3',  # Medium Orchid
            'Failed_Breakout_Bullish': '#4682B4',  # Steel Blue
            'Failed_Breakout_Bearish': '#CD5C5C',  # Indian Red
            'Institutional_Accumulation': '#FFD700',  # Gold
            'Institutional_Distribution': '#B8860B'   # Dark Golden Rod
        }
        
        # Category colors for simplified visualizations
        self.category_colors = {
            'Bullish': '#32CD32',  # Lime Green
            'Neutral': '#D3D3D3',  # Light Gray
            'Bearish': '#FF4500',  # Orange Red
            'Reversal': '#00BFFF',  # Deep Sky Blue
            'Exhaustion': '#9370DB',  # Medium Purple
            'Institutional': '#FFD700'   # Gold
        }
        
        logger.info(f"Initialized Performance Metrics Generator with results directory: {results_dir}")
    
    def load_results(self, file_pattern='*.csv'):
        """
        Load market regime results from files.
        
        Args:
            file_pattern (str, optional): File pattern to match result files
            
        Returns:
            dict: Dictionary mapping file names to result dataframes
        """
        try:
            import glob
            
            # Find result files
            result_files = glob.glob(os.path.join(self.results_dir, file_pattern))
            
            if not result_files:
                logger.warning(f"No result files found matching pattern: {file_pattern}")
                return {}
            
            # Load each file
            results = {}
            
            for file_path in result_files:
                try:
                    file_name = os.path.basename(file_path)
                    
                    # Load data
                    data = pd.read_csv(file_path)
                    
                    # Ensure datetime is in datetime format
                    if 'datetime' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                        try:
                            data['datetime'] = pd.to_datetime(data['datetime'])
                        except:
                            logger.warning(f"Failed to convert datetime column in {file_name}")
                    
                    # Add to results
                    results[file_name] = data
                    
                    logger.info(f"Loaded {file_name} with {len(data)} rows")
                
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return {}
    
    def generate_regime_distribution_metrics(self, results):
        """
        Generate market regime distribution metrics.
        
        Args:
            results (dict): Dictionary mapping file names to result dataframes
            
        Returns:
            dict: Dictionary of regime distribution metrics
        """
        try:
            # Initialize metrics
            metrics = {}
            
            for file_name, data in results.items():
                if 'market_regime' not in data.columns:
                    logger.warning(f"No market_regime column in {file_name}")
                    continue
                
                # Calculate regime distribution
                regime_counts = data['market_regime'].value_counts()
                regime_percentage = regime_counts / len(data) * 100
                
                # Calculate category distribution
                category_counts = {}
                category_percentage = {}
                
                for category, regimes in self.regime_categories.items():
                    category_mask = data['market_regime'].isin(regimes)
                    category_counts[category] = category_mask.sum()
                    category_percentage[category] = category_mask.sum() / len(data) * 100
                
                # Calculate confidence metrics
                confidence_metrics = {}
                
                if 'market_regime_confidence' in data.columns:
                    # Overall confidence
                    confidence_metrics['overall'] = {
                        'mean': data['market_regime_confidence'].mean(),
                        'median': data['market_regime_confidence'].median(),
                        'min': data['market_regime_confidence'].min(),
                        'max': data['market_regime_confidence'].max(),
                        'std': data['market_regime_confidence'].std()
                    }
                    
                    # Confidence by regime
                    regime_confidence = data.groupby('market_regime')['market_regime_confidence'].agg(
                        ['mean', 'median', 'min', 'max', 'std', 'count']
                    )
                    
                    confidence_metrics['by_regime'] = regime_confidence.to_dict()
                    
                    # Confidence by category
                    category_confidence = {}
                    
                    for category, regimes in self.regime_categories.items():
                        category_mask = data['market_regime'].isin(regimes)
                        if category_mask.sum() > 0:
                            category_confidence[category] = {
                                'mean': data.loc[category_mask, 'market_regime_confidence'].mean(),
                                'median': data.loc[category_mask, 'market_regime_confidence'].median(),
                                'min': data.loc[category_mask, 'market_regime_confidence'].min(),
                                'max': data.loc[category_mask, 'market_regime_confidence'].max(),
                                'std': data.loc[category_mask, 'market_regime_confidence'].std(),
                                'count': category_mask.sum()
                            }
                    
                    confidence_metrics['by_category'] = category_confidence
                
                # Add to metrics
                metrics[file_name] = {
                    'regime_counts': regime_counts.to_dict(),
                    'regime_percentage': regime_percentage.to_dict(),
                    'category_counts': category_counts,
                    'category_percentage': category_percentage,
                    'confidence_metrics': confidence_metrics
                }
                
                logger.info(f"Generated regime distribution metrics for {file_name}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating regime distribution metrics: {str(e)}")
            return {}
    
    def generate_regime_transition_metrics(self, results):
        """
        Generate market regime transition metrics.
        
        Args:
            results (dict): Dictionary mapping file names to result dataframes
            
        Returns:
            dict: Dictionary of regime transition metrics
        """
        try:
            # Initialize metrics
            metrics = {}
            
            for file_name, data in results.items():
                if 'market_regime' not in data.columns or 'datetime' not in data.columns:
                    logger.warning(f"Missing required columns in {file_name}")
                    continue
                
                # Sort by datetime
                data = data.sort_values('datetime')
                
                # Calculate regime transitions
                transitions = {}
                transition_counts = {}
                
                # Initialize transition matrix
                all_regimes = set(data['market_regime'].unique())
                transition_matrix = pd.DataFrame(0, index=all_regimes, columns=all_regimes)
                
                # Calculate transitions
                prev_regime = None
                
                for i, row in data.iterrows():
                    current_regime = row['market_regime']
                    
                    if prev_regime is not None and prev_regime != current_regime:
                        transition = (prev_regime, current_regime)
                        
                        if transition not in transitions:
                            transitions[transition] = 0
                        
                        transitions[transition] += 1
                        
                        # Update transition matrix
                        transition_matrix.loc[prev_regime, current_regime] += 1
                    
                    prev_regime = current_regime
                
                # Calculate transition probabilities
                transition_probs = transition_matrix.copy()
                
                for i in transition_probs.index:
                    row_sum = transition_probs.loc[i].sum()
                    if row_sum > 0:
                        transition_probs.loc[i] = transition_probs.loc[i] / row_sum
                
                # Calculate category transitions
                category_transitions = {}
                category_transition_matrix = pd.DataFrame(
                    0, 
                    index=self.regime_categories.keys(), 
                    columns=self.regime_categories.keys()
                )
                
                # Map regimes to categories
                regime_to_category = {}
                for category, regimes in self.regime_categories.items():
                    for regime in regimes:
                        regime_to_category[regime] = category
                
                # Calculate category for each row
                data['category'] = data['market_regime'].map(
                    lambda x: next((cat for cat, regimes in self.regime_categories.items() 
                                   if x in regimes), 'Other')
                )
                
                # Calculate category transitions
                prev_category = None
                
                for i, row in data.iterrows():
                    current_category = row['category']
                    
                    if prev_category is not None and prev_category != current_category:
                        cat_transition = (prev_category, current_category)
                        
                        if cat_transition not in category_transitions:
                            category_transitions[cat_transition] = 0
                        
                        category_transitions[cat_transition] += 1
                        
                        # Update category transition matrix
                        category_transition_matrix.loc[prev_category, current_category] += 1
                    
                    prev_category = current_category
                
                # Calculate category transition probabilities
                category_transition_probs = category_transition_matrix.copy()
                
                for i in category_transition_probs.index:
                    row_sum = category_transition_probs.loc[i].sum()
                    if row_sum > 0:
                        category_transition_probs.loc[i] = category_transition_probs.loc[i] / row_sum
                
                # Add to metrics
                metrics[file_name] = {
                    'transitions': transitions,
                    'transition_matrix': transition_matrix.to_dict(),
                    'transition_probabilities': transition_probs.to_dict(),
                    'category_transitions': category_transitions,
                    'category_transition_matrix': category_transition_matrix.to_dict(),
                    'category_transition_probabilities': category_transition_probs.to_dict()
                }
                
                logger.info(f"Generated regime transition metrics for {file_name}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating regime transition metrics: {str(e)}")
            return {}
    
    def generate_component_contribution_metrics(self, results):
        """
        Generate component contribution metrics.
        
        Args:
            results (dict): Dictionary mapping file names to result dataframes
            
        Returns:
            dict: Dictionary of component contribution metrics
        """
        try:
            # Initialize metrics
            metrics = {}
            
            for file_name, data in results.items():
                if 'component_contributions' not in data.columns or 'market_regime' not in data.columns:
                    logger.warning(f"Missing required columns in {file_name}")
                    continue
                
                # Extract component contributions
                component_data = []
                
                for i, row in data.iterrows():
                    if pd.notna(row['component_contributions']):
                        try:
                            contributions = json.loads(row['component_contributions'])
                            
                            for component, details in contributions.items():
                                if 'signal' in details and 'weight' in details:
                                    entry = {
                                        'datetime': row['datetime'] if 'datetime' in row else None,
                                        'market_regime': row['market_regime'],
                                        'component': component,
                                        'signal': details['signal'],
                                        'weight': details['weight']
                                    }
                                    
                                    # Add category
                                    entry['category'] = next(
                                        (cat for cat, regimes in self.regime_categories.items() 
                                         if row['market_regime'] in regimes), 
                                        'Other'
                                    )
                                    
                                    component_data.append(entry)
                        except:
                            logger.warning(f"Error parsing component_contributions in row {i}")
                
                if not component_data:
                    logger.warning(f"No component contribution data found in {file_name}")
                    continue
                
                # Create dataframe
                component_df = pd.DataFrame(component_data)
                
                # Calculate component signal distribution
                component_signal_counts = component_df.groupby(['component', 'signal']).size().unstack(fill_value=0)
                
                # Calculate component weight statistics
                component_weights = component_df.groupby('component')['weight'].agg(
                    ['mean', 'median', 'min', 'max', 'std', 'count']
                )
                
                # Calculate component-regime correlation
                component_regime_counts = component_df.groupby(['component', 'market_regime']).size().unstack(fill_value=0)
                
                # Calculate component-category correlation
                component_category_counts = component_df.groupby(['component', 'category']).size().unstack(fill_value=0)
                
                # Calculate signal-regime correlation
                signal_regime_counts = component_df.groupby(['signal', 'market_regime']).size().unstack(fill_value=0)
                
                # Add to metrics
                metrics[file_name] = {
                    'component_signal_counts': component_signal_counts.to_dict(),
                    'component_weights': component_weights.to_dict(),
                    'component_regime_counts': component_regime_counts.to_dict(),
                    'component_category_counts': component_category_counts.to_dict(),
                    'signal_regime_counts': signal_regime_counts.to_dict()
                }
                
                logger.info(f"Generated component contribution metrics for {file_name}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating component contribution metrics: {str(e)}")
            return {}
    
    def generate_time_based_metrics(self, results):
        """
        Generate time-based metrics.
        
        Args:
            results (dict): Dictionary mapping file names to result dataframes
            
        Returns:
            dict: Dictionary of time-based metrics
        """
        try:
            # Initialize metrics
            metrics = {}
            
            for file_name, data in results.items():
                if 'market_regime' not in data.columns or 'datetime' not in data.columns:
                    logger.warning(f"Missing required columns in {file_name}")
                    continue
                
                # Sort by datetime
                data = data.sort_values('datetime')
                
                # Add time-based columns
                data['hour'] = data['datetime'].dt.hour
                data['minute'] = data['datetime'].dt.minute
                data['day_of_week'] = data['datetime'].dt.day_name()
                data['day_of_month'] = data['datetime'].dt.day
                data['month'] = data['datetime'].dt.month_name()
                
                # Calculate time of day if not present
                if 'time_of_day' not in data.columns:
                    data['time_of_day'] = data.apply(
                        lambda row: self._get_time_of_day(row['hour'], row['minute']), 
                        axis=1
                    )
                
                # Calculate regime by time of day
                time_of_day_regime_counts = data.groupby(['time_of_day', 'market_regime']).size().unstack(fill_value=0)
                
                # Calculate regime by hour
                hour_regime_counts = data.groupby(['hour', 'market_regime']).size().unstack(fill_value=0)
                
                # Calculate regime by day of week
                day_of_week_regime_counts = data.groupby(['day_of_week', 'market_regime']).size().unstack(fill_value=0)
                
                # Calculate regime by day of month
                day_of_month_regime_counts = data.groupby(['day_of_month', 'market_regime']).size().unstack(fill_value=0)
                
                # Calculate regime by month
                month_regime_counts = data.groupby(['month', 'market_regime']).size().unstack(fill_value=0)
                
                # Calculate regime duration
                regime_durations = []
                current_regime = None
                regime_start = None
                
                for i, row in data.iterrows():
                    if current_regime is None:
                        current_regime = row['market_regime']
                        regime_start = row['datetime']
                    elif row['market_regime'] != current_regime:
                        # Regime changed, calculate duration
                        duration = (row['datetime'] - regime_start).total_seconds() / 60  # Duration in minutes
                        
                        regime_durations.append({
                            'regime': current_regime,
                            'start': regime_start,
                            'end': row['datetime'],
                            'duration_minutes': duration
                        })
                        
                        # Update for next regime
                        current_regime = row['market_regime']
                        regime_start = row['datetime']
                
                # Add last regime if exists
                if current_regime is not None and regime_start is not None:
                    duration = (data['datetime'].iloc[-1] - regime_start).total_seconds() / 60
                    
                    regime_durations.append({
                        'regime': current_regime,
                        'start': regime_start,
                        'end': data['datetime'].iloc[-1],
                        'duration_minutes': duration
                    })
                
                # Create regime duration dataframe
                regime_duration_df = pd.DataFrame(regime_durations)
                
                # Calculate duration statistics by regime
                if not regime_duration_df.empty:
                    duration_stats = regime_duration_df.groupby('regime')['duration_minutes'].agg(
                        ['mean', 'median', 'min', 'max', 'std', 'count']
                    )
                else:
                    duration_stats = pd.DataFrame()
                
                # Add to metrics
                metrics[file_name] = {
                    'time_of_day_regime_counts': time_of_day_regime_counts.to_dict(),
                    'hour_regime_counts': hour_regime_counts.to_dict(),
                    'day_of_week_regime_counts': day_of_week_regime_counts.to_dict(),
                    'day_of_month_regime_counts': day_of_month_regime_counts.to_dict(),
                    'month_regime_counts': month_regime_counts.to_dict(),
                    'regime_durations': regime_duration_df.to_dict() if not regime_duration_df.empty else {},
                    'duration_statistics': duration_stats.to_dict() if not duration_stats.empty else {}
                }
                
                logger.info(f"Generated time-based metrics for {file_name}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating time-based metrics: {str(e)}")
            return {}
    
    def _get_time_of_day(self, hour, minute):
        """
        Get time of day category.
        
        Args:
            hour (int): Hour (0-23)
            minute (int): Minute (0-59)
            
        Returns:
            str: Time of day category
        """
        time_val = hour + minute / 60.0
        
        if 9.25 <= time_val < 9.75:  # 9:15-9:45
            return 'opening'
        elif 9.75 <= time_val < 12.0:  # 9:45-12:00
            return 'morning'
        elif 12.0 <= time_val < 13.0:  # 12:00-13:00
            return 'lunch'
        elif 13.0 <= time_val < 14.5:  # 13:00-14:30
            return 'afternoon'
        elif 14.5 <= time_val < 15.5:  # 14:30-15:30
            return 'closing'
        else:
            return 'after_hours'
    
    def generate_multi_timeframe_metrics(self, results):
        """
        Generate multi-timeframe metrics.
        
        Args:
            results (dict): Dictionary mapping file names to result dataframes
            
        Returns:
            dict: Dictionary of multi-timeframe metrics
        """
        try:
            # Initialize metrics
            metrics = {}
            
            # Group results by timeframe
            timeframe_results = {}
            
            for file_name, data in results.items():
                # Extract timeframe from filename
                timeframe = None
                
                for tf in ['5m', '15m', '1h', '1d']:
                    if tf in file_name:
                        timeframe = tf
                        break
                
                if timeframe is None:
                    logger.warning(f"Could not determine timeframe for {file_name}")
                    continue
                
                if timeframe not in timeframe_results:
                    timeframe_results[timeframe] = []
                
                timeframe_results[timeframe].append((file_name, data))
            
            # Calculate metrics for each timeframe
            for timeframe, results_list in timeframe_results.items():
                timeframe_metrics = {}
                
                for file_name, data in results_list:
                    if 'market_regime' not in data.columns or 'datetime' not in data.columns:
                        logger.warning(f"Missing required columns in {file_name}")
                        continue
                    
                    # Calculate regime distribution
                    regime_counts = data['market_regime'].value_counts()
                    regime_percentage = regime_counts / len(data) * 100
                    
                    # Calculate category distribution
                    category_counts = {}
                    category_percentage = {}
                    
                    for category, regimes in self.regime_categories.items():
                        category_mask = data['market_regime'].isin(regimes)
                        category_counts[category] = category_mask.sum()
                        category_percentage[category] = category_mask.sum() / len(data) * 100
                    
                    # Add to metrics
                    timeframe_metrics[file_name] = {
                        'regime_counts': regime_counts.to_dict(),
                        'regime_percentage': regime_percentage.to_dict(),
                        'category_counts': category_counts,
                        'category_percentage': category_percentage
                    }
                
                metrics[timeframe] = timeframe_metrics
                
                logger.info(f"Generated multi-timeframe metrics for {timeframe}")
            
            # Calculate cross-timeframe agreement if possible
            if len(timeframe_results) > 1:
                # Find common datetimes across timeframes
                common_datetimes = None
                
                for timeframe, results_list in timeframe_results.items():
                    for _, data in results_list:
                        if 'datetime' in data.columns:
                            if common_datetimes is None:
                                common_datetimes = set(data['datetime'])
                            else:
                                common_datetimes &= set(data['datetime'])
                
                if common_datetimes and len(common_datetimes) > 0:
                    # Convert to list and sort
                    common_datetimes = sorted(list(common_datetimes))
                    
                    # Calculate agreement for each common datetime
                    agreement_data = []
                    
                    for dt in common_datetimes:
                        regimes_at_dt = {}
                        
                        for timeframe, results_list in timeframe_results.items():
                            for file_name, data in results_list:
                                if 'datetime' in data.columns and 'market_regime' in data.columns:
                                    # Find row with this datetime
                                    dt_data = data[data['datetime'] == dt]
                                    
                                    if not dt_data.empty:
                                        regimes_at_dt[timeframe] = dt_data['market_regime'].iloc[0]
                        
                        # Calculate agreement
                        if regimes_at_dt:
                            regime_counts = {}
                            
                            for timeframe, regime in regimes_at_dt.items():
                                if regime not in regime_counts:
                                    regime_counts[regime] = 0
                                
                                regime_counts[regime] += 1
                            
                            # Find most common regime
                            most_common_regime = max(regime_counts.items(), key=lambda x: x[1])
                            agreement_score = most_common_regime[1] / len(regimes_at_dt)
                            
                            # Add to agreement data
                            agreement_data.append({
                                'datetime': dt,
                                'most_common_regime': most_common_regime[0],
                                'agreement_score': agreement_score,
                                'regimes': regimes_at_dt
                            })
                    
                    if agreement_data:
                        # Create dataframe
                        agreement_df = pd.DataFrame(agreement_data)
                        
                        # Calculate agreement statistics
                        agreement_stats = {
                            'mean': agreement_df['agreement_score'].mean(),
                            'median': agreement_df['agreement_score'].median(),
                            'min': agreement_df['agreement_score'].min(),
                            'max': agreement_df['agreement_score'].max(),
                            'std': agreement_df['agreement_score'].std()
                        }
                        
                        # Calculate most common regimes
                        most_common_regimes = agreement_df['most_common_regime'].value_counts().to_dict()
                        
                        # Add to metrics
                        metrics['cross_timeframe_agreement'] = {
                            'agreement_data': agreement_df.to_dict(),
                            'agreement_statistics': agreement_stats,
                            'most_common_regimes': most_common_regimes
                        }
                        
                        logger.info(f"Generated cross-timeframe agreement metrics")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating multi-timeframe metrics: {str(e)}")
            return {}
    
    def generate_all_metrics(self):
        """
        Generate all performance metrics.
        
        Returns:
            dict: Dictionary of all metrics
        """
        try:
            # Load results
            results = self.load_results()
            
            if not results:
                logger.warning("No results found")
                return {}
            
            # Generate metrics
            metrics = {
                'regime_distribution': self.generate_regime_distribution_metrics(results),
                'regime_transition': self.generate_regime_transition_metrics(results),
                'component_contribution': self.generate_component_contribution_metrics(results),
                'time_based': self.generate_time_based_metrics(results),
                'multi_timeframe': self.generate_multi_timeframe_metrics(results)
            }
            
            # Save metrics
            self.save_metrics(metrics)
            
            # Generate visualizations
            self.generate_visualizations(metrics, results)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating all metrics: {str(e)}")
            return {}
    
    def save_metrics(self, metrics):
        """
        Save metrics to file.
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        try:
            # Create metrics directory
            metrics_dir = os.path.join(self.output_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Save each metric type
            for metric_type, metric_data in metrics.items():
                # Create metric type directory
                metric_dir = os.path.join(metrics_dir, metric_type)
                os.makedirs(metric_dir, exist_ok=True)
                
                # Save metric data
                for file_name, data in metric_data.items():
                    # Create file name
                    metric_file = os.path.join(metric_dir, f"{file_name.replace('.csv', '')}_metrics.json")
                    
                    # Convert pandas objects to dict
                    serializable_data = {}
                    
                    for key, value in data.items():
                        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                            serializable_data[key] = value.to_dict()
                        else:
                            serializable_data[key] = value
                    
                    # Save to file
                    with open(metric_file, 'w') as f:
                        json.dump(serializable_data, f, indent=2)
                    
                    logger.info(f"Saved {metric_type} metrics for {file_name} to {metric_file}")
            
            # Save summary metrics
            summary_metrics = self.generate_summary_metrics(metrics)
            
            summary_file = os.path.join(metrics_dir, 'summary_metrics.json')
            
            with open(summary_file, 'w') as f:
                json.dump(summary_metrics, f, indent=2)
            
            logger.info(f"Saved summary metrics to {summary_file}")
        
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def generate_summary_metrics(self, metrics):
        """
        Generate summary metrics.
        
        Args:
            metrics (dict): Dictionary of metrics
            
        Returns:
            dict: Dictionary of summary metrics
        """
        try:
            # Initialize summary metrics
            summary = {}
            
            # Regime distribution summary
            if 'regime_distribution' in metrics:
                regime_dist = metrics['regime_distribution']
                
                if regime_dist:
                    # Calculate average regime distribution
                    all_regimes = set()
                    regime_counts = {}
                    
                    for file_name, data in regime_dist.items():
                        if 'regime_percentage' in data:
                            all_regimes.update(data['regime_percentage'].keys())
                            
                            for regime, percentage in data['regime_percentage'].items():
                                if regime not in regime_counts:
                                    regime_counts[regime] = []
                                
                                regime_counts[regime].append(percentage)
                    
                    # Calculate average percentage
                    avg_regime_percentage = {}
                    
                    for regime, percentages in regime_counts.items():
                        avg_regime_percentage[regime] = sum(percentages) / len(percentages)
                    
                    # Calculate average category distribution
                    all_categories = set()
                    category_counts = {}
                    
                    for file_name, data in regime_dist.items():
                        if 'category_percentage' in data:
                            all_categories.update(data['category_percentage'].keys())
                            
                            for category, percentage in data['category_percentage'].items():
                                if category not in category_counts:
                                    category_counts[category] = []
                                
                                category_counts[category].append(percentage)
                    
                    # Calculate average percentage
                    avg_category_percentage = {}
                    
                    for category, percentages in category_counts.items():
                        avg_category_percentage[category] = sum(percentages) / len(percentages)
                    
                    # Calculate average confidence
                    avg_confidence = {}
                    
                    for file_name, data in regime_dist.items():
                        if 'confidence_metrics' in data and 'overall' in data['confidence_metrics']:
                            for metric, value in data['confidence_metrics']['overall'].items():
                                if metric not in avg_confidence:
                                    avg_confidence[metric] = []
                                
                                avg_confidence[metric].append(value)
                    
                    # Calculate average
                    avg_confidence_metrics = {}
                    
                    for metric, values in avg_confidence.items():
                        avg_confidence_metrics[metric] = sum(values) / len(values)
                    
                    summary['regime_distribution'] = {
                        'avg_regime_percentage': avg_regime_percentage,
                        'avg_category_percentage': avg_category_percentage,
                        'avg_confidence': avg_confidence_metrics
                    }
            
            # Regime transition summary
            if 'regime_transition' in metrics:
                regime_trans = metrics['regime_transition']
                
                if regime_trans:
                    # Calculate average transition probabilities
                    all_transitions = set()
                    transition_probs = {}
                    
                    for file_name, data in regime_trans.items():
                        if 'transition_probabilities' in data:
                            for from_regime, to_regimes in data['transition_probabilities'].items():
                                for to_regime, prob in to_regimes.items():
                                    transition = (from_regime, to_regime)
                                    all_transitions.add(transition)
                                    
                                    if transition not in transition_probs:
                                        transition_probs[transition] = []
                                    
                                    transition_probs[transition].append(prob)
                    
                    # Calculate average probability
                    avg_transition_probs = {}
                    
                    for transition, probs in transition_probs.items():
                        avg_transition_probs[transition] = sum(probs) / len(probs)
                    
                    # Calculate average category transition probabilities
                    all_cat_transitions = set()
                    cat_transition_probs = {}
                    
                    for file_name, data in regime_trans.items():
                        if 'category_transition_probabilities' in data:
                            for from_cat, to_cats in data['category_transition_probabilities'].items():
                                for to_cat, prob in to_cats.items():
                                    cat_transition = (from_cat, to_cat)
                                    all_cat_transitions.add(cat_transition)
                                    
                                    if cat_transition not in cat_transition_probs:
                                        cat_transition_probs[cat_transition] = []
                                    
                                    cat_transition_probs[cat_transition].append(prob)
                    
                    # Calculate average probability
                    avg_cat_transition_probs = {}
                    
                    for transition, probs in cat_transition_probs.items():
                        avg_cat_transition_probs[transition] = sum(probs) / len(probs)
                    
                    summary['regime_transition'] = {
                        'avg_transition_probabilities': avg_transition_probs,
                        'avg_category_transition_probabilities': avg_cat_transition_probs
                    }
            
            # Component contribution summary
            if 'component_contribution' in metrics:
                comp_contrib = metrics['component_contribution']
                
                if comp_contrib:
                    # Calculate average component weights
                    all_components = set()
                    component_weights = {}
                    
                    for file_name, data in comp_contrib.items():
                        if 'component_weights' in data:
                            for component, weight_data in data['component_weights'].items():
                                all_components.add(component)
                                
                                if component not in component_weights:
                                    component_weights[component] = {}
                                
                                for metric, value in weight_data.items():
                                    if metric not in component_weights[component]:
                                        component_weights[component][metric] = []
                                    
                                    component_weights[component][metric].append(value)
                    
                    # Calculate average weights
                    avg_component_weights = {}
                    
                    for component, weight_data in component_weights.items():
                        avg_component_weights[component] = {}
                        
                        for metric, values in weight_data.items():
                            avg_component_weights[component][metric] = sum(values) / len(values)
                    
                    summary['component_contribution'] = {
                        'avg_component_weights': avg_component_weights
                    }
            
            # Time-based summary
            if 'time_based' in metrics:
                time_based = metrics['time_based']
                
                if time_based:
                    # Calculate average regime duration
                    all_regimes = set()
                    regime_durations = {}
                    
                    for file_name, data in time_based.items():
                        if 'duration_statistics' in data:
                            for regime, duration_data in data['duration_statistics'].items():
                                all_regimes.add(regime)
                                
                                if regime not in regime_durations:
                                    regime_durations[regime] = {}
                                
                                for metric, value in duration_data.items():
                                    if metric not in regime_durations[regime]:
                                        regime_durations[regime][metric] = []
                                    
                                    regime_durations[regime][metric].append(value)
                    
                    # Calculate average durations
                    avg_regime_durations = {}
                    
                    for regime, duration_data in regime_durations.items():
                        avg_regime_durations[regime] = {}
                        
                        for metric, values in duration_data.items():
                            avg_regime_durations[regime][metric] = sum(values) / len(values)
                    
                    summary['time_based'] = {
                        'avg_regime_durations': avg_regime_durations
                    }
            
            # Multi-timeframe summary
            if 'multi_timeframe' in metrics and 'cross_timeframe_agreement' in metrics['multi_timeframe']:
                cross_tf = metrics['multi_timeframe']['cross_timeframe_agreement']
                
                if 'agreement_statistics' in cross_tf:
                    summary['multi_timeframe'] = {
                        'cross_timeframe_agreement': cross_tf['agreement_statistics']
                    }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary metrics: {str(e)}")
            return {}
    
    def generate_visualizations(self, metrics, results):
        """
        Generate visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            results (dict): Dictionary mapping file names to result dataframes
        """
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Generate regime distribution visualizations
            self.generate_regime_distribution_visualizations(metrics, viz_dir)
            
            # Generate regime transition visualizations
            self.generate_regime_transition_visualizations(metrics, viz_dir)
            
            # Generate component contribution visualizations
            self.generate_component_contribution_visualizations(metrics, viz_dir)
            
            # Generate time-based visualizations
            self.generate_time_based_visualizations(metrics, viz_dir)
            
            # Generate multi-timeframe visualizations
            self.generate_multi_timeframe_visualizations(metrics, viz_dir)
            
            # Generate summary visualizations
            self.generate_summary_visualizations(metrics, viz_dir)
            
            # Generate combined visualizations
            self.generate_combined_visualizations(results, viz_dir)
            
            logger.info(f"Generated all visualizations in {viz_dir}")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def generate_regime_distribution_visualizations(self, metrics, viz_dir):
        """
        Generate regime distribution visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            viz_dir (str): Visualizations directory
        """
        try:
            # Create regime distribution directory
            regime_dir = os.path.join(viz_dir, 'regime_distribution')
            os.makedirs(regime_dir, exist_ok=True)
            
            if 'regime_distribution' not in metrics:
                logger.warning("No regime distribution metrics found")
                return
            
            regime_dist = metrics['regime_distribution']
            
            for file_name, data in regime_dist.items():
                # Create file prefix
                prefix = file_name.replace('.csv', '')
                
                # 1. Regime distribution bar chart
                if 'regime_percentage' in data:
                    regime_percentage = data['regime_percentage']
                    
                    # Sort regimes by category
                    sorted_regimes = []
                    
                    for category, regimes in self.regime_categories.items():
                        for regime in regimes:
                            if regime in regime_percentage:
                                sorted_regimes.append(regime)
                    
                    # Add any remaining regimes
                    for regime in regime_percentage:
                        if regime not in sorted_regimes:
                            sorted_regimes.append(regime)
                    
                    # Create dataframe
                    regime_df = pd.DataFrame({
                        'Regime': sorted_regimes,
                        'Percentage': [regime_percentage.get(regime, 0) for regime in sorted_regimes]
                    })
                    
                    # Create color map
                    colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in sorted_regimes]
                    
                    # Plot
                    plt.figure(figsize=(14, 8))
                    bars = plt.bar(regime_df['Regime'], regime_df['Percentage'], color=colors)
                    plt.title(f'Market Regime Distribution - {prefix}')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Percentage (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add percentage labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.5,
                            f'{height:.1f}%',
                            ha='center',
                            va='bottom',
                            rotation=0
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(regime_dir, f"{prefix}_regime_distribution.png"))
                    plt.close()
                
                # 2. Category distribution pie chart
                if 'category_percentage' in data:
                    category_percentage = data['category_percentage']
                    
                    # Create dataframe
                    category_df = pd.DataFrame({
                        'Category': list(category_percentage.keys()),
                        'Percentage': list(category_percentage.values())
                    })
                    
                    # Create color map
                    colors = [self.category_colors.get(category, '#CCCCCC') for category in category_df['Category']]
                    
                    # Plot
                    plt.figure(figsize=(10, 8))
                    plt.pie(
                        category_df['Percentage'],
                        labels=category_df['Category'],
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors
                    )
                    plt.title(f'Market Regime Category Distribution - {prefix}')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(os.path.join(regime_dir, f"{prefix}_category_distribution.png"))
                    plt.close()
                
                # 3. Confidence distribution histogram
                if 'confidence_metrics' in data and 'by_regime' in data['confidence_metrics']:
                    regime_confidence = data['confidence_metrics']['by_regime']
                    
                    # Create dataframe
                    confidence_data = []
                    
                    for regime, metrics_data in regime_confidence.items():
                        if 'mean' in metrics_data:
                            confidence_data.append({
                                'Regime': regime,
                                'Mean Confidence': metrics_data['mean']
                            })
                    
                    confidence_df = pd.DataFrame(confidence_data)
                    
                    # Sort by regime category
                    sorted_regimes = []
                    
                    for category, regimes in self.regime_categories.items():
                        for regime in regimes:
                            if regime in confidence_df['Regime'].values:
                                sorted_regimes.append(regime)
                    
                    # Add any remaining regimes
                    for regime in confidence_df['Regime']:
                        if regime not in sorted_regimes:
                            sorted_regimes.append(regime)
                    
                    # Filter and sort dataframe
                    confidence_df = confidence_df[confidence_df['Regime'].isin(sorted_regimes)]
                    confidence_df['Regime'] = pd.Categorical(
                        confidence_df['Regime'],
                        categories=sorted_regimes,
                        ordered=True
                    )
                    confidence_df = confidence_df.sort_values('Regime')
                    
                    # Create color map
                    colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in confidence_df['Regime']]
                    
                    # Plot
                    plt.figure(figsize=(14, 8))
                    bars = plt.bar(confidence_df['Regime'], confidence_df['Mean Confidence'], color=colors)
                    plt.title(f'Market Regime Confidence - {prefix}')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Mean Confidence')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add confidence labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.02,
                            f'{height:.2f}',
                            ha='center',
                            va='bottom',
                            rotation=0
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(regime_dir, f"{prefix}_regime_confidence.png"))
                    plt.close()
            
            logger.info(f"Generated regime distribution visualizations in {regime_dir}")
        
        except Exception as e:
            logger.error(f"Error generating regime distribution visualizations: {str(e)}")
    
    def generate_regime_transition_visualizations(self, metrics, viz_dir):
        """
        Generate regime transition visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            viz_dir (str): Visualizations directory
        """
        try:
            # Create regime transition directory
            transition_dir = os.path.join(viz_dir, 'regime_transition')
            os.makedirs(transition_dir, exist_ok=True)
            
            if 'regime_transition' not in metrics:
                logger.warning("No regime transition metrics found")
                return
            
            regime_trans = metrics['regime_transition']
            
            for file_name, data in regime_trans.items():
                # Create file prefix
                prefix = file_name.replace('.csv', '')
                
                # 1. Transition matrix heatmap
                if 'transition_probabilities' in data:
                    # Create dataframe
                    transition_probs = pd.DataFrame(data['transition_probabilities'])
                    
                    # Plot
                    plt.figure(figsize=(14, 12))
                    sns.heatmap(
                        transition_probs,
                        annot=True,
                        fmt='.2f',
                        cmap='viridis',
                        linewidths=0.5,
                        cbar_kws={'label': 'Transition Probability'}
                    )
                    plt.title(f'Market Regime Transition Probabilities - {prefix}')
                    plt.xlabel('To Regime')
                    plt.ylabel('From Regime')
                    plt.tight_layout()
                    plt.savefig(os.path.join(transition_dir, f"{prefix}_transition_matrix.png"))
                    plt.close()
                
                # 2. Category transition matrix heatmap
                if 'category_transition_probabilities' in data:
                    # Create dataframe
                    category_probs = pd.DataFrame(data['category_transition_probabilities'])
                    
                    # Plot
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        category_probs,
                        annot=True,
                        fmt='.2f',
                        cmap='viridis',
                        linewidths=0.5,
                        cbar_kws={'label': 'Transition Probability'}
                    )
                    plt.title(f'Market Regime Category Transition Probabilities - {prefix}')
                    plt.xlabel('To Category')
                    plt.ylabel('From Category')
                    plt.tight_layout()
                    plt.savefig(os.path.join(transition_dir, f"{prefix}_category_transition_matrix.png"))
                    plt.close()
                
                # 3. Transition network graph
                if 'transition_probabilities' in data:
                    try:
                        import networkx as nx
                        
                        # Create dataframe
                        transition_probs = pd.DataFrame(data['transition_probabilities'])
                        
                        # Create graph
                        G = nx.DiGraph()
                        
                        # Add nodes
                        for regime in transition_probs.index:
                            # Get category
                            category = next(
                                (cat for cat, regimes in self.regime_categories.items() if regime in regimes),
                                'Other'
                            )
                            
                            # Add node
                            G.add_node(regime, category=category)
                        
                        # Add edges
                        for from_regime in transition_probs.index:
                            for to_regime in transition_probs.columns:
                                prob = transition_probs.loc[from_regime, to_regime]
                                
                                if prob > 0.05:  # Only add significant transitions
                                    G.add_edge(from_regime, to_regime, weight=prob)
                        
                        # Plot
                        plt.figure(figsize=(16, 12))
                        
                        # Create position layout
                        pos = nx.spring_layout(G, k=0.3, iterations=50)
                        
                        # Draw nodes
                        for category, regimes in self.regime_categories.items():
                            # Filter nodes in this category
                            category_nodes = [n for n, d in G.nodes(data=True) 
                                             if d.get('category') == category]
                            
                            if category_nodes:
                                nx.draw_networkx_nodes(
                                    G, pos,
                                    nodelist=category_nodes,
                                    node_color=self.category_colors.get(category, '#CCCCCC'),
                                    node_size=500,
                                    alpha=0.8,
                                    label=category
                                )
                        
                        # Draw edges
                        edges = G.edges(data=True)
                        weights = [d['weight'] * 5 for _, _, d in edges]
                        
                        nx.draw_networkx_edges(
                            G, pos,
                            width=weights,
                            alpha=0.7,
                            edge_color='gray',
                            arrows=True,
                            arrowsize=15,
                            arrowstyle='-|>'
                        )
                        
                        # Draw labels
                        nx.draw_networkx_labels(
                            G, pos,
                            font_size=8,
                            font_family='sans-serif'
                        )
                        
                        plt.title(f'Market Regime Transition Network - {prefix}')
                        plt.axis('off')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(transition_dir, f"{prefix}_transition_network.png"))
                        plt.close()
                    
                    except ImportError:
                        logger.warning("NetworkX not available, skipping transition network graph")
            
            logger.info(f"Generated regime transition visualizations in {transition_dir}")
        
        except Exception as e:
            logger.error(f"Error generating regime transition visualizations: {str(e)}")
    
    def generate_component_contribution_visualizations(self, metrics, viz_dir):
        """
        Generate component contribution visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            viz_dir (str): Visualizations directory
        """
        try:
            # Create component contribution directory
            component_dir = os.path.join(viz_dir, 'component_contribution')
            os.makedirs(component_dir, exist_ok=True)
            
            if 'component_contribution' not in metrics:
                logger.warning("No component contribution metrics found")
                return
            
            comp_contrib = metrics['component_contribution']
            
            for file_name, data in comp_contrib.items():
                # Create file prefix
                prefix = file_name.replace('.csv', '')
                
                # 1. Component signal distribution
                if 'component_signal_counts' in data:
                    # Create dataframe
                    signal_counts = pd.DataFrame(data['component_signal_counts'])
                    
                    # Plot
                    plt.figure(figsize=(14, 10))
                    signal_counts.plot(kind='bar', stacked=True)
                    plt.title(f'Component Signal Distribution - {prefix}')
                    plt.xlabel('Component')
                    plt.ylabel('Count')
                    plt.legend(title='Signal')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(component_dir, f"{prefix}_component_signal_distribution.png"))
                    plt.close()
                
                # 2. Component weight distribution
                if 'component_weights' in data:
                    # Create dataframe
                    weight_data = []
                    
                    for component, weights in data['component_weights'].items():
                        if 'mean' in weights:
                            weight_data.append({
                                'Component': component,
                                'Mean Weight': weights['mean']
                            })
                    
                    weight_df = pd.DataFrame(weight_data)
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    bars = plt.bar(weight_df['Component'], weight_df['Mean Weight'])
                    plt.title(f'Component Weight Distribution - {prefix}')
                    plt.xlabel('Component')
                    plt.ylabel('Mean Weight')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add weight labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.01,
                            f'{height:.2f}',
                            ha='center',
                            va='bottom',
                            rotation=0
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(component_dir, f"{prefix}_component_weight_distribution.png"))
                    plt.close()
                
                # 3. Component-regime correlation heatmap
                if 'component_regime_counts' in data:
                    # Create dataframe
                    regime_counts = pd.DataFrame(data['component_regime_counts'])
                    
                    # Normalize by row
                    regime_probs = regime_counts.copy()
                    
                    for i in regime_probs.index:
                        row_sum = regime_probs.loc[i].sum()
                        if row_sum > 0:
                            regime_probs.loc[i] = regime_probs.loc[i] / row_sum
                    
                    # Plot
                    plt.figure(figsize=(16, 10))
                    sns.heatmap(
                        regime_probs,
                        annot=True,
                        fmt='.2f',
                        cmap='viridis',
                        linewidths=0.5,
                        cbar_kws={'label': 'Probability'}
                    )
                    plt.title(f'Component-Regime Correlation - {prefix}')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Component')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(component_dir, f"{prefix}_component_regime_correlation.png"))
                    plt.close()
                
                # 4. Component-category correlation heatmap
                if 'component_category_counts' in data:
                    # Create dataframe
                    category_counts = pd.DataFrame(data['component_category_counts'])
                    
                    # Normalize by row
                    category_probs = category_counts.copy()
                    
                    for i in category_probs.index:
                        row_sum = category_probs.loc[i].sum()
                        if row_sum > 0:
                            category_probs.loc[i] = category_probs.loc[i] / row_sum
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(
                        category_probs,
                        annot=True,
                        fmt='.2f',
                        cmap='viridis',
                        linewidths=0.5,
                        cbar_kws={'label': 'Probability'}
                    )
                    plt.title(f'Component-Category Correlation - {prefix}')
                    plt.xlabel('Market Regime Category')
                    plt.ylabel('Component')
                    plt.tight_layout()
                    plt.savefig(os.path.join(component_dir, f"{prefix}_component_category_correlation.png"))
                    plt.close()
            
            logger.info(f"Generated component contribution visualizations in {component_dir}")
        
        except Exception as e:
            logger.error(f"Error generating component contribution visualizations: {str(e)}")
    
    def generate_time_based_visualizations(self, metrics, viz_dir):
        """
        Generate time-based visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            viz_dir (str): Visualizations directory
        """
        try:
            # Create time-based directory
            time_dir = os.path.join(viz_dir, 'time_based')
            os.makedirs(time_dir, exist_ok=True)
            
            if 'time_based' not in metrics:
                logger.warning("No time-based metrics found")
                return
            
            time_based = metrics['time_based']
            
            for file_name, data in time_based.items():
                # Create file prefix
                prefix = file_name.replace('.csv', '')
                
                # 1. Regime by time of day
                if 'time_of_day_regime_counts' in data:
                    # Create dataframe
                    time_regime_counts = pd.DataFrame(data['time_of_day_regime_counts'])
                    
                    # Plot
                    plt.figure(figsize=(14, 10))
                    time_regime_counts.plot(kind='bar', stacked=True)
                    plt.title(f'Market Regime by Time of Day - {prefix}')
                    plt.xlabel('Time of Day')
                    plt.ylabel('Count')
                    plt.legend(title='Market Regime')
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(time_dir, f"{prefix}_regime_by_time_of_day.png"))
                    plt.close()
                
                # 2. Regime by hour
                if 'hour_regime_counts' in data:
                    # Create dataframe
                    hour_regime_counts = pd.DataFrame(data['hour_regime_counts'])
                    
                    # Plot
                    plt.figure(figsize=(14, 10))
                    hour_regime_counts.plot(kind='bar', stacked=True)
                    plt.title(f'Market Regime by Hour - {prefix}')
                    plt.xlabel('Hour')
                    plt.ylabel('Count')
                    plt.legend(title='Market Regime')
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(time_dir, f"{prefix}_regime_by_hour.png"))
                    plt.close()
                
                # 3. Regime by day of week
                if 'day_of_week_regime_counts' in data:
                    # Create dataframe
                    day_regime_counts = pd.DataFrame(data['day_of_week_regime_counts'])
                    
                    # Reorder days
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_regime_counts = day_regime_counts.reindex(
                        [d for d in days_order if d in day_regime_counts.index]
                    )
                    
                    # Plot
                    plt.figure(figsize=(14, 10))
                    day_regime_counts.plot(kind='bar', stacked=True)
                    plt.title(f'Market Regime by Day of Week - {prefix}')
                    plt.xlabel('Day of Week')
                    plt.ylabel('Count')
                    plt.legend(title='Market Regime')
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(time_dir, f"{prefix}_regime_by_day_of_week.png"))
                    plt.close()
                
                # 4. Regime duration box plot
                if 'duration_statistics' in data:
                    # Create dataframe
                    duration_data = []
                    
                    for regime, stats in data['duration_statistics'].items():
                        if 'mean' in stats and 'median' in stats and 'min' in stats and 'max' in stats:
                            duration_data.append({
                                'Regime': regime,
                                'Mean Duration (min)': stats['mean'],
                                'Median Duration (min)': stats['median'],
                                'Min Duration (min)': stats['min'],
                                'Max Duration (min)': stats['max']
                            })
                    
                    duration_df = pd.DataFrame(duration_data)
                    
                    # Sort by regime category
                    sorted_regimes = []
                    
                    for category, regimes in self.regime_categories.items():
                        for regime in regimes:
                            if regime in duration_df['Regime'].values:
                                sorted_regimes.append(regime)
                    
                    # Add any remaining regimes
                    for regime in duration_df['Regime']:
                        if regime not in sorted_regimes:
                            sorted_regimes.append(regime)
                    
                    # Filter and sort dataframe
                    duration_df = duration_df[duration_df['Regime'].isin(sorted_regimes)]
                    duration_df['Regime'] = pd.Categorical(
                        duration_df['Regime'],
                        categories=sorted_regimes,
                        ordered=True
                    )
                    duration_df = duration_df.sort_values('Regime')
                    
                    # Create color map
                    colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in duration_df['Regime']]
                    
                    # Plot mean duration
                    plt.figure(figsize=(14, 8))
                    bars = plt.bar(duration_df['Regime'], duration_df['Mean Duration (min)'], color=colors)
                    plt.title(f'Market Regime Mean Duration - {prefix}')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Mean Duration (minutes)')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add duration labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 1,
                            f'{height:.1f}',
                            ha='center',
                            va='bottom',
                            rotation=0
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(time_dir, f"{prefix}_regime_mean_duration.png"))
                    plt.close()
            
            logger.info(f"Generated time-based visualizations in {time_dir}")
        
        except Exception as e:
            logger.error(f"Error generating time-based visualizations: {str(e)}")
    
    def generate_multi_timeframe_visualizations(self, metrics, viz_dir):
        """
        Generate multi-timeframe visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            viz_dir (str): Visualizations directory
        """
        try:
            # Create multi-timeframe directory
            multi_dir = os.path.join(viz_dir, 'multi_timeframe')
            os.makedirs(multi_dir, exist_ok=True)
            
            if 'multi_timeframe' not in metrics:
                logger.warning("No multi-timeframe metrics found")
                return
            
            multi_tf = metrics['multi_timeframe']
            
            # 1. Timeframe comparison
            timeframe_data = {}
            
            for timeframe, tf_data in multi_tf.items():
                if timeframe != 'cross_timeframe_agreement':
                    # Calculate average regime distribution
                    regime_percentages = {}
                    
                    for file_name, data in tf_data.items():
                        if 'regime_percentage' in data:
                            for regime, percentage in data['regime_percentage'].items():
                                if regime not in regime_percentages:
                                    regime_percentages[regime] = []
                                
                                regime_percentages[regime].append(percentage)
                    
                    # Calculate average percentage
                    avg_regime_percentage = {}
                    
                    for regime, percentages in regime_percentages.items():
                        avg_regime_percentage[regime] = sum(percentages) / len(percentages)
                    
                    timeframe_data[timeframe] = avg_regime_percentage
            
            if timeframe_data:
                # Create dataframe
                tf_df = pd.DataFrame(timeframe_data)
                
                # Plot
                plt.figure(figsize=(14, 10))
                tf_df.plot(kind='bar')
                plt.title('Market Regime Distribution by Timeframe')
                plt.xlabel('Market Regime')
                plt.ylabel('Percentage (%)')
                plt.legend(title='Timeframe')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(multi_dir, "regime_distribution_by_timeframe.png"))
                plt.close()
            
            # 2. Cross-timeframe agreement
            if 'cross_timeframe_agreement' in multi_tf:
                cross_tf = multi_tf['cross_timeframe_agreement']
                
                if 'agreement_data' in cross_tf:
                    # Create dataframe
                    agreement_df = pd.DataFrame(cross_tf['agreement_data'])
                    
                    if 'datetime' in agreement_df.columns and 'agreement_score' in agreement_df.columns:
                        # Plot agreement score over time
                        plt.figure(figsize=(14, 8))
                        plt.plot(agreement_df['datetime'], agreement_df['agreement_score'])
                        plt.title('Multi-Timeframe Agreement Score Over Time')
                        plt.xlabel('Datetime')
                        plt.ylabel('Agreement Score')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(multi_dir, "timeframe_agreement_score.png"))
                        plt.close()
                    
                    if 'most_common_regime' in agreement_df.columns:
                        # Plot most common regime over time
                        plt.figure(figsize=(14, 8))
                        
                        # Create numeric mapping for regimes
                        regimes = agreement_df['most_common_regime'].unique()
                        regime_map = {regime: i for i, regime in enumerate(regimes)}
                        
                        # Convert regimes to numeric values
                        numeric_regimes = agreement_df['most_common_regime'].map(regime_map)
                        
                        # Plot
                        plt.scatter(agreement_df['datetime'], numeric_regimes, alpha=0.7)
                        plt.yticks(range(len(regimes)), regimes)
                        plt.title('Most Common Market Regime Across Timeframes')
                        plt.xlabel('Datetime')
                        plt.ylabel('Market Regime')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(multi_dir, "most_common_regime.png"))
                        plt.close()
                
                if 'most_common_regimes' in cross_tf:
                    # Create dataframe
                    most_common = pd.DataFrame({
                        'Regime': list(cross_tf['most_common_regimes'].keys()),
                        'Count': list(cross_tf['most_common_regimes'].values())
                    })
                    
                    # Sort by count
                    most_common = most_common.sort_values('Count', ascending=False)
                    
                    # Create color map
                    colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in most_common['Regime']]
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    bars = plt.bar(most_common['Regime'], most_common['Count'], color=colors)
                    plt.title('Most Common Market Regimes Across Timeframes')
                    plt.xlabel('Market Regime')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add count labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.5,
                            f'{height:.0f}',
                            ha='center',
                            va='bottom',
                            rotation=0
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(multi_dir, "most_common_regimes_count.png"))
                    plt.close()
            
            logger.info(f"Generated multi-timeframe visualizations in {multi_dir}")
        
        except Exception as e:
            logger.error(f"Error generating multi-timeframe visualizations: {str(e)}")
    
    def generate_summary_visualizations(self, metrics, viz_dir):
        """
        Generate summary visualizations.
        
        Args:
            metrics (dict): Dictionary of metrics
            viz_dir (str): Visualizations directory
        """
        try:
            # Create summary directory
            summary_dir = os.path.join(viz_dir, 'summary')
            os.makedirs(summary_dir, exist_ok=True)
            
            # Generate summary metrics if not already generated
            summary_metrics = self.generate_summary_metrics(metrics)
            
            # 1. Average regime distribution
            if 'regime_distribution' in summary_metrics and 'avg_regime_percentage' in summary_metrics['regime_distribution']:
                avg_regime_percentage = summary_metrics['regime_distribution']['avg_regime_percentage']
                
                # Sort regimes by category
                sorted_regimes = []
                
                for category, regimes in self.regime_categories.items():
                    for regime in regimes:
                        if regime in avg_regime_percentage:
                            sorted_regimes.append(regime)
                
                # Add any remaining regimes
                for regime in avg_regime_percentage:
                    if regime not in sorted_regimes:
                        sorted_regimes.append(regime)
                
                # Create dataframe
                regime_df = pd.DataFrame({
                    'Regime': sorted_regimes,
                    'Percentage': [avg_regime_percentage.get(regime, 0) for regime in sorted_regimes]
                })
                
                # Create color map
                colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in sorted_regimes]
                
                # Plot
                plt.figure(figsize=(14, 8))
                bars = plt.bar(regime_df['Regime'], regime_df['Percentage'], color=colors)
                plt.title('Average Market Regime Distribution')
                plt.xlabel('Market Regime')
                plt.ylabel('Percentage (%)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.5,
                        f'{height:.1f}%',
                        ha='center',
                        va='bottom',
                        rotation=0
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, "avg_regime_distribution.png"))
                plt.close()
            
            # 2. Average category distribution
            if 'regime_distribution' in summary_metrics and 'avg_category_percentage' in summary_metrics['regime_distribution']:
                avg_category_percentage = summary_metrics['regime_distribution']['avg_category_percentage']
                
                # Create dataframe
                category_df = pd.DataFrame({
                    'Category': list(avg_category_percentage.keys()),
                    'Percentage': list(avg_category_percentage.values())
                })
                
                # Create color map
                colors = [self.category_colors.get(category, '#CCCCCC') for category in category_df['Category']]
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.pie(
                    category_df['Percentage'],
                    labels=category_df['Category'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors
                )
                plt.title('Average Market Regime Category Distribution')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, "avg_category_distribution.png"))
                plt.close()
            
            # 3. Average regime duration
            if 'time_based' in summary_metrics and 'avg_regime_durations' in summary_metrics['time_based']:
                avg_regime_durations = summary_metrics['time_based']['avg_regime_durations']
                
                # Create dataframe
                duration_data = []
                
                for regime, stats in avg_regime_durations.items():
                    if 'mean' in stats:
                        duration_data.append({
                            'Regime': regime,
                            'Mean Duration (min)': stats['mean']
                        })
                
                duration_df = pd.DataFrame(duration_data)
                
                # Sort by regime category
                sorted_regimes = []
                
                for category, regimes in self.regime_categories.items():
                    for regime in regimes:
                        if regime in duration_df['Regime'].values:
                            sorted_regimes.append(regime)
                
                # Add any remaining regimes
                for regime in duration_df['Regime']:
                    if regime not in sorted_regimes:
                        sorted_regimes.append(regime)
                
                # Filter and sort dataframe
                duration_df = duration_df[duration_df['Regime'].isin(sorted_regimes)]
                duration_df['Regime'] = pd.Categorical(
                    duration_df['Regime'],
                    categories=sorted_regimes,
                    ordered=True
                )
                duration_df = duration_df.sort_values('Regime')
                
                # Create color map
                colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in duration_df['Regime']]
                
                # Plot
                plt.figure(figsize=(14, 8))
                bars = plt.bar(duration_df['Regime'], duration_df['Mean Duration (min)'], color=colors)
                plt.title('Average Market Regime Duration')
                plt.xlabel('Market Regime')
                plt.ylabel('Mean Duration (minutes)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add duration labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 1,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        rotation=0
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, "avg_regime_duration.png"))
                plt.close()
            
            # 4. Average component weights
            if 'component_contribution' in summary_metrics and 'avg_component_weights' in summary_metrics['component_contribution']:
                avg_component_weights = summary_metrics['component_contribution']['avg_component_weights']
                
                # Create dataframe
                weight_data = []
                
                for component, weights in avg_component_weights.items():
                    if 'mean' in weights:
                        weight_data.append({
                            'Component': component,
                            'Mean Weight': weights['mean']
                        })
                
                weight_df = pd.DataFrame(weight_data)
                
                # Plot
                plt.figure(figsize=(12, 8))
                bars = plt.bar(weight_df['Component'], weight_df['Mean Weight'])
                plt.title('Average Component Weight Distribution')
                plt.xlabel('Component')
                plt.ylabel('Mean Weight')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add weight labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.01,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        rotation=0
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, "avg_component_weight_distribution.png"))
                plt.close()
            
            logger.info(f"Generated summary visualizations in {summary_dir}")
        
        except Exception as e:
            logger.error(f"Error generating summary visualizations: {str(e)}")
    
    def generate_combined_visualizations(self, results, viz_dir):
        """
        Generate combined visualizations.
        
        Args:
            results (dict): Dictionary mapping file names to result dataframes
            viz_dir (str): Visualizations directory
        """
        try:
            # Create combined directory
            combined_dir = os.path.join(viz_dir, 'combined')
            os.makedirs(combined_dir, exist_ok=True)
            
            # 1. Market regime over time for all files
            regime_time_data = []
            
            for file_name, data in results.items():
                if 'datetime' in data.columns and 'market_regime' in data.columns:
                    # Extract timeframe from filename
                    timeframe = None
                    
                    for tf in ['5m', '15m', '1h', '1d']:
                        if tf in file_name:
                            timeframe = tf
                            break
                    
                    if timeframe is None:
                        timeframe = 'unknown'
                    
                    # Add to data
                    for i, row in data.iterrows():
                        regime_time_data.append({
                            'datetime': row['datetime'],
                            'market_regime': row['market_regime'],
                            'timeframe': timeframe
                        })
            
            if regime_time_data:
                # Create dataframe
                regime_time_df = pd.DataFrame(regime_time_data)
                
                # Sort by datetime
                regime_time_df = regime_time_df.sort_values('datetime')
                
                # Create separate plot for each timeframe
                for timeframe in regime_time_df['timeframe'].unique():
                    # Filter data
                    tf_data = regime_time_df[regime_time_df['timeframe'] == timeframe]
                    
                    # Create numeric mapping for regimes
                    regimes = tf_data['market_regime'].unique()
                    regime_map = {regime: i for i, regime in enumerate(regimes)}
                    
                    # Convert regimes to numeric values
                    numeric_regimes = tf_data['market_regime'].map(regime_map)
                    
                    # Create color map
                    colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in tf_data['market_regime']]
                    
                    # Plot
                    plt.figure(figsize=(16, 10))
                    plt.scatter(tf_data['datetime'], numeric_regimes, c=colors, alpha=0.7)
                    plt.yticks(range(len(regimes)), regimes)
                    plt.title(f'Market Regime Over Time - {timeframe}')
                    plt.xlabel('Datetime')
                    plt.ylabel('Market Regime')
                    plt.grid(True, alpha=0.3)
                    
                    # Format x-axis
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.gcf().autofmt_xdate()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(combined_dir, f"regime_over_time_{timeframe}.png"))
                    plt.close()
            
            # 2. Combined regime distribution
            regime_counts = {}
            
            for file_name, data in results.items():
                if 'market_regime' in data.columns:
                    # Count regimes
                    for regime in data['market_regime'].unique():
                        if regime not in regime_counts:
                            regime_counts[regime] = 0
                        
                        regime_counts[regime] += (data['market_regime'] == regime).sum()
            
            if regime_counts:
                # Sort regimes by category
                sorted_regimes = []
                
                for category, regimes in self.regime_categories.items():
                    for regime in regimes:
                        if regime in regime_counts:
                            sorted_regimes.append(regime)
                
                # Add any remaining regimes
                for regime in regime_counts:
                    if regime not in sorted_regimes:
                        sorted_regimes.append(regime)
                
                # Create dataframe
                regime_df = pd.DataFrame({
                    'Regime': sorted_regimes,
                    'Count': [regime_counts.get(regime, 0) for regime in sorted_regimes]
                })
                
                # Calculate percentage
                total_count = regime_df['Count'].sum()
                regime_df['Percentage'] = regime_df['Count'] / total_count * 100
                
                # Create color map
                colors = [self.regime_colors.get(regime, '#CCCCCC') for regime in sorted_regimes]
                
                # Plot
                plt.figure(figsize=(14, 8))
                bars = plt.bar(regime_df['Regime'], regime_df['Percentage'], color=colors)
                plt.title('Combined Market Regime Distribution')
                plt.xlabel('Market Regime')
                plt.ylabel('Percentage (%)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.5,
                        f'{height:.1f}%',
                        ha='center',
                        va='bottom',
                        rotation=0
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(combined_dir, "combined_regime_distribution.png"))
                plt.close()
            
            # 3. Combined category distribution
            category_counts = {}
            
            for file_name, data in results.items():
                if 'market_regime' in data.columns:
                    # Count categories
                    for category, regimes in self.regime_categories.items():
                        if category not in category_counts:
                            category_counts[category] = 0
                        
                        category_counts[category] += data['market_regime'].isin(regimes).sum()
            
            if category_counts:
                # Create dataframe
                category_df = pd.DataFrame({
                    'Category': list(category_counts.keys()),
                    'Count': list(category_counts.values())
                })
                
                # Calculate percentage
                total_count = category_df['Count'].sum()
                category_df['Percentage'] = category_df['Count'] / total_count * 100
                
                # Create color map
                colors = [self.category_colors.get(category, '#CCCCCC') for category in category_df['Category']]
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.pie(
                    category_df['Percentage'],
                    labels=category_df['Category'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors
                )
                plt.title('Combined Market Regime Category Distribution')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(os.path.join(combined_dir, "combined_category_distribution.png"))
                plt.close()
            
            logger.info(f"Generated combined visualizations in {combined_dir}")
        
        except Exception as e:
            logger.error(f"Error generating combined visualizations: {str(e)}")

def main():
    """
    Main function.
    """
    try:
        # Parse command line arguments
        import argparse
        
        parser = argparse.ArgumentParser(description='Generate performance metrics and visualizations')
        parser.add_argument('--results_dir', type=str, default='/home/ubuntu/market_regime_testing/test_results',
                           help='Directory containing market regime results')
        parser.add_argument('--output_dir', type=str, default=None,
                           help='Output directory for metrics and visualizations')
        
        args = parser.parse_args()
        
        # Create performance metrics generator
        generator = PerformanceMetricsGenerator(args.results_dir, args.output_dir)
        
        # Generate all metrics
        metrics = generator.generate_all_metrics()
        
        logger.info("Performance metrics and visualizations generation complete")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
