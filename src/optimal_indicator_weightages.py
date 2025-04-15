"""
Optimal Indicator Weightages Analysis

This script analyzes the processed market regime data to determine optimal
weightages for each indicator based on historical performance, time-of-day patterns,
and volatility conditions.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("optimal_weightages.log")
    ]
)
logger = logging.getLogger(__name__)

class OptimalWeightageAnalyzer:
    """
    A class for analyzing and determining optimal indicator weightages
    for market regime classification.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the OptimalWeightageAnalyzer.
        
        Args:
            data_dir (str, optional): Directory containing processed market regime data.
        """
        self.data_dir = data_dir or '/home/ubuntu/market_regime_testing/output/minute_regime_analysis'
        self.output_dir = os.path.join(self.data_dir, 'optimal_weightages')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default indicator weightages
        self.default_weightages = {
            'trending_oi_pa': 0.30,
            'ema_indicators': 0.20,
            'vwap_indicators': 0.20,
            'greek_sentiment': 0.25,
            'other_indicators': 0.05
        }
        
        # Time periods for time-of-day analysis
        self.time_periods = {
            'opening': ('09:15', '10:00'),
            'morning': ('10:00', '12:00'),
            'midday': ('12:00', '14:00'),
            'closing': ('14:00', '15:30')
        }
        
        # Volatility levels for volatility-based analysis
        self.volatility_levels = {
            'low': (0, 0.01),
            'medium': (0.01, 0.02),
            'high': (0.02, float('inf'))
        }
        
        logger.info("Initialized OptimalWeightageAnalyzer")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """
        Load processed market regime data.
        
        Returns:
            pd.DataFrame: Combined market regime data.
        """
        logger.info("Loading processed market regime data")
        
        try:
            # Look for combined results file
            combined_file = os.path.join(self.data_dir, "combined_minute_regime_classification.csv")
            
            if os.path.exists(combined_file):
                logger.info(f"Loading combined results from {combined_file}")
                data = pd.read_csv(combined_file)
                logger.info(f"Loaded {len(data)} rows from combined results")
                return data
            
            # If combined file doesn't exist, look for individual result files
            result_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                           if f.startswith("minute_regime_classification_") and f.endswith(".csv")]
            
            if result_files:
                logger.info(f"Loading {len(result_files)} individual result files")
                dfs = []
                
                for file in result_files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                        logger.info(f"Loaded {len(df)} rows from {file}")
                    except Exception as e:
                        logger.error(f"Error loading {file}: {str(e)}")
                
                if dfs:
                    data = pd.concat(dfs, ignore_index=True)
                    logger.info(f"Combined {len(data)} rows from individual result files")
                    return data
                else:
                    logger.error("No data loaded from individual result files")
                    return pd.DataFrame()
            
            # If no result files found, look for intermediate results
            intermediate_dir = os.path.join(self.data_dir, 'intermediate_results')
            
            if os.path.exists(intermediate_dir):
                intermediate_files = [os.path.join(intermediate_dir, f) for f in os.listdir(intermediate_dir) 
                                     if f.endswith(".csv")]
                
                if intermediate_files:
                    logger.info(f"Loading {len(intermediate_files)} intermediate result files")
                    dfs = []
                    
                    for file in intermediate_files:
                        try:
                            df = pd.read_csv(file)
                            dfs.append(df)
                            logger.info(f"Loaded {len(df)} rows from {file}")
                        except Exception as e:
                            logger.error(f"Error loading {file}: {str(e)}")
                    
                    if dfs:
                        data = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Combined {len(data)} rows from intermediate result files")
                        return data
                    else:
                        logger.error("No data loaded from intermediate result files")
                        return pd.DataFrame()
            
            logger.error("No processed market regime data found")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def analyze_historical_performance(self, data):
        """
        Analyze historical performance of indicators.
        
        Args:
            data (pd.DataFrame): Market regime data.
            
        Returns:
            dict: Optimal weightages based on historical performance.
        """
        logger.info("Analyzing historical performance of indicators")
        
        try:
            # Ensure required columns exist
            required_cols = [
                'trending_oi_pa_component', 'ema_regime_component', 
                'vwap_regime_component', 'greek_regime_component',
                'combined_regime_component'
            ]
            
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for historical performance analysis: {missing_cols}")
                return self.default_weightages
            
            # Calculate correlation between each indicator component and combined component
            correlations = {}
            
            for indicator, component_col in {
                'trending_oi_pa': 'trending_oi_pa_component',
                'ema_indicators': 'ema_regime_component',
                'vwap_indicators': 'vwap_regime_component',
                'greek_sentiment': 'greek_regime_component'
            }.items():
                if component_col in data.columns:
                    corr = data[component_col].corr(data['combined_regime_component'])
                    correlations[indicator] = abs(corr)  # Use absolute correlation
                else:
                    correlations[indicator] = 0.5  # Default if column doesn't exist
            
            # Normalize correlations to sum to 0.95 (leaving 0.05 for other indicators)
            total_corr = sum(correlations.values())
            
            if total_corr > 0:
                weightages = {
                    indicator: (corr / total_corr) * 0.95
                    for indicator, corr in correlations.items()
                }
                weightages['other_indicators'] = 0.05
            else:
                weightages = self.default_weightages
            
            logger.info(f"Historical performance weightages: {weightages}")
            return weightages
        
        except Exception as e:
            logger.error(f"Error analyzing historical performance: {str(e)}")
            return self.default_weightages
    
    def analyze_time_of_day_patterns(self, data):
        """
        Analyze time-of-day patterns for indicator weightages.
        
        Args:
            data (pd.DataFrame): Market regime data.
            
        Returns:
            dict: Optimal weightages for different times of day.
        """
        logger.info("Analyzing time-of-day patterns for indicator weightages")
        
        try:
            # Ensure datetime column exists
            if 'datetime' not in data.columns:
                logger.warning("Missing datetime column for time-of-day analysis")
                return {period: self.default_weightages for period in self.time_periods}
            
            # Convert datetime to pandas datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                data['datetime'] = pd.to_datetime(data['datetime'])
            
            # Extract time of day
            data['time'] = data['datetime'].dt.strftime('%H:%M')
            
            # Define time periods
            time_period_weightages = {}
            
            for period, (start_time, end_time) in self.time_periods.items():
                # Filter data for this time period
                period_data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]
                
                if period_data.empty:
                    logger.warning(f"No data for time period {period} ({start_time} - {end_time})")
                    time_period_weightages[period] = self.default_weightages
                    continue
                
                # Analyze historical performance for this time period
                period_weightages = self.analyze_historical_performance(period_data)
                time_period_weightages[period] = period_weightages
                
                logger.info(f"Time period {period} ({start_time} - {end_time}) weightages: {period_weightages}")
            
            return time_period_weightages
        
        except Exception as e:
            logger.error(f"Error analyzing time-of-day patterns: {str(e)}")
            return {period: self.default_weightages for period in self.time_periods}
    
    def analyze_volatility_based_weightages(self, data):
        """
        Analyze volatility-based weightages for indicators.
        
        Args:
            data (pd.DataFrame): Market regime data.
            
        Returns:
            dict: Optimal weightages for different volatility levels.
        """
        logger.info("Analyzing volatility-based weightages for indicators")
        
        try:
            # Check if volatility column exists
            if 'volatility' not in data.columns:
                # Use absolute value of combined component as a proxy for volatility
                if 'combined_regime_component' in data.columns:
                    data['volatility'] = data['combined_regime_component'].abs()
                else:
                    logger.warning("Missing volatility and combined_regime_component columns for volatility-based analysis")
                    return {level: self.default_weightages for level in self.volatility_levels}
            
            # Define volatility levels
            volatility_level_weightages = {}
            
            for level, (min_vol, max_vol) in self.volatility_levels.items():
                # Filter data for this volatility level
                level_data = data[(data['volatility'] >= min_vol) & (data['volatility'] < max_vol)]
                
                if level_data.empty:
                    logger.warning(f"No data for volatility level {level} ({min_vol} - {max_vol})")
                    volatility_level_weightages[level] = self.default_weightages
                    continue
                
                # Analyze historical performance for this volatility level
                level_weightages = self.analyze_historical_performance(level_data)
                volatility_level_weightages[level] = level_weightages
                
                logger.info(f"Volatility level {level} ({min_vol} - {max_vol}) weightages: {level_weightages}")
            
            return volatility_level_weightages
        
        except Exception as e:
            logger.error(f"Error analyzing volatility-based weightages: {str(e)}")
            return {level: self.default_weightages for level in self.volatility_levels}
    
    def analyze_indicator_contributions(self, data):
        """
        Analyze individual indicator contributions to regime classification.
        
        Args:
            data (pd.DataFrame): Market regime data.
            
        Returns:
            dict: Indicator contribution analysis.
        """
        logger.info("Analyzing individual indicator contributions to regime classification")
        
        try:
            # Ensure required columns exist
            component_cols = {
                'trending_oi_pa': 'trending_oi_pa_component',
                'ema_indicators': 'ema_regime_component',
                'vwap_indicators': 'vwap_regime_component',
                'greek_sentiment': 'greek_regime_component'
            }
            
            missing_cols = [col for indicator, col in component_cols.items() if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for indicator contribution analysis: {missing_cols}")
            
            # Calculate contribution statistics for each indicator
            contributions = {}
            
            for indicator, component_col in component_cols.items():
                if component_col in data.columns:
                    # Calculate basic statistics
                    mean = float(data[component_col].mean())
                    std = float(data[component_col].std())
                    min_val = float(data[component_col].min())
                    max_val = float(data[component_col].max())
                    
                    # Calculate correlation with combined component
                    if 'combined_regime_component' in data.columns:
                        corr = float(data[component_col].corr(data['combined_regime_component']))
                    else:
                        corr = None
                    
                    # Calculate regime-specific contributions
                    regime_contributions = {}
                    
                    if 'regime' in data.columns:
                        for regime in data['regime'].unique():
                            regime_data = data[data['regime'] == regime]
                            regime_mean = float(regime_data[component_col].mean())
                            regime_contributions[regime] = regime_mean
                    
                    # Store contribution analysis
                    contributions[indicator] = {
                        'mean': mean,
                        'std': std,
                        'min': min_val,
                        'max': max_val,
                        'correlation_with_combined': corr,
                        'regime_specific_contributions': regime_contributions
                    }
                else:
                    contributions[indicator] = {
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'correlation_with_combined': None,
                        'regime_specific_contributions': {}
                    }
            
            logger.info(f"Analyzed contributions for {len(contributions)} indicators")
            return contributions
        
        except Exception as e:
            logger.error(f"Error analyzing indicator contributions: {str(e)}")
            return {}
    
    def generate_optimal_weightages_report(self, data):
        """
        Generate comprehensive report on optimal indicator weightages.
        
        Args:
            data (pd.DataFrame): Market regime data.
            
        Returns:
            dict: Comprehensive report on optimal indicator weightages.
        """
        logger.info("Generating comprehensive report on optimal indicator weightages")
        
        try:
            # Analyze historical performance
            historical_weightages = self.analyze_historical_performance(data)
            
            # Analyze time-of-day patterns
            time_of_day_weightages = self.analyze_time_of_day_patterns(data)
            
            # Analyze volatility-based weightages
            volatility_weightages = self.analyze_volatility_based_weightages(data)
            
            # Analyze indicator contributions
            indicator_contributions = self.analyze_indicator_contributions(data)
            
            # Generate comprehensive report
            report = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_analyzed': len(data),
                'overall_optimal_weightages': historical_weightages,
                'time_of_day_weightages': time_of_day_weightages,
                'volatility_based_weightages': volatility_weightages,
                'indicator_contributions': indicator_contributions,
                'recommendations': {
                    'default_weightages': historical_weightages,
                    'use_time_of_day_adjustment': True,
                    'use_volatility_adjustment': True,
                    'high_confidence_threshold': 0.7,
                    'implementation_notes': [
                        "Use overall optimal weightages as default",
                        "Adjust weightages based on time of day for more accurate regime classification",
                        "Further adjust weightages based on market volatility",
                        "Consider confidence scores when making trading decisions",
                        "Monitor and update weightages periodically based on recent performance"
                    ]
                }
            }
            
            # Save report to file
            report_file = os.path.join(self.output_dir, "optimal_weightages_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"Saved optimal weightages report to {report_file}")
            
            # Generate visualizations
            self.generate_visualizations(data, historical_weightages, time_of_day_weightages, volatility_weightages)
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating optimal weightages report: {str(e)}")
            return {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'default_weightages': self.default_weightages
            }
    
    def generate_visualizations(self, data, historical_weightages, time_of_day_weightages, volatility_weightages):
        """
        Generate visualizations for optimal indicator weightages.
        
        Args:
            data (pd.DataFrame): Market regime data.
            historical_weightages (dict): Historical performance weightages.
            time_of_day_weightages (dict): Time-of-day weightages.
            volatility_weightages (dict): Volatility-based weightages.
        """
        logger.info("Generating visualizations for optimal indicator weightages")
        
        try:
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. Overall optimal weightages
            plt.figure(figsize=(10, 6))
            indicators = list(historical_weightages.keys())
            weights = list(historical_weightages.values())
            
            plt.pie(weights, labels=indicators, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Overall Optimal Indicator Weightages')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'overall_optimal_weightages.png'))
            plt.close()
            
            # 2. Time-of-day weightages
            plt.figure(figsize=(12, 8))
            time_periods = list(time_of_day_weightages.keys())
            indicators = list(historical_weightages.keys())
            
            time_period_data = []
            for period in time_periods:
                for indicator in indicators:
                    time_period_data.append({
                        'Time Period': period,
                        'Indicator': indicator,
                        'Weight': time_of_day_weightages[period][indicator]
                    })
            
            time_period_df = pd.DataFrame(time_period_data)
            time_period_pivot = time_period_df.pivot(index='Time Period', columns='Indicator', values='Weight')
            
            ax = time_period_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))
            plt.title('Indicator Weightages by Time of Day')
            plt.xlabel('Time Period')
            plt.ylabel('Weight')
            plt.legend(title='Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'time_of_day_weightages.png'))
            plt.close()
            
            # 3. Volatility-based weightages
            plt.figure(figsize=(12, 8))
            volatility_levels = list(volatility_weightages.keys())
            
            volatility_data = []
            for level in volatility_levels:
                for indicator in indicators:
                    volatility_data.append({
                        'Volatility Level': level,
                        'Indicator': indicator,
                        'Weight': volatility_weightages[level][indicator]
                    })
            
            volatility_df = pd.DataFrame(volatility_data)
            volatility_pivot = volatility_df.pivot(index='Volatility Level', columns='Indicator', values='Weight')
            
            ax = volatility_pivot.plot(kind='bar', stacked=True, figsize=(12, 8))
            plt.title('Indicator Weightages by Volatility Level')
            plt.xlabel('Volatility Level')
            plt.ylabel('Weight')
            plt.legend(title='Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'volatility_weightages.png'))
            plt.close()
            
            # 4. Indicator correlations with combined component
            if 'combined_regime_component' in data.columns:
                plt.figure(figsize=(10, 6))
                correlations = {}
                
                for indicator, component_col in {
                    'trending_oi_pa': 'trending_oi_pa_component',
                    'ema_indicators': 'ema_regime_component',
                    'vwap_indicators': 'vwap_regime_component',
                    'greek_sentiment': 'greek_regime_component'
                }.items():
                    if component_col in data.columns:
                        correlations[indicator] = abs(data[component_col].corr(data['combined_regime_component']))
                
                if correlations:
                    indicators = list(correlations.keys())
                    corr_values = list(correlations.values())
                    
                    plt.bar(indicators, corr_values)
                    plt.title('Indicator Correlations with Combined Component')
                    plt.xlabel('Indicator')
                    plt.ylabel('Absolute Correlation')
                    plt.ylim(0, 1)
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'indicator_correlations.png'))
                plt.close()
            
            logger.info(f"Saved visualizations to {viz_dir}")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

def main():
    """
    Main function to analyze optimal indicator weightages.
    """
    # Initialize analyzer
    analyzer = OptimalWeightageAnalyzer()
    
    # Load data
    data = analyzer.load_data()
    
    if data.empty:
        logger.error("No data loaded, cannot determine optimal weightages")
        return
    
    # Generate optimal weightages report
    report = analyzer.generate_optimal_weightages_report(data)
    
    logger.info("Optimal weightages analysis completed")
    logger.info(f"Overall optimal weightages: {report['overall_optimal_weightages']}")

if __name__ == "__main__":
    main()
