"""
Test Complete Market Regime System

This script tests the complete market regime system with sample data,
validating the entire pipeline from data preprocessing to market regime identification.
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

# Add project root to path
sys.path.append('/home/ubuntu/market_regime_testing')

# Import the unified pipeline
from unified_market_regime_pipeline import MarketRegimePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_market_regime_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Load data from file.
    
    Args:
        data_path (str): Path to data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            data = pd.read_excel(data_path)
        else:
            logger.error(f"Unsupported file format: {data_path}")
            return None
        
        logger.info(f"Loaded data from {data_path} with {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        return None

def preprocess_data(data):
    """
    Preprocess data for market regime analysis.
    
    Args:
        data (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    try:
        # Make a copy
        df = data.copy()
        
        # Ensure datetime is in datetime format
        if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except:
                logger.warning("Failed to convert datetime column to datetime format")
                
                # Try to create datetime from date and time columns
                if 'date' in df.columns and 'time' in df.columns:
                    try:
                        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                    except:
                        logger.warning("Failed to create datetime from date and time columns")
        
        # Ensure required columns exist
        required_columns = ['strike', 'option_type', 'open_interest', 'price', 'underlying_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to find alternative column names
            column_mapping = {
                'strike': ['Strike', 'strike_price', 'STRIKE'],
                'option_type': ['type', 'call_put', 'cp', 'option_type'],
                'open_interest': ['OI', 'OPEN_INTEREST', 'oi'],
                'price': ['close', 'Close', 'CLOSE', 'last_price'],
                'underlying_price': ['underlying', 'Underlying', 'spot_price', 'index_price']
            }
            
            for missing_col in missing_columns:
                for alt_col in column_mapping.get(missing_col, []):
                    if alt_col in df.columns:
                        df[missing_col] = df[alt_col]
                        logger.info(f"Using {alt_col} as {missing_col}")
                        break
            
            # Check again after mapping
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns after mapping: {missing_columns}")
        
        # Sort by datetime and strike
        if 'datetime' in df.columns and 'strike' in df.columns:
            df = df.sort_values(['datetime', 'strike'])
        
        logger.info(f"Preprocessed data with {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return data

def test_market_regime_pipeline(data_path, output_dir, dte=None, timeframe='5m'):
    """
    Test the market regime pipeline with sample data.
    
    Args:
        data_path (str): Path to data file
        output_dir (str): Output directory
        dte (int, optional): Days to expiry
        timeframe (str, optional): Timeframe of the data
        
    Returns:
        pd.DataFrame: Processed data with market regime
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = load_data(data_path)
        
        if data is None:
            logger.error("Failed to load data")
            return None
        
        # Preprocess data
        preprocessed_data = preprocess_data(data)
        
        # Initialize pipeline
        config = {
            'output_dir': output_dir,
            'component_weights': {
                'greek_sentiment': 0.20,
                'trending_oi_pa': 0.30,
                'iv_skew': 0.20,
                'ema_indicators': 0.15,
                'vwap_indicators': 0.15
            },
            'use_multi_timeframe': False,
            'use_time_of_day_adjustments': True
        }
        
        pipeline = MarketRegimePipeline(config)
        
        # Process data
        processed_data = pipeline.process_data(preprocessed_data, dte, timeframe)
        
        # Prepare for consolidator
        consolidator_data = pipeline.prepare_for_consolidator(processed_data, dte)
        
        # Save output
        output_file = os.path.join(output_dir, f"consolidator_{timeframe}_dte{dte}.csv")
        consolidator_data.to_csv(output_file, index=False)
        
        logger.info(f"Saved consolidator data to {output_file}")
        
        # Analyze results
        analyze_results(processed_data, output_dir, timeframe, dte)
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error testing market regime pipeline: {str(e)}")
        return None

def test_multi_timeframe_pipeline(data_files, output_dir, dte=None):
    """
    Test the multi-timeframe pipeline with sample data.
    
    Args:
        data_files (dict): Dictionary mapping timeframes to data files
        output_dir (str): Output directory
        dte (int, optional): Days to expiry
        
    Returns:
        dict: Dictionary mapping timeframes to processed data
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipeline
        config = {
            'output_dir': output_dir,
            'component_weights': {
                'greek_sentiment': 0.20,
                'trending_oi_pa': 0.30,
                'iv_skew': 0.20,
                'ema_indicators': 0.15,
                'vwap_indicators': 0.15
            },
            'use_multi_timeframe': True,
            'timeframe_weights': {
                '5m': 0.20,
                '15m': 0.30,
                '1h': 0.30,
                '1d': 0.20
            },
            'use_time_of_day_adjustments': True
        }
        
        pipeline = MarketRegimePipeline(config)
        
        # Run multi-timeframe pipeline
        results = pipeline.run_multi_timeframe_pipeline(data_files, output_dir, dte)
        
        # Analyze results for each timeframe
        for timeframe, processed_data in results.items():
            analyze_results(processed_data, output_dir, timeframe, dte)
        
        # Analyze multi-timeframe agreement
        analyze_multi_timeframe_agreement(results, output_dir, dte)
        
        return results
    
    except Exception as e:
        logger.error(f"Error testing multi-timeframe pipeline: {str(e)}")
        return {}

def analyze_results(data, output_dir, timeframe, dte=None):
    """
    Analyze market regime results.
    
    Args:
        data (pd.DataFrame): Processed data with market regime
        output_dir (str): Output directory
        timeframe (str): Timeframe of the data
        dte (int, optional): Days to expiry
    """
    try:
        # Create analysis directory
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create filename prefix
        dte_str = f"_dte{dte}" if dte is not None else ""
        prefix = f"{timeframe}{dte_str}"
        
        # 1. Market regime distribution
        regime_counts = data['market_regime'].value_counts()
        
        # Save to CSV
        regime_counts.to_csv(os.path.join(analysis_dir, f"regime_distribution_{prefix}.csv"))
        
        # Plot
        plt.figure(figsize=(12, 8))
        regime_counts.plot(kind='bar')
        plt.title(f'Market Regime Distribution ({timeframe})')
        plt.xlabel('Market Regime')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"regime_distribution_{prefix}.png"))
        plt.close()
        
        # 2. Market regime confidence
        if 'market_regime_confidence' in data.columns:
            regime_confidence = data.groupby('market_regime')['market_regime_confidence'].mean()
            
            # Save to CSV
            regime_confidence.to_csv(os.path.join(analysis_dir, f"regime_confidence_{prefix}.csv"))
            
            # Plot
            plt.figure(figsize=(12, 8))
            regime_confidence.plot(kind='bar')
            plt.title(f'Market Regime Confidence ({timeframe})')
            plt.xlabel('Market Regime')
            plt.ylabel('Average Confidence')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"regime_confidence_{prefix}.png"))
            plt.close()
        
        # 3. Market regime over time
        if 'datetime' in data.columns and 'market_regime' in data.columns:
            # Save to CSV
            time_regimes = data[['datetime', 'market_regime', 'market_regime_confidence']]
            time_regimes.to_csv(os.path.join(analysis_dir, f"regime_over_time_{prefix}.csv"), index=False)
            
            # Plot
            plt.figure(figsize=(14, 8))
            
            # Create numeric mapping for regimes
            regimes = data['market_regime'].unique()
            regime_map = {regime: i for i, regime in enumerate(regimes)}
            
            # Convert regimes to numeric values
            numeric_regimes = data['market_regime'].map(regime_map)
            
            # Plot
            plt.scatter(data['datetime'], numeric_regimes, alpha=0.7)
            plt.yticks(range(len(regimes)), regimes)
            plt.title(f'Market Regime Over Time ({timeframe})')
            plt.xlabel('Datetime')
            plt.ylabel('Market Regime')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"regime_over_time_{prefix}.png"))
            plt.close()
        
        # 4. Component contributions
        if 'component_contributions' in data.columns:
            # Extract component contributions
            component_data = []
            
            for i, row in data.iterrows():
                if pd.notna(row['component_contributions']):
                    contributions = json.loads(row['component_contributions'])
                    
                    for component, details in contributions.items():
                        if 'signal' in details and 'weight' in details:
                            component_data.append({
                                'datetime': row['datetime'] if 'datetime' in row else None,
                                'market_regime': row['market_regime'],
                                'component': component,
                                'signal': details['signal'],
                                'weight': details['weight']
                            })
            
            if component_data:
                # Create dataframe
                component_df = pd.DataFrame(component_data)
                
                # Save to CSV
                component_df.to_csv(os.path.join(analysis_dir, f"component_contributions_{prefix}.csv"), index=False)
                
                # Plot component signal distribution
                plt.figure(figsize=(14, 10))
                component_signal_counts = component_df.groupby(['component', 'signal']).size().unstack(fill_value=0)
                component_signal_counts.plot(kind='bar', stacked=True)
                plt.title(f'Component Signal Distribution ({timeframe})')
                plt.xlabel('Component')
                plt.ylabel('Count')
                plt.legend(title='Signal')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, f"component_signal_distribution_{prefix}.png"))
                plt.close()
        
        # 5. Time-of-day analysis
        if 'time_of_day' in data.columns and 'market_regime' in data.columns:
            # Save to CSV
            time_of_day_regimes = data.groupby(['time_of_day', 'market_regime']).size().unstack(fill_value=0)
            time_of_day_regimes.to_csv(os.path.join(analysis_dir, f"regime_by_time_of_day_{prefix}.csv"))
            
            # Plot
            plt.figure(figsize=(12, 8))
            time_of_day_regimes.plot(kind='bar', stacked=True)
            plt.title(f'Market Regime by Time of Day ({timeframe})')
            plt.xlabel('Time of Day')
            plt.ylabel('Count')
            plt.legend(title='Market Regime')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"regime_by_time_of_day_{prefix}.png"))
            plt.close()
        
        logger.info(f"Analyzed results for {timeframe}")
    
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")

def analyze_multi_timeframe_agreement(results, output_dir, dte=None):
    """
    Analyze multi-timeframe agreement.
    
    Args:
        results (dict): Dictionary mapping timeframes to processed data
        output_dir (str): Output directory
        dte (int, optional): Days to expiry
    """
    try:
        # Create analysis directory
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create filename prefix
        dte_str = f"_dte{dte}" if dte is not None else ""
        prefix = f"multi_timeframe{dte_str}"
        
        # Extract timeframes and regimes
        timeframes = list(results.keys())
        
        if not timeframes:
            logger.warning("No timeframes to analyze")
            return
        
        # Create agreement matrix
        agreement_data = []
        
        # Get common datetimes
        common_datetimes = set()
        
        for timeframe, data in results.items():
            if 'datetime' in data.columns:
                if not common_datetimes:
                    common_datetimes = set(data['datetime'])
                else:
                    common_datetimes &= set(data['datetime'])
        
        if not common_datetimes:
            logger.warning("No common datetimes to analyze")
            return
        
        # Convert to list and sort
        common_datetimes = sorted(list(common_datetimes))
        
        # Analyze agreement for each common datetime
        for dt in common_datetimes:
            regimes_at_dt = {}
            
            for timeframe, data in results.items():
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
            
            # Save to CSV
            agreement_df.to_csv(os.path.join(analysis_dir, f"timeframe_agreement_{prefix}.csv"), index=False)
            
            # Plot agreement score over time
            plt.figure(figsize=(14, 8))
            plt.plot(agreement_df['datetime'], agreement_df['agreement_score'])
            plt.title('Multi-Timeframe Agreement Score Over Time')
            plt.xlabel('Datetime')
            plt.ylabel('Agreement Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"timeframe_agreement_score_{prefix}.png"))
            plt.close()
            
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
            plt.savefig(os.path.join(analysis_dir, f"most_common_regime_{prefix}.png"))
            plt.close()
            
            # Plot agreement heatmap
            # Extract regimes for each timeframe
            heatmap_data = {}
            
            for timeframe in timeframes:
                heatmap_data[timeframe] = []
                
                for entry in agreement_data:
                    heatmap_data[timeframe].append(entry['regimes'].get(timeframe, 'Unknown'))
            
            # Create dataframe
            heatmap_df = pd.DataFrame(heatmap_data, index=agreement_df['datetime'])
            
            # Create numeric mapping for regimes
            all_regimes = set()
            for timeframe in timeframes:
                all_regimes.update(heatmap_df[timeframe].unique())
            
            regime_map = {regime: i for i, regime in enumerate(sorted(all_regimes))}
            
            # Convert regimes to numeric values
            heatmap_numeric = heatmap_df.copy()
            
            for timeframe in timeframes:
                heatmap_numeric[timeframe] = heatmap_df[timeframe].map(regime_map)
            
            # Plot heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(
                heatmap_numeric.T,
                cmap='viridis',
                cbar_kws={'label': 'Market Regime (Numeric)'}
            )
            plt.title('Market Regime by Timeframe')
            plt.xlabel('Datetime Index')
            plt.ylabel('Timeframe')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"regime_by_timeframe_{prefix}.png"))
            plt.close()
            
            logger.info(f"Analyzed multi-timeframe agreement")
        else:
            logger.warning("No agreement data to analyze")
    
    except Exception as e:
        logger.error(f"Error analyzing multi-timeframe agreement: {str(e)}")

def main():
    """
    Main function.
    """
    try:
        # Create output directory
        output_dir = os.path.join('/home/ubuntu/market_regime_testing/test_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Test single timeframe pipeline
        data_path = '/home/ubuntu/market_regime_testing/data/nifty_options_data.csv'
        dte = 5  # Example DTE
        
        logger.info(f"Testing single timeframe pipeline with {data_path}")
        processed_data = test_market_regime_pipeline(data_path, output_dir, dte, '5m')
        
        # Test multi-timeframe pipeline
        data_files = {
            '5m': '/home/ubuntu/market_regime_testing/data/nifty_options_data_5m.csv',
            '15m': '/home/ubuntu/market_regime_testing/data/nifty_options_data_15m.csv',
            '1h': '/home/ubuntu/market_regime_testing/data/nifty_options_data_1h.csv',
            '1d': '/home/ubuntu/market_regime_testing/data/nifty_options_data_1d.csv'
        }
        
        logger.info(f"Testing multi-timeframe pipeline with {len(data_files)} timeframes")
        multi_timeframe_results = test_multi_timeframe_pipeline(data_files, output_dir, dte)
        
        logger.info("Testing complete")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
