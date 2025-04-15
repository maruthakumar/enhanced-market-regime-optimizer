"""
Script to test Greek sentiment analysis for market regime testing
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_regime_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GreekSentimentTester:
    """
    Class to test Greek sentiment analysis for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the Greek sentiment tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/greek_sentiment')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Greek sentiment parameters
        self.delta_threshold = self.config.get('delta_threshold', 0.5)
        self.gamma_threshold = self.config.get('gamma_threshold', 0.05)
        self.vega_threshold = self.config.get('vega_threshold', 0.1)
        self.theta_threshold = self.config.get('theta_threshold', 0.05)
        self.lookback_period = self.config.get('lookback_period', 5)
        self.atm_range = self.config.get('atm_range', 0.05)  # 5% range around ATM
        
        logger.info(f"Initialized GreekSentimentTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"Greek sentiment parameters: delta_threshold={self.delta_threshold}, gamma_threshold={self.gamma_threshold}")
    
    def load_data(self):
        """
        Load processed data for testing
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info("Loading processed data for Greek sentiment testing")
        
        # Try to load merged data first
        merged_data_path = os.path.join(self.data_dir, "merged_data.csv")
        if os.path.exists(merged_data_path):
            logger.info(f"Loading merged data from {merged_data_path}")
            df = pd.read_csv(merged_data_path)
            logger.info(f"Loaded merged data with {len(df)} rows")
            return df
        
        # If merged data doesn't exist, try to load individual processed files
        logger.info("Merged data not found, looking for individual processed files")
        processed_files = [f for f in os.listdir(self.data_dir) if f.startswith("processed_") and f.endswith(".csv")]
        
        if not processed_files:
            logger.error("No processed data files found")
            return None
        
        logger.info(f"Found {len(processed_files)} processed data files")
        dfs = []
        
        for file_name in processed_files:
            file_path = os.path.join(self.data_dir, file_name)
            logger.info(f"Loading data from {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_name}")
            except Exception as e:
                logger.error(f"Error loading {file_name}: {str(e)}")
        
        if not dfs:
            logger.error("No data loaded from processed files")
            return None
        
        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged {len(dfs)} dataframes with total {len(merged_df)} rows")
        
        return merged_df
    
    def calculate_greek_sentiment(self, df):
        """
        Calculate Greek sentiment indicators
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Greek sentiment indicators
        """
        logger.info("Calculating Greek sentiment indicators")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_columns = ['Strike', 'Option_Type', 'Underlying_Price']
        greek_columns = ['Delta', 'Gamma', 'Vega', 'Theta']
        
        # If Underlying_Price doesn't exist but Price does, use Price
        if 'Underlying_Price' not in result_df.columns and 'Price' in result_df.columns:
            logger.info("Using 'Price' column as 'Underlying_Price'")
            result_df['Underlying_Price'] = result_df['Price']
        
        # Check if required columns exist after adjustments
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        missing_greeks = [col for col in greek_columns if col not in result_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns after adjustments: {missing_columns}")
            return result_df
        
        if missing_greeks:
            logger.warning(f"Missing Greek columns: {missing_greeks}. Will create synthetic Greek data.")
            
            # Create synthetic Greek data
            if 'Strike' in result_df.columns and 'Underlying_Price' in result_df.columns and 'Option_Type' in result_df.columns:
                # Calculate moneyness
                result_df['Moneyness'] = result_df['Strike'] / result_df['Underlying_Price']
                
                # Create synthetic Greeks based on moneyness and option type
                if 'Delta' not in result_df.columns:
                    # For calls: Delta approaches 1 as moneyness decreases (ITM), approaches 0 as moneyness increases (OTM)
                    # For puts: Delta approaches -1 as moneyness increases (ITM), approaches 0 as moneyness decreases (OTM)
                    result_df['Delta'] = np.where(
                        result_df['Option_Type'].str.lower() == 'call',
                        0.5 - 0.5 * (result_df['Moneyness'] - 1) * 10,  # Simplified delta calculation for calls
                        -0.5 + 0.5 * (result_df['Moneyness'] - 1) * 10  # Simplified delta calculation for puts
                    )
                    # Clip delta values to valid range
                    result_df['Delta'] = np.where(
                        result_df['Option_Type'].str.lower() == 'call',
                        result_df['Delta'].clip(0, 1),
                        result_df['Delta'].clip(-1, 0)
                    )
                    logger.info("Created synthetic Delta values")
                
                if 'Gamma' not in result_df.columns:
                    # Gamma is highest for ATM options, decreases for ITM and OTM
                    result_df['Gamma'] = 4 * np.exp(-5 * (result_df['Moneyness'] - 1)**2)
                    logger.info("Created synthetic Gamma values")
                
                if 'Vega' not in result_df.columns:
                    # Vega is highest for ATM options, decreases for ITM and OTM
                    result_df['Vega'] = 10 * np.exp(-5 * (result_df['Moneyness'] - 1)**2)
                    logger.info("Created synthetic Vega values")
                
                if 'Theta' not in result_df.columns:
                    # Theta is negative, highest magnitude for ATM options
                    result_df['Theta'] = -5 * np.exp(-5 * (result_df['Moneyness'] - 1)**2)
                    logger.info("Created synthetic Theta values")
                
                # Add some randomness to synthetic Greeks
                np.random.seed(42)  # For reproducibility
                for greek in ['Delta', 'Gamma', 'Vega', 'Theta']:
                    if greek in result_df.columns:
                        result_df[greek] = result_df[greek] * (1 + np.random.normal(0, 0.1, len(result_df)))
                
                logger.info("Added randomness to synthetic Greek values")
        
        # Ensure datetime is in datetime format
        if 'datetime' in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df['datetime']):
            try:
                result_df['datetime'] = pd.to_datetime(result_df['datetime'])
                logger.info("Converted datetime to datetime format")
            except Exception as e:
                logger.warning(f"Failed to convert datetime to datetime format: {str(e)}")
                
                # If conversion fails, try to create datetime from Date and Time columns
                if 'Date' in result_df.columns and 'Time' in result_df.columns:
                    try:
                        result_df['datetime'] = pd.to_datetime(result_df['Date'].astype(str) + ' ' + result_df['Time'].astype(str))
                        logger.info("Created datetime from Date and Time columns")
                    except Exception as e2:
                        logger.error(f"Failed to create datetime from Date and Time: {str(e2)}")
        
        # Sort by datetime
        if 'datetime' in result_df.columns:
            result_df.sort_values('datetime', inplace=True)
            logger.info("Sorted data by datetime")
        
        try:
            # Process data by datetime to calculate Greek sentiment
            if 'datetime' in result_df.columns:
                # Group by datetime
                datetime_groups = result_df.groupby('datetime')
                
                # Initialize result columns
                result_df['Delta_Sentiment'] = np.nan
                result_df['Gamma_Sentiment'] = np.nan
                result_df['Vega_Sentiment'] = np.nan
                result_df['Theta_Sentiment'] = np.nan
                result_df['Combined_Greek_Sentiment'] = np.nan
                
                # Process each datetime group
                for dt, group in datetime_groups:
                    # Find ATM strikes
                    underlying_price = group['Underlying_Price'].iloc[0]
                    atm_lower = underlying_price * (1 - self.atm_range)
                    atm_upper = underlying_price * (1 + self.atm_range)
                    
                    # Get ATM options
                    atm_options = group[(group['Strike'] >= atm_lower) & (group['Strike'] <= atm_upper)]
                    
                    # Calculate Greek sentiment
                    if len(atm_options) > 0:
                        # Delta sentiment
                        call_delta_sum = atm_options[atm_options['Option_Type'].str.lower() == 'call']['Delta'].sum()
                        put_delta_sum = atm_options[atm_options['Option_Type'].str.lower() == 'put']['Delta'].sum()
                        
                        # Normalize delta sums
                        call_count = len(atm_options[atm_options['Option_Type'].str.lower() == 'call'])
                        put_count = len(atm_options[atm_options['Option_Type'].str.lower() == 'put'])
                        
                        if call_count > 0:
                            call_delta_avg = call_delta_sum / call_count
                        else:
                            call_delta_avg = 0
                        
                        if put_count > 0:
                            put_delta_avg = put_delta_sum / put_count
                        else:
                            put_delta_avg = 0
                        
                        # Calculate delta sentiment
                        # Positive delta sentiment: More call delta (bullish)
                        # Negative delta sentiment: More put delta (bearish)
                        delta_sentiment = call_delta_avg + put_delta_avg  # Put delta is negative
                        
                        # Gamma sentiment
                        gamma_sum = atm_options['Gamma'].sum()
                        gamma_avg = gamma_sum / len(atm_options) if len(atm_options) > 0 else 0
                        
                        # Calculate gamma sentiment
                        # High gamma: Market expecting large moves (volatile)
                        # Low gamma: Market expecting small moves (stable)
                        gamma_sentiment = 1 if gamma_avg > self.gamma_threshold else -1
                        
                        # Vega sentiment
                        vega_sum = atm_options['Vega'].sum()
                        vega_avg = vega_sum / len(atm_options) if len(atm_options) > 0 else 0
                        
                        # Calculate vega sentiment
                        # High vega: Market sensitive to volatility changes (uncertain)
                        # Low vega: Market less sensitive to volatility changes (confident)
                        vega_sentiment = 1 if vega_avg > self.vega_threshold else -1
                        
                        # Theta sentiment
                        theta_sum = atm_options['Theta'].sum()
                        theta_avg = theta_sum / len(atm_options) if len(atm_options) > 0 else 0
                        
                        # Calculate theta sentiment
                        # High theta (more negative): Time decay is significant (option sellers favored)
                        # Low theta (less negative): Time decay is less significant (option buyers favored)
                        theta_sentiment = 1 if theta_avg < -self.theta_threshold else -1
                        
                        # Calculate combined Greek sentiment
                        # Weight delta sentiment more heavily
                        combined_sentiment = 0.5 * delta_sentiment + 0.2 * gamma_sentiment + 0.2 * vega_sentiment + 0.1 * theta_sentiment
                        
                        # Update result dataframe
                        result_df.loc[result_df['datetime'] == dt, 'Delta_Sentiment'] = delta_sentiment
                        result_df.loc[result_df['datetime'] == dt, 'Gamma_Sentiment'] = gamma_sentiment
                        result_df.loc[result_df['datetime'] == dt, 'Vega_Sentiment'] = vega_sentiment
                        result_df.loc[result_df['datetime'] == dt, 'Theta_Sentiment'] = theta_sentiment
                        result_df.loc[result_df['datetime'] == dt, 'Combined_Greek_Sentiment'] = combined_sentiment
            
            # Calculate Greek sentiment regime
            if 'Combined_Greek_Sentiment' in result_df.columns:
                # Define regime thresholds
                result_df['Greek_Sentiment_Regime'] = pd.cut(
                    result_df['Combined_Greek_Sentiment'],
                    bins=[-float('inf'), -0.5, -0.2, 0.2, 0.5, float('inf')],
                    labels=['Strong_Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong_Bullish']
                )
            
            # Calculate confidence based on agreement among Greeks
            if all(col in result_df.columns for col in ['Delta_Sentiment', 'Gamma_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment']):
                # Convert sentiments to directional indicators (-1, 0, 1)
                result_df['Delta_Direction'] = np.sign(result_df['Delta_Sentiment'])
                result_df['Gamma_Direction'] = result_df['Gamma_Sentiment']
                result_df['Vega_Direction'] = result_df['Vega_Sentiment']
                result_df['Theta_Direction'] = result_df['Theta_Sentiment']
                
                # Calculate agreement score (how many Greeks agree with delta)
                result_df['Greek_Agreement'] = (
                    (result_df['Gamma_Direction'] == result_df['Delta_Direction']).astype(int) +
                    (result_df['Vega_Direction'] == result_df['Delta_Direction']).astype(int) +
                    (result_df['Theta_Direction'] == result_df['Delta_Direction']).astype(int)
                ) / 3.0
                
                # Calculate confidence based on agreement
                result_df['Greek_Sentiment_Confidence'] = 0.5 + 0.5 * result_df['Greek_Agreement']
            
            logger.info("Calculated Greek sentiment indicators")
            
        except Exception as e:
            logger.error(f"Error calculating Greek sentiment indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def visualize_greek_sentiment(self, df):
        """
        Visualize Greek sentiment indicators
        
        Args:
            df (pd.DataFrame): Dataframe with Greek sentiment indicators
        """
        logger.info("Visualizing Greek sentiment indicators")
        
        # Check if required columns exist
        sentiment_columns = ['Delta_Sentiment', 'Gamma_Sentiment', 'Vega_Sentiment', 'Theta_Sentiment', 'Combined_Greek_Sentiment']
        regime_columns = ['Greek_Sentiment_Regime']
        confidence_columns = ['Greek_Sentiment_Confidence']
        
        # Check which visualizations we can create
        can_visualize_sentiment = all(col in df.columns for col in sentiment_columns)
        can_visualize_regime = all(col in df.columns for col in regime_columns)
        can_visualize_confidence = all(col in df.columns for col in confidence_columns)
        
        if not (can_visualize_sentiment or can_visualize_regime or can_visualize_confidence):
            logger.error("No Greek sentiment indicators available for visualization")
            return
        
        try:
            # Visualize individual Greek sentiments
            if can_visualize_sentiment:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(5, 1, 1)
                plt.plot(df['Delta_Sentiment'], label='Delta Sentiment', linewidth=2)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Delta Sentiment')
                plt.grid(alpha=0.3)
                
                plt.subplot(5, 1, 2)
                plt.plot(df['Gamma_Sentiment'], label='Gamma Sentiment', linewidth=2)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Gamma Sentiment')
                plt.grid(alpha=0.3)
                
                plt.subplot(5, 1, 3)
                plt.plot(df['Vega_Sentiment'], label='Vega Sentiment', linewidth=2)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Vega Sentiment')
                plt.grid(alpha=0.3)
                
                plt.subplot(5, 1, 4)
                plt.plot(df['Theta_Sentiment'], label='Theta Sentiment', linewidth=2)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Theta Sentiment')
                plt.grid(alpha=0.3)
                
                plt.subplot(5, 1, 5)
                plt.plot(df['Combined_Greek_Sentiment'], label='Combined Greek Sentiment', linewidth=2)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Combined Greek Sentiment')
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                sentiment_plot_path = os.path.join(self.output_dir, 'greek_sentiments.png')
                plt.savefig(sentiment_plot_path)
                logger.info(f"Saved Greek sentiments plot to {sentiment_plot_path}")
                
                plt.close()
            
            # Visualize Greek sentiment regime
            if can_visualize_regime:
                # Create a figure for regime distribution
                plt.figure(figsize=(10, 6))
                
                regime_counts = df['Greek_Sentiment_Regime'].value_counts().sort_index()
                
                # Create a colormap for regimes
                regime_colors = {
                    'Strong_Bullish': 'green',
                    'Bullish': 'lightgreen',
                    'Neutral': 'gray',
                    'Bearish': 'lightcoral',
                    'Strong_Bearish': 'red'
                }
                
                # Plot regime distribution
                bars = plt.bar(regime_counts.index, regime_counts.values, 
                              color=[regime_colors.get(regime, 'blue') for regime in regime_counts.index])
                
                plt.title('Greek Sentiment Regime Distribution')
                plt.xlabel('Regime')
                plt.ylabel('Count')
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save plot
                regime_plot_path = os.path.join(self.output_dir, 'greek_sentiment_regime_distribution.png')
                plt.savefig(regime_plot_path)
                logger.info(f"Saved Greek sentiment regime distribution plot to {regime_plot_path}")
                
                plt.close()
                
                # Create a figure for regime over time
                if 'Underlying_Price' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Plot price
                    plt.plot(df['Underlying_Price'], label='Price', alpha=0.5, color='blue')
                    
                    # Plot regime as background color
                    x_points = range(len(df))
                    
                    # Plot colored background for each regime
                    for regime, color in regime_colors.items():
                        mask = df['Greek_Sentiment_Regime'] == regime
                        if mask.any():
                            plt.fill_between(x_points, 0, df['Underlying_Price'].max(), where=mask, color=color, alpha=0.2, label=regime)
                    
                    plt.title('Greek Sentiment Regime Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    regime_time_plot_path = os.path.join(self.output_dir, 'greek_sentiment_regime_time.png')
                    plt.savefig(regime_time_plot_path)
                    logger.info(f"Saved Greek sentiment regime over time plot to {regime_time_plot_path}")
                    
                    plt.close()
            
            # Visualize Greek sentiment confidence
            if can_visualize_confidence:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df['Greek_Sentiment_Confidence'], label='Confidence', linewidth=2)
                
                plt.title('Greek Sentiment Confidence')
                plt.xlabel('Time')
                plt.ylabel('Confidence')
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Neutral Confidence')
                plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='High Confidence')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                confidence_plot_path = os.path.join(self.output_dir, 'greek_sentiment_confidence.png')
                plt.savefig(confidence_plot_path)
                logger.info(f"Saved Greek sentiment confidence plot to {confidence_plot_path}")
                
                plt.close()
                
                # Create a figure for combined sentiment and confidence
                if 'Combined_Greek_Sentiment' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Create scatter plot with confidence as size
                    plt.scatter(range(len(df)), df['Combined_Greek_Sentiment'], 
                               s=df['Greek_Sentiment_Confidence'] * 100, alpha=0.5,
                               c=df['Combined_Greek_Sentiment'], cmap='RdYlGn')
                    
                    plt.title('Greek Sentiment with Confidence')
                    plt.xlabel('Time')
                    plt.ylabel('Sentiment')
                    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    plt.colorbar(label='Sentiment')
                    plt.grid(alpha=0.3)
                    
                    # Save plot
                    combined_plot_path = os.path.join(self.output_dir, 'greek_sentiment_with_confidence.png')
                    plt.savefig(combined_plot_path)
                    logger.info(f"Saved Greek sentiment with confidence plot to {combined_plot_path}")
                    
                    plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing Greek sentiment indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_greek_sentiment(self):
        """
        Test Greek sentiment analysis
        
        Returns:
            pd.DataFrame: Dataframe with Greek sentiment indicators
        """
        logger.info("Testing Greek sentiment analysis")
        
        # Load data
        df = self.load_data()
        
        if df is None or len(df) == 0:
            logger.error("No data available for testing")
            return None
        
        # Calculate Greek sentiment
        result_df = self.calculate_greek_sentiment(df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "greek_sentiment_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved Greek sentiment results to {output_path}")
        
        # Visualize results
        self.visualize_greek_sentiment(result_df)
        
        # Log summary statistics
        if 'Greek_Sentiment_Regime' in result_df.columns:
            regime_counts = result_df['Greek_Sentiment_Regime'].value_counts()
            logger.info(f"Greek Sentiment Regime distribution: {regime_counts.to_dict()}")
        
        logger.info("Greek sentiment analysis testing completed")
        
        return result_df

def main():
    """
    Main function to run the Greek sentiment analysis testing
    """
    logger.info("Starting Greek sentiment analysis testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/greek_sentiment',
        'delta_threshold': 0.5,
        'gamma_threshold': 0.05,
        'vega_threshold': 0.1,
        'theta_threshold': 0.05,
        'lookback_period': 5,
        'atm_range': 0.05
    }
    
    # Create Greek sentiment tester
    tester = GreekSentimentTester(config)
    
    # Test Greek sentiment
    result_df = tester.test_greek_sentiment()
    
    logger.info("Greek sentiment analysis testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
