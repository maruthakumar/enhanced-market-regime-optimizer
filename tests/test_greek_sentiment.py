"""
Test script for the enhanced Greek sentiment analysis implementation.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_greek_sentiment():
    """Test the enhanced Greek sentiment implementation."""
    logger.info("Testing enhanced Greek sentiment implementation")
    
    try:
        # Import the Greek sentiment module
        from utils.feature_engineering.greek_sentiment import GreekSentiment
        logger.info("Successfully imported GreekSentiment module")
        
        # Create a simple config
        config = {
            'lookback_period': 20,
            'use_synthetic_data': True,
            'max_delta': 0.5,
            'min_delta': 0.1,
            'current_week_weight': 0.7,
            'next_week_weight': 0.2,
            'current_month_weight': 0.1,
            'vega_weight': 1.0,
            'delta_weight': 1.0,
            'theta_weight': 0.5,
            'gamma_weight': 0.3
        }
        
        # Create a GreekSentiment instance
        greek_sentiment = GreekSentiment(config)
        logger.info("Created GreekSentiment instance")
        
        # Generate synthetic test data
        logger.info("Generating synthetic test data")
        
        # Create a date range
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        
        # Create synthetic data frame with separate expiry cycles
        data = []
        
        for date in dates:
            # Generate times for each date
            for hour in range(9, 16):
                for minute in [0, 15, 30, 45]:
                    if hour == 9 and minute < 15:
                        continue  # Skip times before market open
                    if hour == 15 and minute > 30:
                        continue  # Skip times after market close
                    
                    time_str = f"{hour:02d}:{minute:02d}:00"
                    
                    # Generate rows for each expiry
                    for expiry in ['current_week', 'next_week', 'current_month']:
                        row = {
                            'Date': date.strftime('%Y-%m-%d'),
                            'Time': time_str,
                            'Expiry': expiry
                        }
                        data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Generated synthetic data with {len(df)} rows")
        
        # Calculate sentiment
        logger.info("Calculating Greek sentiment")
        result = greek_sentiment.calculate_features(df)
        
        # Check that sentiment columns were added
        if 'Greek_Sentiment' in result.columns:
            logger.info("Greek sentiment calculation successful")
            logger.info(f"Greek sentiment range: {result['Greek_Sentiment'].min()} to {result['Greek_Sentiment'].max()}")
            
            # Count regimes
            regime_counts = result['Greek_Sentiment_Regime'].value_counts()
            logger.info(f"Regime counts: {regime_counts.to_dict()}")
            
            # Plot sentiment over time if matplotlib is available
            try:
                plt.figure(figsize=(12, 6))
                
                # Get first date's data
                first_date = result['Date'].iloc[0]
                day_data = result[result['Date'] == first_date]
                
                # Convert time to datetime for proper sorting
                day_data['TimeObj'] = pd.to_datetime(day_data['Time'])
                day_data = day_data.sort_values('TimeObj')
                
                # Plot sentiment value
                plt.plot(day_data['Time'], day_data['Greek_Sentiment'], marker='o', label='Greek Sentiment')
                
                # Add horizontal lines for regime thresholds
                plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Strong Bullish')
                plt.axhline(y=0.3, color='g', linestyle=':', alpha=0.5, label='Bullish')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axhline(y=-0.3, color='r', linestyle=':', alpha=0.5, label='Bearish')
                plt.axhline(y=-0.7, color='r', linestyle='--', alpha=0.5, label='Strong Bearish')
                
                plt.title(f'Greek Sentiment on {first_date}')
                plt.xlabel('Time')
                plt.ylabel('Sentiment Score')
                plt.legend()
                plt.grid(alpha=0.3)
                
                # Save plot
                output_dir = 'output/test_greek_sentiment'
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(f'{output_dir}/greek_sentiment_test.png')
                logger.info(f"Sentiment plot saved to {output_dir}/greek_sentiment_test.png")
                
                plt.close()
            except Exception as e:
                logger.error(f"Error creating plot: {str(e)}")
            
            # Also save to CSV for inspection
            result.to_csv(f'{output_dir}/greek_sentiment_test.csv', index=False)
            logger.info(f"Sentiment data saved to {output_dir}/greek_sentiment_test.csv")
            
        else:
            logger.error("Greek sentiment calculation failed - sentiment column not found")
            return False
        
        logger.info("Greek sentiment test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in Greek sentiment test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_greek_sentiment()
    if success:
        print("\nGreek sentiment test completed successfully")
    else:
        print("\nGreek sentiment test failed")
        sys.exit(1) 