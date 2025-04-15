"""
Example script for running the minute-by-minute market regime classifier on a dataset.

This script demonstrates how to use the minute regime classifier to process
market data and generate regime classifications with confidence scores.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the minute regime classifier
from src.minute_regime_classifier import MinuteRegimeClassifier, process_date_range

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("minute_regime_classifier_example.log")
    ]
)
logger = logging.getLogger(__name__)

def run_classifier_example():
    """
    Run the minute regime classifier on a sample dataset.
    """
    logger.info("Running minute regime classifier example")
    
    # Define date range to process
    start_date = '2025-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Processing data from {start_date} to {end_date}")
    
    # Process data for the specified date range
    results = process_date_range(start_date, end_date)
    
    if results:
        logger.info(f"Analysis completed successfully")
        logger.info(f"Results saved to {results.get('output_dir')}")
        logger.info(f"Combined results file: {results.get('combined_file')}")
        logger.info(f"Report file: {results.get('report_file')}")
        
        # Display sample of the results
        combined_file = results.get('combined_file')
        if combined_file and os.path.exists(combined_file):
            df = pd.read_csv(combined_file)
            logger.info(f"Sample of classified regimes (first 5 rows):")
            logger.info(df.head(5)[['datetime', 'regime', 'confidence_score']].to_string())
    else:
        logger.error("Analysis failed")

if __name__ == "__main__":
    run_classifier_example()
