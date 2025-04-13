import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import configparser
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config_manager import ConfigManager
from src.indicator_factory import IndicatorFactory
from src.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRegimeTestFramework:
    """
    Framework for testing market regime classification with sample data.
    """
    
    def __init__(self, config_path: str = None, output_dir: str = None):
        """
        Initialize the test framework.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.
            output_dir: Directory to save test results. If None, uses current directory.
        """
        self.config_manager = ConfigManager(config_path)
        self.indicator_factory = IndicatorFactory(self.config_manager)
        self.market_regime_classifier = MarketRegimeClassifier(self.config_manager)
        self.output_dir = output_dir or os.getcwd()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Preprocess the input data for market regime classification.
        
        Args:
            data_path: Path to the input data file
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing data from {data_path}")
        
        # Read the data
        data = pd.read_csv(data_path)
        
        # Convert date and time columns to datetime
        if 'date' in data.columns and 'time' in data.columns:
            # Convert date from YYMMDD format to YYYY-MM-DD
            data['date'] = data['date'].astype(str).apply(
                lambda x: f"20{x[:2]}-{x[2:4]}-{x[4:6]}" if len(x) == 6 else x
            )
            
            # Combine date and time into a datetime column
            data['datetime'] = pd.to_datetime(
                data['date'] + ' ' + data['time'].astype(str).str.zfill(4),
                format='%Y-%m-%d %H%M',
                errors='coerce'
            )
            
            # Set datetime as index
            data.set_index('datetime', inplace=True)
        
        # Calculate DTE (Days to Expiry)
        if 'expiry' in data.columns and 'date' in data.columns:
            # Convert expiry from YYMMDD format to YYYY-MM-DD
            data['expiry_date'] = data['expiry'].astype(str).apply(
                lambda x: f"20{x[:2]}-{x[2:4]}-{x[4:6]}" if len(x) == 6 else x
            )
            
            # Calculate DTE
            data['expiry_date'] = pd.to_datetime(data['expiry_date'], format='%Y-%m-%d', errors='coerce')
            data['trade_date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
            data['DTE'] = (data['expiry_date'] - data['trade_date']).dt.days
            
            # Drop temporary columns
            data.drop(['expiry_date', 'trade_date'], axis=1, inplace=True)
        
        # Rename columns to match expected format
        column_mapping = {
            'put_implied_volatility': 'Put_IV',
            'call_implied_volatility': 'Call_IV',
            'put_delta': 'Delta',
            'put_gamma': 'Gamma',
            'put_theta': 'Theta',
            'put_vega': 'Vega',
            'underlying_price': 'Close',
            'ATM_STRADDLE': 'ATM_Straddle_Premium',
            'CE_open_at_atm_strike': 'ATM_CE_Premium',
            'PE_open_at_atm_strike': 'ATM_PE_Premium'
        }
        
        data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns}, inplace=True)
        
        # Add High, Low columns if not present (for ATR calculation)
        if 'High' not in data.columns and 'Close' in data.columns:
            data['High'] = data['Close']
        
        if 'Low' not in data.columns and 'Close' in data.columns:
            data['Low'] = data['Close']
        
        # Add Volume column if not present
        if 'Volume' not in data.columns:
            if 'CE_volume' in data.columns and 'PE_volume' in data.columns:
                data['Volume'] = data['CE_volume'] + data['PE_volume']
            else:
                data['Volume'] = 1000  # Default value
        
        # Add OI column if not present
        if 'OI' not in data.columns:
            if 'CE_oi' in data.columns and 'PE_oi' in data.columns:
                data['OI'] = data['CE_oi'] + data['PE_oi']
            else:
                data['OI'] = 1000  # Default value
        
        # Calculate IV if not present
        if 'IV' not in data.columns and 'Put_IV' in data.columns and 'Call_IV' in data.columns:
            data['IV'] = (data['Put_IV'] + data['Call_IV']) / 2
        
        # Calculate Call/Put ratio if not present
        if 'Call_Put_Ratio' not in data.columns and 'CE_volume' in data.columns and 'PE_volume' in data.columns:
            data['Call_Put_Ratio'] = data['CE_volume'] / data['PE_volume'].replace(0, 1)
        
        # Calculate Put Skew if not present
        if 'Put_Skew' not in data.columns and 'Put_IV' in data.columns and 'Call_IV' in data.columns:
            data['Put_Skew'] = data['Put_IV'] / data['Call_IV']
        
        logger.info(f"Preprocessing complete. Data shape: {data.shape}")
        return data
    
    def run_test(self, data_path: str, output_prefix: str = "test_result") -> pd.DataFrame:
        """
        Run market regime classification test on the input data.
        
        Args:
            data_path: Path to the input data file
            output_prefix: Prefix for output files
            
        Returns:
            DataFrame with market regime classifications
        """
        logger.info(f"Running market regime classification test on {data_path}")
        
        # Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Classify market regimes
        result = self.market_regime_classifier.classify_market_regime(data)
        
        # Combine with original data
        combined_result = pd.concat([data, result], axis=1)
        
        # Save results to CSV
        output_path = os.path.join(self.output_dir, f"{output_prefix}.csv")
        combined_result.to_csv(output_path)
        logger.info(f"Test results saved to {output_path}")
        
        # Generate visualizations
        self._generate_visualizations(combined_result, output_prefix)
        
        return combined_result
    
    def _generate_visualizations(self, result: pd.DataFrame, output_prefix: str) -> None:
        """
        Generate visualizations of market regime classifications.
        
        Args:
            result: DataFrame with market regime classifications
            output_prefix: Prefix for output files
        """
        logger.info("Generating visualizations")
        
        # Create a figure for market regime visualization
        plt.figure(figsize=(15, 10))
        
        # Plot underlying price
        if 'Close' in result.columns:
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(result.index, result['Close'], label='Underlying Price')
            ax1.set_title('Underlying Price')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
        
        # Plot market regime
        if 'Market_Regime' in result.columns:
            ax2 = plt.subplot(3, 1, 2)
            
            # Create a numeric representation of market regimes
            regime_categories = result['Market_Regime'].unique()
            regime_map = {regime: i for i, regime in enumerate(regime_categories)}
            result['Regime_Numeric'] = result['Market_Regime'].map(regime_map)
            
            # Plot regime as a step function
            ax2.step(result.index, result['Regime_Numeric'], where='post')
            
            # Set y-ticks to regime names
            ax2.set_yticks(range(len(regime_categories)))
            ax2.set_yticklabels(regime_categories)
            ax2.set_title('Market Regime')
            ax2.grid(True)
        
        # Plot confidence score
        if 'Confidence_Score' in result.columns:
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(result.index, result['Confidence_Score'], label='Confidence Score')
            ax3.set_title('Confidence Score')
            ax3.set_ylabel('Score')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{output_prefix}_regimes.png"))
        plt.close()
        
        # Create a figure for component weights visualization
        weight_cols = [col for col in result.columns if col.endswith('_Weight')]
        if weight_cols:
            plt.figure(figsize=(15, 8))
            
            for col in weight_cols:
                plt.plot(result.index, result[col], label=col)
            
            plt.title('Dynamic Component Weights')
            plt.ylabel('Weight')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{output_prefix}_weights.png"))
            plt.close()
        
        logger.info("Visualizations generated")
    
    def analyze_results(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market regime classification results.
        
        Args:
            result: DataFrame with market regime classifications
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing market regime classification results")
        
        analysis = {}
        
        # Count occurrences of each market regime
        if 'Market_Regime' in result.columns:
            regime_counts = result['Market_Regime'].value_counts()
            analysis['regime_counts'] = regime_counts.to_dict()
            
            # Calculate percentage of each regime
            regime_percentages = (regime_counts / len(result)) * 100
            analysis['regime_percentages'] = regime_percentages.to_dict()
        
        # Calculate average confidence score
        if 'Confidence_Score' in result.columns:
            analysis['avg_confidence_score'] = result['Confidence_Score'].mean()
        
        # Calculate regime transitions
        if 'Market_Regime' in result.columns:
            transitions = (result['Market_Regime'] != result['Market_Regime'].shift(1)).sum()
            analysis['regime_transitions'] = int(transitions)
            analysis['avg_regime_duration'] = len(result) / max(transitions, 1)
        
        # Calculate average component weights
        weight_cols = [col for col in result.columns if col.endswith('_Weight')]
        if weight_cols:
            for col in weight_cols:
                analysis[f'avg_{col}'] = result[col].mean()
        
        # Save analysis to file
        analysis_path = os.path.join(self.output_dir, "analysis_results.txt")
        with open(analysis_path, 'w') as f:
            f.write("Market Regime Classification Analysis\n")
            f.write("====================================\n\n")
            
            if 'regime_counts' in analysis:
                f.write("Regime Counts:\n")
                for regime, count in analysis['regime_counts'].items():
                    f.write(f"  {regime}: {count}\n")
                f.write("\n")
            
            if 'regime_percentages' in analysis:
                f.write("Regime Percentages:\n")
                for regime, percentage in analysis['regime_percentages'].items():
                    f.write(f"  {regime}: {percentage:.2f}%\n")
                f.write("\n")
            
            if 'avg_confidence_score' in analysis:
                f.write(f"Average Confidence Score: {analysis['avg_confidence_score']:.4f}\n\n")
            
            if 'regime_transitions' in analysis:
                f.write(f"Number of Regime Transitions: {analysis['regime_transitions']}\n")
                f.write(f"Average Regime Duration: {analysis['avg_regime_duration']:.2f} data points\n\n")
            
            if any(key.startswith('avg_') for key in analysis):
                f.write("Average Component Weights:\n")
                for key, value in analysis.items():
                    if key.startswith('avg_') and key != 'avg_confidence_score':
                        f.write(f"  {key[4:]}: {value:.4f}\n")
        
        logger.info(f"Analysis results saved to {analysis_path}")
        return analysis

def main():
    """Main function to run the test framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Regime Classification Test Framework')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output', type=str, help='Directory to save test results')
    parser.add_argument('--prefix', type=str, default='test_result', help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Create test framework
    test_framework = MarketRegimeTestFramework(args.config, args.output)
    
    # Run test
    result = test_framework.run_test(args.data, args.prefix)
    
    # Analyze results
    analysis = test_framework.analyze_results(result)
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
