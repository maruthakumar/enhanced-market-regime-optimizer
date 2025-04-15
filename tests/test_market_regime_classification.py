"""
Script to test market regime classification for market regime testing
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

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

class MarketRegimeClassificationTester:
    """
    Class to test market regime classification for market regime testing
    """
    
    def __init__(self, config=None):
        """
        Initialize the market regime classification tester
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', '/home/ubuntu/market_regime_testing/processed_data')
        self.output_dir = self.config.get('output_dir', '/home/ubuntu/market_regime_testing/test_results/market_regime')
        self.results_dir = self.config.get('results_dir', '/home/ubuntu/market_regime_testing/test_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Market regime parameters
        self.num_regimes = self.config.get('num_regimes', 18)  # 18 market regimes as specified
        self.lookback_period = self.config.get('lookback_period', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Component weights
        self.weights = {
            'greek_sentiment': self.config.get('greek_sentiment_weight', 0.20),
            'trending_oi_pa': self.config.get('trending_oi_pa_weight', 0.30),
            'iv_skew': self.config.get('iv_skew_weight', 0.15),
            'technical_indicators': self.config.get('technical_indicators_weight', 0.20),
            'ce_pe_percentile': self.config.get('ce_pe_percentile_weight', 0.15)
        }
        
        # Normalize weights to ensure they sum to 1
        weight_sum = sum(self.weights.values())
        if weight_sum != 1.0:
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        # Define market regime mapping
        self.regime_mapping = {
            # Strong Bullish Regimes
            0: {'name': 'Strong_Bullish_Low_Vol', 'description': 'Strong bullish trend with low volatility'},
            1: {'name': 'Strong_Bullish_Medium_Vol', 'description': 'Strong bullish trend with medium volatility'},
            2: {'name': 'Strong_Bullish_High_Vol', 'description': 'Strong bullish trend with high volatility'},
            
            # Bullish Regimes
            3: {'name': 'Bullish_Low_Vol', 'description': 'Bullish trend with low volatility'},
            4: {'name': 'Bullish_Medium_Vol', 'description': 'Bullish trend with medium volatility'},
            5: {'name': 'Bullish_High_Vol', 'description': 'Bullish trend with high volatility'},
            
            # Neutral Regimes
            6: {'name': 'Neutral_Low_Vol', 'description': 'Neutral trend with low volatility'},
            7: {'name': 'Neutral_Medium_Vol', 'description': 'Neutral trend with medium volatility'},
            8: {'name': 'Neutral_High_Vol', 'description': 'Neutral trend with high volatility'},
            
            # Bearish Regimes
            9: {'name': 'Bearish_Low_Vol', 'description': 'Bearish trend with low volatility'},
            10: {'name': 'Bearish_Medium_Vol', 'description': 'Bearish trend with medium volatility'},
            11: {'name': 'Bearish_High_Vol', 'description': 'Bearish trend with high volatility'},
            
            # Strong Bearish Regimes
            12: {'name': 'Strong_Bearish_Low_Vol', 'description': 'Strong bearish trend with low volatility'},
            13: {'name': 'Strong_Bearish_Medium_Vol', 'description': 'Strong bearish trend with medium volatility'},
            14: {'name': 'Strong_Bearish_High_Vol', 'description': 'Strong bearish trend with high volatility'},
            
            # Transitional Regimes
            15: {'name': 'Transition_Bullish_to_Bearish', 'description': 'Transition from bullish to bearish'},
            16: {'name': 'Transition_Bearish_to_Bullish', 'description': 'Transition from bearish to bullish'},
            17: {'name': 'Choppy_Volatile', 'description': 'Choppy market with high volatility'}
        }
        
        logger.info(f"Initialized MarketRegimeClassificationTester with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
        logger.info(f"Market regime parameters: num_regimes={self.num_regimes}, lookback_period={self.lookback_period}")
        logger.info(f"Component weights: {self.weights}")
    
    def load_component_results(self):
        """
        Load results from individual component tests
        
        Returns:
            dict: Dictionary of component results
        """
        logger.info("Loading component test results")
        
        component_results = {}
        
        # Load Greek sentiment results
        greek_sentiment_path = os.path.join(self.results_dir, "greek_sentiment/greek_sentiment_results.csv")
        if os.path.exists(greek_sentiment_path):
            logger.info(f"Loading Greek sentiment results from {greek_sentiment_path}")
            try:
                component_results['greek_sentiment'] = pd.read_csv(greek_sentiment_path)
                logger.info(f"Loaded Greek sentiment results with {len(component_results['greek_sentiment'])} rows")
            except Exception as e:
                logger.error(f"Error loading Greek sentiment results: {str(e)}")
        else:
            logger.warning(f"Greek sentiment results not found at {greek_sentiment_path}")
        
        # Load trending OI with PA results
        trending_oi_pa_path = os.path.join(self.results_dir, "trending_oi_pa/trending_oi_pa_aggregated.csv")
        if os.path.exists(trending_oi_pa_path):
            logger.info(f"Loading trending OI with PA results from {trending_oi_pa_path}")
            try:
                component_results['trending_oi_pa'] = pd.read_csv(trending_oi_pa_path)
                logger.info(f"Loaded trending OI with PA results with {len(component_results['trending_oi_pa'])} rows")
            except Exception as e:
                logger.error(f"Error loading trending OI with PA results: {str(e)}")
        else:
            logger.warning(f"Trending OI with PA results not found at {trending_oi_pa_path}")
        
        # Load IV skew results
        iv_skew_path = os.path.join(self.results_dir, "iv_skew_atm_straddle/iv_skew_results.csv")
        if os.path.exists(iv_skew_path):
            logger.info(f"Loading IV skew results from {iv_skew_path}")
            try:
                component_results['iv_skew'] = pd.read_csv(iv_skew_path)
                logger.info(f"Loaded IV skew results with {len(component_results['iv_skew'])} rows")
            except Exception as e:
                logger.error(f"Error loading IV skew results: {str(e)}")
        else:
            logger.warning(f"IV skew results not found at {iv_skew_path}")
        
        # Load technical indicators results
        technical_indicators_path = os.path.join(self.results_dir, "technical_indicators/technical_indicators_results.csv")
        if os.path.exists(technical_indicators_path):
            logger.info(f"Loading technical indicators results from {technical_indicators_path}")
            try:
                component_results['technical_indicators'] = pd.read_csv(technical_indicators_path)
                logger.info(f"Loaded technical indicators results with {len(component_results['technical_indicators'])} rows")
            except Exception as e:
                logger.error(f"Error loading technical indicators results: {str(e)}")
        else:
            logger.warning(f"Technical indicators results not found at {technical_indicators_path}")
        
        # Load CE/PE percentile results
        ce_pe_percentile_path = os.path.join(self.results_dir, "ce_pe_percentile/ce_pe_percentile_results.csv")
        if os.path.exists(ce_pe_percentile_path):
            logger.info(f"Loading CE/PE percentile results from {ce_pe_percentile_path}")
            try:
                component_results['ce_pe_percentile'] = pd.read_csv(ce_pe_percentile_path)
                logger.info(f"Loaded CE/PE percentile results with {len(component_results['ce_pe_percentile'])} rows")
            except Exception as e:
                logger.error(f"Error loading CE/PE percentile results: {str(e)}")
        else:
            logger.warning(f"CE/PE percentile results not found at {ce_pe_percentile_path}")
        
        # Check if we have any component results
        if not component_results:
            logger.error("No component results found")
            return None
        
        return component_results
    
    def align_component_data(self, component_results):
        """
        Align component data by datetime
        
        Args:
            component_results (dict): Dictionary of component results
            
        Returns:
            pd.DataFrame: Aligned data
        """
        logger.info("Aligning component data by datetime")
        
        # Check if we have any component results
        if not component_results:
            logger.error("No component results to align")
            return None
        
        # Ensure datetime is in datetime format for all components
        for component, df in component_results.items():
            if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                try:
                    component_results[component]['datetime'] = pd.to_datetime(df['datetime'])
                    logger.info(f"Converted datetime to datetime format for {component}")
                except Exception as e:
                    logger.warning(f"Failed to convert datetime to datetime format for {component}: {str(e)}")
        
        # Get unique datetimes from all components
        all_datetimes = set()
        for component, df in component_results.items():
            if 'datetime' in df.columns:
                all_datetimes.update(df['datetime'].astype(str).tolist())
        
        if not all_datetimes:
            logger.error("No datetimes found in any component")
            return None
        
        logger.info(f"Found {len(all_datetimes)} unique datetimes across all components")
        
        # Create a base dataframe with all datetimes
        aligned_df = pd.DataFrame({'datetime': list(all_datetimes)})
        aligned_df['datetime'] = pd.to_datetime(aligned_df['datetime'])
        aligned_df.sort_values('datetime', inplace=True)
        aligned_df.reset_index(drop=True, inplace=True)
        
        # Add component data to aligned dataframe
        for component, df in component_results.items():
            if 'datetime' not in df.columns:
                logger.warning(f"No datetime column in {component}, skipping")
                continue
            
            # Identify columns to merge
            merge_columns = []
            
            if component == 'greek_sentiment':
                if 'Greek_Sentiment_Regime' in df.columns:
                    merge_columns.append('Greek_Sentiment_Regime')
                if 'Greek_Sentiment_Confidence' in df.columns:
                    merge_columns.append('Greek_Sentiment_Confidence')
                if 'Combined_Greek_Sentiment' in df.columns:
                    merge_columns.append('Combined_Greek_Sentiment')
            
            elif component == 'trending_oi_pa':
                if 'Overall_OI_Pattern' in df.columns:
                    merge_columns.append('Overall_OI_Pattern')
                if 'Direction_of_chng' in df.columns:
                    merge_columns.append('Direction_of_chng')
                if 'Direction_pct' in df.columns:
                    merge_columns.append('Direction_pct')
                if 'Net_PCR' in df.columns:
                    merge_columns.append('Net_PCR')
                if 'Sentiment' in df.columns:
                    merge_columns.append('Sentiment')
            
            elif component == 'iv_skew':
                if 'IV_Skew_Regime' in df.columns:
                    merge_columns.append('IV_Skew_Regime')
                if 'IV_Skew_Confidence' in df.columns:
                    merge_columns.append('IV_Skew_Confidence')
                if 'IV_Percentile' in df.columns:
                    merge_columns.append('IV_Percentile')
                if 'ATM_Straddle_Premium' in df.columns:
                    merge_columns.append('ATM_Straddle_Premium')
            
            elif component == 'technical_indicators':
                if 'Technical_Market_Regime' in df.columns:
                    merge_columns.append('Technical_Market_Regime')
                if 'Technical_Regime_Confidence' in df.columns:
                    merge_columns.append('Technical_Regime_Confidence')
                if 'EMA_Market_Regime' in df.columns:
                    merge_columns.append('EMA_Market_Regime')
                if 'VWAP_Market_Regime' in df.columns:
                    merge_columns.append('VWAP_Market_Regime')
                if 'RSI_Market_Regime' in df.columns:
                    merge_columns.append('RSI_Market_Regime')
                if 'MACD_Market_Regime' in df.columns:
                    merge_columns.append('MACD_Market_Regime')
                if 'BB_Market_Regime' in df.columns:
                    merge_columns.append('BB_Market_Regime')
                if 'ATR_Market_Regime' in df.columns:
                    merge_columns.append('ATR_Market_Regime')
            
            elif component == 'ce_pe_percentile':
                if 'CE_PE_Regime' in df.columns:
                    merge_columns.append('CE_PE_Regime')
                if 'CE_PE_Confidence' in df.columns:
                    merge_columns.append('CE_PE_Confidence')
                if 'CE_Percentile' in df.columns:
                    merge_columns.append('CE_Percentile')
                if 'PE_Percentile' in df.columns:
                    merge_columns.append('PE_Percentile')
                if 'CE_PE_Ratio' in df.columns:
                    merge_columns.append('CE_PE_Ratio')
            
            # Add underlying price if available
            if 'Underlying_Price' in df.columns:
                merge_columns.append('Underlying_Price')
            elif 'Close' in df.columns:
                merge_columns.append('Close')
            
            # If no specific columns found, use all columns except datetime
            if not merge_columns:
                merge_columns = [col for col in df.columns if col != 'datetime']
                logger.warning(f"No specific columns identified for {component}, using all columns: {merge_columns}")
            
            # Merge component data
            if merge_columns:
                # Create a copy of the dataframe with only the columns we want to merge
                merge_df = df[['datetime'] + merge_columns].copy()
                
                # Add component prefix to column names to avoid conflicts
                rename_dict = {col: f"{component}_{col}" for col in merge_columns}
                merge_df.rename(columns=rename_dict, inplace=True)
                
                # Merge with aligned dataframe
                aligned_df = pd.merge(aligned_df, merge_df, on='datetime', how='left')
                logger.info(f"Merged {component} data with {len(merge_columns)} columns")
        
        # Ensure we have a price column
        price_columns = [col for col in aligned_df.columns if col.endswith('_Underlying_Price') or col.endswith('_Close')]
        if price_columns:
            # Use the first available price column
            aligned_df['Price'] = aligned_df[price_columns[0]].copy()
            logger.info(f"Using {price_columns[0]} as Price column")
        else:
            logger.warning("No price column found in any component")
        
        logger.info(f"Aligned data has {len(aligned_df)} rows and {len(aligned_df.columns)} columns")
        
        return aligned_df
    
    def calculate_market_regime(self, aligned_df):
        """
        Calculate market regime based on component data
        
        Args:
            aligned_df (pd.DataFrame): Aligned component data
            
        Returns:
            pd.DataFrame: Data with market regime
        """
        logger.info("Calculating market regime based on component data")
        
        # Make a copy to avoid modifying the original
        result_df = aligned_df.copy()
        
        # Initialize market regime columns
        result_df['Market_Regime'] = np.nan
        result_df['Market_Regime_ID'] = np.nan
        result_df['Market_Regime_Confidence'] = np.nan
        result_df['Market_Regime_Description'] = np.nan
        
        # Check if we have the necessary component data
        required_components = ['greek_sentiment', 'trending_oi_pa', 'technical_indicators']
        missing_components = []
        
        for component in required_components:
            component_columns = [col for col in result_df.columns if col.startswith(f"{component}_")]
            if not component_columns:
                missing_components.append(component)
        
        if missing_components:
            logger.warning(f"Missing required component data: {missing_components}")
            logger.warning("Will proceed with available components")
        
        try:
            # Calculate trend direction score
            trend_direction_score = np.zeros(len(result_df))
            trend_direction_confidence = np.zeros(len(result_df))
            component_count = np.zeros(len(result_df))
            
            # Greek sentiment contribution
            if 'greek_sentiment_Greek_Sentiment_Regime' in result_df.columns:
                # Map regime to score
                regime_scores = {
                    'Strong_Bullish': 2,
                    'Bullish': 1,
                    'Neutral': 0,
                    'Bearish': -1,
                    'Strong_Bearish': -2
                }
                
                # Calculate score
                greek_scores = result_df['greek_sentiment_Greek_Sentiment_Regime'].map(regime_scores).fillna(0)
                
                # Get confidence if available
                if 'greek_sentiment_Greek_Sentiment_Confidence' in result_df.columns:
                    greek_confidence = result_df['greek_sentiment_Greek_Sentiment_Confidence'].fillna(0.5)
                else:
                    greek_confidence = np.ones(len(result_df)) * 0.5
                
                # Add to total score with weight
                trend_direction_score += greek_scores * self.weights['greek_sentiment']
                trend_direction_confidence += greek_confidence * self.weights['greek_sentiment']
                component_count += np.where(pd.notna(result_df['greek_sentiment_Greek_Sentiment_Regime']), 1, 0)
                
                logger.info("Added Greek sentiment contribution to trend direction score")
            
            # Trending OI with PA contribution
            if 'trending_oi_pa_Overall_OI_Pattern' in result_df.columns:
                # Map pattern to score
                pattern_scores = {
                    'Strong_Bullish': 2,
                    'Mild_Bullish': 1,
                    'Neutral': 0,
                    'Mild_Bearish': -1,
                    'Strong_Bearish': -2
                }
                
                # Calculate score
                oi_scores = result_df['trending_oi_pa_Overall_OI_Pattern'].map(pattern_scores).fillna(0)
                
                # Get confidence based on direction percentage if available
                if 'trending_oi_pa_Direction_pct' in result_df.columns:
                    direction_pct = result_df['trending_oi_pa_Direction_pct'].abs().fillna(0)
                    oi_confidence = direction_pct / 100  # Normalize to 0-1 range
                    oi_confidence = oi_confidence.clip(0, 1)  # Ensure within range
                else:
                    oi_confidence = np.ones(len(result_df)) * 0.5
                
                # Add to total score with weight
                trend_direction_score += oi_scores * self.weights['trending_oi_pa']
                trend_direction_confidence += oi_confidence * self.weights['trending_oi_pa']
                component_count += np.where(pd.notna(result_df['trending_oi_pa_Overall_OI_Pattern']), 1, 0)
                
                logger.info("Added trending OI with PA contribution to trend direction score")
            
            # IV skew contribution
            if 'iv_skew_IV_Skew_Regime' in result_df.columns:
                # Map regime to score
                regime_scores = {
                    'Strong_Bullish': 2,
                    'Bullish': 1,
                    'Neutral': 0,
                    'Bearish': -1,
                    'Strong_Bearish': -2
                }
                
                # Calculate score
                iv_scores = result_df['iv_skew_IV_Skew_Regime'].map(regime_scores).fillna(0)
                
                # Get confidence if available
                if 'iv_skew_IV_Skew_Confidence' in result_df.columns:
                    iv_confidence = result_df['iv_skew_IV_Skew_Confidence'].fillna(0.5)
                else:
                    iv_confidence = np.ones(len(result_df)) * 0.5
                
                # Add to total score with weight
                trend_direction_score += iv_scores * self.weights['iv_skew']
                trend_direction_confidence += iv_confidence * self.weights['iv_skew']
                component_count += np.where(pd.notna(result_df['iv_skew_IV_Skew_Regime']), 1, 0)
                
                logger.info("Added IV skew contribution to trend direction score")
            
            # Technical indicators contribution
            if 'technical_indicators_Technical_Market_Regime' in result_df.columns:
                # Map regime to score
                regime_scores = {
                    'Strong_Bullish': 2,
                    'Bullish': 1,
                    'Neutral': 0,
                    'Bearish': -1,
                    'Strong_Bearish': -2
                }
                
                # Calculate score
                tech_scores = result_df['technical_indicators_Technical_Market_Regime'].map(regime_scores).fillna(0)
                
                # Get confidence if available
                if 'technical_indicators_Technical_Regime_Confidence' in result_df.columns:
                    tech_confidence = result_df['technical_indicators_Technical_Regime_Confidence'].fillna(0.5)
                else:
                    tech_confidence = np.ones(len(result_df)) * 0.5
                
                # Add to total score with weight
                trend_direction_score += tech_scores * self.weights['technical_indicators']
                trend_direction_confidence += tech_confidence * self.weights['technical_indicators']
                component_count += np.where(pd.notna(result_df['technical_indicators_Technical_Market_Regime']), 1, 0)
                
                logger.info("Added technical indicators contribution to trend direction score")
            
            # CE/PE percentile contribution
            if 'ce_pe_percentile_CE_PE_Regime' in result_df.columns:
                # Map regime to score
                regime_scores = {
                    'Strong_Bullish': 2,
                    'Bullish': 1,
                    'Neutral': 0,
                    'Bearish': -1,
                    'Strong_Bearish': -2
                }
                
                # Calculate score
                ce_pe_scores = result_df['ce_pe_percentile_CE_PE_Regime'].map(regime_scores).fillna(0)
                
                # Get confidence if available
                if 'ce_pe_percentile_CE_PE_Confidence' in result_df.columns:
                    ce_pe_confidence = result_df['ce_pe_percentile_CE_PE_Confidence'].fillna(0.5)
                else:
                    ce_pe_confidence = np.ones(len(result_df)) * 0.5
                
                # Add to total score with weight
                trend_direction_score += ce_pe_scores * self.weights['ce_pe_percentile']
                trend_direction_confidence += ce_pe_confidence * self.weights['ce_pe_percentile']
                component_count += np.where(pd.notna(result_df['ce_pe_percentile_CE_PE_Regime']), 1, 0)
                
                logger.info("Added CE/PE percentile contribution to trend direction score")
            
            # Normalize confidence by component count
            trend_direction_confidence = np.where(component_count > 0, 
                                                trend_direction_confidence / np.where(component_count > 0, component_count, 1), 
                                                0.5)
            
            # Add trend direction score and confidence to result dataframe
            result_df['Trend_Direction_Score'] = trend_direction_score
            result_df['Trend_Direction_Confidence'] = trend_direction_confidence
            
            # Calculate volatility score
            volatility_score = np.zeros(len(result_df))
            volatility_confidence = np.zeros(len(result_df))
            vol_component_count = np.zeros(len(result_df))
            
            # ATR contribution
            if 'technical_indicators_ATR_Market_Regime' in result_df.columns:
                # Map regime to score
                regime_scores = {
                    'High_Volatility': 2,
                    'Above_Average_Volatility': 1,
                    'Average_Volatility': 0,
                    'Below_Average_Volatility': -1,
                    'Low_Volatility': -2
                }
                
                # Calculate score
                atr_scores = result_df['technical_indicators_ATR_Market_Regime'].map(regime_scores).fillna(0)
                
                # Add to total score
                volatility_score += atr_scores
                vol_component_count += np.where(pd.notna(result_df['technical_indicators_ATR_Market_Regime']), 1, 0)
                
                logger.info("Added ATR contribution to volatility score")
            
            # Bollinger Band width contribution
            if 'technical_indicators_BB_Market_Regime' in result_df.columns:
                # Map regime to score
                regime_scores = {
                    'Expansion': 2,
                    'Strong_Bullish': 1,
                    'Strong_Bearish': 1,
                    'Neutral': 0,
                    'Bullish': 0,
                    'Bearish': 0,
                    'Squeeze': -2
                }
                
                # Calculate score
                bb_scores = result_df['technical_indicators_BB_Market_Regime'].map(regime_scores).fillna(0)
                
                # Add to total score
                volatility_score += bb_scores
                vol_component_count += np.where(pd.notna(result_df['technical_indicators_BB_Market_Regime']), 1, 0)
                
                logger.info("Added Bollinger Band width contribution to volatility score")
            
            # IV percentile contribution
            if 'iv_skew_IV_Percentile' in result_df.columns:
                # Calculate score based on percentile
                iv_percentile = result_df['iv_skew_IV_Percentile'].fillna(50)
                
                # Map percentile to score
                # High percentile = high volatility
                iv_vol_scores = np.where(iv_percentile > 80, 2,
                                      np.where(iv_percentile > 60, 1,
                                             np.where(iv_percentile < 20, -2,
                                                    np.where(iv_percentile < 40, -1, 0))))
                
                # Add to total score
                volatility_score += iv_vol_scores
                vol_component_count += np.where(pd.notna(result_df['iv_skew_IV_Percentile']), 1, 0)
                
                logger.info("Added IV percentile contribution to volatility score")
            
            # ATM straddle premium contribution
            if 'iv_skew_ATM_Straddle_Premium' in result_df.columns:
                # Calculate rolling mean and std of ATM straddle premium
                atm_premium = result_df['iv_skew_ATM_Straddle_Premium'].fillna(method='ffill')
                atm_premium_mean = atm_premium.rolling(window=20, min_periods=1).mean()
                atm_premium_std = atm_premium.rolling(window=20, min_periods=1).std()
                
                # Calculate z-score
                atm_premium_z = (atm_premium - atm_premium_mean) / atm_premium_std.replace(0, 1)
                
                # Map z-score to volatility score
                atm_vol_scores = np.where(atm_premium_z > 2, 2,
                                       np.where(atm_premium_z > 1, 1,
                                              np.where(atm_premium_z < -2, -2,
                                                     np.where(atm_premium_z < -1, -1, 0))))
                
                # Add to total score
                volatility_score += atm_vol_scores
                vol_component_count += np.where(pd.notna(result_df['iv_skew_ATM_Straddle_Premium']), 1, 0)
                
                logger.info("Added ATM straddle premium contribution to volatility score")
            
            # Normalize volatility score by component count
            volatility_score = np.where(vol_component_count > 0, 
                                      volatility_score / np.where(vol_component_count > 0, vol_component_count, 1), 
                                      0)
            
            # Add volatility score to result dataframe
            result_df['Volatility_Score'] = volatility_score
            
            # Detect transitions
            # Calculate rolling mean of trend direction score
            trend_direction_mean = result_df['Trend_Direction_Score'].rolling(window=self.lookback_period, min_periods=1).mean()
            
            # Calculate trend direction change
            trend_direction_change = result_df['Trend_Direction_Score'] - trend_direction_mean
            
            # Identify transitions
            is_transition_bullish_to_bearish = (trend_direction_mean > 0.5) & (trend_direction_change < -1)
            is_transition_bearish_to_bullish = (trend_direction_mean < -0.5) & (trend_direction_change > 1)
            
            # Identify choppy market
            is_choppy = (result_df['Trend_Direction_Score'].rolling(window=self.lookback_period, min_periods=1).std() > 1) & \
                       (result_df['Volatility_Score'] > 1)
            
            # Map trend direction and volatility to market regime
            # Normalize scores to -2 to 2 range
            trend_direction_score_norm = result_df['Trend_Direction_Score'].clip(-2, 2)
            volatility_score_norm = result_df['Volatility_Score'].clip(-2, 2)
            
            # Map to volatility categories
            # -2 to -0.67: Low volatility
            # -0.67 to 0.67: Medium volatility
            # 0.67 to 2: High volatility
            volatility_category = np.where(volatility_score_norm > 0.67, 'High',
                                         np.where(volatility_score_norm < -0.67, 'Low', 'Medium'))
            
            # Map to trend direction categories
            # -2 to -1.33: Strong bearish
            # -1.33 to -0.67: Bearish
            # -0.67 to 0.67: Neutral
            # 0.67 to 1.33: Bullish
            # 1.33 to 2: Strong bullish
            trend_category = np.where(trend_direction_score_norm > 1.33, 'Strong_Bullish',
                                    np.where(trend_direction_score_norm > 0.67, 'Bullish',
                                           np.where(trend_direction_score_norm < -1.33, 'Strong_Bearish',
                                                  np.where(trend_direction_score_norm < -0.67, 'Bearish', 'Neutral'))))
            
            # Combine trend and volatility categories to get market regime
            market_regime = np.where(is_transition_bullish_to_bearish, 'Transition_Bullish_to_Bearish',
                                   np.where(is_transition_bearish_to_bullish, 'Transition_Bearish_to_Bullish',
                                          np.where(is_choppy, 'Choppy_Volatile',
                                                 trend_category + '_' + volatility_category + '_Vol')))
            
            # Map market regime to regime ID
            regime_id_map = {regime['name']: regime_id for regime_id, regime in self.regime_mapping.items()}
            market_regime_id = pd.Series(market_regime).map(regime_id_map).values
            
            # Map market regime to description
            regime_desc_map = {regime['name']: regime['description'] for regime_id, regime in self.regime_mapping.items()}
            market_regime_desc = pd.Series(market_regime).map(regime_desc_map).values
            
            # Add market regime to result dataframe
            result_df['Market_Regime'] = market_regime
            result_df['Market_Regime_ID'] = market_regime_id
            result_df['Market_Regime_Description'] = market_regime_desc
            
            # Calculate market regime confidence
            # Higher confidence when trend direction confidence is high and components agree
            result_df['Market_Regime_Confidence'] = result_df['Trend_Direction_Confidence']
            
            # Adjust confidence for transitions and choppy markets
            result_df.loc[is_transition_bullish_to_bearish | is_transition_bearish_to_bullish, 'Market_Regime_Confidence'] *= 0.8
            result_df.loc[is_choppy, 'Market_Regime_Confidence'] *= 0.7
            
            logger.info("Calculated market regime based on component data")
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result_df
    
    def visualize_market_regime(self, result_df):
        """
        Visualize market regime
        
        Args:
            result_df (pd.DataFrame): Data with market regime
        """
        logger.info("Visualizing market regime")
        
        # Check if we have market regime data
        if 'Market_Regime' not in result_df.columns:
            logger.error("No market regime data available for visualization")
            return
        
        try:
            # Create market regime distribution chart
            plt.figure(figsize=(12, 8))
            
            # Count occurrences of each regime
            regime_counts = result_df['Market_Regime'].value_counts()
            
            # Create a colormap for regimes
            regime_colors = {
                'Strong_Bullish_Low_Vol': 'darkgreen',
                'Strong_Bullish_Medium_Vol': 'green',
                'Strong_Bullish_High_Vol': 'lightgreen',
                'Bullish_Low_Vol': 'mediumseagreen',
                'Bullish_Medium_Vol': 'seagreen',
                'Bullish_High_Vol': 'springgreen',
                'Neutral_Low_Vol': 'lightgray',
                'Neutral_Medium_Vol': 'gray',
                'Neutral_High_Vol': 'darkgray',
                'Bearish_Low_Vol': 'lightcoral',
                'Bearish_Medium_Vol': 'indianred',
                'Bearish_High_Vol': 'firebrick',
                'Strong_Bearish_Low_Vol': 'darkred',
                'Strong_Bearish_Medium_Vol': 'red',
                'Strong_Bearish_High_Vol': 'tomato',
                'Transition_Bullish_to_Bearish': 'orange',
                'Transition_Bearish_to_Bullish': 'yellowgreen',
                'Choppy_Volatile': 'purple'
            }
            
            # Plot regime distribution
            bars = plt.bar(regime_counts.index, regime_counts.values, 
                          color=[regime_colors.get(regime, 'blue') for regime in regime_counts.index])
            
            plt.title('Market Regime Distribution')
            plt.xlabel('Regime')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            regime_plot_path = os.path.join(self.output_dir, 'market_regime_distribution.png')
            plt.savefig(regime_plot_path)
            logger.info(f"Saved market regime distribution plot to {regime_plot_path}")
            
            plt.close()
            
            # Create regime over time chart
            if 'Price' in result_df.columns:
                plt.figure(figsize=(15, 10))
                
                # Plot price
                plt.subplot(3, 1, 1)
                plt.plot(result_df['datetime'], result_df['Price'], label='Price', color='black')
                plt.title('Price Over Time')
                plt.grid(alpha=0.3)
                
                # Plot market regime
                plt.subplot(3, 1, 2)
                
                # Create a colormap for regimes
                cmap = plt.cm.get_cmap('RdYlGn', len(self.regime_mapping))
                regime_colors = {regime_id: cmap(i) for i, regime_id in enumerate(sorted(self.regime_mapping.keys()))}
                
                # Plot regime as colored points
                for regime_id in sorted(self.regime_mapping.keys()):
                    mask = result_df['Market_Regime_ID'] == regime_id
                    if mask.any():
                        plt.scatter(result_df.loc[mask, 'datetime'], 
                                   [regime_id] * mask.sum(), 
                                   color=regime_colors[regime_id], 
                                   label=self.regime_mapping[regime_id]['name'],
                                   s=50)
                
                plt.yticks(sorted(self.regime_mapping.keys()), 
                          [self.regime_mapping[regime_id]['name'] for regime_id in sorted(self.regime_mapping.keys())],
                          fontsize=8)
                plt.title('Market Regime Over Time')
                plt.grid(alpha=0.3)
                
                # Plot confidence
                plt.subplot(3, 1, 3)
                plt.plot(result_df['datetime'], result_df['Market_Regime_Confidence'], label='Confidence', color='blue')
                plt.axhline(y=self.confidence_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({self.confidence_threshold})')
                plt.title('Market Regime Confidence')
                plt.xlabel('Time')
                plt.ylabel('Confidence')
                plt.legend()
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                regime_time_plot_path = os.path.join(self.output_dir, 'market_regime_time.png')
                plt.savefig(regime_time_plot_path)
                logger.info(f"Saved market regime over time plot to {regime_time_plot_path}")
                
                plt.close()
            
            # Create component contribution chart
            plt.figure(figsize=(12, 8))
            
            # Plot trend direction score
            plt.subplot(2, 1, 1)
            plt.plot(result_df['datetime'], result_df['Trend_Direction_Score'], label='Trend Direction', color='blue')
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            plt.title('Trend Direction Score')
            plt.grid(alpha=0.3)
            
            # Plot volatility score
            plt.subplot(2, 1, 2)
            plt.plot(result_df['datetime'], result_df['Volatility_Score'], label='Volatility', color='red')
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            plt.title('Volatility Score')
            plt.xlabel('Time')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            component_plot_path = os.path.join(self.output_dir, 'component_contribution.png')
            plt.savefig(component_plot_path)
            logger.info(f"Saved component contribution plot to {component_plot_path}")
            
            plt.close()
            
            # Create daily market regime calendar
            if 'datetime' in result_df.columns:
                # Convert datetime to date
                result_df['date'] = pd.to_datetime(result_df['datetime']).dt.date
                
                # Group by date and get most common regime
                daily_regime = result_df.groupby('date')['Market_Regime'].agg(lambda x: x.value_counts().index[0])
                daily_regime_id = result_df.groupby('date')['Market_Regime_ID'].agg(lambda x: x.value_counts().index[0])
                daily_confidence = result_df.groupby('date')['Market_Regime_Confidence'].mean()
                
                # Create a dataframe with daily regime
                daily_df = pd.DataFrame({
                    'date': daily_regime.index,
                    'Market_Regime': daily_regime.values,
                    'Market_Regime_ID': daily_regime_id.values,
                    'Market_Regime_Confidence': daily_confidence.values
                })
                
                # Save daily regime to CSV
                daily_output_path = os.path.join(self.output_dir, "daily_market_regime.csv")
                daily_df.to_csv(daily_output_path, index=False)
                logger.info(f"Saved daily market regime to {daily_output_path}")
                
                # Create calendar visualization
                # Get min and max dates
                min_date = daily_df['date'].min()
                max_date = daily_df['date'].max()
                
                # Create a date range
                date_range = pd.date_range(min_date, max_date)
                
                # Create a dataframe with all dates
                calendar_df = pd.DataFrame({'date': date_range})
                calendar_df['date'] = calendar_df['date'].dt.date
                
                # Merge with daily regime
                calendar_df = pd.merge(calendar_df, daily_df, on='date', how='left')
                
                # Create calendar visualization
                # Get number of weeks
                num_weeks = len(calendar_df) // 7 + 1
                
                # Create figure
                fig, ax = plt.subplots(figsize=(15, num_weeks * 1.5))
                
                # Create a grid
                calendar_df['day_of_week'] = pd.to_datetime(calendar_df['date']).dt.dayofweek
                calendar_df['week'] = (pd.to_datetime(calendar_df['date']) - pd.to_datetime(min_date)).dt.days // 7
                
                # Plot each day
                for _, row in calendar_df.iterrows():
                    if pd.isna(row['Market_Regime']):
                        continue
                    
                    # Get color based on regime
                    color = regime_colors.get(row['Market_Regime'], 'blue')
                    
                    # Plot rectangle
                    rect = plt.Rectangle((row['day_of_week'], -row['week']), 1, 1, 
                                       color=color, alpha=0.7)
                    ax.add_patch(rect)
                    
                    # Add date text
                    plt.text(row['day_of_week'] + 0.5, -row['week'] + 0.7, 
                            str(row['date']), 
                            ha='center', va='center', 
                            fontsize=8)
                    
                    # Add regime text
                    plt.text(row['day_of_week'] + 0.5, -row['week'] + 0.3, 
                            str(int(row['Market_Regime_ID'])), 
                            ha='center', va='center', 
                            fontsize=10, fontweight='bold')
                
                # Set limits
                plt.xlim(0, 7)
                plt.ylim(-num_weeks, 1)
                
                # Set ticks
                plt.xticks(np.arange(7) + 0.5, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                plt.yticks([])
                
                # Remove spines
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                plt.title('Daily Market Regime Calendar')
                
                # Add legend
                legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7, label=f"{regime_id}: {regime['name']}") 
                                 for regime_id, regime in self.regime_mapping.items() 
                                 if regime['name'] in regime_colors]
                
                plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
                
                plt.tight_layout()
                
                # Save plot
                calendar_plot_path = os.path.join(self.output_dir, 'market_regime_calendar.png')
                plt.savefig(calendar_plot_path, bbox_inches='tight')
                logger.info(f"Saved market regime calendar plot to {calendar_plot_path}")
                
                plt.close()
            
            logger.info("Completed visualization of market regime")
            
        except Exception as e:
            logger.error(f"Error visualizing market regime: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def test_market_regime_classification(self):
        """
        Test market regime classification
        
        Returns:
            pd.DataFrame: Data with market regime
        """
        logger.info("Testing market regime classification")
        
        # Load component results
        component_results = self.load_component_results()
        
        if component_results is None:
            logger.error("No component results available for testing")
            return None
        
        # Align component data
        aligned_df = self.align_component_data(component_results)
        
        if aligned_df is None:
            logger.error("Failed to align component data")
            return None
        
        # Calculate market regime
        result_df = self.calculate_market_regime(aligned_df)
        
        # Save results
        output_path = os.path.join(self.output_dir, "market_regime_results.csv")
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved market regime results to {output_path}")
        
        # Save regime mapping
        regime_mapping_path = os.path.join(self.output_dir, "market_regime_mapping.json")
        with open(regime_mapping_path, 'w') as f:
            json.dump(self.regime_mapping, f, indent=4)
        logger.info(f"Saved market regime mapping to {regime_mapping_path}")
        
        # Visualize results
        self.visualize_market_regime(result_df)
        
        # Log summary statistics
        if 'Market_Regime' in result_df.columns:
            regime_counts = result_df['Market_Regime'].value_counts()
            logger.info(f"Market Regime distribution: {regime_counts.to_dict()}")
        
        logger.info("Market regime classification testing completed")
        
        return result_df

def main():
    """
    Main function to run the market regime classification testing
    """
    logger.info("Starting market regime classification testing")
    
    # Create configuration
    config = {
        'data_dir': '/home/ubuntu/market_regime_testing/processed_data',
        'output_dir': '/home/ubuntu/market_regime_testing/test_results/market_regime',
        'results_dir': '/home/ubuntu/market_regime_testing/test_results',
        'num_regimes': 18,
        'lookback_period': 5,
        'confidence_threshold': 0.7,
        'greek_sentiment_weight': 0.20,
        'trending_oi_pa_weight': 0.30,
        'iv_skew_weight': 0.15,
        'technical_indicators_weight': 0.20,
        'ce_pe_percentile_weight': 0.15
    }
    
    # Create market regime classification tester
    tester = MarketRegimeClassificationTester(config)
    
    # Test market regime classification
    result_df = tester.test_market_regime_classification()
    
    logger.info("Market regime classification testing completed")
    
    return result_df

if __name__ == "__main__":
    main()
