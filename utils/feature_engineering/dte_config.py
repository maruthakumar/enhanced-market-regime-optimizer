"""
DTE Configuration Module

This module provides DTE (Days To Expiry) configuration functionality
for feature engineering modules.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DTEConfig:
    """
    DTE (Days To Expiry) configuration.
    
    This class handles DTE-related configuration and filtering for feature engineering.
    """
    
    def __init__(self, config=None):
        """
        Initialize DTE configuration.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        
        # DTE column name
        self.dte_column = self.config.get('dte_column', 'DTE')
        
        # DTE bucket boundaries
        self.dte_buckets = self.config.get('dte_buckets', [0, 7, 14, 30, 60, 90])
        
        # DTE bucket names
        self.dte_bucket_names = self.config.get('dte_bucket_names', [
            'weekly', 'biweekly', 'monthly', 'quarterly', 'long_term'
        ])
        
        # Specific DTE configuration
        self.use_specific_dte = self.config.get('use_specific_dte', False)
        self.default_dte = self.config.get('default_dte', 7)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate DTE configuration.
        """
        # Ensure buckets are sorted
        self.dte_buckets = sorted(self.dte_buckets)
        
        # Ensure bucket names match buckets
        if len(self.dte_bucket_names) != len(self.dte_buckets) - 1:
            logger.warning(f"DTE bucket names don't match buckets. Adjusting names.")
            # Generate default names based on bucket boundaries
            self.dte_bucket_names = [
                f"{self.dte_buckets[i]}-{self.dte_buckets[i+1]}"
                for i in range(len(self.dte_buckets) - 1)
            ]
    
    def filter_by_dte(self, data_frame, dte):
        """
        Filter data by specific DTE.
        
        Args:
            data_frame (pd.DataFrame): Input data
            dte (int): DTE value to filter by
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if self.dte_column not in data_frame.columns:
            logger.warning(f"DTE column {self.dte_column} not found in data")
            return data_frame
        
        # Filter by specific DTE (allowing 1 day tolerance)
        return data_frame[
            (data_frame[self.dte_column] >= dte - 1) & 
            (data_frame[self.dte_column] <= dte + 1)
        ]
    
    def add_dte_bucket_column(self, data_frame):
        """
        Add DTE bucket column to data.
        
        Args:
            data_frame (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with DTE bucket column
        """
        if self.dte_column not in data_frame.columns:
            logger.warning(f"DTE column {self.dte_column} not found in data")
            return data_frame
        
        # Make a copy to avoid modifying the original
        df = data_frame.copy()
        
        # Initialize bucket column
        df['DTE_Bucket'] = len(self.dte_buckets) - 1  # Default to the last bucket
        df['DTE_Bucket_Name'] = self.dte_bucket_names[-1] if self.dte_bucket_names else 'unknown'
        
        # Assign buckets
        for i in range(len(self.dte_buckets) - 1):
            mask = (df[self.dte_column] >= self.dte_buckets[i]) & (df[self.dte_column] < self.dte_buckets[i+1])
            df.loc[mask, 'DTE_Bucket'] = i
            if i < len(self.dte_bucket_names):
                df.loc[mask, 'DTE_Bucket_Name'] = self.dte_bucket_names[i]
        
        return df
