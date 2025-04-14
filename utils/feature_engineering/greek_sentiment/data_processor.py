"""
Optimized data processing module for the enhanced market regime optimizer.
This module provides efficient data processing functionality for handling
large datasets with Greek sentiment analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for efficient processing of large datasets with Greek sentiment analysis."""
    
    def __init__(self, config):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config (ConfigParser or dict): Configuration parameters
        """
        self.config = config
        
        # Handle different config types
        if isinstance(config, dict):
            self.data_dir = config.get('market_data_dir') if 'market_data_dir' in config else '../data/market_data'
            base_dir = config.get('base_dir') if 'base_dir' in config else '../output'
            self.chunk_size = config.get('chunk_size', 100000)
            self.use_parallel = config.get('use_parallel', True)
            self.num_processes = config.get('num_processes', mp.cpu_count() - 1)
            self.memory_limit = config.get('memory_limit_mb', 1000)  # Memory limit in MB
        else:
            self.data_dir = config.get('data_processing', 'market_data_dir', fallback='../data/market_data')
            base_dir = config.get('output', 'base_dir', fallback='../output')
            self.chunk_size = config.getint('data_processing', 'chunk_size', fallback=100000)
            self.use_parallel = config.getboolean('data_processing', 'use_parallel', fallback=True)
            self.num_processes = config.getint('data_processing', 'num_processes', fallback=mp.cpu_count() - 1)
            self.memory_limit = config.getint('data_processing', 'memory_limit_mb', fallback=1000)
            
        self.output_dir = os.path.join(base_dir, 'processed_data')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ensure at least one process
        self.num_processes = max(1, self.num_processes)
    
    def process_data_file(self, file_path, output_file=None, columns=None):
        """
        Process a single data file efficiently.
        
        Args:
            file_path (str): Path to the data file
            output_file (str, optional): Path to save processed data
            columns (list, optional): List of columns to load (None for all)
            
        Returns:
            DataFrame or None: Processed data or None if saved to file
        """
        logger.info(f"Processing data file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        # Determine output file if not specified
        if output_file is None:
            file_name = os.path.basename(file_path)
            output_file = os.path.join(self.output_dir, f"processed_{file_name}")
        
        # Check file size to determine processing approach
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # For small files, process in one go
        if file_size_mb < self.memory_limit:
            logger.info("Processing file in one batch")
            try:
                # Load data with specific columns if provided
                if columns is not None:
                    data = pd.read_csv(file_path, usecols=columns)
                else:
                    data = pd.read_csv(file_path)
                
                # Process data
                processed_data = self._process_dataframe(data)
                
                # Save processed data
                processed_data.to_csv(output_file, index=False)
                logger.info(f"Saved processed data to {output_file}")
                
                return processed_data
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                return None
        
        # For large files, process in chunks
        else:
            logger.info(f"Processing file in chunks of {self.chunk_size} rows")
            
            # Initialize chunk counter
            chunk_count = 0
            total_rows = 0
            
            try:
                # Process file in chunks
                chunk_reader = pd.read_csv(
                    file_path, 
                    chunksize=self.chunk_size,
                    usecols=columns
                )
                
                # Process each chunk
                for i, chunk in enumerate(chunk_reader):
                    chunk_count += 1
                    rows_in_chunk = len(chunk)
                    total_rows += rows_in_chunk
                    
                    logger.info(f"Processing chunk {chunk_count} with {rows_in_chunk} rows")
                    
                    # Process chunk
                    processed_chunk = self._process_dataframe(chunk)
                    
                    # Save chunk (append after first chunk)
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0
                    processed_chunk.to_csv(output_file, mode=mode, header=header, index=False)
                    
                    # Clear memory
                    del chunk, processed_chunk
                    gc.collect()
                
                logger.info(f"Processed {total_rows} rows in {chunk_count} chunks")
                logger.info(f"Saved processed data to {output_file}")
                
                return None  # Data saved to file, don't return in memory
                
            except Exception as e:
                logger.error(f"Error processing file in chunks: {str(e)}")
                return None
    
    def process_multiple_files(self, file_paths, output_file=None, columns=None):
        """
        Process multiple data files efficiently, optionally in parallel.
        
        Args:
            file_paths (list): List of paths to data files
            output_file (str, optional): Path to save combined processed data
            columns (list, optional): List of columns to load (None for all)
            
        Returns:
            DataFrame or None: Combined processed data or None if saved to file
        """
        logger.info(f"Processing {len(file_paths)} data files")
        
        # Determine output file if not specified
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f"combined_processed_data_{timestamp}.csv")
        
        # For parallel processing
        if self.use_parallel and len(file_paths) > 1 and self.num_processes > 1:
            logger.info(f"Using parallel processing with {self.num_processes} processes")
            
            # Create temporary output files for each input file
            temp_output_files = [
                os.path.join(self.output_dir, f"temp_processed_{i}_{os.path.basename(file_path)}")
                for i, file_path in enumerate(file_paths)
            ]
            
            # Create processing function with fixed parameters
            process_func = partial(
                self._process_file_wrapper,
                columns=columns
            )
            
            # Create process arguments
            process_args = list(zip(file_paths, temp_output_files))
            
            # Process files in parallel
            with mp.Pool(processes=self.num_processes) as pool:
                pool.starmap(process_func, process_args)
            
            # Combine temporary files
            self._combine_csv_files(temp_output_files, output_file)
            
            # Clean up temporary files
            for temp_file in temp_output_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logger.info(f"Saved combined processed data to {output_file}")
            return None  # Data saved to file, don't return in memory
            
        # For sequential processing
        else:
            logger.info("Processing files sequentially")
            
            all_processed_data = []
            total_rows = 0
            
            for i, file_path in enumerate(file_paths):
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                # Process file
                processed_data = self.process_data_file(file_path, columns=columns)
                
                if processed_data is not None:
                    rows = len(processed_data)
                    total_rows += rows
                    logger.info(f"Processed {rows} rows from file")
                    
                    # Append to combined data
                    all_processed_data.append(processed_data)
                    
                    # Clear memory
                    del processed_data
                    gc.collect()
            
            # Combine all processed data
            if all_processed_data:
                combined_data = pd.concat(all_processed_data, ignore_index=True)
                
                # Save combined data
                combined_data.to_csv(output_file, index=False)
                logger.info(f"Saved combined processed data with {total_rows} rows to {output_file}")
                
                return combined_data
            else:
                logger.warning("No data processed from any file")
                return None
    
    def calculate_greek_sentiment(self, data, output_file=None):
        """
        Calculate Greek sentiment indicators from options data.
        
        Args:
            data (DataFrame): Options data
            output_file (str, optional): Path to save sentiment data
            
        Returns:
            DataFrame: Data with Greek sentiment indicators
        """
        logger.info("Calculating Greek sentiment indicators")
        
        # Check if we have the necessary columns
        required_columns = ['Delta', 'Vega', 'Theta']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for Greek sentiment: {missing_columns}")
            return data
        
        # Calculate sentiment indicators
        try:
            # Group by date if available
            date_col = None
            for col in ['datetime', 'Date', 'date']:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col is not None:
                logger.info(f"Grouping by {date_col} for sentiment calculation")
                
                # Calculate sentiment by date
                sentiment_data = data.groupby(date_col).apply(self._calculate_sentiment_for_group)
                sentiment_data = sentiment_data.reset_index()
            else:
                logger.warning("No date column found, calculating sentiment for entire dataset")
                sentiment_data = self._calculate_sentiment_for_group(data)
            
            # Save sentiment data if output file specified
            if output_file is not None:
                sentiment_data.to_csv(output_file, index=False)
                logger.info(f"Saved Greek sentiment data to {output_file}")
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error calculating Greek sentiment: {str(e)}")
            return data
    
    def _process_dataframe(self, data):
        """
        Process a DataFrame with common data cleaning and preparation steps.
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            DataFrame: Processed data
        """
        # Make a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Handle missing values
        for col in processed_data.columns:
            # For numeric columns, fill with median
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
            # For categorical/string columns, fill with 'Unknown'
            elif pd.api.types.is_string_dtype(processed_data[col]) or pd.api.types.is_categorical_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].fillna('Unknown')
        
        # Convert date columns to datetime if present
        date_columns = ['date', 'Date', 'datetime', 'timestamp']
        for col in date_columns:
            if col in processed_data.columns:
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col])
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
        
        # Add any additional processing steps here
        
        return processed_data
    
    def _calculate_sentiment_for_group(self, group):
        """
        Calculate Greek sentiment indicators for a group of options data.
        
        Args:
            group (DataFrame): Group of options data
            
        Returns:
            DataFrame: Sentiment indicators for the group
        """
        # Calculate aggregate Greek values
        delta_sum = group['Delta'].sum()
        vega_sum = group['Vega'].sum()
        
        # Calculate sentiment indicators
        delta_sentiment = delta_sum / len(group) if len(group) > 0 else 0
        vega_sentiment = vega_sum / len(group) if len(group) > 0 else 0
        
        # Calculate Theta sentiment if available
        theta_sentiment = 0
        if 'Theta' in group.columns:
            theta_sum = group['Theta'].sum()
            theta_sentiment = theta_sum / len(group) if len(group) > 0 else 0
        
        # Create sentiment DataFrame
        sentiment_data = pd.DataFrame({
            'Delta_Sentiment': [delta_sentiment],
            'Vega_Sentiment': [vega_sentiment],
            'Theta_Sentiment': [theta_sentiment]
        })
        
        # Add date column if available in group
        date_col = None
        for col in ['datetime', 'Date', 'date']:
            if col in group.columns:
                date_col = col
                break
        
        if date_col is not None:
            sentiment_data[date_col] = group[date_col].iloc[0]
        
        return sentiment_data
    
    def _process_file_wrapper(self, file_path, output_file, columns=None):
        """
        Wrapper function for processing a file in parallel.
        
        Args:
            file_path (str): Path to the data file
            output_file (str): Path to save processed data
            columns (list, optional): List of columns to load (None for all)
        """
        try:
            self.process_data_file(file_path, output_file, columns)
        except Exception as e:
            logger.error(f"Error in process_file_wrapper for {file_path}: {str(e)}")
    
    def _combine_csv_files(self, input_files, output_file):
        """
        Combine multiple CSV files into a single file.
        
        Args:
            input_files (list): List of input CSV files
            output_file (str): Path to save combined data
        """
        logger.info(f"Combining {len(input_files)} CSV files")
        
        # Check if input files exist
        existing_files = [f for f in input_files if os.path.exists(f)]
        
        if not existing_files:
            logger.warning("No input files found to combine")
            return
        
        try:
            # Write first file with header
            with open(output_file, 'w') as outfile:
                with open(existing_files[0], 'r') as infile:
                    outfile.write(infile.read())
            
            # Append remaining files without header
            for file_path in existing_files[1:]:
                with open(output_file, 'a') as outfile:
                    with open(file_path, 'r') as infile:
                        # Skip header line
                        next(infile)
                        outfile.write(infile.read())
            
            logger.info(f"Combined {len(existing_files)} files into {output_file}")
            
        except Exception as e:
            logger.error(f"Error combining CSV files: {str(e)}")


def process_data_efficiently(file_paths, config, output_file=None, columns=None):
    """
    Process data files efficiently with optimized memory usage.
    
    Args:
        file_paths (list or str): Path(s) to data file(s)
        config (ConfigParser or dict): Configuration parameters
        output_file (str, optional): Path to save processed data
        columns (list, optional): List of columns to load (None for all)
        
    Returns:
        DataFrame or None: Processed data or None if saved to file
    """
    processor = DataProcessor(config)
    
    # Handle single file or list of files
    if isinstance(file_paths, str):
        return processor.process_data_file(file_paths, output_file, columns)
    else:
        return processor.process_multiple_files(file_paths, output_file, columns)


def calculate_greek_sentiment_efficiently(data, config, output_file=None):
    """
    Calculate Greek sentiment indicators efficiently.
    
    Args:
        data (DataFrame): Options data
        config (ConfigParser or dict): Configuration parameters
        output_file (str, optional): Path to save sentiment data
        
    Returns:
        DataFrame: Data with Greek sentiment indicators
    """
    processor = DataProcessor(config)
    return processor.calculate_greek_sentiment(data, output_file)
