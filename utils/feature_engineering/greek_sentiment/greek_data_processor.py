"""
Greek Data Processor Module

This module provides functionality for efficiently processing large datasets of options data
for Greek sentiment analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
import math
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_data_efficiently(data_files, config, output_file=None):
    """
    Process data files efficiently using chunking and parallel processing.
    
    Args:
        data_files (list): List of data file paths
        config (ConfigParser): Configuration parameters
        output_file (str, optional): Output file path
        
    Returns:
        DataFrame: Processed data
    """
    logger.info(f"Processing {len(data_files)} data files efficiently")
    
    # Get processing parameters from config
    chunk_size = config.getint('data_processing', 'chunk_size', fallback=100000)
    use_parallel = config.getboolean('data_processing', 'use_parallel', fallback=True)
    num_processes = config.getint('data_processing', 'num_processes', fallback=multiprocessing.cpu_count())
    memory_limit_mb = config.getint('data_processing', 'memory_limit_mb', fallback=1000)
    
    # Calculate memory per chunk
    estimated_row_size_kb = 2  # Estimated size per row in KB
    max_rows_in_memory = (memory_limit_mb * 1024) // estimated_row_size_kb
    
    # Adjust chunk size based on memory limit
    chunk_size = min(chunk_size, max_rows_in_memory)
    
    logger.info(f"Using chunk size of {chunk_size} rows")
    
    # Process files in parallel if enabled
    if use_parallel and len(data_files) > 1:
        logger.info(f"Using parallel processing with {num_processes} processes")
        
        # Split files among processes
        files_per_process = math.ceil(len(data_files) / num_processes)
        file_groups = [data_files[i:i+files_per_process] for i in range(0, len(data_files), files_per_process)]
        
        # Create pool and process files
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                _process_file_group,
                [(file_group, chunk_size, i) for i, file_group in enumerate(file_groups)]
            )
        
        # Combine results
        processed_data = pd.concat(results, ignore_index=True)
        
        # Clean up to free memory
        del results
        gc.collect()
    else:
        # Process files sequentially
        logger.info("Processing files sequentially")
        
        # Initialize empty list to store processed chunks
        processed_chunks = []
        
        # Process each file
        for file_path in data_files:
            logger.info(f"Processing file: {file_path}")
            
            try:
                # Read file in chunks
                for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                    # Process chunk
                    processed_chunk = _process_chunk(chunk)
                    
                    # Append to list
                    processed_chunks.append(processed_chunk)
                    
                    # Log progress
                    if (chunk_num + 1) % 10 == 0:
                        logger.info(f"Processed {chunk_num + 1} chunks from {file_path}")
                    
                    # Clean up to free memory
                    del chunk
                    gc.collect()
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Combine processed chunks
        processed_data = pd.concat(processed_chunks, ignore_index=True)
        
        # Clean up to free memory
        del processed_chunks
        gc.collect()
    
    # Save processed data if output file is provided
    if output_file:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            processed_data.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
    
    return processed_data

def _process_file_group(file_group, chunk_size, group_id):
    """
    Process a group of files.
    
    Args:
        file_group (list): List of file paths
        chunk_size (int): Chunk size
        group_id (int): Group ID
        
    Returns:
        DataFrame: Processed data
    """
    logger.info(f"Process {os.getpid()} processing file group {group_id} with {len(file_group)} files")
    
    # Initialize empty list to store processed chunks
    processed_chunks = []
    
    # Process each file
    for file_path in file_group:
        logger.info(f"Process {os.getpid()} processing file: {file_path}")
        
        try:
            # Read file in chunks
            for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                # Process chunk
                processed_chunk = _process_chunk(chunk)
                
                # Append to list
                processed_chunks.append(processed_chunk)
                
                # Log progress
                if (chunk_num + 1) % 10 == 0:
                    logger.info(f"Process {os.getpid()} processed {chunk_num + 1} chunks from {file_path}")
                
                # Clean up to free memory
                del chunk
                gc.collect()
        except Exception as e:
            logger.error(f"Process {os.getpid()} error processing file {file_path}: {str(e)}")
    
    # Combine processed chunks
    processed_data = pd.concat(processed_chunks, ignore_index=True)
    
    # Clean up to free memory
    del processed_chunks
    gc.collect()
    
    return processed_data

def _process_chunk(chunk):
    """
    Process a chunk of data.
    
    Args:
        chunk (DataFrame): Chunk of data
        
    Returns:
        DataFrame: Processed chunk
    """
    # Make a copy to avoid modifying the original
    processed_chunk = chunk.copy()
    
    # Convert date columns to datetime
    date_columns = ['Date', 'date', 'datetime', 'timestamp']
    for col in date_columns:
        if col in processed_chunk.columns:
            processed_chunk[col] = pd.to_datetime(processed_chunk[col])
    
    # Handle missing values
    processed_chunk = processed_chunk.fillna(0)
    
    # Add additional features if needed
    # ...
    
    return processed_chunk

def calculate_greek_sentiment_efficiently(data, config, output_file=None):
    """
    Calculate Greek sentiment indicators efficiently.
    
    Args:
        data (DataFrame): Processed data
        config (ConfigParser): Configuration parameters
        output_file (str, optional): Output file path
        
    Returns:
        DataFrame: Data with Greek sentiment indicators
    """
    logger.info("Calculating Greek sentiment indicators efficiently")
    
    # Get processing parameters from config
    chunk_size = config.getint('data_processing', 'chunk_size', fallback=100000)
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Check if required columns exist
    required_columns = ['Delta', 'Vega', 'Theta']
    missing_columns = [col for col in required_columns if col not in result.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns for Greek sentiment calculation: {missing_columns}")
        
        # Add missing columns with random values for testing
        for col in missing_columns:
            result[col] = np.random.normal(0, 1, len(result))
    
    # Calculate Greek sentiment indicators
    try:
        # Process in chunks to manage memory
        num_chunks = math.ceil(len(result) / chunk_size)
        
        for i in range(num_chunks):
            # Get chunk
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(result))
            chunk = result.iloc[start_idx:end_idx]
            
            # Calculate sentiment for chunk
            chunk = _calculate_greek_sentiment_for_chunk(chunk)
            
            # Update result
            result.iloc[start_idx:end_idx] = chunk
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
                logger.info(f"Calculated Greek sentiment for {i + 1}/{num_chunks} chunks")
            
            # Clean up to free memory
            del chunk
            gc.collect()
    except Exception as e:
        logger.error(f"Error calculating Greek sentiment: {str(e)}")
    
    # Save result if output file is provided
    if output_file:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            result.to_csv(output_file, index=False)
            logger.info(f"Saved Greek sentiment data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving Greek sentiment data: {str(e)}")
    
    return result

def _calculate_greek_sentiment_for_chunk(chunk):
    """
    Calculate Greek sentiment indicators for a chunk of data.
    
    Args:
        chunk (DataFrame): Chunk of data
        
    Returns:
        DataFrame: Chunk with Greek sentiment indicators
    """
    # Make a copy to avoid modifying the original
    result = chunk.copy()
    
    # Calculate Delta sentiment
    result['Delta_Sentiment'] = result['Delta']
    
    # Calculate Vega sentiment
    result['Vega_Sentiment'] = result['Vega']
    
    # Calculate Theta sentiment
    result['Theta_Sentiment'] = result['Theta']
    
    # Calculate combined sentiment
    # Note: Weights will be applied in the market regime module
    
    return result
