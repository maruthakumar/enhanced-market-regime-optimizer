"""
Helper functions for the pipeline.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def ensure_directory_exists(directory_path):
    """
    Ensure that the given directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Directory path
        
    Returns:
        bool: True if directory already existed or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {str(e)}")
        return False

def save_to_csv(data_frame, file_path):
    """
    Save DataFrame to CSV file.
    
    Args:
        data_frame (DataFrame): Data to save
        file_path (str): Path to save file
        
    Returns:
        bool: True if file was saved successfully
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        data_frame.to_csv(file_path, index=False)
        logging.info(f"Saved data to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {str(e)}")
        return False

def plot_and_save(plot_func, file_path, *args, **kwargs):
    """
    Create a plot and save it to a file.
    
    Args:
        plot_func (callable): Function to create plot
        file_path (str): Path to save plot
        *args: Arguments to pass to plot function
        **kwargs: Keyword arguments to pass to plot function
        
    Returns:
        bool: True if plot was saved successfully
    """
    try:
        # Create directory if it doesn't exist
        ensure_directory_exists(os.path.dirname(file_path))
        
        # Create figure
        plt.figure(figsize=kwargs.pop('figsize', (10, 6)))
        
        # Call plot function
        plot_func(*args, **kwargs)
        
        # Add title and labels if provided
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        if 'xlabel' in kwargs:
            plt.xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            plt.ylabel(kwargs['ylabel'])
        
        # Add grid if requested
        plt.grid(kwargs.pop('grid', True))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(file_path, dpi=kwargs.pop('dpi', 300))
        
        # Close figure
        plt.close()
        
        logging.info(f"Saved plot to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving plot to {file_path}: {str(e)}")
        return False

def format_percentage(value):
    """
    Format value as percentage string.
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.2f}%"

def format_currency(value):
    """
    Format value as currency string.
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${value:.2f}"

def format_time(seconds):
    """
    Format seconds as time string.
    
    Args:
        seconds (float): Seconds to format
        
    Returns:
        str: Formatted time string
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}" 