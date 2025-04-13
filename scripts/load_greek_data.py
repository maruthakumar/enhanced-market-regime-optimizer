import os
import pandas as pd
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_greek_files(directory):
    """
    Load all formatted Greek data files from a directory.
    
    Args:
        directory (str): Path to directory containing formatted Greek data files
        
    Returns:
        pandas.DataFrame: Combined data from all files
    """
    logger.info(f"Loading Greek data files from {directory}")
    
    # Get all CSV files in directory
    file_pattern = os.path.join(directory, "formatted_greek_data_*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        logger.warning(f"No Greek data files found in {directory}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(files)} Greek data files")
    
    # Load each file and concatenate
    dataframes = []
    for file in files:
        try:
            logger.info(f"Loading {file}")
            df = pd.read_csv(file)
            dataframes.append(df)
            logger.info(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dataframes:
        logger.warning("No data loaded from any files")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined data contains {len(combined)} rows")
    
    return combined

def save_combined_data(combined_df, output_file):
    """
    Save combined data to a single file.
    
    Args:
        combined_df (pandas.DataFrame): Combined Greek data
        output_file (str): Path to output file
    """
    if combined_df.empty:
        logger.warning("No data to save")
        return
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved combined data ({len(combined_df)} rows) to {output_file}")
    except Exception as e:
        logger.error(f"Error saving combined data: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and combine Greek data files')
    parser.add_argument('--input-dir', default='data/market_data/formatted',
                       help='Directory containing formatted Greek data files')
    parser.add_argument('--output-file', default='data/input/Python_Multi_Zone_Files/formatted_greek_data.csv',
                       help='Path to output combined file')
    parser.add_argument('--create-legacy-file', action='store_true',
                       help='Create a copy in the legacy location for backward compatibility')
    args = parser.parse_args()
    
    # Load Greek data files
    combined_df = load_greek_files(args.input_dir)
    
    if combined_df.empty:
        return
    
    # Save combined data
    save_combined_data(combined_df, args.output_file)
    
    # Create legacy file if requested
    if args.create_legacy_file:
        legacy_file = 'data/input/Python_Multi_Zone_Files/formatted_greek_data.csv'
        if args.output_file != legacy_file:
            save_combined_data(combined_df, legacy_file)
            logger.info(f"Created legacy file at {legacy_file}")

if __name__ == "__main__":
    main() 