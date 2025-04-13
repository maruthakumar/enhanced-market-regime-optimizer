import os
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import logging
import argparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_zip_files(directory, force_extract=False):
    """Extract all zip files in the given directory if not already extracted."""
    # Check if files already extracted
    temp_dir = os.path.join(directory, "temp")
    if os.path.exists(temp_dir) and not force_extract:
        files_in_temp = glob.glob(os.path.join(temp_dir, "**/*.csv"), recursive=True)
        if files_in_temp:
            logging.info(f"Found {len(files_in_temp)} files already extracted. Use --force-extract to re-extract.")
            return
    
    # Extract each zip file
    zip_files = glob.glob(os.path.join(directory, "*.zip"))
    total_zips = len(zip_files)
    
    for i, zip_file in enumerate(zip_files, 1):
        zip_name = os.path.basename(zip_file)
        logging.info(f"Extracting {zip_name} ({i}/{total_zips})")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        logging.info(f"Extracted {zip_name}")

def get_csv_files(directory, max_files=None):
    """Get all CSV files in the specified directory."""
    all_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    
    if max_files and len(all_files) > max_files:
        logging.info(f"Limiting to {max_files} files out of {len(all_files)}")
        return all_files[:max_files]
    
    return all_files

def format_option_data(csv_files, output_dir):
    """Process option data files and format for Greek sentiment indicator."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a single output file
    combined_output = os.path.join(output_dir, "formatted_greek_data.csv")
    
    # Process each file and collect data
    total_files = len(csv_files)
    
    # Create column list for output
    columns = ['date', 'time', 'expiry', 'strike', 'underlying_price', 
               'call_delta', 'call_vega', 'call_theta', 'call_gamma',
               'put_delta', 'put_vega', 'put_theta', 'put_gamma']
    
    # Process in chunks to avoid memory issues
    chunk_size = 10
    total_chunks = (total_files + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_files)
        
        logging.info(f"Processing chunk {chunk_idx+1}/{total_chunks} (files {start_idx+1}-{end_idx}/{total_files})")
        
        chunk_files = csv_files[start_idx:end_idx]
        chunk_data = []
        
        for i, file in enumerate(chunk_files):
            file_name = os.path.basename(file)
            logging.info(f"Processing {file_name} ({start_idx+i+1}/{total_files})")
            
            try:
                # Read the CSV file
                df = pd.read_csv(file)
                
                # Check for required columns
                required_cols = ['date', 'time', 'expiry', 'strike', 'underlying_price',
                                'call_delta', 'call_vega', 'call_theta', 'call_gamma',
                                'put_delta', 'put_vega', 'put_theta', 'put_gamma']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logging.warning(f"File {file_name} missing columns: {missing_cols}")
                    continue
                
                # Select only the columns we need
                df = df[required_cols].copy()
                
                # Filter for options with delta in the desired range (0.5 to 0.1 for calls, -0.5 to -0.1 for puts)
                filtered_df = df[
                    ((df['call_delta'] <= 0.5) & (df['call_delta'] >= 0.1)) | 
                    ((df['put_delta'] >= -0.5) & (df['put_delta'] <= -0.1))
                ]
                
                if filtered_df.empty:
                    logging.warning(f"No relevant options found in {file_name}")
                    continue
                    
                chunk_data.append(filtered_df)
                
            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")
        
        # Save chunk data to file
        if chunk_data:
            chunk_df = pd.concat(chunk_data, ignore_index=True)
            
            # Save to file (append if not first chunk)
            if chunk_idx == 0:
                chunk_df.to_csv(combined_output, index=False)
                logging.info(f"Created output file {combined_output} with {len(chunk_df)} rows")
            else:
                # Append without headers
                chunk_df.to_csv(combined_output, mode='a', header=False, index=False)
                logging.info(f"Appended {len(chunk_df)} rows to {combined_output}")
    
    # Also create a compatibility file in the legacy location
    legacy_dir = 'data/input/Python_Multi_Zone_Files'
    legacy_file = os.path.join(legacy_dir, "formatted_greek_data.csv")
    
    os.makedirs(legacy_dir, exist_ok=True)
    
    if os.path.exists(combined_output):
        import shutil
        shutil.copy2(combined_output, legacy_file)
        logging.info(f"Created compatibility file at {legacy_file}")
        
        # Check if file exists and return total row count
        try:
            with open(combined_output, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            logging.info(f"Total rows in output file: {line_count}")
        except Exception as e:
            logging.error(f"Error counting rows: {e}")
            
        return True
    else:
        logging.error("No output file was created")
        return False

def main():
    parser = argparse.ArgumentParser(description='Format market data for Greek sentiment indicator')
    parser.add_argument('--market_data_dir', default='data/market_data',
                       help='Directory containing market data')
    parser.add_argument('--output', default='data/market_data/formatted',
                       help='Output directory')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--force-extract', action='store_true', 
                       help='Force re-extraction of zip files')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    logger.info(f"Processing market data from {args.market_data_dir}")
    
    # Make sure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Extract zip files
    extract_zip_files(args.market_data_dir, args.force_extract)
    
    # Get all CSV files
    csv_files = get_csv_files(os.path.join(args.market_data_dir, "temp"), args.max_files)
    
    if not csv_files:
        logger.warning("No CSV files found in the extracted directories.")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Format the option data
    success = format_option_data(csv_files, args.output)
    
    elapsed = time.time() - start_time
    
    if success:
        logger.info(f"Data processing complete in {elapsed:.2f} seconds")
    else:
        logger.error(f"Data processing failed after {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 