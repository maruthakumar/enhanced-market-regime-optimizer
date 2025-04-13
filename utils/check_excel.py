"""
Check if Excel files can be read with pandas
"""
import sys
import subprocess
import pandas as pd
import os

# Install openpyxl if not installed
try:
    import openpyxl
    print("openpyxl is already installed.")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    print("openpyxl has been installed.")

def check_excel_file(file_path):
    """Read an Excel file and display its structure."""
    try:
        # Get sheet names first
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        print(f"Successfully read Excel file: {file_path}")
        print(f"Sheet names: {sheet_names}")
        
        # Check each sheet
        for sheet in sheet_names:
            print(f"\nSheet: {sheet}")
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                print(f"Columns: {df.columns.tolist()}")
                print(f"Number of rows: {len(df)}")
                if len(df) > 0:
                    print("Sample data:")
                    print(df.head(5))
                
                # If this is a transaction sheet, show more details
                if 'Trans' in sheet:
                    if 'Entry Date' in df.columns and 'Strategy Name' in df.columns and 'Net PNL' in df.columns:
                        print("\nThis appears to be a transaction sheet with required columns")
                        entry_dates = pd.to_datetime(df['Entry Date']).dt.date.unique() if len(df) > 0 else []
                        print(f"Unique entry dates: {len(entry_dates)} dates")
                        if len(entry_dates) > 0:
                            print(f"First date: {min(entry_dates)}, Last date: {max(entry_dates)}")
                        
                        strategies = df['Strategy Name'].unique() if len(df) > 0 else []
                        print(f"Unique strategies: {len(strategies)} strategies")
                        if len(strategies) > 0 and len(strategies) < 10:
                            print(f"Strategies: {strategies}")
            except Exception as e:
                print(f"Error reading sheet {sheet}: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            check_excel_file(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        print("Please provide the path to an Excel file as a command line argument.") 