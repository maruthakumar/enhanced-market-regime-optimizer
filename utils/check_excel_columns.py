"""
Check Excel file columns to understand their structure.
"""
import pandas as pd
import os
import glob

def inspect_excel_files():
    """Check Excel files in the Python_Multi_Zone_Files directory."""
    excel_dir = "data/input/Python_Multi_Zone_Files"
    excel_files = glob.glob(os.path.join(excel_dir, "*.xlsx"))
    
    print(f"Found {len(excel_files)} Excel files in {excel_dir}")
    
    for file in excel_files:
        try:
            print(f"\nInspecting file: {os.path.basename(file)}")
            
            # First, get sheet names
            xl = pd.ExcelFile(file)
            sheet_names = xl.sheet_names
            print(f"Sheets: {sheet_names}")
            
            # Check each sheet
            for sheet in sheet_names:
                print(f"\n  Sheet: {sheet}")
                try:
                    # Read a few rows
                    df = pd.read_excel(file, sheet_name=sheet, nrows=10)
                    print(f"  Columns: {df.columns.tolist()}")
                    print(f"  Sample data:")
                    print(df.head(3))
                except Exception as e:
                    print(f"  Error reading sheet: {str(e)}")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

if __name__ == "__main__":
    inspect_excel_files() 