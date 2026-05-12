import os
import glob
import pandas as pd
from pathlib import Path

# Find parquet files for each bus
bus_183_files = glob.glob(r'data/bus=183/**/*.parquet', recursive=True)
bus_208_files = glob.glob(r'data/bus=208/**/*.parquet', recursive=True)

print(f'Found {len(bus_183_files)} files for bus=183')
print(f'Found {len(bus_208_files)} files for bus=208')
print()

# Sample and analyze schemas
def analyze_bus(bus_name, files, sample_size=3):
    if not files:
        print(f'{bus_name}: No files found')
        return None
    
    print(f'=== {bus_name} ===')
    print(f'Sample file: {files[0]}')
    
    # Read first file to get schema and sample data
    df = pd.read_parquet(files[0])
    cols = set(df.columns)
    total_rows = len(df)
    
    # Sample a few more files to check consistency
    all_cols = cols.copy()
    for f in files[1:sample_size]:
        df_temp = pd.read_parquet(f)
        all_cols.update(df_temp.columns)
        total_rows += len(df_temp)
    
    print(f'Total rows (sampled {min(sample_size, len(files))} files): {total_rows}')
    print(f'Columns found ({len(all_cols)}): {sorted(all_cols)}')
    print()
    
    return {'bus': bus_name, 'cols': all_cols, 'files': files[:sample_size]}

bus_183_info = analyze_bus('BUS 183', bus_183_files)
bus_208_info = analyze_bus('BUS 208', bus_208_files)

# Compare columns
if bus_183_info and bus_208_info:
    common_cols = bus_183_info['cols'] & bus_208_info['cols']
    only_183 = bus_183_info['cols'] - bus_208_info['cols']
    only_208 = bus_208_info['cols'] - bus_183_info['cols']
    
    print('=== COLUMN COMPARISON ===')
    print(f'Common columns ({len(common_cols)}): {sorted(common_cols)}')
    print(f'Only in BUS 183 ({len(only_183)}): {sorted(only_183) if only_183 else "None"}')
    print(f'Only in BUS 208 ({len(only_208)}): {sorted(only_208) if only_208 else "None"}')
    print()
    
    print('=== EXAMPLE PATHS ===')
    print(f'BUS 183: {bus_183_info["files"][0]}')
    print(f'BUS 208: {bus_208_info["files"][0]}')
