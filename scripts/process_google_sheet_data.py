import pandas as pd
import os
from urllib.parse import quote

def get_master_rainfall_data(sheet_url):
    print("--- Starting Master Data Processing Script (v2) ---")

    try:
        google_sheet_id = sheet_url.split('/d/')[1].split('/')[0]
        base_export_url = f'https://docs.google.com/spreadsheets/d/{google_sheet_id}/gviz/tq?tqx=out:csv&sheet='
        print("Successfully created base URL for CSV export.")
    except IndexError:
        print(f"Error: Could not parse the Google Sheet ID from the URL: {sheet_url}")
        return None

    subdivisions = [
        'ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA',
        'NAGA MANI MIZO TRIPURA', 'SUB HIMALAYAN WEST BENGAL & SIKKIM',
        'GANGETIC WEST BENGAL', 'ORISSA', 'JHARKHAND', 'BIHAR', 'EAST UTTAR PRADESH',
        'WEST UTTAR PRADESH', 'UTTARAKHAND', 'HARYANA DELHI & CHANDIGARH', 'PUNJAB',
        'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN', 'EAST RAJASTHAN',
        'WEST MADHYA PRADESH', 'EAST MADHYA PRADESH', 'GUJARAT REGION',
        'SAURASHTRA & KUTCH', 'KONKAN & GOA', 'MADHYA MAHARASHTRA',
        'MARATHWADA', 'VIDARBHA', 'CHHATTISGARH', 'COASTAL ANDHRA PRADESH',
        'TELANGANA', 'RAYALSEEMA', 'TAMILNADU', 'COASTAL KARNATAKA',
        'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA', 'KERALA', 'LAKSHADWEEP'
    ]
    
    all_dataframes = []
    
    print(f"\nProcessing {len(subdivisions)} subdivisions...")
    for i, sheet_name in enumerate(subdivisions):
        print(f"  {i+1}/{len(subdivisions)}: Processing '{sheet_name}'...")
        
        try:
            full_url = base_export_url + quote(sheet_name)
            df_raw = pd.read_csv(full_url, on_bad_lines='skip')
            
            # --- THIS IS THE CORRECTED PART ---
            # This new list matches your 15-column structure.
            column_headers = [
                'Year', 'DataType', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual'
            ]
            
            if len(df_raw.columns) != len(column_headers):
                print(f"    -> Warning: Column count mismatch for {sheet_name}. Expected {len(column_headers)}, found {len(df_raw.columns)}. Skipping this sheet.")
                continue

            df_raw.columns = column_headers
            
            # --- THIS IS THE NEW, MORE RELIABLE CLEANING METHOD ---
            # We explicitly remove the rows where DataType is 'NORM'.
            df_cleaned = df_raw[df_raw['DataType'] == 'ACTL'].copy()

            # Ensure 'Year' column is a number
            df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce')
            df_cleaned = df_cleaned.dropna(subset=['Year'])
            df_cleaned['Year'] = df_cleaned['Year'].astype(int)
            
            # Add the Subdivision name
            df_cleaned['Subdivision'] = sheet_name
            
            # We don't need the 'DataType' column in our final master file
            df_cleaned = df_cleaned.drop(columns=['DataType'])
            
            all_dataframes.append(df_cleaned)
            
        except Exception as e:
            print(f"    -> Failed to process {sheet_name}. Error: {e}")

    if not all_dataframes:
        print("\nNo data was processed successfully. Could not create a master file.")
        return None
        
    print("\nCombining all processed data into one master file...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Re-order the columns to be more logical
    cols_to_order = ['Subdivision', 'Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    master_df = master_df[cols_to_order]

    print("Master DataFrame created successfully!")
    return master_df

if __name__ == "__main__":
    google_sheet_url = "https://docs.google.com/spreadsheets/d/1CRJxniW_QOj6V5ugHbG-6bMPM-I9aNoCba0Pd1nBqrw/edit?usp=sharing"
    
    master_data = get_master_rainfall_data(google_sheet_url)
    
    if master_data is not None:
        output_folder = 'data'
        output_filename = os.path.join(output_folder, 'master_rainfall_india.csv')
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        master_data.to_csv(output_filename, index=False)
        
        print(f"\n>>> FINAL SUCCESS! Master data file saved to '{output_filename}'")
        print("\nHere's a preview of the final combined data:")
        print(master_data.head())
        print("\n")
        print(master_data.tail())