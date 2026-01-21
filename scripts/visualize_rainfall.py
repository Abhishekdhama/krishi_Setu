import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_rainfall_trend(subdivision_name):
    print(f"--- Generating Enhanced Rainfall Trend Plot for: {subdivision_name} ---")

    master_file_path = os.path.join('data', 'master_rainfall_india.csv')

    try:
        df = pd.read_csv(master_file_path)
        print("Master rainfall data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The master file was not found at '{master_file_path}'")
        return

    subdivision_df = df[df['Subdivision'] == subdivision_name].copy()

    if subdivision_df.empty:
        print(f"Error: No data found for subdivision '{subdivision_name}'.")
        return

    subdivision_df = subdivision_df.sort_values(by='Year')
    
    subdivision_df['10_Yr_Rolling_Avg'] = subdivision_df['Annual'].rolling(window=10).mean()
    print("10-Year rolling average calculated.")

    min_year = subdivision_df['Year'].min()
    max_year = subdivision_df['Year'].max()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(subdivision_df['Year'], subdivision_df['Annual'], 
            color='lightgray', 
            marker='o', 
            markersize=4, 
            linestyle='--', 
            label='Annual Rainfall')
            
    ax.plot(subdivision_df['Year'], subdivision_df['10_Yr_Rolling_Avg'], 
            color='navy', 
            linewidth=2.5, 
            label='10-Year Rolling Average')

    mean_rainfall = subdivision_df['Annual'].mean()
    ax.axhline(y=mean_rainfall, color='red', linestyle='--', linewidth=2, label=f'Long-term Average ({mean_rainfall:.2f} mm)')

    ax.set_title(f'Annual Rainfall Trend in {subdivision_name} ({min_year}-{max_year})', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Rainfall (in mm)', fontsize=12, fontweight='bold')
    
    ax.legend(loc='best', fontsize=12)
    ax.grid(True)

    fig.tight_layout()

    output_folder = 'plots'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clean_subdivision_name = subdivision_name.lower().replace(' ', '_').replace('&', 'and')
    output_filename = os.path.join(output_folder, f'{clean_subdivision_name}_rainfall_trend_with_avg.png')

    plt.savefig(output_filename, dpi=300)
    print(f"\n>>> SUCCESS! Enhanced plot saved to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    target_region = 'WEST UTTAR PRADESH'
    plot_rainfall_trend(target_region)