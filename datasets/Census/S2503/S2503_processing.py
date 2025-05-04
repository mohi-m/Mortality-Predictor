import os
import pandas as pd
from typing import List, Dict

# Constants
INPUT_FOLDER = "datasets/Census/S2503"
OUTPUT_FILE = "datasets/Census/S2503/processed/S2503_combined.csv"

# LA County Areas
AREAS: List[str] = [
    "Illinois"
]

# Method to get column mapping based on the year
def get_column_mapping(year: int) -> Dict[str, str]:
    """
    Get the column mapping based on the year.
    Args:
        year (int): The year to determine the column mapping.
    Returns:
        Dict[str, str]: The column mapping dictionary.
    """
    # Column mapping for renaming
    column_mapping: Dict[str, str] = {
        f"Occupied housing units!!HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN {year} INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)": "Median_household_income",
        "Occupied housing units!!MONTHLY HOUSING COSTS!!Median (dollars)": "Median_housing_costs"
    }
    return column_mapping


def extract_year_from_filename(filename: str) -> int:
    """
    Extract the year from the filename.
    Args:
        filename (str): The filename to extract the year from.
    Returns:
        int: The extracted year.
    """
    try:
        # Assuming the filename format is ACSDP5Y{year}.DP05-...
        year_str = filename.split("Y")[1].split(".")[0]
        return int(year_str)
    except Exception as e:
        print(f"Error extracting year from filename: {e}")
        raise


def process_demographic_data(year: int, input_file: str) -> pd.DataFrame:
    """
    Process demographic data from ACS S2503 dataset.
    Returns:
        pd.DataFrame: Processed demographic data
    """
    try:
        # Read and filter data
        df = pd.read_csv(input_file)

        # index of rows with "    Occupied housing units" in the Label (Grouping) column
        total_rows = df[
            df["Label (Grouping)"].str.contains("    Occupied housing units", na=False)
        ].index
        # Increase the index by 1 to get the next row
        total_rows += 1
        # Filter the DataFrame to keep only the row after the row with "    Occupied housing units" in the Label (Grouping) column
        df = df.iloc[total_rows]

        # Add and organize area information
        df.reset_index(drop=True, inplace=True)
        df["Area"] = AREAS

        df = process_new_demographic_data(year, df)
        
        # Convert all the numeric columns to numeric type
        for col in df.columns[2:]:
            if df[col].dtype == 'object':  # Check if the column contains strings
                df[col] = df[col].str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        print(f"Error processing data for {input_file}: {e}")
        raise

def process_new_demographic_data(year: int, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process new demographic data from ACS S2503 dataset.
    Returns:
        pd.DataFrame: Processed demographic data
    """
    column_mapping = get_column_mapping(year)
    # Select and rename columns
    columns_to_keep = ["Area"] + list(column_mapping.keys())
    new_df = new_df[columns_to_keep]
    new_df = new_df.rename(columns=column_mapping)

    # Add year and reorder columns
    new_df["Year"] = year
    final_columns = ["Year", "Area"] + list(column_mapping.values())
    new_df = new_df[final_columns]
    return new_df


def process_all_files(input_folder: str) -> pd.DataFrame:
    """
    Process all demographic data files in the input folder.
    Returns:
        pd.DataFrame: Combined demographic data from all files
    """
    combined_df = pd.DataFrame()
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            year = extract_year_from_filename(file)
            input_file_path = os.path.join(input_folder, file)
            df = process_demographic_data(year, input_file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Convert the 'Year' column to datetime format
    combined_df["Year"] = pd.to_datetime(combined_df["Year"], format="%Y")

    combined_df = extrapolate_2024_data(combined_df)

    # Create an empty DataFrame to store the results
    interpolated_df = pd.DataFrame()

    # Perform interpolation and extrapolation for each Area separately
    for area in AREAS:
        area_df = combined_df[combined_df["Area"] == area].copy()
        area_df.set_index("Year", inplace=True)
        area_df = area_df.resample("D").asfreq().interpolate(method="linear", limit_direction="both")
        area_df = area_df.resample("W-SAT").mean(numeric_only=True)  # Resample to weekly frequency ending on Saturday
        area_df["Area"] = area  # Ensure the 'Area' column is preserved
        area_df = area_df.reset_index()  # Reset index to make 'Year' a column
        area_df = area_df[["Year", "Area"] + [col for col in area_df.columns if col not in ["Year", "Area"]]]  # Reorder columns
        interpolated_df = pd.concat([interpolated_df, area_df])

    # Reset the index and rename the 'Year' column to 'Date'
    interpolated_df.reset_index(inplace=True)
    interpolated_df.rename(columns={"Year": "Date"}, inplace=True)

    # Round all numeric columns to whole numbers.
    numeric_columns = [col for col in interpolated_df.columns if interpolated_df[col].dtype in ['float64', 'int64']]
    for col in numeric_columns:
        interpolated_df[col] = interpolated_df[col].round().astype("Int64")

    combined_df = interpolated_df

    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Combined DataFrame:", combined_df.head())
    return combined_df

def extrapolate_2024_data(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrapolate data for 2024 based on 2022-2023 trend.
    
    Args:
        combined_df (pd.DataFrame): Input DataFrame with historical data
    
    Returns:
        pd.DataFrame: DataFrame with 2024 projections added
    """
    for area in AREAS:
        area_data = combined_df[combined_df['Area'] == area].copy()
        years_2022_2023 = area_data[area_data['Year'].dt.year.isin([2022, 2023])]
        
        if len(years_2022_2023) >= 2:
            # Calculate the rate of change
            last_row = years_2022_2023.iloc[-1].copy()
            prev_row = years_2022_2023.iloc[-2]
            
            # Create new row for 2024
            new_row = last_row.copy()
            new_row['Year'] = pd.to_datetime('2024-01-01')
            
            # Update numeric columns based on trend
            for col in last_row.index:
                if col not in ['Year', 'Area'] and pd.api.types.is_numeric_dtype(combined_df[col]):
                    change = last_row[col] - prev_row[col]
                    new_row[col] = last_row[col] + change
            
            # Add the new row
            combined_df = pd.concat([combined_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return combined_df


def main():
    """Main execution function"""
    try:
        df = process_all_files(INPUT_FOLDER)
        # Save the processed data to a CSV file
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Data successfully processed and saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to process data: {e}")


if __name__ == "__main__":
    main()
