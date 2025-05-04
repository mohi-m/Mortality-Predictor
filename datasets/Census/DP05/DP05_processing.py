import os
import pandas as pd
from typing import List, Dict

# Constants
INPUT_FOLDER = "datasets/Census/DP05"
OUTPUT_FILE = "datasets/Census/DP05/processed/DP05_combined.csv"

# LA County Areas
AREAS: List[str] = [
    "Illinois"
]

# Column mapping for renaming
COLUMN_MAPPING_NEW: Dict[str, str] = {
    "SEX AND AGE!!Total population": "Total population",
    "SEX AND AGE!!Total population!!Sex ratio (males per 100 females)": "Sex ratio",
    "SEX AND AGE!!Total population!!Median age (years)": "Median age",
    "SEX AND AGE!!Total population!!Under 18 years": "Under 18 years",
    "SEX AND AGE!!Total population!!18 years and over": "18 years and over",
    "SEX AND AGE!!Total population!!62 years and over": "62 years and over",
    "Total housing units": "Total housing units",
}

COLUMN_MAPPING_OLD: Dict[str, str] = {
    "SEX AND AGE!!Total population": "Total population",
    "SEX AND AGE!!Total population!!Sex ratio (males per 100 females)": "Sex ratio",
    "SEX AND AGE!!Median age (years)": "Median age",
    "SEX AND AGE!!Total population!!Under 18 years": "Under 18 years",
    "SEX AND AGE!!18 years and over": "18 years and over",
    "SEX AND AGE!!62 years and over": "62 years and over",
    "HISPANIC OR LATINO AND RACE!!Total housing units": "Total housing units",
}


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
    Process demographic data from ACS DP05 dataset.
    Returns:
        pd.DataFrame: Processed demographic data
    """
    try:
        # Read and filter data
        df = pd.read_csv(input_file)
        df = df[df["Label (Grouping)"] == "    Estimate"]

        # Add and organize area information
        df.reset_index(drop=True, inplace=True)
        df["Area"] = AREAS

        if year < 2017:
            df = process_old_demographic_data(year, df)
        else:
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


def process_old_demographic_data(year: int, old_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process old demographic data from ACS DP05 dataset.
    Returns:
        pd.DataFrame: Processed demographic data
    """
    male_column = 'SEX AND AGE!!Male'
    female_column = 'SEX AND AGE!!Female'
    if year > 2012:
        male_column = 'SEX AND AGE!!Total population!!Male'
        female_column = 'SEX AND AGE!!Total population!!Female'

    

    # Create a new column 'SEX AND AGE!!Total population!!Sex ratio (males per 100 females)'
    # using the "SEX AND AGE!!Male","SEX AND AGE!!Female" columns
    # Remove commas and convert columns to numeric to ensure calculations work
    old_df[male_column] = pd.to_numeric(old_df[male_column].str.replace(",", ""), errors="coerce")
    old_df[female_column] = pd.to_numeric(old_df[female_column].str.replace(",", ""), errors="coerce")
    
    old_df["SEX AND AGE!!Total population!!Sex ratio (males per 100 females)"] = (
        old_df[male_column] / old_df[female_column] * 100
    )

    # Create a new column 'SEX AND AGE!!Total population!!Under 18 years'
    # using 'SEX AND AGE!!Total population' and 'SEX AND AGE!!18 years and over' columns
    old_df["SEX AND AGE!!Total population"] = pd.to_numeric(old_df["SEX AND AGE!!Total population"].str.replace(",", ""), errors="coerce")
    old_df["SEX AND AGE!!18 years and over"] = pd.to_numeric(old_df["SEX AND AGE!!18 years and over"].str.replace(",", ""), errors="coerce")

    old_df["SEX AND AGE!!Total population!!Under 18 years"] = (
        old_df["SEX AND AGE!!Total population"]
        - old_df["SEX AND AGE!!18 years and over"]
    )

    # Select and rename columns
    columns_to_keep = ["Area"] + list(COLUMN_MAPPING_OLD.keys())
    old_df = old_df[columns_to_keep]
    old_df = old_df.rename(columns=COLUMN_MAPPING_OLD)

    # Add year and reorder columns
    old_df["Year"] = year
    final_columns = ["Year", "Area"] + list(COLUMN_MAPPING_OLD.values())
    old_df = old_df[final_columns]
    return old_df


def process_new_demographic_data(year: int, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process new demographic data from ACS DP05 dataset.
    Returns:
        pd.DataFrame: Processed demographic data
    """
    # Select and rename columns
    columns_to_keep = ["Area"] + list(COLUMN_MAPPING_NEW.keys())
    new_df = new_df[columns_to_keep]
    new_df = new_df.rename(columns=COLUMN_MAPPING_NEW)

    # Add year and reorder columns
    new_df["Year"] = year
    final_columns = ["Year", "Area"] + list(COLUMN_MAPPING_NEW.values())
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

    # Round all numeric columns to whole numbers except 'Sex ratio' and 'Median age'.
    numeric_columns = [col for col in interpolated_df.columns if interpolated_df[col].dtype in ['float64', 'int64']]
    for col in numeric_columns:
        if col not in ["Sex ratio", "Median age"]:
            interpolated_df[col] = interpolated_df[col].round().astype("Int64")
        else:
            interpolated_df[col] = interpolated_df[col].round(2)  # Round to 2 decimal places for 'Sex ratio' and 'Median age'

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
