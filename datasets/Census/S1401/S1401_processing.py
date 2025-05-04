import os
import pandas as pd
from typing import List, Dict

# Constants
INPUT_FOLDER = "datasets/Census/S1401"
OUTPUT_FILE = "datasets/Census/S1401/processed/S1401_combined.csv"

# LA County Areas
AREAS: List[str] = [
    "Illinois"
]

# Column mapping for renaming
COLUMN_MAPPING_2017_TO_2023: Dict[str, str] = {
    "Population 3 years and over enrolled in school": "Total enrolled in school",
    "Population 3 years and over enrolled in school!!Nursery school, preschool": "Nursery school, preschool",
    "Population 3 years and over enrolled in school!!Kindergarten to 12th grade": "Kindergarten to 12th grade",
    "Population 3 years and over enrolled in school!!Kindergarten to 12th grade!!Kindergarten": "Kindergarten",
    "Population 3 years and over enrolled in school!!Kindergarten to 12th grade!!Elementary: grade 1 to grade 4": "Elementary",
    "Population 3 years and over enrolled in school!!Kindergarten to 12th grade!!Elementary: grade 5 to grade 8": "Middle school",
    "Population 3 years and over enrolled in school!!Kindergarten to 12th grade!!High school: grade 9 to grade 12": "High school",
    "Population 3 years and over enrolled in school!!College, undergraduate": "College, undergraduate",
    "Population 3 years and over enrolled in school!!Graduate, professional school": "Graduate, professional school",
}

COLUMN_MAPPING_2013_TO_2016: Dict[str, str] = {
    "Population 3 years and over enrolled in school": "Total enrolled in school",
    "Nursery school, preschool": "Nursery school, preschool",
    "Kindergarten to 12th grade": "Kindergarten to 12th grade",
    "Kindergarten": "Kindergarten",
    "Elementary: grade 1 to grade 4": "Elementary",
    "Elementary: grade 5 to grade 8": "Middle school",
    "High school: grade 9 to grade 12": "High school",
    "College, undergraduate": "College, undergraduate",
    "Graduate, professional school": "Graduate, professional school",
}

COLUMN_MAPPING_TILL_2012: Dict[str, str] = {
    "Population 3 years and over enrolled in school": "Total enrolled in school",
    "Nursery school, preschool": "Nursery school, preschool",
    "Kindergarten to 12th grade": "Kindergarten to 12th grade",
    "Kindergarten to 12th grade!!Kindergarten": "Kindergarten",
    "Kindergarten to 12th grade!!Elementary: grade 1 to grade 4": "Elementary",
    "Kindergarten to 12th grade!!Elementary: grade 5 to grade 8": "Middle school",
    "Kindergarten to 12th grade!!High school: grade 9 to grade 12": "High school",
    "College, undergraduate": "College, undergraduate",
    "Graduate, professional school": "Graduate, professional school",
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

        # index of rows with "    Total" in the Label (Grouping) column
        total_rows = df[
            df["Label (Grouping)"].str.contains("    Total", na=False)
        ].index
        # Increase the index by 1 to get the next row
        total_rows += 1
        # Filter the DataFrame to keep only the rows with "    Total" in the Label (Grouping) column
        df = df.iloc[total_rows]

        # Add and organize area information
        df.reset_index(drop=True, inplace=True)
        df["Area"] = AREAS

        if year < 2013:
            df = process_demographic_data_using_mapping(
                year, df, COLUMN_MAPPING_TILL_2012
            )
        elif year <= 2016:
            df = process_demographic_data_using_mapping(
                year, df, COLUMN_MAPPING_2013_TO_2016
            )
        else:
            df = process_demographic_data_using_mapping(
                year, df, COLUMN_MAPPING_2017_TO_2023
            )

        # Convert all the numeric columns to numeric type
        for col in df.columns[2:]:
            if df[col].dtype == "object":  # Check if the column contains strings
                df[col] = df[col].str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print(f"Processed DataFrame shape for {input_file}: {df.shape}")

        return df

    except Exception as e:
        print(f"Error processing data for {input_file}: {e}")
        raise


def process_demographic_data_using_mapping(
    year: int, df: pd.DataFrame, mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Process demographic data from ACS DP05 dataset based on mapping.
    Returns:
        pd.DataFrame: Processed demographic data
    """
    # Select and rename columns
    columns_to_keep = ["Area"] + list(mapping.keys())
    df = df[columns_to_keep]
    df = df.rename(columns=mapping)

    # Add year and reorder columns
    df["Year"] = year
    final_columns = ["Year", "Area"] + list(mapping.values())
    df = df[final_columns]
    return df


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

    # Perform interpolation for each Area separately
    for area in AREAS:
        area_df = combined_df[combined_df["Area"] == area].copy()
        area_df.set_index("Year", inplace=True)
        area_df = area_df.resample("D").asfreq().interpolate(method="linear", limit_direction="both")
        area_df = area_df.resample("W-SAT").mean(numeric_only=True)  # Resample to weekly frequency
        area_df["Area"] = area  # Ensure the 'Area' column is preserved
        area_df = area_df.reset_index()  # Reset index to make 'Year' a column
        area_df = area_df[["Year", "Area"] + [col for col in area_df.columns if col not in ["Year", "Area"]]]  # Reorder columns
        interpolated_df = pd.concat([interpolated_df, area_df])

    # Reset the index and rename the 'Year' column to 'Date'
    interpolated_df.reset_index(inplace=True)
    interpolated_df.rename(columns={"Year": "Date"}, inplace=True)

    # Round all numeric columns to whole numbers
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
