library(tidyverse)
library(dplyr)
library(lubridate)
library(readr)


# Step 1: Read all CSV files and combine them
file_list <- list.files(path = "./datasets/Climate datasets", pattern = "*.csv", full.names = TRUE)
df_list <- lapply(file_list, read_csv)
df <- bind_rows(df_list)

# Step 2: Convert date column to Date format (update column name if needed)
df$date <- as.Date(df$date, format="%Y-%m-%d")

# Step 3: Group by week and calculate the mean for numerical columns
weekly_df <- df %>%
  mutate(Week_Ending_Date = floor_date(date, "week", week_start = 7) + days(6)) %>%
  group_by(Week_Ending_Date) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = "drop")

# Step 4: Save the weekly data to a new CSV file
write_csv(weekly_df, "Climate_weekly_data.csv")

print("Weekly data saved successfully!")
