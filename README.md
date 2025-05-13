# Mortality-Predictor

A model built to predict US state level mortality rate (and its causes) based on weather and socioeconomic conditions. This project combines multiple data sources including mortality data, climate data, and census data to analyze and predict mortality rates in Illinois.

## Project Structure

```
├── datasets/
│   ├── Weekly_Provisional_Counts_of_Deaths_by_State_and_Select_Causes__2020-2023_20250216.csv
│   ├── Census/                  # Census data from 2020-2023
│   │   ├── DP05/               # Demographic data
│   │   ├── S1401/              # School enrollment data
│   │   └── S2503/              # Employment data
│   └── Climate datasets/        # Climate data from 2020-2023
│       └── Data Cleaning/       # Processed climate data
├── exploratory_analysis/        # Visualizations and analysis outputs
├── main.ipynb                   # Python notebook for statistical analysis
└── main.rmd                     # R notebook for statistical analysis
```

## Prerequisites

### For Python Components

- Python 3.8+
- Required packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

### For R Components

- R 4.0+
- Required packages:
  ```r
  install.packages(c(
    "tidyverse", "lubridate", "corrplot", "glmnet",
    "caret", "xgboost", "prophet", "future"
  ))
  ```

## Getting Started

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd Mortality-Predictor
   ```

2. Data Processing:

   - Run the Python preprocessing scripts in the Census data folders to process raw census data
   - Run the R script in `Climate datasets/Data Cleaning/` to process climate data

3. Analysis:
   - Open `main.rmd` in RStudio or VS Code with R extensions
   - Install required R packages as listed in the Prerequisites section
   - Run the R notebook to perform:
     - Data cleaning and merging
     - Feature correlation analysis
     - LASSO regression modeling
     - Prophet forecasting
     - XGBoost modeling with grid search

## Analysis Components

### Data Sources

- Mortality data from CDC (2020-2023)
- Climate data from NOAA (2020-2023)
- Census data (ACS 5-year estimates 2020-2023)
  - DP05: Demographic estimates
  - S1401: School enrollment
  - S2503: Employment status

### Models and Analysis

1. LASSO Regression

   - Features: Climate variables
   - Target: Non-COVID mortality rates
   - Cross-validation with polynomial features

2. Prophet Time Series

   - Forecasting mortality trends
   - Seasonal decomposition
   - Cross-validation

3. XGBoost
   - Combined climate and socioeconomic features
   - Grid search for hyperparameter tuning
   - Feature importance analysis

## Results

The analysis results can be found in the `exploratory_analysis` folder, including:

- Time series visualizations of mortality rates
- Correlation analysis with socioeconomic factors
- Model performance metrics and predictions