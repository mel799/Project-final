# Project-final
### Reasearch Question 
Which regression performs best to forecast Switzerland’s National Electricity Consumption, Linear regression, or  Machine Learning regression Random Forest, Lasso or  XGBoost ? 

## Create Setup
# Create envrironement
conda env create -f environment.yml
conda activate electricity-forecast


## Usage
python main.py

Expected output: 
- Loaded data 
    - 2 folders are created in the data folder, "clean" and "dataset"
- Data analysis 
- Trained regression models
    - the models are saved in the folder "results/models
- Performance comparaison using regression metrics (RMSE, MAE, R²)
    - Evaluation results of the best regression model are stored in results/models/model_scores.txt
- Best Model visualization 
    - stored in results/figures

## Project Structure

pproject-final/
├── main.py                 # Main entry point
├── src/
│   ├── data_loader    # Load & preprocess electricity data
│       └──GDP_monthly.py   # transform data to monthly
│       └──load_gdp         # loan and clean the GDP data
│       └── ...
│   ├── data_preprocessing
│       └── add features    # add features to the dataset
│       └── merge_data      # creates the dataset file
│       └── temp_heat_need  # adds the head feature
│   ├── models.py           # Train regression models
│   └── evaluation
│       └──eda_analysis     #analysis the dataset and patterns
│   └── model_visualisation.py
│   └── models_evaluation.py
│   └── models_regression.py
├── test
│   └── random_foreset_tuning.py 
├── data/
│   └── raw/
│       └── climate_Swiss.txt       # Swiss meteo data
│       └── electricity.csv         # Switzerland electricity consumption data
│       └── GDP_quarterly.cvs       # Swiss GDP quarterly
│       └── population_data_yearly  # population per year
├── results/
│   ├── eda                 # eda results and visualization
│   └── figures             # Best model figure
│   └── models              # Models predictions and scores
└── environment.yml         # conda environment

## Models Compared
- Linear Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

## Results
- Best model: XGBoost
RMSE: 107.10
MAE : 63.80
R2  : 0.9558
- Winner: XGBoost best results

## Requirements
- Python 3.11
- scikit-learn, pandas, matplotlib, seaborn, numpy