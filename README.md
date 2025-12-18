# Project-final
### Reasearch Question 
Which regression performs best to forecast Switzerland’s National Electricity Consumption, Linear regression, or  Machine Learning regression Random Forest, Lasso or  XGBoost ? 

## Create Setup and envrironement
# Steps to Run the Project

In a new directory of your choice, follow these steps:

### 1. Clone the project from GitHub
Open a terminal and run:

```bash
git clone https://github.com/mel799/Project-final.git
cd Project-final
```

### 2. Create a new Conda environment
```bash
conda env create -f environment.yml -n electricity-project
```

### 3. Activate the environement
```bash
conda activate electricity-project
```

## Usage

### 4. Run the project
```bash
python main.py
```

All required datasets are already included in the repository in the following directory:data/raw/ 
No additional data download is required

Expected output: 
- Loaded data 
    - 2 folders are created in the data folder, "clean" and "dataset"
- EDA analysis 
- Trained regression models
- Performance comparaison using regression metrics (RMSE, MAE, R²)
    - Evaluation results of the best regression model are stored in results/models/model_scores.txt
- Best Model visualization 
    - stored in results/figures

## Project Structure

```
Project-final/
├── main.py                 # Main entry point
├── src/
│   ├── data_loader         # Load data
│       └──GDP_monthly.py   # transform data to monthly
│       └──load_gdp.py       # loand and clean the GDP data
│       └── load_population.py
│       └── ...
│   ├── data_preprocessing  
│       └── add features    # add features to the dataset
│       └── merge_data      # creates the dataset file
│       └── temp_heat_need  # adds the head feature
│   └── evaluation
│       └──eda_analysis         #analysis the dataset and patterns
│   └── model_visualisation.py  # creates graphics visualization
│   └── models_evaluation.py    # evalutates all the models with metrics
│   └── models_regression.py    # Regression models
├── test
│   └── random_foreset_tuning.py    #tune the random forest regression
├── data/
│   └── raw/
│       └── climate_Swiss.txt       # Swiss meteo data
│       └── electricity.csv         # Switzerland electricity consumption data
│       └── GDP_quarterly.cvs       # Swiss GDP quarterly
│       └── population_data_yearly  # population per year
└── environment.yml         # conda environment to download
└── PROPOSAL.md            # project proposal submitted in November
├── Project_report.tex      # Final report (LaTeX)
├── project_report.pdf     # Final report (pdf)
└── README.md               # Project documentation
```

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

## Requirements
- Python 3.11
- scikit-learn, pandas, matplotlib, seaborn, numpy, jupyter, xgboost, pyyaml
