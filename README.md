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
- Trained regression models
- Performance comparaison using regression metrics (RMSE, MAE, R²)
- Evaluation results of the best regression model
- Best Model visualization

## Project Structure

project-final/
├── main.py                 # Main entry point
├── src/
│   ├── data_loader.py      # Load & preprocess electricity data
│   ├── models.py           # Train regression models
│   └── evaluation.py       # Regression metrics & comparison
├── data/
│   └── raw
        └──electricity.csv     # Switzerland electricity consumption data
├── results/
│   ├── metrics.csv         # Model performance metrics
│   └── plots/              # Forecast & comparison plots
├── requirements.txt        # pip dependencies
└── environment.yml         # conda environment


## Results
- Random Forest: 0.967 accuracy
- KNN: 0.933 accuracy
- Logistic Regression: 0.967 accuracy
- Winner: Tie between Random Forest and Logistic Regression

## Requirements
- Python 3.11
- scikit-learn, pandas, matplotlib, seaborn