# Project-final
### REasearch Question 
Which regression performs best to forecast Switzerland’s National Electricity Consumption, Linear regression, or  Machine Learning regression, Random Forest, Lasso or  XGBoost ? 

## Create Setup
# Create envrironement
 
## Usage
python main.py

Expected output: Accuracy comparison between three models.
## Project Structure

my-iris-comparison/
├── main.py              # Main entry point
├── src/                 # Source code
│   ├── data_loader.py   # Data loading/preprocessing
│   ├── models.py        # Model training
│   └── evaluation.py    # Evaluation metrics
├── results/             # Output plots and metrics
└── environment.yml      # Dependencies

## Results
- Random Forest: 0.967 accuracy
- KNN: 0.933 accuracy
- Logistic Regression: 0.967 accuracy
- Winner: Tie between Random Forest and Logistic Regression

## Requirements
- Python 3.11
- scikit-learn, pandas, matplotlib, seaborn