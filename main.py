#Data Loading
from src.data_loader.load_temp import load_temperature
from src.data_loader.load_pop import load_population
from src.data_loader.load_gdp import load_gdp
from src.data_loader.load_elec_consumption import load_consumption
from src.data_loader.pop_yearly_monthly import make_population_monthly
from src.data_loader.GDP_monthly import make_gdp_monthly

#data Preprocessing
from src.data_preprocessing.temp_heat_need import add_heat_need
from src.data_preprocessing.add_features import add_features
from src.data_preprocessing.merge_data import merge_all
# file name in import should match the actual file name 
# second term is the name of the function inside that file

# EDA Analysis
from src.evaluation.eda_analysis import run_eda

# Models
from src.models_regression import (
    naive_last_month, naive_last_year,
    train_linear_regression, predict_linear_regression,
    train_lasso, predict_lasso,
    train_lasso_gridsearch,
    train_random_forest, predict_random_forest,
    train_xgboost, predict_xgboost,
    train_xgboost_gridsearch
)
#Models Evaluation
from src.models_evaluation import (
    evaluate_baseline, evaluate_linear_regression,
    evaluate_random_forest, evaluate_xgboost, evaluate_lasso,
    save_scores
)


def main():
    #------------------------------------------------------------------------------
    #Load Data and Preprocess
    #------------------------------------------------------------------------------
    print("=== Loading temperature data ===")
    temp_df = load_temperature()
    print("=== Adding heat need ===")
    temp_df = add_heat_need(temp_df)

    print("=== Loading population data ===")
    pop_df = load_population()
    print("=== Converting population to monthly ===")
    pop_monthly = make_population_monthly()

    print("=== Loading GDP data ===")
    gdp_df = load_gdp()
    print("=== Converting GDP to monthly ===")
    gdp_monthly = make_gdp_monthly()

    print("=== Loading electricity consumption data ===")
    cons_df = load_consumption()

    print("=== Merging all datasets ===")
    merged_df = merge_all(temp_df, pop_monthly, gdp_monthly, cons_df)

    print("=== Adding features ===")
    final_df = add_features(merged_df)

    #Exploratory Data Analysis
    print("=== Running EDA ===")
    run_eda(final_df, do_clustering=True, verbose=False)
    
    #-----------------------------------------------------------------------------
    # Train and test split, Modeling, Evaluation
    #-----------------------------------------------------------------------------

    # Split train / test
    split_date = "2019-01-01"
    train = final_df[final_df["date"] < split_date].copy()
    test = final_df[final_df["date"] >= split_date].copy()

    # Storage for all model results
    scores = {}

    # --- BASELINE MODELS -----------------------------------------------------
    print("=== Running Baseline Regression Models ===")
    test = naive_last_month(test)
    test = naive_last_year(test)

    scores["Naive Last Month"] = evaluate_baseline(test, "naive_last_month")
    scores["Naive Last Year"] = evaluate_baseline(test, "naive_last_year")

    # Save baseline predictions
    output_path = "results/models/baseline_predictions.csv"
    cols_to_save = [
        "date", "electricity_consumption",
        "naive_last_month", "naive_last_year"
    ]
    test.dropna(subset=cols_to_save).to_csv(output_path, index=False)
    print(f"Baseline predictions saved to: {output_path}")

    # --- LINEAR REGRESSION ---------------------------------------------------
    print("\n=== Training Linear Regression ===")

    feature_cols = [
        "temp", "heat_need", "gdp_real", "population",
        "month", "year",
        "consumption_lag_1", "consumption_lag_12",
        "temp_lag_1", "gdp_lag_1"
    ]

    # Train + Predict
    lr_model = train_linear_regression(train, feature_cols)
    test = predict_linear_regression(lr_model, test, feature_cols)

    # Evaluate
    scores["Linear Regression"] = evaluate_linear_regression(test)

    # Save Linear regression predictions
    print("\n=== Saving Linear Regression Predictions ===")
    output_path = "results/models/linear_regression_predictions.csv"
    cols_to_save = ["date", "electricity_consumption", "prediction_lr"]
    test.dropna(subset=cols_to_save).to_csv(output_path, index=False)
    print(f"Linear Regression predictions saved to: {output_path}")

    # --- RANDOM FOREST REGRESSION ----------------------------------------
    print("\n=== Training Random Forest Regression ===")
    rf_model = train_random_forest(train, feature_cols)
    test = predict_random_forest(rf_model, test, feature_cols)
    
    scores["Random Forest"] = evaluate_random_forest(test)
    # Save RF predictions
    output_path = "results/models/random_forest_predictions.csv"
    cols_to_save = ["date", "electricity_consumption", "prediction_rf"]
    test.dropna(subset=cols_to_save).to_csv(output_path, index=False)
    print(f"Random Forest predictions saved to: {output_path}")
    
    # --- XGBOOST WITH GRID SEARCH ----------------------------------------------

    print("\n=== Tuning XGBoost with GridSearchCV ===")
    
    xgb_model, grid = train_xgboost_gridsearch(train, feature_cols)
    test = predict_xgboost(xgb_model, test, feature_cols)
    scores["XGBoost (Tuned)"] = evaluate_xgboost(test)
    # Save tuned XGB prediction
    output_path = "results/models/xgboost_tuned_predictions.csv"
    cols_to_save = ["date", "electricity_consumption", "prediction_xgb"]
    test.dropna(subset=cols_to_save).to_csv(output_path, index=False)
    
    print(f"Tuned XGBoost predictions saved to: {output_path}")

    # --- LASSO REGRESSION ---------------------------------------------------------
    print("\n=== Training LASSO Regression ===")
    lasso_model = train_lasso(train, feature_cols, alpha=0.1)
    test = predict_lasso(lasso_model, test, feature_cols)
    scores["LASSO"] = evaluate_lasso(test)
    
    # Save predictions
    output_path = "results/models/lasso_predictions.csv"
    cols_to_save = ["date", "electricity_consumption", "prediction_lasso"]
    test.dropna(subset=cols_to_save).to_csv(output_path, index=False)
    
    print(f"LASSO predictions saved to: {output_path}")

    # --- LASSO REGRESSION (GRID SEARCH) ------------------------------------------
    print("\n=== Tuning LASSO Regression (GridSearchCV) ===")
    lasso_tuned_model, lasso_grid = train_lasso_gridsearch(train, feature_cols)
    test = predict_lasso(lasso_tuned_model, test, feature_cols)
    
    scores["LASSO (Tuned)"] = evaluate_lasso(test)
    
    # Save tuned predictions
    output_path = "results/models/lasso_tuned_predictions.csv"
    cols_to_save = ["date", "electricity_consumption", "prediction_lasso"]
    test.dropna(subset=cols_to_save).to_csv(output_path, index=False)
    
    print(f"Tuned LASSO predictions saved to: {output_path}")
    
    # --- Save all models scores ---------------------------------------------
    print("\n=== Saving Model Score Summary ===")
    score_file_path = "results/models/model_scores.txt"
    save_scores(scores, score_file_path)
    print("\n=== Done ===")

    #Conclusion : winner = may(resutls from evaluation files)
    print("\n=== Final Model Selection ===")

    # Choose best model based on lowest RMSE
    best_model_name = min(scores, key=lambda k: scores[k]["rmse"])
    best_model_metrics = scores[best_model_name]

    print(f"Best model: {best_model_name}")
    print(f"RMSE: {best_model_metrics['rmse']:.2f}")
    print(f"MAE : {best_model_metrics['mae']:.2f}")
    print(f"R2  : {best_model_metrics['r2']:.4f}")



if __name__ == "__main__":
    main()