
# Project proposal

### Title: Forecasting Switzerland’s National Electricity Consumption
### Category: Data Science / Predictive Modeling

## Reasearch Question: Which regression model performs best to forecast Switzerland’s National Electricity Consumption, Linear regression, or Machine Learning regression; Random Forest, Lasso or XGBoost?

### Motivation:

Electricity is an essential resource for our society, but it remains limited and demand continues
to grow due to population increase and economic development. In the context of global warming,
it is becoming crucial to better understand and anticipate energy needs in order to promote the
transition to renewable sources and ensure the stability of the electricity grid. This project aims to
develop a predictive model capable of estimating electricity consumption in Switzerland using
variables such as weather, population, and seasonal effects. By focusing on a national-scale
forecast, these data provide actionable information for policymakers, electricity grid operators,
and researchers to optimize energy efficiency, anticipate consumption peaks, and facilitate the
integration of renewable energies.

### Planned approach and technologies:

Collect Data: National electricity consumption, Weather data (temperature, humidity,
precipitation), Population and socio-economic data from the Swiss Federal Statistical
Office (total population, GDP). Observe patterns, trends, variance and covariance in the
variables depending on this, creates adapted features.
Build and test Models: Start with linear regression. Try more advanced models. Measure
how good the predictions are using metrics like RMSE and R². Compare the predictions
with the actual consumption.
Baseline Models
Naïve model and Linear Regression: to understand the impact of individual variables on
electricity consumption.
Machine Learning Models:
Try the following ML models: Random Forest Regressor, XGBoost, Lasso, to capture
nonlinear relationships and interactions between features.
Evaluate the Regressions with RAE; RSME; R^
Optimize and Visualize: Make code run faster using tools like Numba. Illustrate results
with graphics

### Expected challenges and how you’ll address them:

Data completeness, some historical consumption or weather data may have gaps.

### Solution: 

Identify missing data, handle missing values, check the impact.

### Success criteria (how will you know it’s working?)

Data is available, time-series predictability,Electricity consumption is usually somewhat regular (daily, weekly,
seasonal cycles).

###Stretch goals (if time permits)

Compare electricity consumption with solar and hydro production.