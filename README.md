# zillow_clustering_project

## About the Project
with the use of clustering I am trying to create a model that can find and predict drivers of LOG ERROR

### Data Dictionary:
   | __Most Valuable Features__ |
   | Column/Feature | Description |
    |--- | --- |
    | __bathroomcnt__ | The number of bathrooms in the home. |\n
    | __bedroomcnt__ | The number of bedrooms in the home. |\n
    | __tax_value__ | The tax-assessed value of the home. <br> __Not__ the home's ultimate sale price. |\n
    | __sq_ft__ | The home's square footage. |\n
    | __year_built__ | The year the home was built. |\n
    | __lot_size__ | The square footage of the lot on which <br> the home is built. |__latitude__ | A measurement of distance north or south of the Equator.
    |__longitude__ | the measurement east or west of the prime meridian.
### Goals

To use clustering to create a viable model to better catch andd predict LOG ERROR


### Initial Questions

1. What columns have the ;argest correlation with LOG ERROR

2. Can we get viable clusters frrom bed and bathroomss

3. Does location matter?

4. How important is square footage?

### Planning

#### To prepare this data i used the following steps

1. Acquire the data using built functions

2. Clean and split data using built functions

3. Use matplotlib, seaborn, and dtale for exploratory data analysis

4. Find possible relational predictors

5. Use features with best correlation to create clusters of features

6. Create baseline 

7. Start modeling using selected features, with OLS, lasso lars, and polynomial models

8. Choose the best model and test (logistic regression)

### How to Reproduce

to reproduce this project you will need to: 

- have a copy of Zillow.csv

- clone this repository

- use the functions in .py files to acquire and clean data

- used libraries are numpy, pandas, matplotlib, seaborn, and sklearn


# Conclusion:

## Takeaway:

- while we were able to find some correlations to the target variable, over all they were not very large.

- Bedrroms, Bathrooms, Square Footage, Lat & long, and Tax Valuation seemed to be the larrgest driver of this particular dataset.

- 13% increase is a good start, and with more time I'm certain we could increase that number.

## Recomendations:

- As the model is currently better than the baseline I wouldd recommend implementation until a better model can be realized.

## Next Steps?

- more feature engineering.

- Find a method to handle outliers.

- _And/Or_ convert MinMaxscaler to a robust scaling model.
