# House Price Prediction

Predicting house sale prices in Ames, Iowa using Python and machine learning.

## Data
From the Kaggle competition "House Prices: Advanced Regression Techniques":
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Download train.csv and put it in a folder called data. Data files are not stored in this repository.

## Notebooks
Listed in the order I built them.
- 01_first_look.ipynb: load the data, check its shape and missing values
- 02_python_basics.ipynb: Python practice with variables, lists, dictionaries, and loops
- 03_functions_and_data.ipynb: functions and the first pandas load
- 04_select_filter_sort.ipynb: selecting, filtering, and sorting the houses
- 05_clean_and_group.ipynb: filling missing values and comparing neighborhoods
- 06_charts.ipynb: charts, plus the comparison of quality, location, and size

## What I found so far
- The training data has 1,460 houses and 81 columns.
- About 17% of houses (242 of 1,460) have 4 or more bedrooms.
- Missing values: PoolQC has the most (1,453 of 1,460 rows). This does not mean the data is broken. In this dataset a missing value usually means the house has no such feature, as explained in data_description.txt. The same is true for Alley, Fence, and the garage columns, so I filled them with "None". LotFrontage was filled with the median, and GarageYrBlt with 0 as a placeholder for no garage.
- Price depends on quality, location, and size. Quality has the strongest link to price (correlation 0.79), followed by location (0.74) and size (0.71). The factors are close together, and the location figure is slightly generous because each house is part of its neighborhood's average.
- Location: NoRidge has the highest average price (about 335,000) and MeadowV the lowest (about 99,000), a gap of about 3.4 times. NridgHt and StoneBr charge about 158 per square foot, against a typical 120.
- Size alone does not set the price. The two largest houses (5,642 and 4,676 sq ft, both in Edwards) sold for only 160,000 and 184,750, while large houses in NoRidge sold for 625,000 to 755,000.
- Not in this dataset, but important to buyers: land ownership type and property documentation. A future model could include these.

## Progress
- [x] First look at the data
- [x] Cleaning and charts
- [ ] SQL analysis
- [ ] Models
- [ ] Live app