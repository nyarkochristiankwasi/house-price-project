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
- 07_sql_setup.ipynb: load the data into a SQLite database and run first SQL queries
- 08_statistics_basics.ipynb: mean, median, standard deviation, and the shape of the price distribution
- 09_sql_queries.ipynb: SQL queries on the housing data using WHERE, ORDER BY, IN, LIKE, and DISTINCT
- 10_sql_groupby.ipynb: GROUP BY, HAVING, and JOIN on the housing data, plus a cap rate example
- 11_sql_vs_pandas.ipynb: checks that SQL and pandas agree, plus five SQL practice problems
- 12_project2_sql_analysis.ipynb: Project 2, five business questions about Ames house prices answered with SQL
- 13_train_test_split.ipynb: splitting the data into training and test sets before modeling
- 14_first_model.ipynb: first linear regression models, from one feature to five, with errors on the test set

## What I found so far
- The training data has 1,460 houses and 81 columns.
- About 17% of houses (242 of 1,460) have 4 or more bedrooms.
- Missing values: PoolQC has the most (1,453 of 1,460 rows). This does not mean the data is broken. In this dataset a missing value usually means the house has no such feature, as explained in data_description.txt. The same is true for Alley, Fence, and the garage columns, so I filled them with "None". LotFrontage was filled with the median, and GarageYrBlt with 0 as a placeholder for no garage.
- Price depends on quality, location, and size. Quality has the strongest link to price (correlation 0.79), followed by location (0.74) and size (0.71). The factors are close together, and the location figure is slightly generous because each house is part of its neighborhood's average.
- Location: NoRidge has the highest average price (about 335,000) and MeadowV the lowest (about 99,000), a gap of about 3.4 times. NridgHt and StoneBr charge about 158 per square foot, against a typical 120.
- Size alone does not set the price. The two largest houses (5,642 and 4,676 sq ft, both in Edwards) sold for only 160,000 and 184,750, while large houses in NoRidge sold for 625,000 to 755,000.
- Not in this dataset, but important to buyers: land ownership type and property documentation. A future model could include these.
- SQL check: I loaded the data into SQLite, and a query for houses over 400,000 with 4 or more bedrooms returned the same 14 houses as pandas. Seven of the 14 are in NoRidge and three are in NridgHt.
- Prices are right-skewed. The mean (about 180,921) is above the median (163,000) because a long tail of expensive houses pulls it up. The standard deviation is about 79,443, and about 78% of houses fall within one standard deviation of the mean.
- SQL check: 46 houses were built before 1950 with a quality score of 7 or higher, and 26 of them sold above the overall median (163,000). Most are in OldTown (20) and Crawfor (15), and Crawfor's typical price (about 230,000) is well above OldTown's (about 158,000), so neighborhood matters alongside quality.
- SQL summaries: average price rises at every step of overall quality (about 50,000 at quality 1 to 438,588 at quality 10), but not at every step of bedroom count. Three-bedroom houses are the most common (804 of 1,460).
- JOIN check: 636 of the 1,460 houses (about 44%) sold above their own neighborhood's average. The 1892 OldTown house with quality 10 sold 346,775 above its neighborhood average, so quality can outweigh location.
- SQL and pandas gave identical answers on three cross-checks. Garage size and central air both go with higher prices (3-car garages average about 309,600, about 3 times the no-garage average of 103,300), though both partly reflect house size, age, and quality. StoneBr and NridgHt charge the most per square foot (about 165).
- Project 2 (SQL analysis): price depends on quality, location, and size together. Average price rises at every step of quality (about 50,150 to 438,588), NridgHt averages about 2.5 times BrkSide among the larger neighborhoods, and StoneBr and NridgHt charge the most per square foot (about 165). 636 of 1,460 houses sold above their neighborhood average.
- Train/test split: I split the 1,460 houses 80/20 (1,168 for training, 292 for testing) with a fixed random seed. The average prices of the two sets (181,442 and 178,840) are close to the overall mean, so the test set is a fair sample.
- First models: guessing the average price for every house is off by about 62,600. A linear regression on living area alone cuts that to about 38,300 (R2 0.55), adding overall quality brings it to about 28,600 (R2 0.75), and five features reach about 25,400 (R2 0.79), nearly 60% lower than the baseline. Garage size and year built added little because they overlap with size and quality.


## Progress
- [x] First look at the data
- [x] Cleaning and charts
- [x] SQL analysis
- [ ] Models
- [ ] Live app