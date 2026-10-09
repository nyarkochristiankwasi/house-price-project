# House Price Prediction

Predicting house sale prices in Ames, Iowa using Python and machine learning.

## Data
From the Kaggle competition "House Prices: Advanced Regression Techniques":
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Download train.csv and put it in a folder called data. Data files are not stored in this repository.

## What I found so far
- The training data has 1,460 houses and 81 columns.
- PoolQC has the most missing values: 1,453 of 1,460 rows. This does not mean the data is broken. In this dataset a missing value means the house has no pool, as explained in data_description.txt. The same is true for columns like Alley and Fence.
- About 17% of houses (242 of 1460) have 4 or more bedrooms.
- Size alone does not set the price. The two largest houses (5,642 and 4,676 sq ft, both in Edwards) sold for about the average price, while large houses in NoRidge sold for 625,000 to 755,000. Location may matter more than size, which I will test next.
- Not in this dataset, but important to buyers: land ownership type and property documentation. A future model could include these.

## Progress
- [x] First look at the data
- [ ] Cleaning and charts
- [ ] SQL analysis
- [ ] Models
- [ ] Live app