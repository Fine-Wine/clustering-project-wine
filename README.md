# clustering-project-wine
 
# Project Description
 
The quality of wine from https://data.world/food/wine-quality will be investigated using a variety of target variables to predict what affects the quality of wine using clustering and classification techniques, and make predictions on wine quality. We have decided to look into five different areas that may affect wine quality.
 
# Project Goal
 
* Construct an ML Regression model that predicts wine quality using features of white and red wines.
* Find the key drivers of wine quality. 
* Deliver a report that explains what steps were taken, why and what the outcome was.
* Make recommendations on what works or doesn't work in predicting wine quality.

 
# Initial Thoughts
 
Our initial hypothesis is that the alcohol content of wine is one the biggest drivers of wine quality.
 
# The Plan
 
* Aquire data from data.world/food/wine-quality database
 
* Prepare data
   * Cleaned up data
       * Dropped unnecessary columns  
       * Placed underscores back in columns that were missing
       * Dropped duplicate columns
       * Created a scaled dataframe for modeling purposes with identified features:
            * created dummies
            * dropped columns
            * scaled the data
       * split the data  
 
* Explore data in search of drivers of wine quality
   * Answer the following initial questions
       * Does the quantity of alcohol effect the quality of the wine?
       * Does the density of the wine effect the quality of the wine?
       * Does the amount of sugar effect the quality of the wine?
       * Does the acidity of the wine effect the quality of the wine?
      
* Develop a Model to predict the quality of wine
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest validate and difference accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|'red_or_white'|	 Specifies if the wine is red or white 0 = white, 1 = red|
|'fixed_acidity'|	Set of low volatility organic acids such as malic, lactic, tartaric or citric acids|
|'volatile_acidity'| set of short chain organic acids that can be extracted from the sample by means of a distillation process|
|'citric_acid'|	Increase acidity, complement a specific flavor or prevent ferric hazes|
|'residual sugar'|	consists mostly of the grape sugars that are left over after the fermentation process but can also be created by small amounts added before bottling| 
|'chlorides'|	Adds to the saltiness of a wine, which can contribute to or detract from the overall taste and quality of the wine|
|'free_sulfur'|	Free sulfites are those available to react and thus exhibit both germicidal and antioxidant properties| 
|'total_sulfur'|	The portion of SO2 that is free in the wine plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugars|
|'density'|	Mass of a unit volume of a material substance, the thickness|
|'pH'|	Way to measure ripeness in relation to acidity|
|'sulphates'|	Antimicrobial agents produced as a byproduct of yeast metabolism during fermentation|
|'alcohol'|	Percent of alcohol in the wine by volume| 
|'quality'|	A score given to the wine from 0-10 |
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from https://data.world/food/wine-quality
3) Read the data into a .csv and merge/append the data
4) Run notebook.
 
# Takeaways and Conclusions
* Bathrooms, bedrooms, and square footage are the top three features with the strongest correlation.
    * Square footage has the strongest correlation of all the features.
* There is a moderate corellation between square footage and property value. 
    * Further investigation into size of the property and its value is necessary in determing the optimal square footage.
* There is a slight correlation between the number of bedrooms and property value. It is difficult to determine the maximum number of bedrooms based on the chart.
* The optimal number of bathrooms to optimize property value appears to be 3.5. However more investigation is required to confirm this finding. Further analysis on optimizing the cost of building a property, location, and comparing that the total value of the propery is neccessary.
* The optimal square footage is between 1001-2000 square feet. However, there is a moderate negative correlation and the large number of homes within this range could be causing this correlation.
 
# Recommendations
* The square footage of a house appears to be optimal between 1000-2000 square feet.
* The number of bedrooms does not apeear to be a significant factor in determining the property value. However, targeted analysis on specific bedroom counts could produce different conclusions.
* The number of optimal bathrooms appears to be 3.5, although more analysis into the increase in value is necessarry to confirm this conclusion.
