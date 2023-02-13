# Predicting-Restaurant-Closures-in-NYC

##### Link to Code:


### Abstract

The restaurant industry is essential to New York City’s social and economic fabric. From small, family-owned restaurants and food carts to four-star world-famous establishments, the city abounds with restaurants that offer cuisines from every corner of the globe. Restaurants lend vitality to each neighborhood and are integral to Manhattan’s central business districts and the city’s vibrant tourism industry, attracting millions of visitors each year. However, a lot of restaurants in NYC are vulnerable to closures within five years. This paper aims to investigate the key factors that contribute to restaurant closures in New York. We used machine learning techniques like Logistic Regression and Random Forest to make early predictions on restaurant closures. To improve our model, we performed the bias variance analysis, error analysis, dimensionality reduction and feature engineering after which we got an accuracy of 70.69% on our test data using the Logistic Regression model. The key factors that indicated increased likelihood of a restaurant closure were the latitude and longitude of the restaurant, number of inspection visits, and the type of cuisine. We concluded that the location and cuisine of the restaurant could have a huge impact on the longevity of a restaurant’s business.

### 1. Introduction
The motivation behind our project is to predict restaurant closures to ensure maintaining the continuity and stability of the restaurant industry. We believe that forecasting restaurant closures based on cuisines and neighborhoods would help upcoming restaurants to position themselves better in the market. The paper also investigates correlations between various factors that lead to restaurant closures. We explore different aspects of restaurant closures and aim to have an pplication result which would give a holistic analysis at the end of this project.

The Department of Health and Mental Hygiene (DOHMH) of NYC conducts restaurant inspections regularly and updates the dataset daily which can be downloaded from the NYC Open Data website. Based on these inspections each restaurant is assigned either a Critical, Not Critical, or Not Applicable flag. For this project, we are interested in predicting whether a restaurant is likely to receive a critical flag based on important metrics like the grade received after inspection, location, and cuisine of the restaurant.

The Covid-19 pandemic created a huge setback to the restaurant industry in NYC. People avoided eating out or visiting restaurants which made it extremely difficult for restaurant owners to keep the business going. They did not have enough resources to hire staff and maintain the high standards of cleanliness which led to increased restaurant closures. In our project we analyze our dataset in two parts, pre-covid and post-covid to understand how restaurant inspections were different and how they contributed to these closures.

## 2. Context
Machine Learning models are widely used in businesses that need to make accurate time-critical decisions which play an extremely important role in the success of the business.

In the food industry, small and medium-sized restaurants often have trouble forecasting sales or closures due to a lack of data or funds for data analysis. This is where machine learning plays an important role in making sure that all restaurants are equipped with the same kind of data to give them a fair chance to compete in a cost-effective way. Previously, machine learning models have been built to analyze different contributing factors that lead to poor performance of a restaurant. Prior work explored the reviews from online restaurant finder services like Yelp to predict restaurant closures. Another factor that has been explored is how real-time events around the restaurant affect the business on a day-to-day basis. Few researchers also built a time-series model to forecast restaurant sales based on different points of the year.

Through this project we wanted to explore how restaurant inspections could be a contributing factor to restaurant closures with a focus on New York City. To understand our project in depth, one needs to know Supervised Learning algorithms like Logistic Regression and Random Forest, and techniques like Error analysis and Feature Engineering, which we have explained further.

## 3. Methods
Since we have a labeled dataset, the method we used for this machine learning project is supervised learning. To predict restaurant closures based on cuisines and neighborhoods we want to build a classification model. We tried different algorithms such as logistic regression for its simplicity, and random forest which works well with large datasets, to see which one would give us the better performance.

#### LOGISTIC REGRESSION
Logistic regression is a statistical method for predicting binary outcomes. It is used to model the probability of a certain class or event occurring. In logistic regression, the outcome is modeled using one or more independent variables that are typically numeric, with a logistic function. The model estimates the probability that an event will occur by fitting the data to the logistic function. The result is a probability value between 0 and 1, which can be used to make predictions about future events. The sigmoid function is given by 

<img width="154" alt="image" src="https://user-images.githubusercontent.com/66789469/218352876-2b4f12fd-5d9c-411b-b5dc-1668f4248be9.png">

#### RANDOM FOREST
Random forest is an ensemble machine learning method for classification and regression. It is a type of supervised learning algorithm that can be used to build predictive models from a data set. In a random forest, a large number of decision trees are trained on a data set, and the final prediction is made by aggregating the predictions of individual trees. This allows the random forest to capture the underlying structure of the data and make more accurate predictions than any individual decision tree.

#### METRIC CHOSEN
For our project, the negative cases (accurately detecting restaurants that are not likely to close) are also important. In such situations, it is useful to account for Sensitivity and Specificity. The metric we have chosen is Balanced Accuracy.

<img width="232" alt="image" src="https://user-images.githubusercontent.com/66789469/218353078-4b7b4dc0-0a1d-4ce1-a7fc-05f4b56cb296.png">

 ### 4. Setup and Experimental Analysis
 
 #### 4.1 DATASET CLEANING
 
For data cleaning, we first removed unnecessary columns, then handled missing values and finally handled outliers.
1. Removing unnecessary columns - We focused on Cuisine and Borough and then dropped the columns that were not highly correlated with those.
2. Handling missing values - To update the missing values in the Grade column, we analyzed the existing correlation between the Action and the Grade column and updated the column accordingly. We did the similar approach between the Grade and the Score column.
3. Handling outliers - In the no. of inspections column, some restaurants had over 150 inspections while the others had less than 70 inspections. We normalized the no. of inspections column at 70.

#### 4.2 DATASET ANALYSIS [FEATURES]
In this project we analyze key features of the DOHMH Dataset like Borough, Cuisine Type and Grade received after inspection in depth.
Grade Column Key -
Grade A - Not Critical. Grade B, C - Critical. Grade P, N, Z - Pending inspection of three types.
In this dataset, the restaurant closures are predicted by the Critical and Non Critical Flags. Critical indicates that the violations are severe and non-critical indicates that the violations are not severe.

##### BOROUGH
<img width="268" alt="image" src="https://user-images.githubusercontent.com/66789469/218354625-5715f272-f35f-4fd3-9af9-5845de587219.png">
Manhattan has the highest A grade among the other boroughs. C grade is the least common.

<img width="276" alt="image" src="https://user-images.githubusercontent.com/66789469/218354657-de46b2f3-3673-4547-9638-8eda51c56d2e.png">
11-20 visits being most common, followed by 6-10 visits. Having only 1 visit is typically the least common. This is not surprising as restaurants can expect to get an inspection at least once a year and therefore even restaurants with an A grading and no violations will likely have as many inspections as years they have been in business.

##### CUISINE
<img width="273" alt="image" src="https://user-images.githubusercontent.com/66789469/218354703-659c699e-535e-4e91-b21f-5a009af1167d.png">
The top 5 most popular cuisines are American, Chinese, Coffee/Tea, Pizza and Italian

<img width="272" alt="image" src="https://user-images.githubusercontent.com/66789469/218354932-b5653b9a-3e1d-4bcb-9596-c9aead5edcc2.png">
It looks like 'Coffee/Tea', ‘Juice, Smoothies, Fruit Salads’, and 'Other' have more restaurants with fewer inspection visits, indicating they either have fewer violations or they consist of a lot of newer restaurants and therefore have not had as many visits yet. Latin American, Chinese, and Thai seem to have the greatest number of inspection visits

##### GRADE
<img width="303" alt="image" src="https://user-images.githubusercontent.com/66789469/218354979-8b106a35-2a6b-4d59-ad30-f3e4a33fd811.png">

The restaurants that are not yet graded (N) are most likely to only have 2-5 visits, likely because they are new restaurants. We also see that restaurants with a 'B' or 'C' grade are likely to have more inspection visits than restaurants with an 'A' grade, likely because these restaurants have violations (leading to their 'B' or 'C' grades) and therefore need extra inspections.

#### 4.3 DATASET SPLIT

We split the DOHMH Restaurant inspection dataset into three:
The top 5 most popular cuisines are American, Chinese, Coffee/Tea, Pizza and Italian
1. Training set: 80% of total data
2. Development set: 10% of total data
3. Testing set: 10% of total data

#### 4.4 EXPERIMENTS

##### EXPERIMENT 1: EXPLORE DIFFERENT MODELS
Since our project is classifying structured data, we chose Logistic regression as our baseline model as it is a simple stochastic classification model.
For our second model we chose the Random Forest which consists of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.

<img width="292" alt="image" src="https://user-images.githubusercontent.com/66789469/218355326-2018070a-44fd-4e12-b0fe-6fe922bb372f.png">

For the logistic regression model we did not use any regularization or penalty. For the random forest model we used the following parameters:
n_estimators=120, criterion = 'entropy',max_depth=None, max_features=10, min_samples_split=8

We observe the Random Forest model has a high variance and is overfitting the data. The development accuracy is significantly lower than the training accuracy. To better understand the kind of wrong classification errors made by our model we plot the confusion matrix

<img width="243" alt="image" src="https://user-images.githubusercontent.com/66789469/218355417-e6feba9d-5407-4c59-8061-825273667b04.png">

##### EXPERIMENT 2: USING THE CLASSIFICATION ALGORITHM FROM ABOVE TO PREDICT RESTAURANT CLOSURES IN EACH BOROUGH BASED ON TOP 20 CUISINES
<img width="287" alt="image" src="https://user-images.githubusercontent.com/66789469/218355458-480606b2-679d-466d-9385-a846bf363f04.png">
Manhattan has the highest chances of restaurant closure among the other boroughs. This could be because of the higher rent prices and competition in Manhattan.

##### EXPERIMENT 3: ANALYZE HOW THE INSPECTION RATES HAVE VARIED BASED ON THE INFLUENCE OF THE PANDEMIC
<img width="293" alt="image" src="https://user-images.githubusercontent.com/66789469/218355512-3b3d88be-a807-440f-a251-cfed57396e75.png">
On analyzing the pre pandemic [before 2020] and post pandemic data [2020 till now], we noticed that there is an increase in re-inspection of restaurants. This could be attributed to increased concern during and after the pandemic. However initial inspection rates decreased by 20%, this could be attributed to fewer restaurant openings during the pandemic.

### 5. Outcomes and Results

#### 5.1 ERROR ANALYSIS

To improve our model performance, we looked into Experiment 1 in further detail and performed an error analysis on it. We analyzed the wrong predictions made by our model and tried to find a pattern. To do this we manually looked into the top 30 important features based on which our model is making these predictions. Through this we realized that our model is overly reliant on particular Inspection scores like 2, 4, 0, 3, 12, etc. rather than a range of scores. Most of the restaurants that were wrongly classified as Critical were based purely on these scores. However, inspection scores are just one of the many factors that determine a restaurant’s likelihood to close. The inspection scores are also highly correlated with the grade received by the restaurant after inspection, which is why we decided to drop the Score column

<img width="306" alt="image" src="https://user-images.githubusercontent.com/66789469/218355644-5bb8ad94-20a4-4a52-84ae-a52d5ef63623.png">

We also dropped the ‘#_of_inspections’ column since it is highly correlated with the ‘count_range’ column. It is better to have a range of values for the number of inspections to prevent overfitting.

<img width="267" alt="image" src="https://user-images.githubusercontent.com/66789469/218355684-845156c2-dc2b-47e4-af99-478d3e9da6a6.png">

#### 5.2 RESULTS
After our feature engineering, we test the accuracy of our models again:

<img width="292" alt="image" src="https://user-images.githubusercontent.com/66789469/218355779-f4b5d03d-4e4b-4e0a-8a42-827db4109d3e.png">

Since the Logistic Regression model produced a higher development accuracy, we used this model to perform predictions on test dataset, which gave us an accuracy:

<img width="289" alt="image" src="https://user-images.githubusercontent.com/66789469/218355832-33077f59-4633-4c64-9841-caba8ec82bcd.png">

#### 5.3 CONCLUSION
Through this project, we wanted to explore how restaurant inspections impact the longevity of a restaurant. We initially hypothesized that the restaurant location and cuisine would be a key factor in the success of the restaurant. From our model’s predictions we were able to extract the most important feature - which was the latitude and longitude of the restaurant. We were also able to identify that restaurants located in Manhattan were more likely to close. Finally, using the model developed in this project, we can predict restaurant closures in NYC with an accuracy of 70.69% Our key takeaway is that restaurant closures not only depend on the location or cuisine but also on their total number of inspection visits.

### 6. References
1. https://data.cityofnewyork.us/Health/DOHMH -New-York-City-Restaurant-Inspection-Results /43nn-pn8j
2. DOHMH New York City Restaurant Inspection Results | Kaggle
3. Using Yelp Data to Predict Restaurant Closure | by Michail Alifierakis
4. NYC-Restaurant-Yelp-and-Inspection-Analysi s [rspiro9]
5. Python | Plotting Google Map using gmplot package - GeeksforGeeks
6. Decision Tree Tutorials & Notes | Machine Learning | HackerEarth
7. Choosing the Best Algorithm for your Classification Model. | by Rahil Shaikh | DataDrivenInvestor

