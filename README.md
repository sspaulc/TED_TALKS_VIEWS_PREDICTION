# TED_TALKS_VIEWS_PREDICTION
We have to predict the number of views a ted talk gets from the given number of features by using regression analysis.

1. Problem statement-- 
TED is devoted to spreading powerful ideas on just about any topic. These datasets contain over 4,000 TED talks including transcripts in many languages. Founded in 1984 by Richard Salman as a nonprofit organization that aimed at bringing experts from the fields of Technology, Entertainment, and Design together, TED Conferences have gone on to become the Mecca of ideas from virtually all walks of life. As of 2015, TED and its sister TEDx chapters have published more than 2000 talks for free consumption by the masses and its speaker list boasts of the likes of Al Gore, Jimmy Wales, Shahrukh Khan, and Bill Gates. The main objective is to build a predictive model, which could help in predicting the views of the videos uploaded on the TEDx website.

2. Columns--
1. Title
2. First Speaker
3. All the Speakers
4. Occupations
5. About Speakers
6. Views
7. Recorded date
8. Published date
9. Event
10. Native language
11. Available language
12. Comments
13. Duration
14. Topics
15. Related talks
16. Url 
17. Description
18. Transcript

3. Steps involved:
 
Data Collection
To proceed with the problem dealing first we will load our dataset that is given to us in .csv file into a dataframe.Mount the drive and load the csv file into a dataframe. 

Exploratory Data Analysis
After loading the dataset we looked for duplicate values in the ‘talk_id’ 
column. There were none. So We performed EDA by comparing our target variable that is Views with other independent variables. This process helped us figuring out various aspects and relationships among the target and the independent variables. It gave us a better idea of which feature behaves in which manner compared to the target variable.
Numerical Variables:
Talk_id
Views
Comments
duration
Textual Variables:
Title
Speaker_1
Recorded_date
Published_date
Event
Native_lang
Url
Description
Dictionaries:
Speakers
Occupations
About_speakers
Related_talks
List:
topics

Out of all the continuous variables, ‘views’ is the target variable.
The target variable ‘views’ was a skewed variable.
 The other continuous variables have distributions as:
Comments

Duration

All of the data had very skewed continuous variable distributions.

Null values Treatment
Our dataset contains around 400 null values which might tend to disturb our mean absolute score hence we have performed KNN nan value imputer for numerical features and replaced categorical features nan 

values with the value ‘Other’. We chose to impute nan values and not drop them due to the size of the data set

Encoding of categorical columns 
We used Target Encoding for replacing the values of categorical variables with the mean of the views. This was done to not increase the dimensions to the data set while also keeping the relationship of variables with views into consideration.

Feature Selection
For Feature Selection we have done the following: we have introduced new numerical features from the categorical features,combined features and also we have used f_regression in which we have taken the features with the maximum f-scores.

Outlier Treatment
We have done outlier treatment on variables like duration and occupation. This was done by replacing outliers with the extreme values at the first and third quartiles. We have done outlier treatment to prevent high errors that were influenced by outliers.

Fitting different models
For modelling we tried various regression algorithms like:
XGBoost Regressor
Extra Trees Regressor

Random Forest Regressor
 
Tuning the hyperparameters for better accuracy
Tuning the hyperparameters of respective algorithms is necessary for less error values,regularization  and to avoid overfitting in case of tree based models.


4.1. Algorithms:
We have used only non-parametric models for prediction because two of the hypotheses such as linearity between output and input variables and errors normally distributed were not met.

XGBoost Regression:
Gradient boosting refers to a class of ensemble machine learning algorithms that can be used for classification or regression predictive modeling problems.Extreme Gradient Boosting, or XGBoost for short, is an efficient open-source implementation of the gradient boosting algorithm. It is computationally effectively faster with better model performance.
XGboost can be optimized by fixing the number of trees, fixing learning rate,tuning gamma, tuning regularization and various hyper parameter tuning.
When we implemented this model we got the following scores :

MAE train: 164091.332037
MAE test: 226944.860549
R2_Score train: 0.918158
R2_Score test: 0.830151
RMSE_Score train: 315411.385197
RMSE_Score test: 454270.753145

.
Extra Trees Regressor:
Extremely Randomized Trees, or Extra Trees for short, is an ensemble machine learning algorithm.
Specifically, it is an ensemble of decision trees and is related to other ensembles of decision trees algorithms such as bootstrap aggregation (bagging) and random forest.
The Extra Trees algorithm works by creating a large number of unpruned decision trees from the training dataset. Predictions are made by averaging the prediction of the decision trees.
The random selection of split points makes the decision trees in the ensemble less correlated, although this increases the variance of the algorithm. This increase in variance can be countered by increasing the number of trees used in the ensemble.
We use the criterion as ‘MAE’ as it uses L1 regularization to select the median and selects the best features for reducing the mean absolute error. 
MAE is used as it is not influenced by outliers.
Scores obtained were:

MAE_train : 207304.048833
MAE_test: 204793.751052
R2_Score_train: 0.796536
R2_Score_test:0.806390
RMSE_Score_train:497317.341381
RMSE_Score_test:485005.01521

Random Forest Regressor:
Every decision tree has high variance, but when we combine all of them together in parallel then the resultant variance is low as each decision tree gets perfectly trained on that particular sample data and hence the output doesn’t depend on one decision tree but multiple decision trees. In the case of a classification problem, the final output is taken by using the majority voting classifier. In the case of a regression problem, the final output is the mean of all the outputs.

A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap and Aggregation, commonly known as bagging. The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees.
Random Forest has multiple decision trees as base learning models. We randomly perform row sampling and feature sampling from the dataset forming sample datasets for every model. This part is called Bootstrap.

MAE train: 186583.315347 
MAE test: 191844.536467
R2_Score train:0.806193
R2_Score test: 0.803246
RMSE_Score_train:485371.330401
RMSE_Score_test:488927.132141


          4.2. Model performance:

Model can be evaluated by various metrics such as:
Root Mean Square Error-
Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.

It gets influenced by outliers.



Mean Absolute Error- Mean Absolute Error is a model evaluation metric used with regression models. The mean absolute error of a model with respect to a test set is the mean of the absolute values of the individual prediction errors on all instances in the test set. Each prediction error is the difference between the true value and the predicted value for the instance. 




We choose MAE and not RMSE as the deciding factor because of the following reasons:
RMSE is heavily influenced by outliers as the higher the values get the more the RMSE increases. MAE doesn’t increase with outliers.
MAE is linear and RMSE is quadratically increasing.

4.3. Hyper parameter tuning:
Hyperparameters are sets of information that are used to control the way of learning an algorithm. Their definitions impact parameters of the models, seen as a way of learning, change from the new hyperparameters. This set of values affects performance, stability and interpretation of a model. Each algorithm requires a specific hyperparameters grid that can be adjusted according to the business problem. Hyperparameters alter the way a model 
learns to trigger this training algorithm after parameters to generate outputs.

We used Grid Search CV, Randomized Search CV and Bayesian Optimization for hyperparameter tuning. This also results in cross validation and in our case we divided the dataset into different folds. The best performance improvement among the three was by Bayesian Optimization.

Grid Search CV-Grid Search combines a selection of hyperparameters established by the scientist and runs through all of them to evaluate the model’s performance. Its advantage is that it is a simple technique that will go through all the programmed combinations.
The common hyperparameters which we extracted were n_estimators, max_depth, verbose=1 and cv = KFold.

Randomized Search CV- In Random Search, the hyperparameters are chosen at random within a range of values that it can assume. The advantage of this method is that there is a greater chance of finding regions of the cost minimization space with more suitable hyperparameters, since the choice for each iteration is random. The disadvantage of this method is that the combination of hyperparameters is beyond the scientist’s control

Bayesian Optimization- Bayesian Hyperparameter optimization is a very efficient and interesting way to find good hyperparameters. In this approach, in naive interpretation way is to use a support model to find the best hyperparameters.A hyperparameter optimization process based on a probabilistic model, often Gaussian Process, will be used to find data from data observed in the later distribution of the performance of the given models or set of tested hyperparameters.
As it is a Bayesian process at each iteration, the distribution of the model’s performance in relation to the hyperparameters used is evaluated and a new probability distribution is generated. With this distribution it is possible to make a more appropriate choice of the set of values that we will use so that our algorithm learns in the best possible way.



5. Conclusion:

That's it! We reached the end of our exercise.
Starting with loading the data so far we have done EDA , null values treatment, encoding of categorical columns, feature selection and then model building.
In all of these models our errors have been in the range of 2,00,000 which is around 10% of the average views. We have been 
able to correctly predict views 90% of the time.
After hyper parameter tuning, we have prevented overfitting and decreased errors by regularizing and reducing learning rate.
Given that only 10% is errors, our models have performed very well on unseen data due to various factors like feature selection,correct model selection,etc.
 
Future work:
We can do a dynamic regression time series modelling due to the availability of the time features.
We can improve the views on the less popular topics by inviting more popular speakers.
We can use topic modelling to tackle views in each topic separately.




