# galvanize_notes
This will be where I store all of the main takeaways from each week.

HIGH LEVEL
-Always check data for missing values. df.isnull().sum() ==> leads to an output for each parameter


## Week 1
### PANDAS
If you index a DataFrame with a single value or a list of values, it selects the columns.
If you use a slice or sequence of booleans, it selects the rows. 
<li> As a reminder, logical operators with arrays AND series look like</li>

```
(golf_df['Humidity']>90) | (golf_df['Outlook']=="Sunny") # '|' is 'OR'
(golf_df['Result']=="Don't Play") & golf_df['Windy'] # '&' is 'AND'
```

## Week 2
#### Day 3 | Maximum likelihood estimator

```
# these are the imports from the module
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optim

from itertools import product
```
MLE is our tool we will use to find the best parameters.
A point on the PDF is called the likelihood for the respectice x-value. On a continuous distribution, the probability of getting a specific number is 0!


## Week 5
#### Day 1 | K-Nearest Neighbors and Cross Validation 

We looked at KNN to determine how we could classify points. We discussed importance of units and selecting number of neighbors to determine the classification of a point. You want to make sure your data is not "fat and short". The # of cols should be less than the square root of n.
```
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
```
We train-test split first to later determine the usefulness of our model.
The RMSE of the train set should always be lower than the RMSE for the test set except in rare cases.
RMSE = the standard deviation of the residuals (square root of the mean square error).

=============================    
For cv, we used the following imports
```
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston #example of loading a dataset
from sklearn.metrics import mean_squared_error
```
To fit the KNN regressor we used
```
reg = KNeighborsRegressor()
reg.fit(X_train, y_train)

def cross_val(X_train, y_train, k=3):
    ''' Returns error for k-fold cross validation. '''
    kf = KFold(n_splits=k)
    error = np.empty(k)
    index = 0
    for train, test in kf.split(X_train):
        reg = KNeighborsRegressor()
        reg.fit(X_train[train], y_train[train])
        pred = reg.predict(X_train[test])
        error[index] = rmse(pred, y_train[test])
        index += 1
    return np.mean(error)
```

#### Day 2 | Predictive Linear Regression and Inferential Regression
Once we get a dataset, we look at its scatter matrix.
```
pd.plotting.scatter_matrix(balance, figsize=(15, 10))
plt.show()
```
For categorical data, we convert the columns to 1/0s and then delete one of the cols to avoid multicollinearity.
```
# Married - 1, Not Married - 0
balance['Married'] = balance['Married'].map({'Yes': 1, 'No': 0})
```
We can also create dummy variable for columns with labels.
```
#Get the Dummy variables
ethnicity_dummy = pd.get_dummies(balance['Ethnicity'])
Only need two of the three values
balance[ ['Asian', 'Caucasian'] ] = ethnicity_dummy[ ['Asian', 'Caucasian'] ]
# Remove the Ethnicity column
del balance['Ethnicity']
# or for drop: df_main.drop(columns=[‘YEAR’], inplace=True)
```
=============================    
For inferential regression, we are concerned with LINH and multicollinearity
Prediction is not as important as our understanding of the variables
```
import statsmodels.formula.api as smf
#example
prestige_model = smf.glm(formula="prestige ~ income + education",
                         data=prestige)
prestige_results = prestige_model.fit()
prestige_results.summary()

#OR
import statsmodels.api as sm
model = sm.OLS(y, X)
results = model.fit()
results.summary()
```
We mkake a QQ plot to see if residuals are normally distributed
```
# for the prestige data
fig, ax = plt.subplots(figsize=((5, 5)))
predictions = prestige_results.predict(prestige)
stats.probplot(prestige['prestige'] - predictions,
               plot=ax);
ax.set_title("QQ Plot vs Normal dist for Prestige Residuals")
```
For collinearity, notice on the output, "the condition number ("Cond. No.") is huge, indicating multicollinearity. In such a case the coefficients can't really be trusted."
#### Day 3 | Algorithmic Complexity and Regularized Regression
Algorithmic complexity is important to see how efficient our code is. A formal definition is, "Algorithmic complexity is a measure of how long an algorithm would take to complete given an input of size n".
```
O(1)
    General rule: most things, specifically anything that doesn't use a loop secretly
    ==
    <
    %     - it might seem like mod would have to iterate through the number. That's good way to think, but unfortunatly is incorrect. As it turns out we found a faster way: a bitwise comparison.
    +
    -
    /
    *
    .format
    return
    len() - python lists secretly remembers it's own length
    x[5]
    if x in dictionary():
    if x in set():

O(n)
    General rule: Any thing that must look at all elements
    min()
    max()
    sum()
    mean()
    replace()
    join()
    set([1,2,3...])
    list([1,2,3...])
    dictionary([1,2,3...])
    Counter([1,2,3...])
    defaultdict([1,2,3...])
    np.array([1,2,3...])
    for x in set:
    for x in dictionary:
    [x for x in list]
    copy()
    if x in list:

O( nlog(n) )
    General rule: sorting algorithms
    sorted()
```
=============================    
For regularization, we are trying to dampen down the coefficients of the parameter by introducing a lambda to our model. Lambda can go from 0 (regular linreg) to infinity. 
```
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler # need to standardized for regularization
from utils import XyScaler

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)
```
"Remember that your predictors and response must be standardized when using ridge regression, and that this standardization must happen inside of the cross validation using only the training set!"

#### Day 4 | Logisitic Regression and Decision Rules    
Logistic regression is a classifier.
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
```
example from grad school admission (yes or no)
```
kfold = KFold(n_splits=10)

accuracies = []
precisions = []
recalls = []

X_train, X_test, y_train, y_test = train_test_split(X, y)

for train_index, test_index in kfold.split(X_train):
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    y_predict = model.predict(X_train.iloc[test_index])
    y_true = y_train.iloc[test_index]
    accuracies.append(accuracy_score(y_true, y_predict))
    precisions.append(precision_score(y_true, y_predict))
    recalls.append(recall_score(y_true, y_predict))

print("Accuracy:", np.average(accuracies))
print("Precision:", np.average(precisions))
print("Recall:", np.average(recalls))
```
This is a little tricky on threshold for the model. When should something be classified as a 1 vs a 0. It seems you need to ensure when you run this, you do it on your training set during crossfold validation so you are not doing it after the fact.

===============================     
For "decision rules":
*"Imbalanced datasets highlight that the misclassification costs implicit in model algorithms don't always align with our judgement of real-world misclassification costs. Sometimes it is equally important to get every instance right. Sometimes it is more important to identify some classes than others. A data scientist should always consider the costs and frequencies of different misclassifications."*
For a certain threshold, we developed a confusion matrix. Col headers were "Actual positive", "Actual negative". Row headers were "Predicted positive", "Predicted negative".
We looked at profit curves and how changing our classification either increased or decreased our profit.


## Week 6
#### Day 1 | Gradient Descent and Perceptrons
Gradient descent has been a concept we have been using implicitly for the past couple weeks. For our cost function, we are trying to make it smaller. We use a "learning rate", alpha, to determine our step.

Choose a starting point, a learning rate, and a threshold
Repeatedly:
Calculate the gradient at the current point,
Multiply the gradient by the negative of the learning rate
Add that to the current point to find a new point
Repeat until within the threshold


```
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
```

#### Day 2 | Time series and Decision trees
Use df.resample('Q-NOV') to get quarterly means that follow the seasons of the year (spring, summer, fall, winter).

===============================     
Decision trees: Goal is to achieve the best classification with the least number of decisions.  
Internal node: tests a predictor  
Branch: predictor value  
Lead: assigns classification  
**Gini impurity**: 1 minus the summation of the proportion of each class squared.
ex: 5 red circles/ 6 green squares ==> 1-(5/11)^2-(6/11)^2  
**Entropy**: oppositve of the summation of the probability times the log2(probability)
information gained equals parent impurity minus summation of child impurities
Classification trees outcomes are discrete. Regression trees outcomes are continuous and use RSS instead of Gini/entropy.
```
from sklearn import tree
tree.DecisionTreeClassifier(class_weight=None, criterion = 'entropy', max_depth = 2, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0, min_impurity_split=None,min_samples_leaf =1, min_samples_split=2,min_weight_fraction_leaf=0,presort=False, random_state=None, splitter ='best')
```
Three laws of recursion
1. must have a base case
2. must change its state and move toward the base case
3. must call itself, recursively

```diff
+ 
- 
! 
@@ @@ 
# 
```

