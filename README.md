# galvanize_notes
This will be where I store all of the main takeaways from each week.

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
We looked at KNN to determine how we could classify points. We discussed importance of units and selecting number of neighbors to determine the classification of a point.
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
from sklearn.datasets import load_boston
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
```
=============================    
For inferential regression, we are concerned with LINH and multicollinearity
Predictiion is not as important as our understanding of the variables
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
#### Day 4 | Logisitic Regression and Decision Rules

