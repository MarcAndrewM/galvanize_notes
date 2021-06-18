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
We looked at KNN to determine how we could classify points. We discussed importance of units and how many neighbors to determine the classification.
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
#### Day 3 | Algorithmic Complexity and Regularized Regression
#### Day 4 | Logisitic Regression and Decision Rules

