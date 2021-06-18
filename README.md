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
```
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
```
We train-test split first to later determine the usefulness of our model.

#### Day 2 | Predictive Linear Regression and Inferential Regression
#### Day 3 | Algorithmic Complexity and Regularized Regression
#### Day 4 | Logisitic Regression and Decision Rules

