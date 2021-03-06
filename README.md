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
# or for drop: df_main.drop(columns=[???YEAR???], inplace=True)
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

## Week 7
#### Day 1 | Image processing and Convolutional neural networks (CNN)    
Images are saved as a matrix of numbers. Color images are typically three equally sized matrices for red, blue, and green. Dim-1= height, dim-2= width, dim-3= color.
```
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize, rotate
ex
coin = io.imread('data/coin.jpg')
```
For images, such as x-rays, we don't necessarily care about color. We care more about intensity (histogram of pixels).    
Edge detection can be accomplished with gradient analysis of the pixel intensities. An operator we can use is the Sobel operator.

===============================   
CNNs    
Create a filter that corresponds to the feature of interest, scan from left to right, register the accordance btw signal and filter at a variety of positions, and output a sequence of arrays.    
Each time the filter moves, it computes the sum of the element-wise products.    
When you convolve, you must determine the padding (adding zeros to prevent overhanging on the edges) and stridig (# of rows/cols to move).    
Activation introduces non-lineraity into our neural network, making it faster. We can use RELU or max pooling.    
We use tensors w/ CNN. Dim-1= vector, Dim-2= matrix, Dim-3= 3-tensor.    
<br>
#### Day 2 | Natural language processing (NLP) and Text classification/Naive Bayes    
NLP deals with how computers interact with language.    
Challenges are sparsity (small # of words in any single doc), ambiguity, dialects.    
**Document** single email, incident report. **Corpus** a collection of documents (X matrix). **Stop words** words not useful in differentiatig documents, typically removed. **Tokens** components of a doc (words). These are the atomic units of the text. They can be stemmed or lemmatized. **N-grams** two words that commonly appear together e.g. Star Wars. **Bag of words** token count is interpreted as importance.    
**Text Processing Workflow**
1. lowercase
2. strip punctuation
3. remove stop words
4. stem/lemmatize (decrease sparsity/increase density)
5. convert to numeric respresentation *counts, term treq, term freq-inv doc freq*
6. train/cluster
7. optional: part of speech tagging    
The ultimate goal of indeixing is to make a signature (vector) for each document.
<br>
===============================    <br>
Naive Bayes is helpful in classifying email (spam/not spam), sentiment analysis.    
Has Laplace smoothing for scenario where word does not appear for a certain calss. We add 1 in this case.    
Think of three classes (sports, art, travel). We take the prior for each (sports, art, travel) and then multiple by all o the conditional probabilities. We do not worrry about the denomiator in this case.    

#### Day 3 | Cluster and Principal component analysis (PCA)    

Clustering- reason to believe that difference exist in the data that have not been explicityly labeled. This unseen label can be deduced.    
"Within cluster variation" (WCV) *a cluster is good when pts in a cluster are near to each other and far away from pts outside of the cluster*    
K-Means clustering- pick a value for k (#of clusters), initialize k randrom pts (centroids), assign each obs pt to the nearest centroid (repeat until convergence). First iteration moves all the centroids to roughly the center. Then assign pts to nearest centroid, over and over.    
**k-means++** after inital random centroid, the subbsequent ones are choosen to fill the gaps. (default in sklearn).    
*Stopping criteria* 1. max_iter is how many iterations to do 2. centrodids don't change 3. centroids don't move by very much.    
**non-deterministic** is due to random initialization and pts might not always be assigned to same cluster.    
Elbow plot helps us evaluate # of Ks to use; ee how RSS descends and when we start to experience diminishing returns- select the elbow part of the plot    
Shilouette score measures "goodness" of cluster. (b-a)/max(a,b) (a= low avg intra-cluster dist, b = high avg nearest cluster dist) best = 1 (a=0), worst = -1 (b=0)

```
import itertools
import scipy.stats as scs
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
```    

=============================== 
<br>
When faced with a large set of correlated variables, principal components allow us to summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set.    
Reduces dimensionality, removes collinearity. We want to capture the pincipal components that capture the most variation (90%).    
1. Standardize columns
2. Create covariance (correlation if standardized) matrix
3. Find the eigenvectors and eigenvalues of the covariance/correlation matrix
4. The eigenvectors are the principal components
We can look at the explained variance by the # of components to see how "deep" we should go.    

#### Day 4 | Singular value decomposition (SVD) and Topic modeling with w/ NMF    
SVD can be a more computational efficient way to find the same eigenvectors as PCA.    
SVD decomposes matrix A (mxn) into U (mxm), S(mxn), and V.T (nxn). We use SVD to determine the *latent features*.    
1. The U matrix relates rows in X to the latent topics, based on the magnitude of the values in the matrix (the larger the value, the more it loads onto that latent topic).
2. The S matrix contains the singular values associated with the latent topics. Squared singular values are the same as eigenvalues.
3. The V matrix relates the topics (rows) to the columns of the X matrix (the movies for example from class)
4. By selecting the number of singular values, you are simultaneously reducing dimensionality and eliminating collinearity, and finding latent topics that can be used to reconstruct your original data.
<br>
=============================== 
<br>
NMF is another approach to compress a matrix, similar to SVD. We get a matrix w/ min loss of fidelity. We do not allow negative numbers.    
We use alterrnating least squares to calculate the matrices.
```
from sklearn.decomposition import NMF
```
