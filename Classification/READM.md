# Classification

1. **Data Visualisation and Feature Selection and Normalisation**

**Data Visualisation:**

The scatter_matrix function from pandas is used to create a scatter matrix. This matrix allows visualizing the correlation between different variables in a DataFrame. Each cell of the matrix represents a scatter plot between two variables, and the diagonal represents a histogram of each variable.

**Feature Selection:**

Univariate Selection, Principal Component Analysis (PCA), Recursive Feature Elimination (RFE), and Feature Importance are all methods used for feature selection in machine learning. Here's how you can use each of them:

1. **Univariate Selection:** Statistical tests can be used to select those features that have the strongest relationship with the output variable. The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.
2. **Recursive Feature Elimination (RFE):** The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
3. **Principal Component Analysis (PCA):** PCA uses linear algebra to transform the dataset into a compressed form. Generally this is called a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal components in the transformed result.
4. **Feature Importance:** Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.

Here is an example of how you can use each of these methods in Python:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

# load your data
# X, Y = load_your_data()

# Univariate Selection
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
print(fit.scores_)

# Recursive Feature Elimination
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# Principal Component Analysis
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

# Feature Importance
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
```

**Normalization:**

Normalization is a technique used to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values or losing information. It is also known as Min-Max scaling.  Here is how it works:

For each feature, calculate the minimum value of that feature in the dataset.

For each feature, calculate the maximum value of that feature in the dataset.

For each value in the feature, apply the following normalization formula:  normalized_value = (value - min_value) / (max_value - min_value)

1. **Classification using the accurate algorithm :**

1. **K-Nearest Neighbors (KNN):** KNN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. It works by finding a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning).
2. **Decision Tree:** Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
3. **Artificial Neural Network (ANN):** ANNs are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.
4. **Naive Bayes:** Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.
5. **Support Vector Machine (SVM):** SVMs are a set of supervised learning methods used for classification, regression and outliers detection. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.
    - **Linear Kernel:** A linear kernel can be used as normal dot product any two given observations. The product between two vectors is the sum of the multiplication of each pair of input values.
    - **Polynomial Kernel:** A polynomial kernel is a more generalized form of the linear kernel. The polynomial kernel can distinguish curved or nonlinear input space.
    - **Gaussian Kernel:** The Gaussian kernel is a function that decreases with distance and ranges between zero (in the limit) and one (when x = x'). The Gaussian kernel is also parameterized by a bandwidth parameter, which determines how fast the similarity metric decreases (to 0) as the examples are further apart.

**************************************Model Evaluation :**************************************

Classification Accuracy, Logarithmic Loss, Area Under ROC Curve, Confusion Matrix, and Classification Report are all metrics used to evaluate the performance of classification models. Here's a brief explanation of how each of them works:

1. **Classification Accuracy:** This is simply the ratio of correct predictions to total predictions made. It's the most common evaluation metric for classification problems, however, it is often misused as it can be misleading when classes are imbalanced.
2. **Logarithmic Loss (Log Loss):** Log Loss quantifies the accuracy of a classifier by penalising false classifications. Minimising the Log Loss is basically equivalent to maximising the accuracy of the classifier.
3. **Area Under ROC Curve (AUC-ROC):** ROC Curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity. The area under the curve (AUC) provides an aggregate measure of performance across all possible classification thresholds. A model whose predictions are 100% correct has an AUC of 1 and a model whose predictions are 100% wrong has an AUC of 0.
4. **Confusion Matrix:** A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. It allows visualization of the performance of an algorithm.
5. **Classification Report:** This is another way to evaluate the classification model, it displays the precision, recall, F1 and support scores for the model. Precision is the ability of a classifier not to label a positive sample as negative, recall is the ability of a classifier to find all positive samples, and the F1 score can be interpreted as a weighted average of the precision and recall.

Here's how you can calculate each of them using scikit-learn in Python:

```python
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Assume that we have a dataset `X` and `y`
# X, y = load_your_dataset()  # replace with your actual data loading code

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Classification Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy}")

# Logarithmic Loss
logloss = metrics.log_loss(y_test, y_pred_proba)
print(f"Logarithmic Loss: {logloss}")

# Area Under ROC Curve
auc = metrics.roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"Area Under ROC Curve: {auc}")

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \\n{cm}")

# Classification Report
cr = metrics.classification_report(y_test, y_pred)
print(f"Classification Report: \\n{cr}")

```

In this code:

- `accuracy_score` is used to calculate the Classification Accuracy.
- `log_loss` is used to calculate the Logarithmic Loss.
- `roc_auc_score` is used to calculate the Area Under ROC Curve.
- `confusion_matrix` is used to generate the Confusion Matrix.
- `classification_report` is used to generate the Classification Report, which includes metrics such as precision, recall, f1-score, and support for each class.

****************************************Model comparison:****************************************

Spot-checking is a way to evaluate machine learning algorithms on your dataset. The idea is to quickly test a diverse set of algorithms on your dataset and get a general idea of how each performs. This can help you to narrow down the best algorithms to fine-tune and optimize.  Here's how it works:

Define a list of models. This list should include a variety of algorithms, including linear models, non-linear models, and ensemble methods. The goal is to get a diverse set of predictions.

Run each model on your dataset. This can be done using a method like cross-validation, which will give you a robust estimate of each model's performance.

Analyze the results. Look at the performance of each model to decide which models are worth further investigation.

**Ensemble learning :**

Bagging, stacking, and boosting are ensemble learning methods used in machine learning. Here's a brief explanation of how each of them works:

1. **Bagging:** Bagging stands for Bootstrap Aggregating. It works by creating multiple subsets of the original dataset, training a model on each subset, and combining the predictions. The goal of bagging is to reduce the variance of a decision tree classifier. A popular example of bagging is the Random Forest algorithm.
2. **Stacking:** Stacking involves training multiple different models and using another machine learning model to combine their predictions. Each algorithm may be suited to a different aspect of the problem. By combining them, we can produce a more accurate prediction.
3. **Boosting:** Boosting works by training a model on the dataset and then creating a second model that attempts to correct the errors from the first model. Models are added until the training set is predicted perfectly or a maximum number of models are added. AdaBoost and Gradient Boosting are examples of boosting algorithms.

Here's how you can use bagging, stacking, and boosting techniques in Python using the `sklearn.ensemble` module:

```python
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
X, y = load_your_dataset()  # replace with your actual data loading code

# prepare models
models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]

# Bagging
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0)
cv_results = model_selection.cross_val_score(bagging, X, y, cv=10, scoring='accuracy')
print(f"Bagging: {cv_results.mean()} ({cv_results.std()})")

# Stacking
stacking = StackingClassifier(estimators=models, final_estimator=LogisticRegression())
cv_results = model_selection.cross_val_score(stacking, X, y, cv=10, scoring='accuracy')
print(f"Stacking: {cv_results.mean()} ({cv_results.std()})")

# Boosting
boosting = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0)
cv_results = model_selection.cross_val_score(boosting, X, y, cv=10, scoring='accuracy')
print(f"Boosting: {cv_results.mean()} ({cv_results.std()})")

```

In this code:

- Bagging is performed using the `BaggingClassifier` with a `DecisionTreeClassifier` as the base estimator and 10 estimators.
- Stacking is performed using the `StackingClassifier` with the previously defined models as the base estimators and a `LogisticRegression` as the final estimator.
- Boosting is performed using the `AdaBoostClassifier` with a `DecisionTreeClassifier` as the base estimator and 10 estimators.

The performance of each technique is evaluated using 10-fold cross-validation and the results are printed out as the mean and standard deviation of the accuracy.

**References :**

[https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/)[https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/)

[https://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-](https://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-scikit-learn/)

[https://machinelearningmastery.com/bagging-ensemble-with-python/](https://machinelearningmastery.com/bagging-ensemble-with-python/)[https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

[https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)
