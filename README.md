# Feature Selection
It is possible to automatically select those features in your data that are most useful or most relevant for the problem you are working on. This is a process called feature selection. Feature selection is different from dimensionality reduction. Both methods seek to reduce the number of attributes in the dataset, but a dimensionality reduction method do so by creating new combinations of attributes, where as feature selection methods include and exclude attributes present in the data without changing them. Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.

There are 2 (3) types of feature selection that differ based on whether features are selected via the target variable or not.

- Supervised (use target variable)
- Unsupervised (ignore target vraible)
- (Dimensionality reduction, see: https://github.com/lucaskienast/Dimensionality-Reduction)

There are three general classes of feature selection algorithms: 

- Filter methods
- Wrapper methods
- Embedded methods

## Installation
Use `git clone` to get a copy of this repository.
```
$ git clone https://github.com/lucaskienast/Feature-Selection.git
$ cd Feature-Selection
```

## Note on Overfitting
It is important to consider feature selection a part of the model selection process. If you do not, you may inadvertently introduce bias into your models which can result in overfitting. You should do feature selection on a different dataset than you train your predictive model on, because the effect of not doing this is you will overfit your training data. 

For example, you must include feature selection within the inner-loop when you are using accuracy estimation methods such as cross-validation. This means that feature selection is performed on the prepared fold right before the model is trained. A mistake would be to perform feature selection first to prepare your data, then perform model selection and training on the selected features. The reason is that the decisions made to select the features were made on the entire training set, that in turn are passed onto the model. 

This may cause a mode a model that is enhanced by the selected features over other models being tested to get seemingly better results, when in fact it is biased result.

## Checklist

1. Do you have domain knowledge? If yes, construct a better set of ad hoc?????? features
2. Are your features commensurate? If no, consider normalizing them.
3. Do you suspect interdependence of features? If yes, expand your feature set by constructing conjunctive features or products of features, as much as your computer resources allow you.
4. Do you need to prune the input variables (e.g. for cost, speed or data understanding reasons)? If no, construct disjunctive features or weighted sums of feature
5. Do you need to assess features individually (e.g. to understand their influence on the system or because their number is so large that you need to do a first filtering)? If yes, use a variable ranking method; else, do it anyway to get baseline results.
6. Do you need a predictor? If no, stop
7. Do you suspect your data is ???dirty??? (has a few meaningless input patterns and/or noisy outputs or wrong class labels)? If yes, detect the outlier examples using the top ranking variables obtained in step 5 as representation; check and/or discard them.
8. Do you know what to try first? If no, use a linear predictor. Use a forward selection method with the ???probe??? method as a stopping criterion or use the 0-norm embedded method for comparison, following the ranking of step 5, construct a sequence of predictors of same nature using increasing subsets of features. Can you match or improve performance with a smaller subset? If yes, try a non-linear predictor with that subset.
9. Do you have new ideas, time, computational resources, and enough examples? If yes, compare several feature selection methods, including your new idea, correlation coefficients, backward selection and embedded methods. Use linear and non-linear predictors. Select the best approach with model selection
10. Do you want a stable solution (to improve performance and/or understanding)? If yes, subsample your data and redo your analysis for several ???bootstrap???.

## Filter Methods
Filter feature selection methods apply a statistical measure to assign a scoring to each feature. The features are ranked by the score and either selected to be kept or removed from the dataset. The methods are often univariate and consider the feature independently, or with regard to the dependent variable.

Summary:

- Functionality: generic set of methods which do not include machine learning algorithms
- Time Complexity: faster compared to wrapper methods
- Overfitting: less prone to overfitting

Examples are: 

- Chi squared test
- Information gain 
- ANOVA test (analysis of variance)
- Feature Importance (Non-Embedded)
- Missing value ratio
- Low variance filter
- High correlation filter
- (Dimensionality Reduction)

### Numerical Input & Output
This is a regression predictive modeling problem with numerical input variables. The most common techniques are to use a correlation coefficient, such as Pearson???s for a linear correlation, or rank-based methods for a nonlinear correlation.

- Pearson???s correlation coefficient (linear) - `sklearn.feature_selection.f_regression()`
- Spearman???s rank coefficient (nonlinear) - `scipy.stats.spearmanr()`
- Mutual information - `sklearn.feature_selection.mutual_info_regression()`

### Numerical Input & Categorical Output
This is a classification predictive modeling problem with numerical input variables. This might be the most common example of a classification problem. Again, the most common techniques are correlation based, although in this case, they must take the categorical target into account. Kendall does assume that the categorical variable is ordinal.

- ANOVA correlation coefficient (linear) - `sklearn.feature_selection.f_classif()`
- Kendall???s rank coefficient (nonlinear) - `scipy.stats.kendalltau()`

### Categorical Input & Numerical Output
This is a regression predictive modeling problem with categorical input variables. This is a strange example of a regression problem (e.g. you would not encounter it often). Nevertheless, you can use the same ???Numerical Input, Categorical Output??? methods (described above), but in reverse.

### Categorical Input & Output
This is a classification predictive modeling problem with categorical input variables. The most common correlation measure for categorical data is the chi-squared test. You can also use mutual information (information gain) from the field of information theory. In fact, mutual information is a powerful method that may prove useful for both categorical and numerical data, e.g. it is agnostic to the data types.

- Chi-Squared test (contingency tables) - `sklearn.feature_selection.chi2`
- Mutual Information - `sklearn.feature_selection.mutual_info_classif()`

### Selection Method
The scikit-learn library also provides many different filtering methods once statistics have been calculated for each input variable with the target. Two of the more popular methods include:

Select the top k variables: SelectKBest - `sklearn.feature_selection.SelectKBest()`
Select the top percentile variables: SelectPercentile - `sklearn.feature_selection.SelectPercentile`

### Transform Variables
Consider transforming the variables in order to access different statistical methods. For example, you can transform a categorical variable to ordinal, even if it is not, and see if any interesting results come out. You can also make a numerical variable discrete (e.g. bins); try categorical-based measures. Some statistical measures assume properties of the variables, such as Pearson???s that assumes a Gaussian probability distribution to the observations and a linear relationship. You can transform the data to meet the expectations of the test and try the test regardless of the expectations and compare results.

## Wrapper Methods
Wrapper methods consider the selection of a set of features as a search problem, where different combinations are prepared, evaluated and compared to other combinations. A predictive model is used to evaluate a combination of features and assign a score based on model accuracy. The search process may be methodical such as a best-first search, it may stochastic such as a random hill-climbing algorithm, or it may use heuristics, like forward and backward passes to add and remove features.

Summary:

- Functionality: evaluates on a specific machine learning algorithm to find optimal features
- Time Complexity: high computation time for a dataset with many features
- Overfitting: high chances of overfitting

Examples:

- Recursive feature elimination
- Forward feature selection
- Backward feature elimination
- Stepwise selection
- Permutation Feature Selection

## Embedded Methods
Embedded methods learn which features best contribute to the accuracy of the model while the model is being created. The most common type of embedded feature selection methods are regularization methods. Regularization methods are also called penalization methods that introduce additional constraints into the optimization of a predictive algorithm (such as a regression algorithm) that bias the model toward lower complexity (fewer coefficients).

Summary:

- Functionality: embeds fix features during model building process & feature selection done by observing each iteration of model training
- Time Complexity: between filter and wrapper methods
- Overfitting: generally used to reduce overfitting by penalizing too many coefficients

Examples:

- LASSO Regression
- Ridge Regression
- ElasticNet Regression
- Feature Importance (Embedded)

See Regularized Regression Methods: https://github.com/lucaskienast/Regression-Models

## References

Analytics Vidhya (2020) A comprehensive guide to Feature Selection using Wrapper methods in Python. Available at: https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/ (Accessed: 1 September 2021)

Bex, T. (2021) Powerful Feature Selection with Recursive Feature Elimination (RFE) of Sklearn. Available at: https://towardsdatascience.com/powerful-feature-selection-with-recursive-feature-elimination-rfe-of-sklearn-23efb2cdb54e (Accessed: 1 September 2021)

Brownlee, J. (2021) An Introduction to Feature Selection. Available at: https://machinelearningmastery.com/an-introduction-to-feature-selection/ (Accessed: 1 September 2021)

Brownlee, J. (2020) Feature Importance and Feature Selection With XGBoost in Python. Available at: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/ (Accessed: 2 September 2021)

Brownlee, J. (2021) Feature Selection For Machine Learning in Python. Available at: https://machinelearningmastery.com/feature-selection-machine-learning-python/ (Accessed: 1 September 2021)

Brownlee, J. (2020) How to Calculate Feature Importance With Python. Available at: https://machinelearningmastery.com/calculate-feature-importance-with-python/ (Accessed: 2 September 2021)

Brownlee, J. (2020) How to Choose a Feature Selection Method For Machine Learning. Available at: https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/ (Accessed: 1 September 2021)

Brownlee, J. (2020) How to Perform Feature Selection with Categorical Data. Available at: https://machinelearningmastery.com/feature-selection-with-categorical-data/ (Accessed: 2 September 2021)

Brownlee, J. (2020) How to Perform Feature Selection for Regression Data. Available at: https://machinelearningmastery.com/feature-selection-for-regression-data/ (Accessed: 2 September 2021)

Brownlee, J. (2020) Ordinal and One-Hot Encodings for Categorical Data. Available at: https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/ (Accessed: 2 September 2021)

Brownlee, J. (2020) Recursive Feature Elimination (RFE) for Feature Selection in Python. Available at: https://machinelearningmastery.com/rfe-feature-selection-in-python/ (Accessed: 1 September 2021)

Davis, D. (2021) Machine Learning Tutorial ??? Feature Engineering and Feature Selection For Beginners. Available at: https://www.freecodecamp.org/news/feature-engineering-and-feature-selection-for-beginners/ (Accessed: 1 September 2021)

Dubey, A. (2018) Feature Selection Using Random Forest. Available at: https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f (Accessed: 1 September 2021)

Effrosynidis, D. (2021) Feature Selection for Machine Learning: 3 Categories and 12 Methods. Available at: https://towardsdatascience.com/feature-selection-for-machine-learning-3-categories-and-12-methods-6a4403f86543 (Accessed: 1 September 2021)

Garg, S. (2021) How To Implement Feature Selection From Scratch In Python. Available at: https://medium.com/mlearning-ai/feature-selection-using-filter-method-python-implementation-from-scratch-375d86389003 (Accessed: 1 September 2021)

Guyon, I. & Elisseeff, A. (2003) 'An Introduction to Variable and Feature Selection', Journal of Machine Learning Research, Vol. 3, pp. 1157-1182. Available at: https://jmlr.csail.mit.edu/papers/volume3/guyon03a/guyon03a.pdf (Accessed: 1 September 2021)

MNU Munich & Pechenova, E. (2018) Feature Subset Selection. Available at: https://www.mathematik.uni-muenchen.de/~deckert/teaching/WS1819/ATML/pechenova_feature_selection.pdf (Accessed: 1 September 2021)

Paul, S. (2020) Beginner's Guide to Feature Selection in Python. Available at: https://www.datacamp.com/community/tutorials/feature-selection-python (Accessed: 1 September 2021)

Scikit-Learn (2021) Feature Selection. Available at: https://scikit-learn.org/stable/modules/feature_selection.html (Accessed: 1 September 2021)

Shaikh, R. (2018) Feature Selection Techniques in Machine Learning with Python. Available at: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e (Accessed: 1 September 2021)

Sharma, P. (2018) The Ultimate Guide to 12 Dimensionality Reduction Techniques (with Python codes). Available at: https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/ (Accessed: 31 August 2021)
