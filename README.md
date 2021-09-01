# Feature Selection
It is possible to automatically select those features in your data that are most useful or most relevant for the problem you are working on. This is a process called feature selection. Feature selection is different from dimensionality reduction. Both methods seek to reduce the number of attributes in the dataset, but a dimensionality reduction method do so by creating new combinations of attributes, where as feature selection methods include and exclude attributes present in the data without changing them. Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.

There are three general classes of feature selection algorithms: 

- Filter methods
- Wrapper methods
- Embedded methods

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
- Correlation coefficient scores

## Wrapper Methods
Wrapper methods consider the selection of a set of features as a search problem, where different combinations are prepared, evaluated and compared to other combinations. A predictive model is used to evaluate a combination of features and assign a score based on model accuracy. The search process may be methodical such as a best-first search, it may stochastic such as a random hill-climbing algorithm, or it may use heuristics, like forward and backward passes to add and remove features.

Summary:

- Functionality: evaluates on a specific machine learning algorithm to find optimal features
- Time Complexity: high computation time for a dataset with many features
- Overfitting: high chances of overfitting

Examples:

- Recursive feature elimination
- Forward selection
- Backward elimination
- Stepwise selection

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

See: https://github.com/lucaskienast/Regression-Models

## References

Analytics Vidhya (2020) A comprehensive guide to Feature Selection using Wrapper methods in Python. Available at: https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/ (Accessed: 1 September 2021)

Brownlee, J. (2021) An Introduction to Feature Selection. Available at: https://machinelearningmastery.com/an-introduction-to-feature-selection/ (Accessed: 1 September 2021)
