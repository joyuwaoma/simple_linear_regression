# simple_linear_regression

![7](https://github.com/joyuwaoma/simple_linear_regression/blob/main/7.png)

## Objective:

1. Use scikit-learn to implement simple linear regression
2. Create, train, and test a linear regression model on real data

For this lab, you will need to have the following packages:
NumPy, Matplotlib, Pandas, Scikit-learn

## Understand the data:

You will use a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Dataset source.

```python
MODEL YEAR e.g. 2014
MAKE e.g. VOLVO
MODEL e.g. S60 AWD
VEHICLE CLASS e.g. COMPACT
ENGINE SIZE e.g. 3.0
CYLINDERS e.g 6
TRANSMISSION e.g. AS6
FUEL TYPE e.g. Z
FUEL CONSUMPTION in CITY(L/100 km) e.g. 13.2
FUEL CONSUMPTION in HWY (L/100 km) e.g. 9.5
FUEL CONSUMPTION COMBINED (L/100 km) e.g. 11.5
FUEL CONSUMPTION COMBINED MPG (MPG) e.g. 25
CO2 EMISSIONS (g/km) e.g. 182
```

**The task will be to create a simple linear regression model from one of these features to predict CO2 emissions of unobserved cars based on that feature.**

## Explore the data:

First, consider a statistical summary of the data.
```python
df.describe()
```
![1](https://github.com/joyuwaoma/simple_linear_regression/blob/main/1.png)

Select a few features that might be indicative of CO2 emission to explore more. 
```python
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)
```
![2](https://github.com/joyuwaoma/simple_linear_regression/blob/main/2a.png)

## Visualize features:

Consider the histograms for each of these features.
```python
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()
```
![3](https://github.com/joyuwaoma/simple_linear_regression/blob/main/3.png)

As you can see, most engines have 4, 6, or 8 cylinders, and engine sizes between 2 and 4 liters.
As you might expect, combined fuel consumption and CO2 emissions have very similar distributions.



Display some scatter plots of these features against the CO2 emissions, to see how linear their relationships are.

```python
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
```
![4](https://github.com/joyuwaoma/simple_linear_regression/blob/main/4.png)
This is an informative result. Three car groups each have a strong linear relationship between their combined fuel consumption and their CO2 emissions. Their intercepts are similar, while they noticeably differ in their slopes.


```python
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
plt.show()
```
![5](https://github.com/joyuwaoma/simple_linear_regression/blob/main/5.png)
Although the relationship between engine size and CO2 emission is quite linear, you can see that their correlation is weaker than that for each of the three fuel consumption groups. Notice that the x-axis range has been expanded to make the two plots more comparable.

## Practice exercise:
1. Plot CYLINDER against CO2 Emission, to see how linear their relationship is.
```python
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='green')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2 Emission")
plt.show()
```
![6](https://github.com/joyuwaoma/simple_linear_regression/blob/main/6.png)

2. Extract the input feature and labels from the dataset, and Create train and test datasets
```python
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

type(X_train), np.shape(X_train), np.shape(X_train)
```
3. Build a simple linear regression model
```python
from sklearn import linear_model

# create a model object
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)
```
Visualize model outputs
```python
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
```
![7](https://github.com/joyuwaoma/simple_linear_regression/blob/main/7.png)

## Model evaluation:

You can compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics play a key role in the development of a model, as they provide insight into areas that require improvement.

There are different model evaluation metrics, let's use MSE here to calculate the accuracy of our model based on the test set: 
* Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just an average error.

* Mean Squared Error (MSE): MSE is the mean of the squared error. In fact, it's the metric used by the model to find the best fit line, and for that reason, it is also called the residual sum of squares.

* Root Mean Squared Error (RMSE). RMSE simply transforms the MSE into the same units as the variables being compared, which can make it easier to interpret.

* R-squared is not an error but rather a popular metric used to estimate the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict( X_test.reshape(-1,1))

print("Mean absolute error: %.2f" % mean_absolute_error(y_test_, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))
print("R2-score: %.2f" % r2_score( y_test_, y_test) )
```
4. Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.
```python
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
```
![7](https://github.com/joyuwaoma/simple_linear_regression/blob/main/7.png)

5. Check the evaluation metrics if you train a regression model using the FUELCONSUMPTION_COMB feature. (Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets)
```python
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```
6. Train a linear regression model using the training data created.
```python
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)
```
7. Use the model to make test predictions on the fuel consumption testing data.
```python
y_test_ = regressor.predict(X_test.reshape(-1,1))
```
8. Calculate and print the Mean Squared Error of the test predictions.
```python
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
```

