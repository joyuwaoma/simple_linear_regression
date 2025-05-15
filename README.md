# simple_linear_regression

![7](https://github.com/joyuwaoma/simple_linear_regression/blob/main/7.png)

## Objective:

1. Use scikit-learn to implement simple linear regression
2. Create, train, and test a linear regression model on real data

For this lab, you will need to have the following packages:
NumPy, Matplotlib, Pandas, Scikit-learn

## Understand the data:

You will use a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Dataset source.

```
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
```
df.describe()
```
![1](https://github.com/joyuwaoma/simple_linear_regression/blob/main/1.png)

Select a few features that might be indicative of CO2 emission to explore more. 
```
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)
```
![2](https://github.com/joyuwaoma/simple_linear_regression/blob/main/2a.png)

## Visualize features:

Consider the histograms for each of these features.
```
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()
```
![3](https://github.com/joyuwaoma/simple_linear_regression/blob/main/3.png)

As you can see, most engines have 4, 6, or 8 cylinders, and engine sizes between 2 and 4 liters.
As you might expect, combined fuel consumption and CO2 emissions have very similar distributions.

