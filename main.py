import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

filepath = 'https://static.junilearning.com/ai_level_2/life_expectancy.csv'
# dataset: https://www.kaggle.com/kumarajarshi/life-expectancy-who
data = pd.read_csv(filepath)
data = data.drop('Country', axis = 1)
data = data.drop('Status', axis = 1)
data.dropna(inplace=True)

x = np.array(data.drop('Life expectancy ', axis = 1))
y = np.array(data['Life expectancy '])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=45)
linear = LinearRegression()
linear.fit(x_train, y_train)
score = linear.score(x_test, y_test)
print(score)

results = linear.predict([[2015,263,62,0.01,71.27962362,65,1154,19.1,83,6,8.16,65,0.1,584.25921,33736494,17.2,17.3,0.479,10.1]])
print(results[0])

x_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_poly,y,test_size=0.25,random_state=45)

linear.fit(x_train, y_train)
score = linear.score(x_test, y_test)
print(score)

prediction = PolynomialFeatures(degree=2, include_bias=False).fit_transform([[2015,263,62,0.01,71.27962362,65,1154,19.1,83,6,8.16,65,0.1,584.25921,33736494,17.2,17.3,0.479,10.1]])
results = linear.predict(prediction)
print(results[0])
