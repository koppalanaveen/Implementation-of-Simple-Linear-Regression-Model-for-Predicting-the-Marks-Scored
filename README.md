# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KOPPALA NAVEEN
RegisterNumber:  212223100023
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```
## Output:

df.head()

![image](https://github.com/user-attachments/assets/64e72f9a-da50-4537-bbab-6137cb91dbd5)

df.tail()

![image](https://github.com/user-attachments/assets/a1f5d9f1-2c5d-4d8d-ab12-8b98ccee66ee)

Array value of X

![image](https://github.com/user-attachments/assets/49534172-7a83-4334-b99e-a628f5c031b2)

Array value of Y

![image](https://github.com/user-attachments/assets/85b76c0f-c23d-4ae9-8c4b-5919993c1cc4)

Values of Y prediction

![image](https://github.com/user-attachments/assets/706c5c34-51b7-423b-851d-f74eb2e490eb)

Array values of Y test

![image](https://github.com/user-attachments/assets/7ce65c28-95c4-474d-b53c-241b85fdf4b7)

Training Set Graph

![image](https://github.com/user-attachments/assets/4b0910fa-5f21-4c4e-b48f-e4a8c5e6888d)

Test Set Graph

![image](https://github.com/user-attachments/assets/6cd43438-e4a8-4a5d-8b27-1dc61ed54de0)

Values of MSE, MAE and RMSE

![image](https://github.com/user-attachments/assets/2b36d58d-24a4-4573-950c-6cf145f19545)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
