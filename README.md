# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: import pandas module and import the required data set.

Step 3: Find the null values and count them.

Step 4: Count number of left values.

Step 5: From sklearn import LabelEncoder to convert string values to numerical values.

Step 6: From sklearn.model_selection import train_test_split.

Step 7: Assign the train dataset and test dataset.

Step 8: From sklearn.tree import DecisionTreeClassifier.

Step 9: Use criteria as entropy.

Step 10: From sklearn import metrics.

Step 11: Find the accuracy of our model and predict the require values.

Step 12: Stop the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RAMYA S
RegisterNumber:  212222040130
*/
import pandas as pd
data=pd.read_csv("Exp_8_Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
![image](https://github.com/user-attachments/assets/ff909076-0a7f-460f-b27d-54f89c02d215)
## info :
![image](https://github.com/user-attachments/assets/3eb2a6f2-6a0e-4fa2-a1d9-9da04b19e53d)
## checking for null values :
![image](https://github.com/user-attachments/assets/acdbef1d-076a-4614-b940-249d95724085)
![image](https://github.com/user-attachments/assets/1228ab70-280b-4077-9e20-37b1e7de5b19)
## Accuracy:
![image](https://github.com/user-attachments/assets/74647f65-8f01-44ad-ab3a-be4a7019758e)
## Predict:
![image](https://github.com/user-attachments/assets/a5060145-2fb2-4fab-8324-248b714e096a)















## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
