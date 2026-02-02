# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Clean and transform the raw data, then divide it into a training set (to build the model) and a testing set (to validate it).

2.Select your input features and establish parameters like the loss function and regularization to prevent overfitting.

3.Fit the model by adjusting parameters to minimize the loss function using the training data.

4.Assess performance on test data using metrics like Accuracy and F1 Score; if results are poor, refine the features or hyperparameters.

5.Apply the finalized model to new data to predict outcomes and analyze the coefficients to understand how each variable influences the final result.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SIBHIRAAJ R
RegisterNumber: 212224230268
```
## HEAD VALUES:

```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
## Output:
<img width="811" height="144" alt="image" src="https://github.com/user-attachments/assets/af259483-caea-4b85-ace9-76cfdabb672f" />

## SALARY DATE:
```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
````
## Output:
<img width="814" height="154" alt="image" src="https://github.com/user-attachments/assets/9a5b3de0-129f-4237-9e16-b98cc3e00ccc" />

## Checking Null function:
```
d1.isnull().sum()
```
## Output:
<img width="304" height="754" alt="image" src="https://github.com/user-attachments/assets/f7ed3ffa-759d-411a-a31a-d8c2f64a77ed" />

## Duplicate data:
```
d1.duplicated().sum()
```
## Output:
<img width="239" height="44" alt="image" src="https://github.com/user-attachments/assets/a565071e-fff0-4ef2-b478-7f5744a17c5e" />

## Data status:
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```
## Output:
<img width="814" height="353" alt="image" src="https://github.com/user-attachments/assets/1f0d5df7-fc30-47f6-9f63-4df1104b6fd0" />

## X data:
```
x=d1.iloc[:, : -1]
```
## Output:
<img width="814" height="378" alt="image" src="https://github.com/user-attachments/assets/6f9518bf-c585-4c6b-be9b-ccde3dbd8233" />

## Y data:
```
y=d1["status"]
y
```
## Output:
<img width="281" height="686" alt="image" src="https://github.com/user-attachments/assets/34970d9b-381d-4c48-9052-5710fac78806" />

## Y prediction value:
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```
## Output:
<img width="817" height="96" alt="image" src="https://github.com/user-attachments/assets/c6fd3048-c290-41e7-a658-76e453e90cf6" />

## Accuracy value:
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
## Output:
<img width="705" height="95" alt="image" src="https://github.com/user-attachments/assets/3e710744-9198-4ca5-a93a-63e7d70fff84" />

## Confusion matrix:
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```
## Output:
<img width="295" height="67" alt="image" src="https://github.com/user-attachments/assets/cbb12d8d-c76f-4657-8f5d-f0da454a7984" />

## Classification report:
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
## Output:
<img width="781" height="251" alt="image" src="https://github.com/user-attachments/assets/604f1339-d189-412a-a807-54016c7dd884" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
