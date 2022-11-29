# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: A.SANJAI
RegisterNumber: 212220040142  
*/
import chardet
file = "/content/spam.csv"
with open(file,'rb') as rawdata:
	result = chardet.detect(rawdata.read(10000))
result
import pandas as pd
dataset = pd.read_csv("/content/spam.csv",encoding="windows-1252")
dataset.head()
dataset.info()
dataset.isnull().sum()
x=dataset["v1"].values
y=dataset["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer() 
x_train=cv.fit_transform(x_train) 
x_test=cv.transform(x_test) 
from sklearn.svm import SVC 
svc=SVC() 
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test) 
y_pred
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y_pred) 
accuracy
```

## Output:
![image](https://user-images.githubusercontent.com/95969295/204564612-bdac4dd0-fe60-4b02-ba85-46723d35bfe0.png)

![image](https://user-images.githubusercontent.com/95969295/204564665-bf3c341b-1e01-4a2a-ba12-8123448b5ffa.png)

![image](https://user-images.githubusercontent.com/95969295/204564713-05bd8df7-e044-4738-af2b-f7107d0e840b.png)

![image](https://user-images.githubusercontent.com/95969295/204564775-81586fe5-96ab-48e9-a341-e69c01b629f6.png)

![image](https://user-images.githubusercontent.com/95969295/204564855-244a5146-fec1-4a68-bfae-777be9881de3.png)

![image](https://user-images.githubusercontent.com/95969295/204564897-fbd89acb-0ab9-4dda-9904-67c044c422d6.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
