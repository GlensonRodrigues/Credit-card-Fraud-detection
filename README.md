# Credit-card-Fraud-detection
# Import all required docs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# read the file in csv format
df=pd.read_csv("creditcard[1].csv")
df **(#to display the file)**

# check for Null values, verify Dtypes, etc. 
df.info()


# check for skewness
df.describe() 

# check for data distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

![download](https://github.com/GlensonRodrigues/Credit-card-Fraud-detection/assets/143869767/28c727ea-e55f-4bf0-9c9b-180edf236319)


df.corr()

# Our data is Int & Float values no Preprocessing is required

# dividing data between Features & Traget
features= df.iloc[:,:-1]
target=df.iloc[:,-1]

# imported train test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.3,random_state=1)

# classification data hence used Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
a=accuracy_score(ytest,ypred)
cm=confusion_matrix(ytest,ypred)
cr=classification_report(ytest,ypred)
print(a)
print(cm)
print(cr)

    accuracy                           1.00     85443
   macro avg       0.81      0.82      0.82     85443
weighted avg       1.00      1.00      1.00     85443

  
