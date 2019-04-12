

from sklearn.model_selection import train_test_split
import os
import numpy
import pandas as pd
import mlflow.sklearn
from sklearn.svm import SVC

df=pd.read_csv('Iris.csv', delimiter = ',')
X=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

svn = SVC()
svn.fit(X_train, y_train)
yPrediction = svn.predict(X_test)

cwd = os.getcwd()
model_path=cwd+'/model'
mlflow.sklearn.save_model(svn, model_path,serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)