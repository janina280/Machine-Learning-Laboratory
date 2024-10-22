import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

trainData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
testData = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

trainData = trainData.drop('Id', axis=1)
testData = testData.drop('Id', axis=1)

numericalColumns = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
trainData = trainData.select_dtypes(include=numericalColumns)
testData = testData.select_dtypes(include=numericalColumns)

X = trainData.drop(['SalePrice'], axis=1)
y = trainData['SalePrice']

X = X.fillna(0)

XTrain, XValidate, yTrain, yValidate = train_test_split(X, y, test_size = 0.25)

model = LinearRegression()
model.fit(XTrain, yTrain)

yPrediction = model.predict(XValidate)
plt.scatter(yValidate, yPrediction)
plt.title('Predicted vs Real Sale Price')
plt.xlabel('Real Sale Price')
plt.ylabel('Predicted Price')

print("The score is: ")
print(model.score(XValidate, yValidate))

XTest = testData.fillna(0)
testPrediction = model.predict(XTest)

sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
salePrediction = pd.DataFrame({'Id': sample.Id, 'SalePrice': testPrediction})
salePrediction.to_csv('submission.csv', index=False)