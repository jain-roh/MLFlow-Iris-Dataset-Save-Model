import mlflow
import mlflow.sklearn
import os
import pandas as pd
import numpy as np
import sys

SepalLengthCm=float(sys.argv[1])
SepalWidthCm=float(sys.argv[2])
PetalLengthCm=float(sys.argv[3])
PetalWidthCm=float(sys.argv[4])
df = pd.DataFrame(data=np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]]), columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm'])
mlflow.log_param("SepalLengthCm",SepalLengthCm)
mlflow.log_param("SepalWidthCm",SepalWidthCm)
mlflow.log_param("PetalLengthCm",PetalLengthCm)
mlflow.log_param("PetalWidthCm",PetalWidthCm)

cwd = os.getcwd()
model_path=cwd+'/model'
loaded_model=None
if os.path.isdir(cwd+"/model"):
        loaded_model = mlflow.sklearn.load_model(model_path)
else:
    import TrainAndSaveModel
    loaded_model = mlflow.sklearn.load_model(model_path)

yTPred=loaded_model.predict(df)
print(yTPred[0])








