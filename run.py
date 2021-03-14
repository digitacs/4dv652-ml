import d6tflow
import cfg
from tasks import aimo
import visualize

params_model = {
    'modelName':'aimo_v13',
    'saveAs':'joblib',
    'targetScore': 'AimoScore',
    'targetClassifier': 'WeakLinks',
    'aimoModel': 'ols', 
    'weakestLinkModel': 'gls', 
    'datasetFileName':'AimoScoreWeakLinks',
    'checkCorrelationWith'
    'trainingToTestRate': 0.8,
    'columnsToDiscard':["ID","Date","EstimatedScore"],
    'kernel':'poly',
    'c':0.6,
    'gamma':0.5,
    'degree':3
}

# run workflow for model 
d6tflow.run(aimo.TaskPrepareData(**params_model))
d6tflow.run(aimo.TaskSavePreparedDataToCsv(**params_model))
d6tflow.run(aimo.TaskTrainAndTestSplit(**params_model))
d6tflow.run(aimo.TaskTrainSVM(**params_model))

# TODO: Update save method, pass model to task as an argument instead
d6tflow.run(aimo.TaskSaveModel(**params_model))


# compare results from new model
model = aimo.TaskTrainSVM(**params_model).output().load()
X_test = aimo.TaskTrainAndTestSplit(**params_model).output()["X_test"].load()
y_test = aimo.TaskTrainAndTestSplit(**params_model).output()["y_test"].load()["WeakLinks"]

print("Model accuracy: ",model.score(X_test, y_test))
# 0.59
