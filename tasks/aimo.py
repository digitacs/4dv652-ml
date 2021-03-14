import d6tflow
import luigi
from joblib import dump
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

class TaskGetAimoScoresData(d6tflow.tasks.TaskPqPandas):  # save dataframe as parquet

    def run(self):
         # Load the data
        AimoScore = pd.read_excel('./datasets/AimoScore_WeakLink_big_scores.xls')
        print("AimoScore size: {}".format(len(AimoScore)))
        self.save(AimoScore)  # quickly save dataframe


class TaskGetWeakestLinkData(d6tflow.tasks.TaskPqPandas):  # save dataframe as parquet

    def run(self):
         # Load the data
        WeakLinks = pd.read_excel('./datasets/scores_and_weak_links.xlsx', engine='openpyxl')
        print("WeakLinks size: {}".format(len(WeakLinks)))
        self.save(WeakLinks)  # quickly save dataframe


@d6tflow.requires({'AimoScore':TaskGetAimoScoresData, 'WeakestLink':TaskGetWeakestLinkData})  # define dependency
class TaskMergeData(d6tflow.tasks.TaskPqPandas):  # save dataframe as parquet

    def run(self):
        # Load the data
        AimoScore = self.inputLoad()['AimoScore']
        WeakestLink = self.inputLoad()['WeakestLink']
        merged = pd.merge(AimoScore, WeakestLink,on=["ID"])
        self.save(merged)
    
@d6tflow.requires(TaskMergeData)
class TaskPrepareData(d6tflow.tasks.TaskPqPandas):  
    targetScore = luigi.Parameter(default="Target")
    targetClassifier = luigi.Parameter(default="Target")
    columnsToDiscard = luigi.Parameter(default=[])


    def correlatedColumnsFirstHalf(self,data,minThreshold):
        corr = data.corr()
        pairs = []
        keep = []

        for i in corr.columns:
            for j in corr[i][abs(corr[i])>=minThreshold].index:
                if i != j:
                    if i not in keep:
                        if(i != self.targetScore):
                            pairs.append(i)
                            keep.append(j)
        
        return pairs

    def run(self):
        # Load the data
        data = self.inputLoad()
        # Find the weakest link for each data point
        data[self.targetClassifier] = data.loc[:,"ForwardHead":"RightHeelRises"].idxmax(axis=1)
        # Discard extra columns
        data.drop(data.columns.to_series()["ForwardHead":"RightHeelRises"], axis=1, inplace=True)
        # Remove Duplicates
        data = data.drop_duplicates(subset="ID")
        # Remove clusters with very few samples
        temp = (data[self.targetClassifier].value_counts()<10)
        temp = temp[~temp == False]
        data = data[~data[self.targetClassifier].isin(temp.index)]

        # Removing the identical variables
        data = data.drop(
            columns=self.correlatedColumnsFirstHalf(data,minThreshold=1)
        )

        # Removing the symmetric variables
        data = data.drop(
            columns=self.correlatedColumnsFirstHalf(data,minThreshold=0.8)
        )

        #Discard extra columns
        for column in self.columnsToDiscard:
            data = data.drop(
                columns=[column]
            )

        self.save(data)

@d6tflow.requires(TaskPrepareData)
class TaskSavePreparedDataToCsv(d6tflow.tasks.TaskPqPandas):  
    datasetFileName = luigi.FloatParameter(default="preparedData")

    def run(self):
        # Load the data
        data = self.inputLoad()
        data.to_csv('./datasets/'+self.datasetFileName+'.csv')


@d6tflow.requires(TaskPrepareData)
class TaskTrainAndTestSplit(d6tflow.tasks.TaskPqPandas):  
    trainingToTestRate = luigi.FloatParameter(default=0.8)
    # So that you get the same results
    seed = luigi.IntParameter(default=0)
    persist = ['X_train', 'X_test', 'y_train', 'y_test']

    def run(self):
        # Load the data
        data = self.inputLoad()

        y_train,y_test ,X_train, X_test  = train_test_split(
            data[["WeakLinks","AimoScore"]], 
            data.drop(columns=["WeakLinks","AimoScore"]), 
            train_size = self.trainingToTestRate,
            random_state = self.seed
        )
        
        self.save({'X_train': X_train, 'X_test': X_test,
                   'y_train': y_train, 'y_test': y_test})
    

@d6tflow.requires(TaskTrainAndTestSplit)
class TaskTrainSVM(d6tflow.tasks.TaskPickle):  
    kernel = luigi.Parameter(default="linear")
    c = luigi.FloatParameter(default=-1)
    gamma = luigi.FloatParameter(default=-1)
    degree = luigi.FloatParameter(default=1)
    targetClassifier = luigi.Parameter(default="Target")

    def run(self):
        # Load the data
        X_train = self.inputLoad()['X_train']
        y_train = self.inputLoad()['y_train']

        if(self.c != -1 and self.gamma != -1):
            model = SVC(kernel=self.kernel,C=self.c, gamma=self.gamma, degree=self.degree)
        elif (self.c != -1):
            model = SVC(kernel=self.kernel,C=self.c, degree=self.degree)
        elif (self.gamma != -1):
            model = SVC(kernel=self.kernel,gamma=self.gamma, degree=self.degree)
        else:
            model = SVC(kernel=self.kernel, degree=self.degree)

        # Create and Train the model
        model.fit(X_train, y_train[self.targetClassifier])

        self.save(model)


@d6tflow.requires(TaskTrainSVM)
class TaskSaveModel(d6tflow.tasks.TaskPickle):  
    modelName = luigi.Parameter(default="model")
    saveAs = luigi.Parameter(default="joblib")

    def run(self):
        model = self.inputLoad()

        if(self.saveAs == "joblib"):
            dump(model, './models/'+self.modelName+'.joblib') 
        else:
            pickle_out = open('./models/'+self.modelName+'.pickle',"wb")
            pickle.dumps(model,pickle_out)
            pickle_out.close()
