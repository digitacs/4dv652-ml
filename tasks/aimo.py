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
    preparation_score_target = luigi.Parameter(default="AimoScore")
    preparation_weakestlink_target = luigi.Parameter(default="WeakLinks")
    preparation_weakestlink_removeLessThan = luigi.FloatParameter(default=0)
    preparation_cleanup_identicalThreshold = luigi.FloatParameter(default=1)
    preparation_cleanup_symmetricThreshold = luigi.FloatParameter(default=0.95)
    preparation_columnsToDiscard = luigi.Parameter(default="")


    def correlatedColumnsFirstHalf(self,data,minThreshold):
        corr = data.corr()
        pairs = []
        keep = []

        for i in corr.columns:
            for j in corr[i][abs(corr[i])>=minThreshold].index:
                if i != j:
                    if i not in keep:
                        if(i != self.preparation_score_target):
                            pairs.append(i)
                            keep.append(j)
        
        return pairs

    def run(self):
        # Load the data
        data = self.inputLoad()
        # Find the weakest link for each data point
        data[self.preparation_weakestlink_target] = data.loc[:,"ForwardHead":"RightHeelRises"].idxmax(axis=1)
        # Discard extra columns
        data.drop(data.columns.to_series()["ForwardHead":"RightHeelRises"], axis=1, inplace=True)
        # Remove Duplicates
        data = data.drop_duplicates(subset="ID")
        # Remove clusters with very few samples
        temp = (data[self.preparation_weakestlink_target].value_counts() < self.preparation_weakestlink_removeLessThan)
        temp = temp[~temp == False]
        data = data[~data[self.preparation_weakestlink_target].isin(temp.index)]

        # Removing the identical variables
        data = data.drop(
            columns=self.correlatedColumnsFirstHalf(data, minThreshold=self.preparation_cleanup_identicalThreshold)
        )

        # Removing the symmetric variables
        data = data.drop(
            columns=self.correlatedColumnsFirstHalf(data, minThreshold=self.preparation_cleanup_symmetricThreshold)
        )

        #Discard extra columns
        for column in self.preparation_columnsToDiscard.split(","):
            if len(column) == 0:
                continue
            data = data.drop(
                columns=[column]
            )

        self.save(data)

@d6tflow.requires(TaskPrepareData)
class TaskSavePreparedDataToCsv(d6tflow.tasks.TaskPqPandas):  
    output_dataset_fileName = luigi.FloatParameter(default="preparedData")

    def run(self):
        # Load the data
        data = self.inputLoad()
        data.to_csv('./datasets/'+self.output_dataset_fileName+'.csv')


@d6tflow.requires(TaskPrepareData)
class TaskTrainAndTestSplit(d6tflow.tasks.TaskPqPandas):  
    training_trainingToTestRate = luigi.FloatParameter(default=0.8)
    training_seed = luigi.IntParameter(default=0)
    persist = ['X_train', 'X_test', 'y_train', 'y_test']

    def run(self):
        # Load the data
        data = self.inputLoad()

        y_train,y_test ,X_train, X_test  = train_test_split(
            data[["WeakLinks","AimoScore"]], 
            data.drop(columns=["WeakLinks","AimoScore"]), 
            train_size = self.training_trainingToTestRate,
            random_state = self.training_seed
        )
        
        self.save({'X_train': X_train, 'X_test': X_test,
                   'y_train': y_train, 'y_test': y_test})
    

@d6tflow.requires(TaskTrainAndTestSplit)
class TaskTrainSVM(d6tflow.tasks.TaskPickle):  
    training_model_parameters_kernel = luigi.Parameter(default="linear")
    training_model_parameters_c = luigi.FloatParameter(default=-1)
    training_model_parameters_gamma = luigi.FloatParameter(default=-1)
    training_model_parameters_degree = luigi.FloatParameter(default=1)
    training_target = luigi.Parameter(default="Target")

    def run(self):
        # Load the data
        X_train = self.inputLoad()['X_train']
        y_train = self.inputLoad()['y_train']

        if( self.training_model_parameters_c != -1 and self.training_model_parameters_gamma != -1 ):
            model = SVC( 
                kernel=self.training_model_parameters_kernel,
                C=self.training_model_parameters_c,
                gamma=self.training_model_parameters_gamma,
                degree=self.training_model_parameters_degree
            )
        elif ( self.training_model_parameters_c != -1):
            model = SVC(
                kernel=self.training_model_parameters_kernel,
                C=self.training_model_parameters_c,
                degree=self.training_model_parameters_degree
            )
        elif ( self.training_model_parameters_gamma != -1):
            model = SVC(
                kernel=self.training_model_parameters_kernel,
                gamma=self.training_model_parameters_gamma,
                degree=self.training_model_parameters_degree
            )
        else:
            model = SVC(
                kernel=self.training_model_parameters_kernel,
                degree=self.training_model_parameters_degree
            )

        # Create and Train the model
        model.fit(X_train, y_train[self.training_target])

        self.save(model)


class TrainModel(object):
    def train(self,config):
            self.config = config
            method_name='train_'+str(self.config["training_model_type"])
            method=getattr(self,method_name,lambda :'Invalid')

            if(self.config["output_dataset_fileName"]):
                d6tflow.run(TaskSavePreparedDataToCsv(**self.config))

            d6tflow.run(TaskTrainAndTestSplit(**self.config))
            self.X_test = TaskTrainAndTestSplit(**self.config).output()["X_test"].load()
            self.y_test = TaskTrainAndTestSplit(**self.config).output()["y_test"].load()["WeakLinks"]
            return method()
    
    def save(self, model):
        if(self.config["output_model_fileName"]):
            if(self.config["output_model_saveAs"] == "joblib"):
                dump(model, './models/'+self.config["output_model_fileName"]+'.joblib') 
            else:
                pickle_out = open('./models/'+self.config["output_model_fileName"]+'.pickle',"wb")
                pickle.dumps(model,pickle_out)
                pickle_out.close()

    def train_SVM(self):
            d6tflow.run(TaskTrainSVM(**self.config))
            model = TaskTrainSVM(**self.config).output().load()
            self.save(model)
            return (model, model.score(self.X_test, self.y_test))
