import d6tflow
import os
import yaml
import cfg
from tasks import aimo
import mlflow
import visualize
import argparse
import time

# Define the program description
descriptionText = '''Aimo Machin Learning Project.
 This is the main entry point of the project and in order to train a model you should pass the configuration file path as a argument'''


# Initiate the parser
parser = argparse.ArgumentParser(description=descriptionText)
parser.add_argument("-V", "--version", help="show program version", action="store_true")

# Add configuration file path argument
parser.add_argument("-c", "--config", help="set the configuration", metavar='file', type=str)

# Read arguments from the command line
args = parser.parse_args()

# Check for --version or -V
if args.version:
    print("AIMO ML Version "+cfg.version)
elif args.config:
    try:
        with open(args.config) as configFile:
            config = yaml.safe_load(configFile)
    except:
        pass

    # run workflow for model 
    config = cfg.flatten_config(config)
    model_type = config["training_model_type"]

    # Log all parameters in MLFlow
    for key in config:
        mlflow.log_param(key, config[key])

    # Train the model
    start_time = time.time()
    trainer = aimo.TrainModel()
    model, accuracy = trainer.train(config)
    process_time = time.time() - start_time
    

    # compare results from new model
    print("Model accuracy: ",accuracy)

    # Log metric in MLFlow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("process_time", process_time)

    # Save the model in MLFlow
    mlflow.sklearn.log_model(model, "model")
