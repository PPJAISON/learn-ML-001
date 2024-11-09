import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses  import dataclass

from src.components.data_transormation import DataTransformation
from src.components.data_transormation import DataTransformationConfig


@dataclass
class DataIngetionConfig:
    #train_data_path: str=os.path.join('')
    train_data_path: str= '/home/bcl6/projects/dev-ops/learn-and-POC/ML-MachineLearning/10-20241022-c/01-ml-with-krishna/artifacts/train.csv'
    test_data_path:  str= '/home/bcl6/projects/dev-ops/learn-and-POC/ML-MachineLearning/10-20241022-c/01-ml-with-krishna/artifacts/test.csv'
    row_data_path:   str= '/home/bcl6/projects/dev-ops/learn-and-POC/ML-MachineLearning/10-20241022-c/01-ml-with-krishna/artifacts/data.csv'


class DataIngestion:
    def __init__(self):
        self.ingetion_config= DataIngetionConfig()

    def initiate_data_ingestion(self):
        print("Entered the data ingestion method")
        logging.info(" Entered the data ingestion method")

        try:
            df=pd.read_csv('/home/bcl6/projects/dev-ops/learn-and-POC/ML-MachineLearning/mlproject-Kirsh-Naik-folder/notebook/data/stud.csv')
            logging.info(' Read the stud data set')


            df.to_csv(self.ingetion_config.row_data_path,index=False, header=True )
            logging.info(' Writing row data')

            train_set, test_set= train_test_split(df, test_size=0.2,random_state=43)

            train_set.to_csv(self.ingetion_config.train_data_path,index=False, header=True )
            test_set.to_csv(self.ingetion_config.test_data_path,index=False, header=True )


            logging.info(' Ingestion of the data is completed')

            return(
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    print("inside data_ingestion")
    obj=DataIngestion()

    #obj.initiate_data_ingestion()
    train_data, test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)