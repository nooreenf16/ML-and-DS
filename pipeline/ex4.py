# implement a classification pipeline which aim at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.


from xml.etree.ElementInclude import include
import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data


class DataPreprocessor(object):

    """ 
    This class is a mandatory API. More about its structure - few lines below.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages. 
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non descriptive columns
    3 ...

    The test data is unavailable when building the ML pipeline, thus it is necessary to determine the 
    preprocessing steps and values on the train set and apply them on the test set.


    *** Mandatory structure ***
    The ***fields*** are ***not*** mandatory
    The ***methods***  - "fit" and "transform" - are ***required***.

    You're more than welcome to use sklearn.pipeline for the "heavy lifting" of the preprocessing tasks, but it is not an obligation. 
    Any class that implements the methods "fit" and "transform", with the required inputs & outps will be accepted. 
    Even if "fit" performs no taksks at all.
    """

    def __init__(self):
        self.transformer: Pipeline = None

    def fit(self, dataset_df):
        """
        Input:
        dataset_df: the training data loaded from the csv file as a dataframe containing only the features
        (not the target - see the main function).

        Output:
        None

        Functionality:
        Based on all the provided training data, this method learns with which values to fill the NA's, 
        how to scale the features, how to encode categorical variables etc.

        *** This method will be called exactly once during evaluation. See the main section for details ***


        Note that implementation below is a boilerplate code which performs very basic categorical and numerical fields
        preprocessing.

        """

        # This section can be hard-coded
        # numerical_columns = ['Transportation expense', 'Height', ] # There are more - what else?
        # numerical_columns = ['Reason', 'Education']
        numerical_columns = ['Reason', 'Month', 'Day', 'Transportation expense',
                             'Residence Distance', 'Service time', 'Son', 'Pet', 'Weight', 'Height', 'Season']
        categorical_columns = ['Education', 'Smoker', 'Drinker', 'Age Group']
        # Handling Numerical Fields
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent"))
        ])

        # Handling Categorical Fields
        categorical_transformer = LabelEncoder()
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordenc', OrdinalEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("drop_columns", 'drop', 'ID'),
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns),
            ]
        )

        self.transformer = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])

        self.transformer.fit(dataset_df)

    def transform(self, df):
        """
        Input:
        df:  *any* data similarly structured to the train data (dataset_df input of "fit")

        Output: 
        A processed dataframe or ndarray containing only the input features (X).

        It should maintain the same row order as the input.
        Note that the labels vector (y) should not exist in the returned ndarray object or dataframe.


        Functionality:
        Based on the information learned in the "fit" method, apply the required transformations to the passed data (df)

        """
        return self.transformer.transform(df)
        # think about if you would like to add additional computed columns.


def train_model(processed_X, y):
    """
    This function gets the data after the pre-processing stage  - after running DataPreprocessor.transform on it, 
    a vector of labels, and returns a trained model. 

    Input:
    processed_X (ndarray or dataframe): the data after the pre-processing stage
    y: a vector of labels

    Output:
    model: an object with a "predict" method, which accepts the ***pre-processed*** data and outputs the prediction


    """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(processed_X, y)

    return model


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    train_csv_path = 'time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)

    X_train = train_dataset_df.iloc[:, :-1]
    y_train = train_dataset_df['TimeOff']
    preprocessor.fit(X_train)
    model = train_model(preprocessor.transform(X_train), y_train)

    ### Evaluation Section ####
    # test_csv_path = 'time_off_data_test.csv'
    # Obviously, this will be different during evaluation. For now, you can keep it to validate proper execution
    test_csv_path = train_csv_path
    test_dataset_df = load_dataset(test_csv_path)

    X_test = test_dataset_df.iloc[:, :-1]
    y_test = test_dataset_df['TimeOff']

    processed_X_test = preprocessor.transform(X_test)
    predictions = model.predict(processed_X_test)
    test_score = accuracy_score(y_test, predictions)
    print("test:", test_score)

    predictions = model.predict(preprocessor.transform(X_train))
    test_score = accuracy_score(y_train, predictions)
    print('train:', test_score)
