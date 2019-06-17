'''
really informative docstring on this module goes here...
by Vassily
06/2019
'''

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import pathlib
from collections import namedtuple
import Utilities as utils

#default name to save the current model
DEFAULT_MODEL_NAME = "last_model.pkl"

#TODO: implement to keep the folders in order
#MODELS_FOLDER_PATH = "."

#custom defined type Model (namedtuple) to keep our model
Model = namedtuple('Model', ['clf', 'binarizer', 'vectorizer', 'transformer'])

class MLClassifierAPI(object):

    #class variable
    #todo: property
    model = None

    def train(self, data, filename="", modelname=""):
        """Gets training data or extracts it from given filename,
        trains the model on this data, saves the model and returns it

           Parameters
           ----------
           data : pd.DataFrame
               The data to train
           filename : str, optional
               Filename to load data from
           modelname : str, optional
                Name by which trained model will be saved

           Returns
           -------
           Model (namedtuple)
               a serializable model object which contains classifier, binarizer, vectorizer, transformer
        """
        #todo: refactor all this data/file loading
        if data is None: #no data was provided, try to get it by filename
            if not filename: #no data to work on, return
                print("Please provide data or data file.")
                return None
            if filename and not pathlib.Path(filename).exists(): #cannot locate the file
                print("File not found:", filename)
                return None
            try:
                data = self.loadData(filename)
            except:
                print("Data couldn't be loaded from file")

        #transform data into Text (X) and Labels (Y) for training
        Text, Labels  = self.extractXY(data)

        #transform labels
        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(Labels)
        Y = multilabel_binarizer.transform(Labels)

        # use resampling to overcome inbalanced classes
        ros = RandomOverSampler(random_state=42)
        ros.fit_sample(data, Y)
        Y = multilabel_binarizer.transform(Labels.iloc[ros.sample_indices_])

        #vectorize and transform texts, cut the outliers
        count_vect = CountVectorizer(min_df = 3, max_df = .99)
        X_counts = count_vect.fit_transform(Text.iloc[ros.sample_indices_])
        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)

        #pick algorithm
        lr = LogisticRegression(solver = 'liblinear')
        #nb_clf = MultinomialNB()
        #sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)

        clf = OneVsRestClassifier(lr)

        #perform grid search
        param_grid = {"estimator__C": np.logspace(-3, 3, 5), "estimator__penalty": ["l1", "l2"]}
        CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

        best_clf = CV_clf.fit(X_tfidf, Y)

        #create named tuple object for the model
        self.model = Model(clf=best_clf,binarizer=multilabel_binarizer, vectorizer=count_vect, transformer=tfidf_transformer)

        #save the model and return it
        self.saveModel(name=modelname)
        return self.model

    def evaluate(self, model, test_data):
        """Evaluates test data by given model and returns its scores and loss
                   Parameters
                   ----------
                   model : Model
                       The model to evaluate; if none is given, loads from memory
                   test_data : pd.DataFrame
                       The data to evaluate upon

                   Returns
                   -------
                   dict
                       a dictionary object which contain scores and loss
                """
        if model is None:
            self.loadModel()
            model = self.model
        #get labels = Y (ground truth) from data, no X is needed
        _ , Labels = self.extractXY(test_data)

        #vectorize labels
        Y_true = model.binarizer.transform(Labels)

        #get predicted labels
        Y_pred = self.classify(model, test_data)

        #calculate loss and scores
        #todo: decide where to keep all methods, here or in utils
        loss = hamming_loss(Y_true,Y_pred)
        score = utils.hamming_score(Y_true, Y_pred)
        accuracy = accuracy_score(Y_true, Y_pred)

        #return the result
        return {"hamming loss":loss, "hamming score":score, "accuracy":accuracy}

    def classify(self, model, data, textlabels=False):
        """Classifies given data by given model and returns labels, either vectorized or "real" (textual)
        This is convenient as the method can be called both from inside the API where vectorized result
        is preferred (for model evaluation) and from outside where user may want to see real labels

               Parameters
               ----------
               model : Model
                   The model to classify by; if none is given, loads from memory
               test_data : pd.DataFrame
                   The data to classify
               textlabels : bool, optional
                    If True, return text labels, otherwise - vector
               Returns
               -------
               list
                   a list of predictions (labels), vectorized (binary) or textual
            """
        #todo: validate data
        if model is None:
            self.loadModel()
            model = self.model

        #extract text (=X) to classify from data, labels (=Y) are irrelevant
        Text , _ = self.extractXY(data)

        #vectorize and transform
        #todo: check if we have vectorizer and transformer
        x_counts = model.vectorizer.transform(Text)
        x_tfidf = model.transformer.transform(x_counts)

        #classify = get predictions (labels)
        prediction = model.clf.predict(x_tfidf)

        if not textlabels:
            return prediction
        else:
            #if text is required, transfrom from binary vector
            return model.binarizer.inverse_transform(prediction)

    #not required explicitly but seems nice to have
    def classify_single_text(self, model, text):
        """Classifies given text by given model and returns its labels
               Parameters
               ----------
               model : Model
                   The model to clasify by; if none is given, loads from memory
               text : str
                   The text to classify

               Returns
               -------
               list
                   list of labels
            """
        if model is None:
            self.loadModel()
            model = self.model

        #iterable expected by vectorizer, so we wrap our text into list
        text = [text]

        #vectorize and transform
        x_counts = model.vectorizer.transform(text)
        x_tfidf = model.transformer.transform(x_counts)

        #classify
        prediction = model.clf.predict(x_tfidf)
        results = model.binarizer.inverse_transform(prediction)

        return results

    def extractXY(self, data):
        """Prepares data for the ML tasks - splits it into X and Y, preprocesses, and returns them
              Parameters
              ----------
              data : pd.DataFrame
                  The data to work upon

              Returns
              -------
              tuple
                  tuple of dataframes
           """
        #TODO: validate data - encoding, format
        Text = self.preproccess(data)
        Labels = data['Solution Type'].str.split(';')

        return Text, Labels

    @staticmethod
    def preproccess(data):
        """Prepprocesses data - gathers all relevant textual columns into one and cleans the text
              ----------
              data : pd.DataFrame
                  The data to work upon

              Returns
              -------
              pd.DataFrame
                  concatenated clean text
                   """
        #TODO: decide on cleaning numbers in ID columns
        text = data['Brand'].fillna("") + ' ' \
               + data['Relevancy'].fillna("") + ' ' \
               + data['Brand Raw'].fillna("") + ' ' \
               + data['Title'].fillna("") + ' ' \
               + data['Description'].fillna("") + ' ' \
               + data['About'].fillna("") + ' ' \
               + data["Source Product Identifier"].fillna("") + ' ' \
               + data["ASIN"].fillna("") + ' ' \
               + data["Source Ingredient"].fillna("") + ' ' \
               + data["Product Category"].fillna("") + ' ' \
               + data["Link"].fillna("")
        text.apply(lambda x: utils.clean_text(x))

        return text

    def saveModel(self, name=""):
        """Saves model by given name; if none given uses the default one
              Parameters
              ----------
              name : str, optional
                  The name to save the model by
        """
        if not name:
            name = DEFAULT_MODEL_NAME

        if self.model is not None:
            joblib.dump(self.model, name)

    def loadModel(self, name=""):
        """Loads model with given name; if none given, brings one from memory;
        if it doesn't exist either, loads the default one (which is also the last saved)
              Parameters
              ----------
              name : str, optional
                  The name of the model to load
        """
        if not name:
            if self.model is not None:  # we can use the one we have loaded
                print("In-memory model will be used.")
                return
            else:
                name = DEFAULT_MODEL_NAME  # get the default/last saved one
                print("Last saved model will be used.")
        try:
            self.model = joblib.load(name)

        except:
            print("Model couldn't be loaded.")

    def loadData(self, filename):
        """Loads data from given filename; if none given, uses the only one it has :)
                Parameters
                ----------
                name : filename
                    The data's filename
                Returns
                -------
                pd.DataFrame
                    loaded data

        """
        if not filename:
            filename = './nlp_eng_task_train.csv'
        data = pd.read_csv(filename, encoding='latin-1').sample(frac=1).drop_duplicates()
        return data

