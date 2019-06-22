import numpy as np
from mnist import MNIST
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from imblearn.metrics import geometric_mean_score
import time
from termcolor import colored
import pickle

from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def load_classifiers(pickle_name):
    pickle_in = open(pickle_name, "rb")
    loaded = pickle.load(pickle_in)
    pickle_in.close()
    return loaded


def save_classifier(model, file_name='best_classifier'):
    pickle_out = open(file_name, "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


train_dataset = load_classifiers('fruit_dataset_train')
train_dataset['data'] = np.array(train_dataset['data'])
train_dataset['target'] = np.array(train_dataset['target'])
test_dataset = load_classifiers('fruit_dataset_test')
test_dataset['data'] = np.array(test_dataset['data'])
test_dataset['target'] = np.array(test_dataset['target'])


def print_metrics(expected, predicted):
    accurracy = accuracy_score(expected, predicted)
    f1score = f1_score(expected, predicted, average='macro')
    gmean_score = geometric_mean_score(expected, predicted)
    print(colored("Accuracy: ", 'green'), accurracy)
    print(colored("F1-Score Macro: ", 'green'), f1score)
    print(colored("Geometric Mean Score: ", 'green'), gmean_score)
    print(colored("Confusion Matrix: \n", 'green'), confusion_matrix(expected, predicted))
    print(colored("Classification Report: \n", 'green'), classification_report(expected, predicted))


train_data = train_dataset['data']
train_targets = train_dataset['target']

test_data = test_dataset['data']
test_targets = test_dataset['target']


#knn = KNeighborsClassifier(n_neighbors=5)
#knn.fit(train_data, train_targets)
#print('KNN Score: ', knn.score(test_data, test_targets))

ext_forest = ExtraTreesClassifier(n_estimators=200, n_jobs=-1)
ext_forest.fit(train_data, train_targets)
print('ExtraTree Score: ', ext_forest.score(test_data, test_targets))

#ext_forest_predicted = ext_forest.predict(test_data)
#print_metrics(test_targets, ext_forest_predicted)


randFor_class = ensemble.RandomForestClassifier(n_estimators=200,
                                                random_state=1)
randFor_class.fit(train_data, train_targets)
score = randFor_class.score(test_data, test_targets)

print('RandomForest Score: ', randFor_class.score(test_data, test_targets))
#randFor_pred = ext_forest.predict(test_data)
#print_metrics(test_targets, randFor_pred)


ensemble = VotingClassifier(estimators=[
    ('RandomForest', randFor_class), ('ExtraTrees', ext_forest)],
                            voting='soft')
ensemble.fit(train_data, train_targets)
print('Ensamble Score: ', ensemble.score(test_data, test_targets))

#ensemble_predicted = ensemble.predict(test_data)
#print_metrics(test_targets, ensemble_predicted)

save_classifier(ext_forest, file_name='best_classifier')