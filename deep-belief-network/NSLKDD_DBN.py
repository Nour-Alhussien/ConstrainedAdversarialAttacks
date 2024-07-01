import sys

import numpy as np
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1337)
from sklearn.model_selection import train_test_split
from sklearn.metrics._classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification

columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes'
    , 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in'
    , 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations'
    , 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login'
    , 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate'
    , 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate'
    , 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate'
    , 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate'
    , 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])

df_train = pd.read_csv(r'\data_set\NSL-KDD\KDDTrain+.txt',
                       header=None, names=columns)
df_test = pd.read_csv(r'\data_set\NSL-KDD\KDDTest+.txt',
                      header=None, names=columns)

df_train["Attack"] = df_train.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_train.drop('attack', axis=1, inplace=True)
df_test["Attack"] = df_test.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_test.drop('attack', axis=1, inplace=True)
print(df_train['Attack'].value_counts())
le = preprocessing.LabelEncoder()

clm = ['protocol_type', 'service', 'flag', 'Attack']
for x in clm:
    df_train[x] = le.fit_transform(df_train[x])
    df_test[x] = le.fit_transform(df_test[x])

x_train = df_train.drop('Attack', axis=1)
y_train = df_train["Attack"]

x_test = df_test.drop('Attack', axis=1)
y_test = df_test["Attack"]

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train_ = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test_ = x_test.reshape(x_test.shape[0], x_test.shape[1])
print(x_train.shape)
print(x_test.shape)
# (125973, 42, 1)


x_train = x_train_
y_train = y_train
x_test = x_test_
y_test = y_test

X_train = x_train
X_test = x_test
Y_train = y_train
Y_test = y_test


# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=300,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

model = DecisionTreeClassifier(max_depth=3)
model.fit(X=X_train, y=Y_train)
print("Training score for DT model  is : ", model.score(X_train, Y_train))
print("Testing score for DT model is : ", model.score(X_test, Y_test))

art_classifier = SklearnClassifier(model=model)
zoo = ZooAttack(classifier=art_classifier, confidence=0.5, targeted=False, learning_rate=1e-1, max_iter=80,
                binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=None,
                use_importance=None, nb_parallel=5, batch_size=1, variable_h=0.8)

x_test_adv = zoo.generate(X_test)
print("Testing_ADV score for DT model is : ", model.score(x_test_adv, Y_test))
dbn_pred_ad = classifier.predict(x_test_adv)
print("Testing_ADV score for BDN model is : ", accuracy_score(dbn_pred_ad, Y_test))


