import warnings

warnings.filterwarnings('ignore')
import numpy as np
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys

np.random.seed(1337)
from sklearn.model_selection import train_test_split
from sklearn.metrics._classification import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification

data = pd.read_csv(r'\data_set\SIMARGL\dataset-part1.csv')
non_numaric_column = []
df_zero = data[data["LABEL"] == "SYN Scan - aggressive"]
df_one = data[data["LABEL"] == 'Normal flow']
print('Number of one :', df_one.shape, 'Number of zero :', df_zero.shape)
data = pd.concat([df_one.iloc[:50000, :], df_zero.iloc[:50000, :]], axis=0)
print(data['LABEL'].value_counts())

label = data['LABEL']

print(label.value_counts())
lab = preprocessing.LabelEncoder()
label = lab.fit_transform(label)
# label = pd.DataFrame(label)
# print(label.value_counts())


for name, dtype in data.dtypes.iteritems():
    if dtype == 'object':
        non_numaric_column.append(name)
        data.drop(name, axis=1, inplace=True)

std_scaler = StandardScaler()
data = std_scaler.fit_transform(data.to_numpy())

X_train, X_test, Y_train, Y_test = train_test_split(data, label,
                                                    test_size=0.3,
                                                    random_state=404)

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=15,
                                         n_iter_backprop=50,
                                         batch_size=300,
                                         activation_function='sigmoid',
                                         dropout_p=0.21)

classifier.fit(X_train, Y_train)
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


