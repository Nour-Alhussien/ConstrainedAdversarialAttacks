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

df = pd.concat(objs=[
    pd.read_parquet(
        r'\WISENT-CIDDS-001\Parquet_format\cidds-001-openstack.parquet'),
    pd.read_parquet(
        r'\WISENT-CIDDS-001\Parquet_format\cidds-001-externalserver.parquet')
], copy=False, sort=False, ignore_index=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df = df.drop(columns=['label', 'attack_id'])
df['proto'] = df['proto'].astype('object')
df['proto'] = df['proto'].str.strip()
df['proto'] = df['proto'].astype('category')
df['proto'] = df['proto'].cat.codes
df['proto'] = df['proto'].astype(np.int32)
df['attack_type'] = df['attack_type'].astype('object')
df.loc[df['attack_type'] != 'benign', 'attack_type'] = 1
df.loc[df['attack_type'] == 'benign', 'attack_type'] = 0
print(df['attack_type'].value_counts())
df['attack_type'] = df['attack_type'].astype(dtype=np.int32)

target = 'attack_type'
conts = list(df.columns.difference([target]).values)

df_zero = df[df["attack_type"] == 0]
df_one = df[df["attack_type"] == 1]

print('Number of one :', df_one.shape, 'Number of zero :', df_zero.shape)
df_train = df.sample(frac=0.2, replace=False, random_state=505)
df_test = df.drop(index=df_train.index)


def xs_y(df_, targ):
    if not isinstance(targ, list):
        xs = df_[df_.columns.difference([targ])].copy()
    else:
        xs = df_[df_.columns.difference(targ)].copy()
    y = df_[targ].copy()
    return xs, y


x_train, y_train = xs_y(df_train, targ=target)
x_test, y_test = xs_y(df_test, targ=target)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(x_train.to_numpy())
X_test = std_scaler.fit_transform(x_test.to_numpy())
Y_train = pd.Series(y_train)
Y_test = pd.Series(y_test)


# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=100,
                                         activation_function='sigmoid',
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

zoo = ZooAttack(classifier=art_classifier, confidence=0.5, targeted=False, learning_rate=1e-1, max_iter=90,
                binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=None,
                use_importance=None, nb_parallel=5, batch_size=1, variable_h=0.8)

x_test_adv = zoo.generate(X_test)
print("Testing_ADV score for DT model is : ", model.score(x_test_adv, Y_test))
dbn_pred_ad = classifier.predict(x_test_adv)
print("Testing_ADV score for BDN model is : ", accuracy_score(dbn_pred_ad, Y_test))
