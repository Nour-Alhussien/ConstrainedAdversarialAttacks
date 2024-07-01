import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np

np.random.seed(1337)
from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.concat(objs=[
    pd.read_csv(
        r"\data_set\training_testing\UNSW_TR.csv"),
    pd.read_csv(r"\data_set\training_testing\UNSW_TS.csv")
], copy=False, sort=False, ignore_index=True)

df_zero = df[df["label"] == 0]
df_one = df[df["label"] == 1]

print('Number of one :', df_one.shape, 'Number of zero :', df_zero.shape)

df = pd.concat([df_one.iloc[:10000, :], df_zero.iloc[:10000, :]], axis=0)

list_drop = ['id', 'attack_cat']
df.drop(list_drop, axis=1, inplace=True)

df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')

DEBUG = 0

for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = ' + str(df_numeric[feature].max()))
        print('75th = ' + str(df_numeric[feature].quantile(0.95)))
        print('median = ' + str(df_numeric[feature].median()))
        print(df_numeric[feature].max() > 10 * df_numeric[feature].median())
        print('----------------------------------------------------')
    if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
        df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')

df_numeric = df.select_dtypes(include=[np.number])
df_before = df_numeric.copy()
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique() > 50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique() > 50:
        if df_numeric[feature].min() == 0:
            df[feature] = np.log(df[feature] + 1)
        else:
            df[feature] = np.log(df[feature])

df_numeric = df.select_dtypes(include=[np.number])

df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')

DEBUG = 0
for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique() > 6)
        print(sum(df[feature].isin(df[feature].value_counts().head().index)))
        print('----------------------------------------------------')

    if df_cat[feature].nunique() > 6:
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')

df['proto'].value_counts().head().index
df['proto'].value_counts().index


best_features = SelectKBest(score_func=chi2, k='all')

X = df.iloc[:, 4:-2]
y = df.iloc[:, -1]
fit = best_features.fit(X, y)

df_scores = pd.DataFrame(fit.scores_)
df_col = pd.DataFrame(X.columns)

feature_score = pd.concat([df_col, df_scores], axis=1)
feature_score.columns = ['feature', 'score']
feature_score.sort_values(by=['score'], ascending=True, inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X.head()
feature_names = list(X.columns)
np.shape(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
np.shape(X)
df_cat.describe(include='all')
len(feature_names)

for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)

for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)

for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)

len(feature_names)

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=2000,
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

zoo = ZooAttack(classifier=art_classifier, confidence=0.5, targeted=False, learning_rate=1e-1, max_iter=80,
                binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=None,
                use_importance=None, nb_parallel=5, batch_size=1, variable_h=0.8)

x_test_adv = zoo.generate(X_test)
print("Testing_ADV score for DT model is : ", model.score(x_test_adv, Y_test))
dbn_pred_ad = classifier.predict(x_test_adv)
print("Testing_ADV score for BDN model is : ", accuracy_score(dbn_pred_ad, Y_test))

