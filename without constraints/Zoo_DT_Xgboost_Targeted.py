import random
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(1337)
import sys
from sklearn.metrics._classification import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd

sys.path.insert(1, r'C:\Users\proam\PycharmProjects\DBN(UNSW)\deep-belief-network')
import numpy as np

from dbn.tensorflow import SupervisedDBNClassification
from dbn.zoo import ZooAttack
from dbn.carlini import CarliniL2Method
from dbn.deepfool import DeepFool
from sklearn.tree import DecisionTreeClassifier
np.random.seed(1337)
from art.estimators.classification import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

f = open("GBC_ZOO_with.txt", "a")
#  Load Dataset
df = pd.read_csv(r'C:\Users\proam\PycharmProjects\DBN(UNSW)\UNSW-NB15_1.csv', header=None)
# df = df.iloc[0:1000]

#  Pre-processing
df = df.fillna(-1)
df.columns = [f'col_{i}_' for i in range(1, 50)]
df.col_1_ = df.col_1_.astype('str')
df = df[df.col_2_ != '0x000b']
df = df[df.col_2_ != '0x000c']
df = df[df.col_2_ != '-']
df.col_2_ = df.col_2_.apply(lambda x: int(x))
df.col_3_ = df.col_3_.astype('str')
df.col_5_ = df.col_5_.astype('str')
df.col_6_ = df.col_6_.astype('str')
df.col_14_ = df.col_14_.astype('str')
df.col_48_ = df.col_48_.astype('str')
df = df[df.col_4_ != '0xc0a8']
df = df[df.col_4_ != '0x20205321']
df = df[df.col_4_ != '-']
df.col_4_ = df.col_4_.apply(lambda x: int(x))
df = df.reset_index(drop=True)
x_train = df.drop(columns=['col_1_', 'col_3_', 'col_48_', 'col_49_']).copy()
y_train = df['col_49_'].copy()
features_list = x_train.columns.tolist()
port_state_condition = {'udp': ['CON'],
                        'arp': ['INT'],
                        'tcp': ['FIN'],
                        'icmp': ['ACC', 'CLO', 'ECO', 'ECR', 'MAS', 'RST', 'TST', 'URH', 'URN', 'no', 'TXD', 'PAR'],
                        'ospf': ['REQ'],
                        }

port_service_condition = {'tcp': ['tcp', 'ftp', 'smtp', 'ftp-data', 'ssh', 'pop3', 'irc', 'ssl'],
                          'udp': ['dns', 'radius', 'snmp', 'dhcp'],
                          }
special_feature = {'dependent_feature': ['col_36_', 'col_20_', 'col_6_', 'col_14_', 'col_33_'],
                   'binary_feature': ['col_39_', 'col_36_'],
                   'nominal_feature': ['col_5_', 'col_6_', 'col_14_'],
                   }

# One Hot Encode the 3 nominal attributes and drop them
enc_5 = OneHotEncoder(handle_unknown='ignore')
enc_6 = OneHotEncoder(handle_unknown='ignore')
enc_14 = OneHotEncoder(handle_unknown='ignore')

nominal_feature = special_feature['nominal_feature']
one_encoder_list = [enc_5, enc_6, enc_14]
for column_name, enc in zip(nominal_feature, one_encoder_list):
    # Create the One Hot Encode DataFrame
    dum = enc.fit_transform(x_train.loc[:, [column_name]]).toarray()
    # Insert into the dataset DataFrame by Series
    for i in range(dum.shape[-1]):
        x_train.insert(x_train.shape[1], column_name + str(i), dum[:, i])
        x_train[column_name + str(i)] = x_train[column_name + str(i)].astype('int32')
    # Drop the old attribute's column
    x_train.drop(column_name, inplace=True, axis=1)

feature_index = {f: np.where(x_train.columns.str.contains(f) == True)[0] for f in features_list}
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(x_train)
x_train = minmax_scaler.transform(x_train)
y_train = y_train.values
x_test = minmax_scaler.transform(x_test)
y_test = y_test.values

test_sample_indxe = np.where(y_test == 1)
# test_sample_indxe_0 = np.where(y_test == 0)
# test_sample_indxe = list(test_sample_indxe)
# test_sample_indxe_0 = list(test_sample_indxe_0)
# merge_lists = np.append(test_sample_indxe[0], test_sample_indxe_0[0][0:len(test_sample_indxe[0])], axis=0)

x_test_sample = x_test[test_sample_indxe].copy()
y_test_sample = y_test[test_sample_indxe].copy()

# Make the dataset balanced
train_indxe_1 = np.where(y_train == 1)[0]
train_indxe_0 = np.where(y_train == 0)[0]
train_indxe_0 = np.random.choice(train_indxe_0, size=len(train_indxe_1), replace=False)
train_indxe = np.concatenate((train_indxe_1, train_indxe_0), axis=None)

test_indxe_1 = np.where(y_test == 1)[0]
test_indxe_0 = np.where(y_test == 0)[0]
test_indxe_0 = np.random.choice(test_indxe_0, size=len(test_indxe_1), replace=False)
test_indxe = np.concatenate((test_indxe_1, test_indxe_0), axis=None)

x_train = x_train[train_indxe].copy()
y_train = y_train[train_indxe].copy()
x_test = x_test[test_indxe].copy()
y_test = y_test[test_indxe].copy()



# //////////////////// DT model ///////////////////
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

art_classifier_dt = SklearnClassifier(model=dt, clip_values=(0, 1))
adversarial_crafter_Zoo_dt = ZooAttack(classifier=art_classifier_dt,
                                       confidence=0.5,
                                       targeted=False,
                                       learning_rate=1e-1,
                                       max_iter=80,
                                       binary_search_steps=10,
                                       initial_const=1e-3,
                                       abort_early=True,
                                       use_resize=False,
                                       use_importance=False,
                                       nb_parallel=1,
                                       batch_size=1,
                                       variable_h=0.2,
                                       feature_index=feature_index,
                                       special_feature=special_feature,
                                       minmax_scaler=minmax_scaler,
                                       one_encoder_list=one_encoder_list,
                                       port_state_condition=port_state_condition,
                                       port_service_condition=port_service_condition,
                                       apply_constraints=False)

x_test_adv_dt = adversarial_crafter_Zoo_dt.generate(x_test_sample.copy())


from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0)

GBC.fit(x_train, y_train)

print('\n[......', 'Results of Zoo on GBC model', '......]\n', file=f)

predictions_test = GBC.predict(x_test_adv_dt)

accuracy = accuracy_score(y_test_sample, predictions_test)
recall = recall_score(y_test_sample, predictions_test, average='weighted')
f1s = f1_score(y_test_sample, predictions_test, average='weighted')
print("\naccuracy:", accuracy, file=f)
print("recall:", recall, file=f)
print("f1s:", f1s, file=f)
print('-------' * 5, file=f)



# art_classifier_GBC = SklearnClassifier(model=GBC, clip_values=(0, 1))
# adversarial_crafter_Zoo_GBC = ZooAttack(classifier=art_classifier_GBC,
#                                         confidence=0.5,
#                                         targeted=False,
#                                         learning_rate=1e-1,
#                                         max_iter=80,
#                                         binary_search_steps=10,
#                                         initial_const=1e-3,
#                                         abort_early=True,
#                                         use_resize=False,
#                                         use_importance=False,
#                                         nb_parallel=1,
#                                         batch_size=1,
#                                         variable_h=0.2,
#                                         feature_index=feature_index,
#                                         special_feature=special_feature,
#                                         minmax_scaler=minmax_scaler,
#                                         one_encoder_list=one_encoder_list,
#                                         port_state_condition=port_state_condition,
#                                         port_service_condition=port_service_condition,
#                                         apply_constraints=True)
#
# x_test_adv_GBC = adversarial_crafter_Zoo_GBC.generate(x_test_sample.copy())
#
#
# length = len(x_train)
# random_int = random.sample(range(0, length), int(len(x_test_adv_dt) * 3.9))
# x_temp_portion = x_train[np.array(random_int)]
# mixed_x = np.append(x_test_adv_dt, x_temp_portion, axis=0)
# y_temp_portion = y_train[np.array(random_int)]
# mixed_y = np.append(y_test_sample, y_temp_portion, axis=0)
#
# GBC.fit(mixed_x, mixed_y)
#

#
# print('\n[......', 'Results of Zoo on GBC model (Trained on mixed)', '......]\n', file=f)
# number_attack_sample, number_fooled, asr = attack_success_rate(x_test_adv_GBC, y_test_sample, GBC)
# print(f'sample length:{len(y_test_sample)}', file=f)
# print(f'number of attack sample:{number_attack_sample}', file=f)
# print(f'number of fooled ones:{number_fooled}', file=f)
# print(f'attack_success_rate:{asr}', file=f)
#
# predictions_test = GBC.predict(x_test_adv_GBC)
#
# accuracy = accuracy_score(y_test_sample, predictions_test)
# recall = recall_score(y_test_sample, predictions_test, average='weighted')
# f1s = f1_score(y_test_sample, predictions_test, average='weighted')
# print("\naccuracy:", accuracy, file=f)
# print("recall:", recall, file=f)
# print("f1s:", f1s, file=f)
# print('-------' * 5, file=f)
