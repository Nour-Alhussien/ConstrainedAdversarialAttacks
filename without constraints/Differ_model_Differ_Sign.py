import random
import warnings

import torch
from sklearn.tree import DecisionTreeClassifier
import copy
import time as time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(1337)
import sys
from sklearn.metrics._classification import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd

sys.path.insert(1, r'C:\Users\proam\PycharmProjects\DBN(UNSW)\deep-belief-network')
import numpy as np
from art.estimators.classification import PyTorchClassifier
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

f = open("Diff_Diff_with_20.txt", "a")
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

# Convert from numpy array to torch tensors
x_train_torch = torch.from_numpy(x_train).float()
y_train_torch = torch.from_numpy(y_train).long()
x_test_torch = torch.from_numpy(x_test).float()
y_test_torch = torch.from_numpy(y_test).long()




class Network(nn.Module):
    ''' A basic neural network model '''

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # python2 : super(MLP, self).__init__()
        super().__init__()  # python2 : super(MLP, self).__init__()d
        # defining the network's operations
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x, softmax=False):
        a = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.float())))))
        if softmax:
            y_pred = F.softmax(a, dim=1)
        else:
            y_pred = a

        return y_pred


"""Define a function to compute the accuracy of the prediction

"""


def evaluate_predictions(predictions, real):
    '''
    Evaluates the accuracy of the predictions
    '''
    n_correct = torch.eq(predictions, real).sum().item()
    accuracy = n_correct / len(predictions) * 100
    return accuracy


"""Train the model

"""
# Set fixed seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Initialising the model
input_size = x_train_torch.shape[1]
hidden_size = [256, 256, 128]
output_size = 2
model_1 = Network(input_size, hidden_size, output_size)
# Setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on : {}".format(device))

# Transferring model and data to GPU
model_1 = model_1.to(device)
x_train_torch = x_train_torch.to(device)
y_train_torch = y_train_torch.to(device)
x_test_torch = x_test_torch.to(device)
y_test_torch = y_test_torch.to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer_1 = optim.Adam(model_1.parameters(), lr=lr)
best_model_weights_1 = copy.deepcopy(model_1.state_dict())
best_accuracy = 0.0
trace = pd.DataFrame(columns=['epoch', 'train_acc', 'test_acc'])
since = time.time()

# start training model 1

for epoch in range(100 + 1):
    # Forward pass
    y_pred = model_1(x_train_torch)
    _, predictions = y_pred.max(dim=1)
    accuracy_train = evaluate_predictions(predictions=predictions.long(), real=y_train_torch)
    loss = criterion(y_pred, y_train_torch)
    if epoch % 10 == 0:
        _, predictions_test = model_1(x_test_torch, softmax=True).max(dim=1)
        accuracy_test = evaluate_predictions(predictions=predictions_test.long(), real=y_test_torch)
        trace = trace.append([{'epoch': epoch,
                               'train_acc': accuracy_train,
                               'test_acc': accuracy_test}])
        if accuracy_test > best_accuracy:
            best_accuracy = accuracy_test
            best_model_weights_1 = copy.deepcopy(model_1.state_dict())
        # Displap statistics
        if epoch % 100 == 0:
            time_elapsed = time.time() - since
            print(
                "epoch: {0:4d} | loss: {1:.4f} | Train accuracy: {2:.4f}% | Test accuracy: {3:.4f}% [{4:.4f}%] | "
                "Running for : {5:.0f}m {6:.0f}s "
                    .format(epoch,
                            loss,
                            accuracy_train,
                            accuracy_test,
                            best_accuracy,
                            time_elapsed // 60,
                            time_elapsed % 60))
    optimizer_1.zero_grad()
    loss.backward()
    optimizer_1.step()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

classifier_1 = PyTorchClassifier(model=model_1, loss=criterion, optimizer=optimizer_1, input_shape=input_size,
                                 nb_classes=output_size)

adversarial_crafter_CarliniL2Method = CarliniL2Method(classifier_1,
                                                      confidence=0.5,
                                                      targeted=False,
                                                      learning_rate=0.01,
                                                      binary_search_steps=10,
                                                      max_iter=80,
                                                      initial_const=0.01,
                                                      max_halving=5,
                                                      max_doubling=5,
                                                      batch_size=128,
                                                      feature_index=feature_index,
                                                      special_feature=special_feature,
                                                      minmax_scaler=minmax_scaler,
                                                      one_encoder_list=one_encoder_list,
                                                      port_state_condition=port_state_condition,
                                                      port_service_condition=port_service_condition,
                                                      apply_constraints=False)

x_test_adv_model_1 = adversarial_crafter_CarliniL2Method.generate(x_test_sample.copy())

# /////////////////////////////////////////////////

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0)
GBC.fit(x_train, y_train)
art_classifier_GBC = SklearnClassifier(model=GBC, clip_values=(0, 1))
adversarial_crafter_Zoo_GBC = ZooAttack(classifier=art_classifier_GBC,
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

x_test_adv_GBC = adversarial_crafter_Zoo_GBC.generate(x_test_sample.copy())



length = len(x_train)
random_int = random.sample(range(0, length), int(len(x_test_adv_model_1) * 4))
x_temp_portion = x_train[np.array(random_int)]
mixed_x = np.append(x_test_adv_model_1, x_temp_portion, axis=0)

y_temp_portion = y_train[np.array(random_int)]
mixed_y = np.append(y_test_sample, y_temp_portion, axis=0)

GBC.fit(mixed_x, mixed_y)



print('\n[......', 'Results of Zoo on GBC model (Trained on mixed)', '......]\n', file=f)


predictions_test = GBC.predict(x_test_adv_GBC)

accuracy = accuracy_score(y_test_sample, predictions_test)
recall = recall_score(y_test_sample, predictions_test, average='weighted')
f1s = f1_score(y_test_sample, predictions_test, average='weighted')
print("\naccuracy:", accuracy, file=f)
print("recall:", recall, file=f)
print("f1s:", f1s, file=f)
print('-------' * 5, file=f)
