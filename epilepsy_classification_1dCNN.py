#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:46:21 2022

@author: sruthisk

Code to change 
- get all perictal channels
- Train-Test split - separate patients data

Done
- Simulate real-time classification of each 2s eeg data

"""

#%% import libs
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn import metrics 
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


dir_to_save='1_epilepsy_codes/'

datadir_perictal = 'EpilepsyData/perictal_modified_dataset'
files_perictal = os.listdir(datadir_perictal)

# datadir_cnt = 'Control_Rest'
datadir_cnt = 'NonSpikeEpilepticData/Blinded_ControlData'
files_cnt1 = os.listdir(datadir_cnt)
files_cnt = [f for f in files_cnt1 if 'edf' in f]
datadir_interictal = 'NonSpikeEpilepticData/Blinded_PatientData'
files_interictal1 = os.listdir(datadir_interictal)
files_interictal = [f for f in files_interictal1 if 'edf' in f]

#########################################################################################################
#%% extract epochs

os.chdir(datadir_perictal)
annotation_desc_2_event_id = {'perictal': 44}
epochs_all = []

chunk_len=2
# TO DO: Instead of choosing these channels - use alll channels - cant make epochs but just add to dataframe or 2d array
sel_channels=['P10','F10','F8','F3','T9','P4', 'T10','P9','O1','F9','T5','O2',
              'T4','T6','F7', 'C4','Fz','C3','T3','Pz','P3','Cz'] #, 'F4' - marked as bad in last file
epochs_all = []
for file in files_perictal:
    raw = mne.io.read_raw_fif(file, preload=True)
    raw,ref_data=mne.io.set_eeg_reference(raw,['A1','A2'] )
    # raw = raw.drop_channels(['A1','A2'])
    assert raw.info['sfreq'] == 256
    raw=raw.pick_channels(sel_channels)
    events_train, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=chunk_len)  # chunk_duration = 5? but for overlapping??
    epochs = mne.Epochs(raw=raw, events=events_train, event_id=annotation_desc_2_event_id, tmin=0., tmax=chunk_len, baseline=None,preload=True)  #reject=reject_criteria gave no epochs for most files
    # epochs.apply_function(normalize_channel_data,channel_wise=True, verbose=True) # zscore the data
    epochs_all.append(epochs)
    
epochs_periictal = mne.concatenate_epochs( epochs_all) #info['nchan'] must match
epochs_periictal.load_data()  # Resampling to save memory.
# epochs_periictal.plot(scalings='auto')
# epochs_periictal_data=epochs_periictal._data
# epochs_periictal[0].plot()
# periictal_data = epochs_periictal_data.reshape(-1,epochs_periictal_data.shape[-1])
# plt.plot(periictal_data)
#Save all epochs
# epochs_periictal.save(dir_to_save+'saved_periictal_epo.fif', overwrite=True)

reject_criteria = dict( eeg=200e-6)       # 150 µV      
flat_criteria = dict(  eeg=1e-6)           # 1 µV

epochs_all = []
os.chdir(datadir_interictal)
for file in files_interictal:
    raw = mne.io.read_raw_edf(file, preload=True)
    raw,ref_data=mne.io.set_eeg_reference(raw,['A1','A2'] )
    assert raw.info['sfreq'] == 256
    raw = raw.drop_channels(['A1','A2'])
    # raw=raw.pick_channels(sel_channels)
    start,stop=30,int(raw.times[-1]-30)
    epoch_events = mne.make_fixed_length_events(raw, start=start,stop=stop,duration=chunk_len*2)
    epochs = mne.Epochs(raw=raw, events=epoch_events, tmin=0., tmax=chunk_len, 
                    reject=reject_criteria, flat=flat_criteria,
                    baseline=None,preload=True)
    # if epochs:    
    #     epochs.apply_function(normalize_channel_data,channel_wise=True, verbose=True)
    epochs_all.append(epochs)
epochs_interictal = mne.concatenate_epochs( epochs_all) #info['nchan'] must match

os.chdir(datadir_cnt)
epochs_all = []
for file in files_cnt:
    raw = mne.io.read_raw_edf(file, preload=True)
    raw,ref_data=mne.io.set_eeg_reference(raw,['A1','A2'] )
    assert raw.info['sfreq'] == 256
    raw = raw.drop_channels(['A1','A2'])
    # raw=raw.pick_channels(sel_channels)
    start,stop=30,int(raw.times[-1]-30)
    epoch_events = mne.make_fixed_length_events(raw, start=start,stop=stop,duration=chunk_len*2)
    epochs = mne.Epochs(raw=raw, events=epoch_events, tmin=0., tmax=chunk_len, 
                    reject=reject_criteria, flat=flat_criteria,
                    baseline=None,preload=True)
    # if epochs:
    #     epochs.apply_function(normalize_channel_data,channel_wise=True, verbose=True)
    epochs_all.append(epochs)
epochs_cnt = mne.concatenate_epochs( epochs_all) #info['nchan'] must match

############################################################################################################
#%% Preprocessing

eeg_periictal = epochs_periictal._data.reshape(-1,epochs_periictal._data.shape[-1])
eeg_interictal = epochs_interictal._data.reshape(-1,epochs_interictal._data.shape[-1])
eeg_cnt = epochs_cnt._data.reshape(-1,epochs_cnt._data.shape[-1])

srate = epochs_cnt.info['sfreq']

ilf = IsolationForest(contamination='auto', max_samples='auto', verbose=0, random_state=42)
good = ilf.fit_predict(eeg_interictal)
print("no of bad epochs = ",(good==-1).sum())
good_eeg_interictal = eeg_interictal[good==1]      

good = ilf.fit_predict(eeg_cnt)
print("no of bad epochs = ",(good==-1).sum())
good_eeg_cnt = eeg_cnt[good==1]  

good = ilf.fit_predict(eeg_periictal)
print("no of bad epochs = ",(good==-1).sum())
good_eeg_periictal = eeg_periictal[good==1] 

# Shuffle data
# np.random.shuffle(good_eeg_interictal) # inplace, return None
# np.random.shuffle(good_eeg_cnt)

# Equalize classes
n_ep = good_eeg_periictal.shape[0]
eeg_interictal_sel = good_eeg_interictal[:n_ep]
eeg_cnt_sel = good_eeg_cnt[:n_ep]
labeltypes = ["control","interictal","periictal"] 
num_classes=3

#%% 'Split data into train and test sets'
# Compile the data [concatenate both epochs fo both types]
data = np.concatenate((eeg_cnt_sel,eeg_interictal_sel,good_eeg_periictal))
labels = np.array((([0]*n_ep)  + ([1]*n_ep) + ([2]*n_ep)))

label_names = {0:'Control', 1: 'Interictal', 2: 'Peri-ictal'}

# data1 = np.expand_dims(data,axis=-1)
# data = np.expand_dims(minmax_scale(data,feature_range=(0, 1),axis=1),axis=-1)    # Normalized the data
# data = np.expand_dims(data,axis=-1) 

trainX, testX, trainy, testy = train_test_split(data, labels, test_size = 0.20)
# Convert labels into categorical array
trainy  = to_categorical(trainy,num_classes=3)
testy   = to_categorical(testy,num_classes=3)

#Normalize train test separately
trainX = np.expand_dims(minmax_scale(trainX,feature_range=(0, 1),axis=1),axis=-1)   
testX_orig = np.expand_dims(testX.copy(),axis=-1) 
testX = np.expand_dims(minmax_scale(testX,feature_range=(0, 1),axis=1),axis=-1)   

############################################################################################################
#%% 'Prepare a Tensorflow 1D Neural Network Model'
n_outputs, n_timesteps, n_features = trainX.shape
n_filters = 20

# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                           padding="same",input_shape=(n_timesteps,n_features)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                           padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                           padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                           padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                           padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=n_filters, kernel_size=3, activation='relu',
                           padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(50, activation='relu'),
    # tf.keras.layers.BatchNormalization()),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# Compile model
# model.compile(loss='mae', optimizer='adam',metrics='accuracy')
# opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20,restore_best_weights=True)

# Summarise the model
model.summary()
    

#%% Train the model
history_1 = model.fit(trainX, trainy, epochs=120, batch_size=32, 
                      verbose=1,validation_split=0.2, shuffle=True) #,callbacks=[es]
plt.figure()
plt.plot(history_1.history["loss"], label="Training Loss")
plt.plot(history_1.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

############################################################################################################
#%% Test the accuracy of the Model

testpredn       = model.predict(testX)

# Reconvert categorical array into label vector
# trainy          = np.argmax(trainy, axis=1)
testy           = np.argmax(testy, axis=1)
testpredn       = np.argmax(testpredn, axis=1)

# Compute F1 score
f1score   = metrics.f1_score(testy,testpredn, average='weighted')
print(f1score)
print ("Classification Report: ")
print (metrics.classification_report(testy,testpredn))
print ("Accuracy Score: ", metrics.accuracy_score(testy,testpredn))

res= np.array([testy,testpredn]).transpose()


def get_distr(all_df,targetcol):
    vc = all_df[targetcol].value_counts().to_frame().reset_index()
    vc['percent'] = vc[targetcol].apply(lambda x : round(100*float(x) / len(all_df), 2))
    vc = vc.rename(columns = {"index" : "Target", targetcol : "Count"})
    print(vc)
# indexes = [i for i, j in enumerate(testpredn) if j == 0]  # wherever prediction was class 0
labels_predicted=pd.DataFrame(testpredn,columns=['class'])
print(get_distr(labels_predicted,'class'))

############################################################################################################
#%% plotting 5 epochs of each - Simulate real-time classification of each 2s eeg data

idx_cnt =  [i for i, x in enumerate(testy) if x == 0]   
idx_interictal = np.where(testy==1)[0]
idx_periictal = np.where(testy==2)[0]

# Take 5 eeg epochs of each class
sel_indices = np.hstack([idx_cnt[10:15], idx_interictal[10:15],idx_periictal[10:15]])
sel_true_labels = ["control"]*5+["interictal"]*5+["periictal"]*5 
sel_data = testX[sel_indices].copy()
sel_data_unscaled = testX_orig[sel_indices].copy()

# for i,sampledata in enumerate(sel_data):
#     plt.cla()
#     plt.plot(sampledata)
#     plt.title(sel_true_labels[i])
#     plt.pause(1)
        
  #%%
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# z = scaler.fit_transform(np.array([4.1525816e-35, 0.0000000e+00, 1.0000000e+00]).reshape(1,-1))

#%% real-time classification
# 2 sublots - 1 subplot is real-time eeg with true label , 2nd plot - class predictions (either type the class prediction or show % belonging to class)
# text_kwargs = dict(ha='center', va='center', fontsize=44, color='C1')

# plt.subplots(figsize=(10, 10))
# fig,ax = plt.figure()
for i in range(len(sel_data)):
    plt.cla()
    plt.plot(sel_data_unscaled[i])
    plt.title(sel_true_labels[i])
        
    sampledata = np.expand_dims(sel_data[i],axis=0)
    
    ## predict the class
    predict_value = model.predict(sampledata)
    print(predict_value)
    classv = np.argmax(predict_value)
    print(classv)
    preds = label_names[classv]
    loss_pred=f" Model Prediction: {preds }"
    plt.text(300,-18e-6, loss_pred,family='cursive', bbox = dict(facecolor = 'red', alpha = 0.5),size=10)  #, **text_kwargs
    # if preds=='Peri-ictal':
    #     plt.facecolor("#f09308")
    plt.ylim((-40e-6,40e-6))
    plt.ylabel('filtered EEG (2s)')
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(0, len(sel_data_unscaled[i]),srate/2)+1,[0,0.5,1,1.5,2])
    # ax.set_xticks(np.arange(0, len(sel_data_unscaled[i])+1, 5))
    plt.pause(1)
    plt.show()

#%%

import matplotlib.pyplot as plt

plt.subplots(figsize=(20, 20))
plt.text(0.6, 0.7, "eggs", size=100, 
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

plt.show()
