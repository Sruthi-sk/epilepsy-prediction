# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:56:59 2022

@author: ws4
"""
#%% import libs

from keras.models import load_model



#%% convert to tflite to use in SBC

# Save model
saved_model_dir='1_epilepsy_codes/epilepsy_model'
model.save(saved_model_dir)
model = load_model(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

TFLITE_FILE_PATH=os.path.join(saved_model_dir,'model_1608.tflite')
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

from tensorflow.lite.tools.flatbuffer_utils import read_model
read_model(TFLITE_FILE_PATH)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
# my_signature = interpreter.get_signature_runner()
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# Test model on our test data
litemodel_preds = np.empty((0),int)
# input_data2 = np.tile(input_data,(2,1,1))
# for sampledata in input_data2:
for sampledata in testX:
    sample = np.float32(sampledata.reshape(input_shape))
    # print(sample.shape,sample.dtype)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data=np.argmax(output_data, axis=1)
    print(output_data)
    litemodel_preds=np.append(litemodel_preds,output_data)

testpredn_lite = np.array(litemodel_preds)
testy           = np.argmax(testy, axis=1)

# Compute F1 score
f1score   = metrics.f1_score(testy,testpredn_lite, average='weighted')
print(f1score)
print ("Classification Report: ")
print (metrics.classification_report(testy,testpredn_lite))
print ("Accuracy Score: ", metrics.accuracy_score(testy,testpredn_lite))

res= np.array([testy,testpredn_lite]).transpose()
labels_predicted=pd.DataFrame(testpredn_lite,columns=['class'])
print(get_distr(labels_predicted,'class'))



#%% For real time
# epochs = mne.make_fixed_length_epochs(eeg,duration=1,overlap=0,preload=True)
# EEG_epoch = epochs._data*1e6
# EEG_data = np.reshape(EEG_data,(-1,512))

# input_data=np.array(EEG_data)
# input_data= np.float32(input_data)

# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter.set_tensor(input_details,input_data)
# interpreter.invoke()
# output_data= interpreter.get_tensor(output_details[0])
# output=np.argmax(output_data)
# print(output)

