import argparse
import os
import tensorflow as tf
import numpy as np 
import tensorflow.keras as keras
from utils import load_of_data,load_data_txt
from metrics import euclidean_distance_square_loss, smooth_accuracy, score 
from sklearn.metrics import roc_curve, auc,classification_report

# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--split_dir', help='Directory for split')
parser.add_argument('-m', '--m', default=3, type=int, help='Number of optical flow pairs per input (default=3)')
parser.add_argument('-d', '--model_dir', default='./saved_models', help='Directory to save trained models')
parser.add_argument('-n', '--model_name', help='Model name to test, e.g.) DCAE, DSVDD, IO-GEN')
parser.add_argument('-v', '--verbose', default=1, help='verbose option, either 0 or 1')
parser.add_argument('-tr','--train_dir')
parser.add_argument('-te','--test_dir')
parser.add_argument('-i','--img_path')
parser.add_argument('-o','--throw_out',type=int,default=1,help='throw out either 1 for true or 0 for false') 

options = parser.parse_args()

split_dir = options.split_dir
m = options.m
model_dir = options.model_dir
model_name = options.model_name
verbose = options.verbose
train_path = options.train_dir
test_path = options.test_dir
img_path = options.img_path
throw_out_ano = options.throw_out
throw_out_ano = (False,True)[throw_out_ano==1]
#result = (on_false, on_true)[condition]

# necessary arguments 
assert img_path != None, 'Please Specifify img_path, -i argument' 
#assert model_name != None, 'Please specify the directory of split to use. Use "-s" argument in execution' 
class_names = ['Normal','Anomalous']
# load data
#train_path = "/content/gdrive/MyDrive/Masters/Datasets/data/dataset_filenames/train_oc_full.txt"
#test_path = "/content/gdrive/MyDrive/Masters/Datasets/data/dataset_filenames/test_oc_full.txt"
# load data
train_x, test_stable_x, test_unstable_x = load_data_txt(train_path,test_path,img_path,throw_out_ano)

# unstable_x locations to confine in time   
n_test_samples = [0, 666, 1333, 4000, 6666, 9333, len(test_unstable_x)]
days = ['D+1', 'D+2', 'D+3 - D+6', 'D+7 - D+10', 'D+11 - D+14', 'D+15 - D+18']  

# test for different models 
print('AUC Scores') 
if model_name == 'DCAE':

    # load model
    ae = keras.models.load_model('./{}/DCAE.h5'.format(model_dir))
    encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('encoded').output)                   
    decoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('decoded').output)                   
    y_test_stable_hat = score(ae.predict(test_stable_x), test_stable_x)

   
    
    y_test_unstable_hat = score(ae.predict(test_unstable_x), test_unstable_x)
    true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    
    fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
    auc_score = auc(fpr, tpr)
    full = np.concatenate([y_test_stable_hat, y_test_unstable_hat],axis=-1)
    print('ALL: {}'.format(auc_score))
    #MSE not between 0 - 1 


    pred_min = np.min(full) 
    pred_max = np.max(full) 
    interval = (pred_max - pred_min) / 1000.
    
    thresholds = np.arange(pred_min,pred_max,interval)
    f1_best = 0.0
    best_thresh = 0.0
    for thr in thresholds:
      pred_class = [1 if x > thr else 0 for x in full]
      curr = classification_report(true_labels, pred_class, target_names=class_names,output_dict=True,zero_division=0)
      #Zero Division=0 same as default but doesnt raise warnings
      ana = curr[class_names[1]]
      f1 = ana['f1-score']
      if f1 > f1_best:
        best_thresh = thr
        f1_best = f1


    print(best_thresh)
    final_pred = [1 if x > best_thresh else 0 for x in full]
    print(classification_report(true_labels, final_pred, target_names=class_names))
    
    
    auc_score = auc(fpr, tpr)
    print('ALL: {}'.format(auc_score))
    print(train_path)
    print(model_name)

elif model_name == 'DSVDD': 

    # load model
    ae = keras.models.load_model('./{}/DCAE.h5'.format(model_dir))
    encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('encoded').output)                   
    dsvdd = keras.models.load_model('./{}/DSVDD.h5'.format(model_dir), \
           custom_objects={'euclidean_distance_square_loss':euclidean_distance_square_loss})

    # Compute Center Feature
    initial_outputs = encoder.predict(train_x)
    center_feat = np.mean(initial_outputs, axis=0)
    target_feat = np.expand_dims(center_feat, 0) 

    y_test_stable_hat = score(dsvdd.predict(test_stable_x), target_feat)
    print("Stable",y_test_stable_hat)
    

    class_names = ['Normal','Anomalous']
    
    y_test_unstable_hat = score(dsvdd.predict(test_unstable_x), target_feat)
    print("Unstable ",y_test_unstable_hat)

    full = np.concatenate([y_test_stable_hat, y_test_unstable_hat],axis=-1)
    true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    
    fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
    pred_min = np.min(full) 
    pred_max = np.max(full) 
    interval = (pred_max - pred_min) / 1000.
    thresholds = np.arange(pred_min,pred_max,interval)
    f1_best = 0.0
    best_thresh = 0.0
    for thr in thresholds:
      pred_class = [1 if x > thr else 0 for x in full]
      curr = classification_report(true_labels, pred_class, target_names=class_names,output_dict=True,zero_division=0)
      #Zero Division=0 same as default but doesnt raise warnings
      ana = curr[class_names[1]]
      f1 = ana['f1-score']
      if f1 > f1_best:
        best_thresh = thr
        f1_best = f1


    print(best_thresh)
    final_pred = [1 if x > best_thresh else 0 for x in full]
    print(classification_report(true_labels, final_pred, target_names=class_names))
    
    
    auc_score = auc(fpr, tpr)
    print('ALL: {}'.format(auc_score))
    print(train_path)
    print(model_name)
    
elif model_name == 'IO-GEN': 

    # load model
    cls = keras.models.load_model('./{}/CLASSIFIER.h5'.format(model_dir), \
          custom_objects={'smooth_accuracy': smooth_accuracy, 'keras': keras})

    y_test_stable_hat = cls.predict(test_stable_x).flatten()
 
    y_test_unstable_hat = cls.predict(test_unstable_x).flatten() 
    full = np.concatenate([y_test_stable_hat, y_test_unstable_hat],axis=-1)
    true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    
    fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
    auc_score = auc(fpr, tpr)
    
    pred_min = np.min(full) 
    pred_max = np.max(full) 
    interval = (pred_max - pred_min) / 1000.
    
    thresholds = np.arange(pred_min,pred_max,interval)
    
    f1_best = 0.0
    best_thresh = 0.0
    for thr in thresholds:
      pred_class = [1 if x > thr else 0 for x in full]
      meme = classification_report(true_labels, pred_class, target_names=class_names,output_dict=True)
      ana = meme[class_names[1]]
      f1 = ana['f1-score']
      if f1 > f1_best:
        best_thresh = thr
        f1_best = f1


    print(best_thresh)
    final_pred = [1 if x > best_thresh else 0 for x in full]
    print(classification_report(true_labels, final_pred, target_names=class_names))
    print('ALL: {}'.format(auc_score))
    print(train_path)
    print(model_name)

else:
    print('Not appropriate model name') 
