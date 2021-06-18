import numpy as np
from PIL import Image
from glob import glob
import os
import pandas as pd
import tensorflow as tf

def load_of_data(split_dir, m, max_m=4):

    trains = pd.read_csv(os.path.join(split_dir, 'train.csv'), names=['paths'])
    tests = pd.read_csv(os.path.join(split_dir, 'test.csv'), names=['paths'])

    train_x = []
    test_stable_x = []
    test_unstable_x = []
    
    # train
    for name in trains['paths']: 
        x = []
        for i in range(m):
            folder, number = name.split('/')
            flow_x = folder + '/flow_x_' + number + '-{}.jpg'.format(i)
            flow_y = folder + '/flow_y_' + number + '-{}.jpg'.format(i)
            x.append(np.asarray(Image.open(flow_x).convert('L'))/255.*2.-1.)
            x.append(np.asarray(Image.open(flow_y).convert('L'))/255.*2.-1.)
        train_x.append(x)
    train_x = np.transpose(np.array(train_x), (0,2,3,1))
    
    # test_stable
    for name in tests['paths']: 
        x = []
        for i in range(m):
            folder, number = name.split('/')
            flow_x = folder + '/flow_x_' + number + '-{}.jpg'.format(i)
            flow_y = folder + '/flow_y_' + number + '-{}.jpg'.format(i)
            x.append(np.asarray(Image.open(flow_x).convert('L'))/255.*2.-1.)
            x.append(np.asarray(Image.open(flow_y).convert('L'))/255.*2.-1.)
        test_stable_x.append(x)
    test_stable_x = np.transpose(np.array(test_stable_x), (0,2,3,1))

    # test_unstable 
    flow_x_list = sorted(glob('./Unstable/flow_x_*.jpg'))
    flow_y_list = sorted(glob('./Unstable/flow_y_*.jpg'))
    
    for i in range(len(flow_x_list)//max_m):
        x = []  
        for j in range(m): 
            x.append(np.asarray(Image.open(flow_x).convert('L'))/255.*2.-1.)
            x.append(np.asarray(Image.open(flow_y).convert('L'))/255.*2.-1.)
        test_unstable_x.append(x)
    test_unstable_x = np.transpose(np.array(test_unstable_x), (0,2,3,1))

    return train_x, test_stable_x, test_unstable_x


def load_data_txt(train_path,test_path):
    train_x = []
    test_stable_x = []
    test_unstable_x = []
    img_path = "/content/gdrive/MyDrive/Masters/Datasets/data/Resize"
    train_file = open(train_path,'r')
    test_file = open(test_path,'r')
    count = 0
    ano_prefix = "ano" 
    
    #Load Train Data
    while True:
        train_line = train_file.readline().strip()
        if not train_line:
            break
        slashIndex = train_line.index('/')
        labCode = train_line[slashIndex+1:slashIndex+4]
        count += 1
        if labCode == ano_prefix:
            continue
        
        if count % 100 == 0:
            print(count)

        flow = np.asarray(Image.open(os.path.join(img_path,train_line)))
        
        #flow = np.asarray(Image.open(train_line))
        flow = tf.convert_to_tensor(flow)
        flow = flow.numpy().astype("float32") / 255.0
        train_x.append(flow)
    #train_x = np.transpose(np.array(train_x), (0,1,3,1))
    count = 0
    while True:
    #Load Test Data
        test_line = test_file.readline().strip()
        if not test_line:
            break

        count += 1
        if count % 100 == 0:
            print(count)

        #flow_te = np.asarray(Image.open(test_line))
        flow_te = np.asarray(Image.open(os.path.join(img_path,test_line)))
        flow_te = tf.convert_to_tensor(flow_te)
        flow_te = flow_te.numpy().astype("float32") / 255.0
        slashIndex = test_line.index('/')
        labCode = test_line[slashIndex+1:slashIndex+4]

        if labCode == ano_prefix:
            test_unstable_x.append(flow_te)
           
        else:
            test_stable_x.append(flow_te)
          
    

    train_x = np.array(train_x)
    test_stable_x = np.array(test_stable_x)
    test_unstable_x = np.array(test_unstable_x)

    #test_unstable_x = np.transpose(np.array(test_unstable_x), (0,1,3,1))
    #test_stable_x = np.transpose(np.array(test_stable_x), (0,1,3,1))
    return train_x, test_stable_x, test_unstable_x


