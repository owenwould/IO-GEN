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


def load_data_txt(train_path,test_path,img_path,throw_out_ano,print_count=False):
    train_x = []
    test_stable_x = []
    test_unstable_x = []
    
    train_file = open(train_path,'r')
    test_file = open(test_path,'r')
    count = 0
    ano_prefix = "ano" 
    normal_unripe_prefix = "nup"
    
    #Load Train Data
    while True:
        train_line = train_file.readline().strip()
        if not train_line:
            break
        slashIndex = train_line.index('/')
        labCode = train_line[slashIndex+1:slashIndex+4]
        count += 1

        if print_count:
            if count % 100 == 0:
                print(count)

        flow = []
       
        flow = np.asarray(Image.open(os.path.join(img_path,train_line)))
        flow = tf.convert_to_tensor(flow)
        flow = flow.numpy().astype("float32") / 127.5 - 1
        if labCode == normal_unripe_prefix:
            continue ##normal_unripe is contains duplications of normal and unripe 


        if labCode == ano_prefix:
            if throw_out_ano:
                #Should the anomalous which is in training be put into test or 
                #not used at all 
                continue
            else:
                test_unstable_x.append(flow) #put ano from train into unstable test
        else:
            train_x.append(flow)
    #train_x = np.transpose(np.array(train_x), (0,1,3,1))
    count = 0
    while True:
    #Load Test Data
        test_line = test_file.readline().strip()
        if not test_line:
            break

        count += 1
        
        if print_count:
            if count % 100 == 0:
                print(count)
        
        flow_te = []
        flow_te = np.asarray(Image.open(os.path.join(img_path,test_line)))
        flow_te = tf.convert_to_tensor(flow_te)
        flow_te = flow_te.numpy().astype("float32") / 127.5 - 1
        slashIndex = test_line.index('/')
        labCode = test_line[slashIndex+1:slashIndex+4]
        if labCode == normal_unripe_prefix:
            continue ##normal_unripe is contains duplications of normal and unripe 

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






def load_data_txt_with_filenames(train_path,test_path,img_path,case_type):
    train_x = []
    test_stable_x = []
    temp_anpo_x = []
    test_unstable_x = []
    ano_keys = []
    unripe_keys = []
    nor_keys = []
    unripe_occ_keys = []
    ripe_occ_keys = []


    #case type one = single (unripe or ripe)
    #case type two = double (unripe + ripe)
    #case type three = full (ripe + unripe +  )

    fn_test_nor = []
    fn_test_ano = []
    fn_train_ano = []

    
    train_file = open(train_path,'r')
    test_file = open(test_path,'r')
    index = 0
    ano_prefix = "ano" 
    normal_unripe_prefix = "nup"
    
    #Load Train Data
    while True:
        train_line = train_file.readline().strip()
        if not train_line:
            break
        slashIndex = train_line.index('/')
        labCode = train_line[slashIndex+1:slashIndex+4]
        
        if labCode == normal_unripe_prefix:
            continue ##normal_unripe is contains duplications of normal and unripe     
       

        flow = []
       
        flow = np.asarray(Image.open(os.path.join(img_path,train_line)))
        flow = tf.convert_to_tensor(flow)
        flow = flow.numpy().astype("float32") / 127.5 - 1
        

        if labCode == ano_prefix:
            temp_anpo_x.append(flow) #put ano from train into unstable test
            basename = os.path.basename(train_line)
            fn_train_ano.append(basename)

        else:
            train_x.append(flow)
        
        index += 1
   

    index = 0
    while True:
    #Load Test Data
        test_line = test_file.readline().strip()
        if not test_line:
            break

        flow_te = []
        flow_te = np.asarray(Image.open(os.path.join(img_path,test_line)))
        flow_te = tf.convert_to_tensor(flow_te)
        flow_te = flow_te.numpy().astype("float32") / 127.5 - 1
        slashIndex = test_line.index('/')
        labCode = test_line[slashIndex+1:slashIndex+4]
        basename = os.path.basename(test_line)
        if labCode == normal_unripe_prefix:
            continue ##normal_unripe is contains duplications of normal and unripe 

        if labCode == ano_prefix:
            test_unstable_x.append(flow_te)
            fn_test_ano.append(basename)
        else:
            test_stable_x.append(flow_te)
            fn_test_nor.append(basename)
            
            if case_type == 0:
                nor_keys.append(index) #Single
            elif case_type == 1:
                if labCode == "nor":
                    nor_keys.append(index)
                else:
                    unripe_keys.append(index)
            elif case_type == 2:
                if labCode == "nor":
                    nor_keys.append(index)
                elif labCode == "unp":
                    unripe_keys.append(index)
                elif labCode == "noc":
                    ripe_occ_keys.append(index)
                else:
                    unripe_occ_keys.append(index)
            
            index += 1 #Keeps filenames aligned with keys , ano will be added at end 

       
       
    start_point = len(fn_test_nor)
    test_unstable_x.extend(temp_anpo_x)
    fn_test_ano.extend(fn_train_ano)

    for ind,x in enumerate(test_unstable_x):
        key = start_point + ind
        ano_keys.append(key)
    
    train_x = np.array(train_x)
    test_stable_x = np.array(test_stable_x)
    test_unstable_x = np.array(test_unstable_x)

    fn_test_nor = np.array(fn_test_nor)
    fn_test_ano = np.array(fn_test_ano)
    filenames = np.concatenate((fn_test_nor,fn_test_ano))

    keys = []

    if case_type == 0:
        keys = [nor_keys,ano_keys]
    elif case_type == 1:
        keys = [nor_keys,unripe_keys,ano_keys]
    else:
        keys = [nor_keys,unripe_keys,ano_keys,ripe_occ_keys,unripe_occ_keys]
    

    
    return test_stable_x, test_unstable_x,filenames,keys



