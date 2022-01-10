# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 00:34:41 2022

@author: noga mudrik
"""

#%% Imports
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
from os.path import dirname, abspath
import os 
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import linear_model
d = dirname(dirname(abspath(__file__)))
os.chdir(d)
import starboost_up
path_data = r'E:\datasetsDART'
from pathlib import Path
import pickle
#%% Create data


def load_data(data_name = 'breast_cancer', drop = True, path_data = r'E:\datasetsDART'):
    """
    Options for data_name:
        - breast_cancer
        - iris
        - wine
        - CT
        - student

    """
    if data_name == 'breast_cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
        problem_type = 'classification'
    elif data_name == 'iris':
        X, y = datasets.load_iris(return_X_y=True) 
        problem_type = 'multi_classification'
    elif data_name == 'wine':
        X, y = datasets.load_wine(return_X_y=True) 
        X = X[y<2,:]
        y = y[y<2]
        problem_type = 'classification'
    elif data_name == 'CT':
        data_all = pd.read_csv(path_data+ '\slice_localization_data_regression\slice_localization_data.csv')
        y = data_all['reference'] 
        cols = data_all.describe().columns
        remove_cols = [col for col in cols if col not in ('reference','patientId')]
        X = data_all[remove_cols]
        split_id = data_all['patientId']
        #X = pd.concat([X, data_all_patient], 1)
        problem_type = 'regression'
        return X,y, problem_type, split_id
    elif data_name == 'student':
        data_all = pd.read_csv(path_data+ '\student\student-mat.csv', sep=';')
        cols = data_all.describe().columns
        remove_cols = [col for col in cols if col not in ['G1','G2','G3']]
        y = data_all['G3'] 
        X = pd.get_dummies(data_all[remove_cols],  drop_first= drop)
        problem_type = 'regression'
    elif data_name == 'mushroom':
        data_all = pd.read_csv(path_data+ '\MushroomDataset\MushroomDataset\secondary_data.csv', sep=';')
        cols = data_all.describe().columns
        remove_cols = [col for col in cols if col != 'class']
        y = pd.get_dummies(data_all['class'] , drop_first = True)
        X = pd.get_dummies(data_all[remove_cols],  drop_first= drop)
        problem_type = 'classification'
    else:
        raise NameError('Unknown data')
    return X,y, problem_type

data_type = input('data type (or def)')
if data_type == 'def':
    X,y, problem_type = load_data( 'breast_cancer', True , path_data)
elif data_type == 'CT':
    X,y, problem_type, split_id = load_data(data_type, True , path_data)
else:
    X,y, problem_type = load_data(data_type, True , path_data)
#X, y = datasets.load_breast_cancer(return_X_y=True)    

#%% funmctions
def micro_f1_score(y_true, y_pred):
    """
    Calculate micro_f1_score
    """


    return metrics.f1_score(y_true, y_pred, average='micro')

def rmse(y_true, y_pred):
    """
    Calculate RMSE
    """
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5

def create_inner_path(list_path):
    if isinstance(list_path, list):
        real_path = ''
        for el in list_path:
            real_path = real_path + '/' + str(el)
    else:
        real_path = list_path
    Path(real_path).mkdir(parents=True, exist_ok=True)
    return real_path


def split_train_test_val(X,y, test_ratio = 0.2, val_ratio = 0, split_id = None, rand_seed = 0):
    """
    A function to split train and test, with an option to use an id column for reference (for instace, in case of different subjects etc.)

    X: data: [samples X features matrix]
    y: labels vector: [samples]
    test_ratio, val_ratio = ratios of the test set and validation set. If no val -> val_ratio =0
    split_id: id to split according to. [samples]
    """
    np.random.seed(0)
    if val_ratio > 0 and test_ratio == 0:
        val_ratio, test_ratio = test_ratio, val_ratio
    if test_ratio + val_ratio > 1:
        raise ValueError('Test and Val ratios should be <1')
    if split_id:
        n_test = int(np.floor(test_ratio *len(np.unique(split_id))))
        choose_test = np.random.choice(np.unique(split_id), n_test)
        X_test = X[[spl for spl in split_id if spl in choose_test] ,:]
        y_test = y[[spl for spl in split_id if spl in choose_test]]
        remained_id = [id_val for id_val in np.unique(split_id) if id_val not in choose_test]
        if val_ratio  > 0:
            n_val = int(np.floor(val_ratio * len(np.unique(split_id))))
            choose_val = np.random.choice(np.unique(remained_id), n_val)
            X_val = X[[spl for spl in split_id if spl in choose_val] ,:]
            y_val = y[[spl for spl in split_id if spl in choose_val]]
            remained_id = [id_val for id_val in np.unique(remained_id) if id_val not in choose_val]
        X_train = X[[spl for spl in split_id if spl in remained_id] ,:]
        y_train = y[[spl for spl in split_id if spl in remained_id]]
        choose_train = remained_id
        if val_ratio  > 0:
            return X_train, y_train , X_val,  y_val, X_test, y_test, choose_train, choose_val,choose_test
        else: X_train, y_train , X_test, y_test, choose_train,choose_test 

    else:
        if val_ratio == 0:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_ratio, random_state=42)
            return X_train, y_train , X_test, y_test

        else:
            X_train, X_test_val, y_train, y_test_val = model_selection.train_test_split(X, y, test_size=test_ratio + val_ratio, random_state=42)
            X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test_val, y_test_val, test_size = val_ratio /(test_ratio + val_ratio), random_state=42)
            return X_train, y_train , X_val,  y_val, X_test, y_test


def run_model(X_fit, y_fit, X_val, y_val, type_model = 'classification', max_depth = 1, n_estimators = 50,  
              learning_rate = 0.1, early_stopping_rounds = False, col_sampling = 1,
                is_DART = True, DART_params = {'n_drop':1, 'dist_drop': 'random' , 'min_1':True, 'weights_list' : None},
                limit_type = False,
              ):

    """

    """
    if type_model == 'classification':
        model = starboost_up.boosting.BoostingClassifier(loss=starboost_up.losses.LogLoss(),
        base_estimator= xgb.XGBRegressor(max_depth = 1), base_estimator_is_tree=True,
        n_estimators=n_estimators,  init_estimator=starboost_up.init.LogOddsEstimator(), 
        learning_rate= learning_rate,   row_sampling=0.8,    col_sampling=col_sampling,    eval_metric=micro_f1_score,
        early_stopping_rounds=early_stopping_rounds,    random_state=42,    type_class ='classification',
        is_DART = is_DART, DART_params = DART_params )

    elif type_model == 'regression':        
        model = starboost_up.boosting.BoostingRegressor(
            loss=sb.losses.L2Loss(),
            base_estimator=tree.DecisionTreeRegressor(max_depth= max_depth),
            base_estimator_is_tree=True,
            n_estimators=n_estimators,         init_estimator=linear_model.LinearRegression(),
            learning_rate=  learning_rate,            row_sampling=0.8,
            col_sampling=col_sampling,            eval_metric=rmse,
            early_stopping_rounds=early_stopping_rounds,            random_state=42    ,
            is_DART =  is_DART, DART_params = DART_params)
    else:
        raise NameError('Unknown problem type')  #
    model = model.fit(X_fit, y_fit, eval_set=(X_val, y_val))

    y_pred = model.predict(X_val)#_proba
    #
    if  type_model == 'regression':          eva = rmse(y_val, y_pred)
    else:        eva = metrics.roc_auc_score(y_val, y_pred)
    evas = {}; inter_predictions = {}
    for ib_num, ib in enumerate(model.iter_predict(X_val)):
        inter_predictions[ib_num] = ib
        if  type_model == 'regression':         evas[ib_num] = rmse(y_val,ib)
        else:         evas[ib_num] = metrics.roc_auc_score(y_val,ib)
    return model, y_pred, eva, evas, inter_predictions



def run_model_x_y(X, y , test_ratio = 0.2, split_id = None, type_model = 'classification', max_depth = 1, n_estimators = 50,  
              learning_rate = 0.1, early_stopping_rounds = False, col_sampling = 0.8,
                is_DART = True, DART_params = {'n_drop':1, 'dist_drop': 'random' , 'min_1':True, 'weights_list' : None}, limit_type = False):

    if split_id:
        X_fit, y_fit , X_val, y_val, choose_train, choose_test = split_train_test_val(X,y, test_ratio = test_ratio, val_ratio = 0, split_id = split_id, rand_seed = 0)
    else:
        X_fit, y_fit , X_val, y_val =  split_train_test_val(X,y, test_ratio = test_ratio, val_ratio = 0, split_id = None, rand_seed = 0)



    model, y_pred, eva, evas, inter_predictions = run_model(X_fit, y_fit, X_val, y_val, type_model = type_model, max_depth = max_depth, n_estimators = n_estimators,  
              learning_rate = learning_rate, early_stopping_rounds =  early_stopping_rounds, col_sampling = col_sampling,
                is_DART =  is_DART,     DART_params = DART_params)
    return  X_fit, X_val, y_fit, y_val, model, y_pred, eva, evas, inter_predictions


#%%
isdart = bool(input('is dart?')    )
ndrop = float(input('ndrop'))

params = {'isdart':isdart, 'n_estimators':150, 'ndrop': ndrop,'min_1' : False, 'limit_type' : False }

if ndrop == 1: ndrop = int(ndrop)
X_fit, X_val, y_fit, y_val, model, y_pred, eva, evas, inter_predictions = run_model_x_y(X, y,
                                                                                        is_DART =isdart , n_estimators= n_estimators, 
                                                                                        DART_params = {'n_drop':ndrop,'min_1':params['min_1']})


real_path = create_inner_path([params['isdart'], params['n_estimators'], params['ndrop'], params['min_1'],params['limit_type']], 
                  ['isdart', 'n_estimators', 'ndrop', 'min_1','limit_type'])


def make_file(array_to_save, path='',file_name ='', to_rewrite= False, type_file = '.npy') :
    """
    This function creates a an npy or jpg or png file
    array_to_save - numpy array to save
    path - path to save 
    file_name - name of saved file 
    to_rewrite - If file already exist -> whether to rewrite it. 
    type_file = '.npy' or 'jpg' or 'png'
    """
    if not type_file.startswith('.'): type_file = '.' + type_file
    if not file_name.endswith(type_file): file_name = file_name +type_file
    my_file = Path('%s\%s'%(path,file_name))
    if not my_file.is_file() or to_rewrite: 
        if len(array_to_save) > 0:
            if type_file == '.npy':
                np.save(my_file, array_to_save)
            elif type_file == '.png' or type_file == '.jpg':
                print('Fig. saved')
                array_to_save.savefig(my_file, dpi=300)  
        else:
            plt.savefig(my_file)



#%% 
name_save = real_path.replace('/','_')[1:];            
if to_update_dict:
    name_cum = 'cum_dict_%s.npy'%data_type
    load_dict = to_load_dict(name_cum)   
    load_dict[name_save] = evas
    save_path =os.getcwd()+ '%s'%('\save_files\save_dicts\%s'%(data_type))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(save_path+'\%s'%name_cum, load_dict)
    print(save_path)
    
    name_cum = 'params_dict_%s.npy'%data_type
    load_dict = to_load_dict(name_cum)   
    load_dict[name_save] = params
    save_path =os.getcwd()+ '%s'%('\save_files\save_dicts\%s'%(data_type))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(save_path+'\%s'%name_cum, load_dict)

    
if to_plot:            
    plt.plot(pd.DataFrame(evas,index =[name_save]).T, color_plot);
    plt.xlabel('Iterations')
    plt.ylabel('AUC score')
    make_file([], whole_path, 'performance_graph.png',True, '.png')

 
# Save the trained model as a pickle string.
pickle.dump(model, open(whole_path+'/model.sav' , 'wb'))