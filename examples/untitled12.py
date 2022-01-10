# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 00:40:33 2022

@author: noga mudrik
"""

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