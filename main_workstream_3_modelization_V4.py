# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:31:30 2023

@author: Amine CHAIBI
"""
#-----------------------------------------------------------------------------------------------------------------------------#
################################################ Charger les fonctions ########################################################
#-----------------------------------------------------------------------------------------------------------------------------#
import os
os.chdir("D:/1_Building_blocks/3_Workstream_3/")

from _0_functions import *
from _1_load_datasets import *

#-----------------------------------------------------------------------------------------------------------------------------#
################################### Pivot agg sur la maille serial_number #####################################################
#-----------------------------------------------------------------------------------------------------------------------------#
maille = 'maille_1'
dataset_filtred = dataset_flagged; filtre = 'global'
# dataset_filtred = dataset_flagged[dataset_flagged['test_type'] == 'TX']; filtre = 'TX'
# dataset_filtred = dataset_flagged[dataset_flagged['test_type'] == 'RX']; filtre = 'RX'

# Choisir les seuils
thr_to_delete = 0.1 # nb NA tolérés
threshold_class = 0.5 # pourcentage pour décider du seuil du vote majoritaire --> par défaut = 0.5

# Transformer les données en pivot table
pivot_init, pivot  = built_pivot_table(dataset_filtred, thr_to_delete, 'test_value', 'serial_number', 'test_desc', 'Yes', 'Yes', 'Yes')

pivot_flagged = pd.merge(pivot,return_for_repair[['serial_number','return']],on='serial_number',how='left')
pivot_flagged['return'] = pivot_flagged['return'].fillna(0)
pivot_flagged = pivot_flagged.sort_values(by=['serial_number'], ascending=True)

target = pivot_flagged.loc[:,'return']

algo = LogisticRegression(); algo_char = 'LogisticRegression'
#algo = DecisionTreeClassifier(); algo_char = 'DecisionTreeClassifier'
#algo = GaussianNB(); algo_char = 'GaussianNB'
#algo = RandomForestClassifier();  algo_char = 'RandomForestClassifier'
#algo = GradientBoostingClassifier() ; algo_char = 'GradientBoostingClassifier'
#algo = AdaBoostClassifier(n_estimators=100) ; algo_char = 'AdaBoostClassifier'
    
metric_df = pd.DataFrame()
nb_split = 10
skf = StratifiedKFold(nb_split)

fold_no = 1
for train_index, test_index in skf.split(pivot_flagged, target):
    train = pivot_flagged.loc[train_index,:]
    test = pivot_flagged.loc[test_index,:]
    metric_df_0 = train_model_StratifiedKFold(pivot_flagged, train, test, algo, fold_no)
    metric_df = metric_df.append(metric_df_0,ignore_index=True)
    fold_no += 1
    
mean_metric = metric_df.mean(axis=0)
std_metric = metric_df.std(axis=0)
metric_df = metric_df.append(mean_metric, ignore_index=True)
metric_df = metric_df.append(std_metric, ignore_index=True)

row_names = pd.DataFrame(pd.Series(range(1, (nb_split+1))), columns =['rownames'])
row_names.reset_index(drop=True, inplace=True)
mean_names = {'rownames':'mean'}
std_names = {'rownames':'std'}
row_names = row_names.append(mean_names, ignore_index=True)
row_names = row_names.append(std_names, ignore_index=True)

metric_df.insert(0, 'KFold', row_names)

os.chdir("D:/1_Building_blocks/3_Workstream_3/3_Outputs")    
metric_df.to_csv('metric_' + maille + '_one_hot_cross_validation_' + filtre + '_' + algo_char + '.csv', index=False, sep=';')

#-----------------------------------------------------------------------------------------------------------------------------#
################################# Pivot agg sur la maille serial_number * frequency * channel #################################
#-----------------------------------------------------------------------------------------------------------------------------#
maille = 'maille_2'
#dataset_filtred = dataset_flagged; filtre = 'global'
#dataset_filtred = dataset_flagged[dataset_flagged['test_type'] == 'TX']; filtre = 'TX'
dataset_filtred = dataset_flagged[dataset_flagged['test_type'] == 'RX']; filtre = 'RX'

# Choisir les seuils
thr_to_delete = 0.5 # nb NA tolérés
threshold_class = 0.5 # pourcentage pour décider du seuil du vote majoritaire --> par défaut = 0.5

# Transformer les données en pivot table
pivot_init_sub, pivot_sub = built_pivot_table_multiple_rows(dataset_filtred, thr_to_delete, 'test_value', 'serial_number', 'frequency', 'channel', 'test_desc', 'Yes', 'Yes', 'Yes')

pivot_sub_flagged = pd.merge(pivot_sub,return_for_repair[['serial_number','return']],on='serial_number',how='left')
pivot_sub_flagged['return'] = pivot_sub_flagged['return'].fillna(0)
pivot_sub_flagged = pivot_sub_flagged.sort_values(by=['serial_number'], ascending=True)

target = pivot_sub_flagged.loc[:,'return']

algo = LogisticRegression(); algo_char = 'LogisticRegression'
#algo = DecisionTreeClassifier(); algo_char = 'DecisionTreeClassifier'
#algo = GaussianNB(); algo_char = 'GaussianNB'
#algo = RandomForestClassifier();  algo_char = 'RandomForestClassifier'
#algo = GradientBoostingClassifier() ; algo_char = 'GradientBoostingClassifier'
#algo = AdaBoostClassifier(n_estimators=100) ; algo_char = 'AdaBoostClassifier'
    
metric_df = pd.DataFrame()
nb_split = 10
skf = StratifiedKFold(nb_split)

fold_no = 1
for train_index, test_index in skf.split(pivot_sub_flagged, target):
    train = pivot_sub_flagged.loc[train_index,:]
    test = pivot_sub_flagged.loc[test_index,:]
    metric_df_0 = train_model_StratifiedKFold(pivot_sub_flagged, train, test, algo, fold_no)
    metric_df = metric_df.append(metric_df_0,ignore_index=True)
    fold_no += 1
    
mean_metric = metric_df.mean(axis=0)
std_metric = metric_df.std(axis=0)
metric_df = metric_df.append(mean_metric, ignore_index=True)
metric_df = metric_df.append(std_metric, ignore_index=True)

row_names = pd.DataFrame(pd.Series(range(1, (nb_split+1))), columns =['rownames'])
row_names.reset_index(drop=True, inplace=True)
mean_names = {'rownames':'mean'}
std_names = {'rownames':'std'}
row_names = row_names.append(mean_names, ignore_index=True)
row_names = row_names.append(std_names, ignore_index=True)

metric_df.insert(0, 'KFold', row_names)

os.chdir("D:/1_Building_blocks/3_Workstream_3/3_Outputs")    
metric_df.to_csv('metric_' + maille + '_one_hot_cross_validation_' + filtre + '_' + algo_char + '.csv', index=False, sep=';')


#-----------------------------------------------------------------------------------------------------------------------------#
########################## Modélisation sans pivot avec one-hot-encoding avec cross validation ################################
#-----------------------------------------------------------------------------------------------------------------------------#
maille = 'maille_3'

dataset_to_one_hot = dataset_flagged.drop(['db_type', 'part_number', 'date_test', 'test_id', 'result_id', 'label', 'desc_init', 'channel_detailed', 'unit_detailed', 'test_id_spec', 'min_value', 'max_value', 'status_init', 'status_octo', 'spec', 'result_detail_description', 'result_detail_label', 'to_take'], axis=1)
dataset_one_hot = pd.get_dummies(dataset_to_one_hot, columns = ['test_desc', 'frequency', 'channel', 'test_type', 'unit_type'])
target = dataset_one_hot.loc[:,'return']

# algo = LogisticRegression(); algo_char = 'LogisticRegression'
# algo = DecisionTreeClassifier(); algo_char = 'DecisionTreeClassifier'
# algo = GaussianNB(); algo_char = 'GaussianNB'
# algo = RandomForestClassifier();  algo_char = 'RandomForestClassifier'
#algo = GradientBoostingClassifier() ; algo_char = 'GradientBoostingClassifier'
algo = AdaBoostClassifier(n_estimators=100) ; algo_char = 'AdaBoostClassifier'
    
metric_df = pd.DataFrame()
nb_split = 10
skf = StratifiedKFold(nb_split)

fold_no = 1
for train_index, test_index in skf.split(dataset_one_hot, target):
    train = dataset_one_hot.loc[train_index,:]
    test = dataset_one_hot.loc[test_index,:]
    metric_df_0 = train_model_StratifiedKFold(dataset_one_hot, train, test, algo, fold_no)
    metric_df = metric_df.append(metric_df_0,ignore_index=True)
    fold_no += 1
    
mean_metric = metric_df.mean(axis=0)
std_metric = metric_df.std(axis=0)
metric_df = metric_df.append(mean_metric, ignore_index=True)
metric_df = metric_df.append(std_metric, ignore_index=True)

row_names = pd.DataFrame(pd.Series(range(1, (nb_split+1))), columns =['rownames'])
row_names.reset_index(drop=True, inplace=True)
mean_names = {'rownames':'mean'}
std_names = {'rownames':'std'}
row_names = row_names.append(mean_names, ignore_index=True)
row_names = row_names.append(std_names, ignore_index=True)

metric_df.insert(0, 'KFold', row_names)

os.chdir("D:/1_Building_blocks/3_Workstream_3/3_Outputs")    
metric_df.to_csv('metric_' + maille + '_one_hot_cross_validation_' + algo_char + '.csv', index=False, sep=';')



######################################################################################################
#--------------------------- K-means sur les dataframe contenant les #-------------------------------#
#-------------------combinaisons de couples de test avec flag 1 ou 0 (outlier ou pas)#---------------#
######################################################################################################
dt_clustering = dt_conca.copy(deep=True)
dt_clustering = dt_clustering.drop(['serial_number','sum_outlier', 'percent_outlier'], axis=1)
 
kmeans = KMeans(n_clusters=3).fit(dt_clustering)
dt_clustering['cluster'] = kmeans.labels_

pca = PCA(n_components=6)
components = pd.DataFrame(pca.fit_transform(dt_clustering))
components['cluster'] = kmeans.labels_

dt_clustering['cluster'] = kmeans.labels_


# Affichage en 2D des deux premières composantes
sns_plt = sns.scatterplot(data = components, 
                x = components.iloc[:,i], 
                y = components.iloc[:,j], 
                hue = "cluster", 
                palette = "muted")
j=j+1
