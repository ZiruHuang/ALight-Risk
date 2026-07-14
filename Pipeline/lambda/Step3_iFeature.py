import shutil
import os
import pandas as pd
import re
import math
import numpy as np
from sklearn.metrics import roc_auc_score
# parameters
# filepath: the path of fasta file
# filename: the name of fasta file,example: 'training.fasta'
# outdir:creat a new folder to store the ifeature results,example: 'training_ifeature'
def fea_extr(filename,outdir):
    fasta = filename
    os.makedirs(outdir, exist_ok=True)
    python_bin = os.environ.get("PYTHON_BIN", "python3")
    ifeature_script = os.environ.get("IFEATURE_SCRIPT", "iFeature.py")
    methods = ['AAC', 'APAAC', 'CKSAAGP', 'DPC',  'CKSAAP', 'DDE', 'GAAC', 'PAAC', 'GDPC', 'GTPC','Moran', 'Geary',
               'NMBroto', 'CTDC', 'CTDT', 'CTDD', 'CTriad', 'KSCTriad','SOCNumber','QSOrder']
    for method in methods:
        cmd = python_bin + " " + ifeature_script + " --file "+ fasta + ' --type ' + method + \
        ' --out ' + outdir + '/'+fasta.split('/')[-1].split('.')[0]+ '_' + method + '.tsv'
        os.system(cmd)
        
# merge label and feature matrix
def merge_label_matrix(dirpath,label):
#     os.chdir(dirpath)
    label = pd.read_csv(label,sep='\t')
    tsv_list = [x for x in os.listdir(dirpath) if x[-3:] == 'tsv']
    for file in tsv_list:
        df = pd.read_csv(dirpath + file,sep='\t')
        df['#'] = label['label']
        df.to_csv(dirpath + file.split('.')[0]+'.csv',index=False)
        os.remove(dirpath + file)
    

def prof_evalue(true,pred):
    N_pos = 0
    N_neg = 0
    TP = 0
    TN = 0
    for i in range(len(true)):
        if true[i]==1:
            N_pos += 1
            if pred[i]==1:
                TP += 1
        else:
            if true[i]==0:
                N_neg += 1
                if pred[i]==0:
                    TN += 1
    FP = N_pos-TP
    FN = N_neg-TN
    SE = TP/N_pos
    SP = TN/N_neg
    ACC = (TP + TN)/(N_pos + N_neg)
    if (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) == 0:
        MCC = 0
    else:
        MCC = (TN * TP - FP * FN)/math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    return SE,SP,ACC,MCC


def pro_test_dimention(test,reduce_file):
    df1 = pd.read_csv(test, sep = ',')
    df2 = pd.read_csv(reduce_file, sep = ',')
    df3 = df1[df2.keys()]
    return df3

fea_extr('2.Training_Data/Lambda/Lambda_trainset-1.fasta','3.iFeature/Lambda/subModel_1')
#fea_extr('2.Training_Data/Lambda/Lambda_trainset-2.fasta','3.iFeature/Lambda/subModel_2')
#fea_extr('2.Training_Data/Lambda/Lambda_trainset-3.fasta','3.iFeature/Lambda/subModel_3')
fea_extr('2.Training_Data/Lambda/Lambda_test_244.fasta','3.iFeature/Lambda/test/')
#fea_extr('2.Training_Data/Lambda/Lambda_all_tr1470.fasta','3.iFeature/Lambda/All_traning_data/')

merge_label_matrix('3.iFeature/Lambda/subModel_1/','2.Training_Data/Lambda/Lambda_trainset-1_label.txt')
#merge_label_matrix('3.iFeature/Lambda/subModel_2/','2.Training_Data/Lambda/Lambda_trainset-2_label.txt')
#merge_label_matrix('3.iFeature/Lambda/subModel_3/','2.Training_Data/Lambda/Lambda_trainset-3_label.txt')
merge_label_matrix('3.iFeature/Lambda/test/','2.Training_Data/Lambda/Lambda_test_244_label.txt')
#merge_label_matrix('3.iFeature/Lambda/All_traning_data/','2.Training_Data/Lambda/Lambda_all_tr1470_label.txt')


