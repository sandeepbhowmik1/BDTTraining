import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from datetime import datetime
import sys , time
#import sklearn_to_tmva
import sklearn
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
try: from sklearn.cross_validation import train_test_split
except : from sklearn.model_selection import train_test_split
import pandas
import matplotlib.mlab as mlab
from scipy.stats import norm
#from pandas import HDFStore,DataFrame
import math
import matplotlib
matplotlib.use('agg')
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
import psutil
import os
import pickle
import root_numpy
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

import ROOT
import glob
from collections import OrderedDict

from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double

ROOT.gROOT.SetBatch(True) ## No printing of plots on screen

from multiprocessing import Pool, Process
p = Pool(16)

## ---- INSTRUCTIONS TO RUN THE SCRIPT -----###
##----- PLEASE DO cmsenv INSIDE A CMSSW_9_4_X AREA BEFORE DOING BDT TRAINING         -----####
##----- (SINCE SOME XGB FEATURES DO NOT WORK INSIDE CMSSW10_X_Y) ---- ###
## python sklearn_Xgboost_evtLevel_TallinnL1Trigger.py --channel TallinnL1PFTau --variables testVars --doXML True --bdtType default --ntrees 1000 --treeDeph 2 --lr 0.01 --mcw 1 --FlattenDF True


def lr_to_str(lr):
    if(lr >= 1.0):
        lr_label = "_lr_"+str(int(lr*1))
    elif((lr >= 0.1) & (lr < 1.0)):
        lr_label = "_lr_0o"+str(int(lr*10))
    elif((lr >= 0.01) & (lr < 0.1)):
        lr_label = "_lr_0o0"+str(int(lr*100))
    elif((lr >= 0.001) & (lr < 0.01)):
        lr_label = "_lr_0o00"+str(int(lr*1000))
    else:
        lr_label = "_indeter"
    return lr_label


startTime = datetime.now()
execfile("../python/data_manager.py")

from optparse import OptionParser
parser = OptionParser()

parser.add_option("--channel ", type="string", dest="channel", help="The ones whose variables implemented now are:\n   - TallinnL1PFTau\n It will create a local folder and store the report*/xml", default='TallinnL1PFTau')
parser.add_option("--variables", type="string", dest="variables", help="  Set of variables to use -- it shall be put by hand in the code, in the fuction trainVars(all)\n all==True -- all variables will be loaded (training + weights) -- it is used only once\n all==False -- only variables of training (not including weights) \n", default="testVars")
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--bdtType", type="string", dest="bdtType", help=" Specify BDT Type \n", default='default')
parser.add_option("--ntrees", type="int", dest="ntrees", help="hyp", default=1000)
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2)
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="int", dest="mcw", help="hyp", default=1)
parser.add_option("--FlattenDF", action="store_true", dest="FlattenDF", help="Make Dataframe flat as function of pT and eta", default=True)
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()

FlattenDF = options.FlattenDF
bdtType=options.bdtType
trainvar=options.variables
channel = options.channel
hyppar=str(options.variables)+"_ntrees_"+str(options.ntrees)+"_depth_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+lr_to_str(options.lr)

file_ = open('roc_%s.log'%channel,'w+')

print (startTime)

if 'TallinnL1PFTau' in channel :  execfile("../cards/info_TallinnL1PFTaus.py")


import shutil,subprocess
proc=subprocess.Popen(['mkdir '+channel],shell=True,stdout=subprocess.PIPE)
out = proc.stdout.read()

weights="MC_weight"
target='target'


output = read_from(channel)

print('output[inputPath]', output["inputPath"])
print('output[treeName]', output["treeName"])
print('output[dirName]', output["dirName"])
print('output[keys]', output["keys"])
print('variables', trainVars(True))
print('bdtType', bdtType)

data=load_data(
    output["inputPath"],
    output["treeName"],
    output["dirName"],
    output["keys"],
    trainVars(True),
    bdtType,
    )
data.dropna(subset=[weights],inplace = True)
data.fillna(0)

### Plot histograms of training variables
hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
if 'default' in bdtType :
    labelBKG = "NeutrinoGun"
    labelSIG = "GluGluHToTauTau"
else:
    labelBKG = "Min-Bias"
    labelSIG = "VBFHToTauTau"

printmin=True
plotResiduals=False
plotAll=False
nbins=15
colorFast='g'
colorFastT='b'
BDTvariables=trainVars(plotAll, options.variables, options.bdtType)

print("BDTvariables =>", BDTvariables)

make_plots(BDTvariables, nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],labelSIG, colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT.pdf",
    printmin,
    plotResiduals,
    [],
    [125]
    )


#print("original data", data)

if(FlattenDF):
    MakeHisto2D(channel, data, 'L1PFTauPt', 'L1PFTauEta', "background", 0)
    MakeHisto2D(channel, data, 'L1PFTauPt', 'L1PFTauEta', "signal", 1)


    data_rewt_bg  =  MakeDataFrameFlat(channel, data, 'L1PFTauPt', 'L1PFTauEta', "background", 0)
    data_rewt_sig =   MakeDataFrameFlat(channel, data, 'L1PFTauPt', 'L1PFTauEta', "signal", 1)
    frames = [data_rewt_bg, data_rewt_sig]
    data_rewt = pd.concat(frames)
    data = data_rewt

    #print("data before flattening", data)
    data['flat_wt'] =  data[weights] * data["Kin_weight"] 
    #data.loc[data['target']==0, [weights]] *= data.loc[data['target']==0, ["Kin_weight"]]
    #data.loc[data['target']==1, [weights]] *= data.loc[data['target']==1, ["Kin_weight"]]
    #print("data after flattening", data)
    weights = "flat_wt"    



## Spltting dataset into test and train
trainVars = BDTvariables + [weights] + ["target", "key"]
traindataset, valdataset  = train_test_split(data[trainVars], test_size=0.2, random_state=7)

order_train = [traindataset, valdataset]
for data_do in order_train :
    if 'default' in bdtType:
        data_do.loc[data_do['target']==0, [weights]] *= 50000/data_do.loc[data_do['target']==0][weights].sum()
        data_do.loc[data_do['target']==1, [weights]] *= 50000/data_do.loc[data_do['target']==1][weights].sum()
        #print("data_do", data_do)


print 'Tot weight of train and validation for signal= ', traindataset.loc[traindataset[target]==1][weights].sum(), valdataset.loc[valdataset[target]==1][weights].sum()
print 'Tot weight of train and validation for bkg= ', traindataset.loc[traindataset[target]==0][weights].sum(),valdataset.loc[valdataset[target]==0][weights].sum()

make_plots(BDTvariables, nbins,
    traindataset.ix[traindataset.target.values == 0],labelBKG, colorFast,
    traindataset.ix[traindataset.target.values == 1],labelSIG, colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT_after_reweighting_train.pdf",
    printmin,
    plotResiduals,
    [],
    [125]
    )

make_plots(BDTvariables, nbins,
    valdataset.ix[valdataset.target.values == 0],labelBKG, colorFast,
    valdataset.ix[valdataset.target.values == 1],labelSIG, colorFastT,
    channel+"/"+bdtType+"_"+trainvar+"_Variables_BDT_after_reweighting_test.pdf",
    printmin,
    plotResiduals,
    [],
    [125]
    )


cls = xgb.XGBClassifier(
    n_estimators = options.ntrees,
    max_depth = options.treeDeph,
    min_child_weight = options.mcw, # min_samples_leaf
    learning_rate = options.lr,
    )


cls.fit(
    traindataset[BDTvariables].values,
    traindataset.target.astype(np.bool),
    sample_weight=(traindataset[weights].astype(np.float64))
    )


print 'traindataset[BDTvariables].columns.values.tolist() : ', traindataset[BDTvariables].columns.values.tolist()

print ("XGBoost trained")
proba = cls.predict_proba(traindataset[BDTvariables].values )
fpr, tpr, thresholds = roc_curve(traindataset[target], proba[:,1],
                                 sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[BDTvariables].values )
fprt, tprt, thresholds = roc_curve(valdataset[target], proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
file_.write("XGBoost_train = %0.8f\n" %train_auc)
file_.write("XGBoost_test = %0.8f\n" %test_auct)
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()

## Save as pkl file
pklpath=channel+"/"+channel+"_XGB_"+trainvar+"_"+bdtType+"_"+str(len(BDTvariables))+"Var"
print ("Done  ",pklpath,hyppar)
if options.doXML==True :
    print ("Date: ", time.asctime( time.localtime(time.time()) ))
    pickle.dump(cls, open(pklpath+".pkl", 'wb'))
    file = open(pklpath+"pkl.log","w")
    file.write(str(BDTvariables)+"\n")
    file.close()
    print ("saved ",pklpath+".pkl")
    print ("variables are: ",pklpath+"_pkl.log")


## Draw ROC curve
fig, ax = plt.subplots(figsize=(8, 8))
train_auc = auc(fpr, tpr, reorder = True)
ax.plot(fpr, tpr, lw=1, color='g',label='XGB train (area = %0.5f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, ls='--',color='g',label='XGB test (area = %0.5f)'%(test_auct))


ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()

fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(BDTvariables)),hyppar)) 
fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(BDTvariables)),hyppar)) 


## feature importance plot
fig, ax = plt.subplots()
f_score_dict =cls.booster().get_fscore()
f_score_dict = {BDTvariables[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.png".format(channel,bdtType,trainvar,str(len(BDTvariables)),hyppar))
fig.savefig("{}/{}_{}_{}_{}_XGB_importance.pdf".format(channel,bdtType,trainvar,str(len(BDTvariables)),hyppar))

### BDT Output Distribution
hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
plt.clf()
y_pred = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, BDTvariables].values)[:, 1] #
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, BDTvariables].values)[:, 1] #
'''for indx in range(0,len(valdataset)) :
test = valdataset.take([indx])
#print indx, '\t', 'test data : '
#print test
pre = cls.predict_proba(test[trainVars(False)].values )[:, 1]
#print 'predict for test data : ',
#print pre'''
plt.figure('XGB',figsize=(6, 6))
values, bins, _ = plt.hist(y_pred , label=("%s (XGB)" % labelBKG), **hist_params)
values, bins, _ = plt.hist(y_predS , label="signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(BDTvariables))+'_'+hyppar+'_XGBclassifier.pdf')
plt.savefig(channel+'/'+bdtType+'_'+trainvar+'_'+str(len(BDTvariables))+'_'+hyppar+'_XGBclassifier.png')


# plot correlation matrix
if options.HypOpt==False :
    for ii in [1,2] :
        if ii == 1 :
            datad=traindataset.loc[traindataset[target].values == 1]
            label="signal"
        else :
            datad=traindataset.loc[traindataset[target].values == 0]
            label="BKG"
        datacorr = datad[BDTvariables].astype(float)  
        correlations = datacorr.corr()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        ticks = np.arange(0,len(BDTvariables),1)
        plt.rc('axes', labelsize=8)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(BDTvariables,rotation=-90)
        ax.set_yticklabels(BDTvariables)
        fig.colorbar(cax)
        fig.tight_layout()
        #plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
        plt.savefig("{}/{}_{}_{}_corr_{}.png".format(channel,bdtType,trainvar,str(len(BDTvariables)),label))
        plt.savefig("{}/{}_{}_{}_corr_{}.pdf".format(channel,bdtType,trainvar,str(len(BDTvariables)),label))
        ax.clear()


process = psutil.Process(os.getpid())
print(process.memory_info().rss)
print(datetime.now() - startTime)
