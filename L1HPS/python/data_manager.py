import itertools as it
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
#from root_numpy import root2array, stretch
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil
from math import sqrt, sin, cos, tan, exp
import ROOT
import math , array
from random import randint
import pandas
import glob
from root_numpy import tree2array

#execfile("../test/example.py") 
execfile("../python/auxillary_functions_L1BDT.py") ## loading all classes and functions needed



def load_data(
    inputPath,
    treeName,
    dirName,
    keys,
    variables,
    bdtType,
    sel = None) :
    print ('bdttype= ', bdtType)
    #my_cols_list=variables+['proces', 'key', 'target', "totalWeight"]
    my_cols_list=variables+['proces', 'key']
    data = pandas.DataFrame(columns=my_cols_list) ## right now an empty dataframe with columns = my_cols_list
    for folderName in keys : ## Loop over file keys
        print ('(folderName, treeName) = ', (folderName, treeName))
        if 'GluGluHToTauTau' in folderName :
            sampleName='GluGluHToTauTau'
            target=1
        elif 'VBFHToTauTau' in folderName :
            sampleName='VBFHToTauTau'
            target=1
        elif 'NeutrinoGun' in folderName :
            sampleName='NeutrinoGun'
            target=0
        elif 'MinBias' in folderName :
            sampleName='MinBias'
            target=0    
        else:
            raise ValueError("Invalid key = %s !!" % folderName)
        inputTree = dirName+'/'+treeName ## Tree path    
        if (bdtType != None):
            #procP1=glob.glob(inputPath+"/"+"NTuple_test_TallinnL1PFTauAnalyzer_"+folderName+"_20190625_3*.root") ## Path to list of files given using *
            procP1=glob.glob(inputPath+"/"+"NTuple_test_TallinnL1PFTauAnalyzer_"+folderName+"_20190716_5*.root") ## Path to list of files given using *
            list=procP1
            print list
        else:
            raise ValueError("bdtType string cannot be empty !!")

        for ii in range(0, len(list)) :
            try: tfile = ROOT.TFile(list[ii])
            except :
                print (list[ii], "FAIL load root file")
                continue
            try: tree = tfile.Get(inputTree)
            except :
                print (inputTree, "FAIL read inputTree", tfile)
                continue
            if tree is not None :
                try: 
                    #chunk_arr = tree2array(tree, selection=sel) ## Converting tree to numpy array
                    L1PFTau_list = MakeL1PFTauCollection(tree, target)
                    chunk_df     = list2df(L1PFTau_list) ## Converting list of L1PFTaus to pandas dataframe                    
                except :
                    print (inputTree, "FAIL load inputTree", tfile)
                    tfile.Close()
                    continue
                else :
                    #chunk_df = pandas.DataFrame(chunk_arr, columns=variables)
                    tfile.Close()
                    chunk_df['proces']=sampleName
                    chunk_df['key']=folderName
                    #chunk_df['target']=target
                    #chunk_df["totalWeight"] = chunk_df["MC_weight"]
                    ## ---- YOU CAN CREATE NEW BRANCHES HERE BY DOING SIMPLE ARITHMETIC OPERATIONS --- ###
                    ## ---- ON THE ALREADY EXISTING ONES AND ADD THEM TO THE PANDAS DATAFRAME      --- ###
                    data=data.append(chunk_df, ignore_index=True)
            else : print ("file "+list[ii]+"was empty")
            tfile.Close()        
        if len(data) == 0 : continue
        nS = len(data.ix[(data.target.values == 1) & (data.key.values==folderName) ])
        nB = len(data.ix[(data.target.values == 0) & (data.key.values==folderName) ])
        print (folderName,"size of sig, bkg, MC_weight, tot weight of data: ", nS, nB , data.ix[ (data.key.values==folderName)]["MC_weight"].sum(), data.ix[(data.key.values==folderName)]["MC_weight"].sum())
        nNW = len(data.ix[(data["MC_weight"].values < 0) & (data.key.values==folderName) ])
        print (folderName, "events with -ve weights", nNW)
    print ('data to list = ', (data.columns.values.tolist()))
    n = len(data)
    print("n", n)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print (treeName," size of sig, bkg: ", nS, nB)
    return data


def make_plots(
    featuresToPlot,nbin,
    data1,label1,color1,
    data2,label2,color2,
    plotname,
    printmin,
    plotResiduals,
    masses = [],
    masses_all = []
    ) :
    print ('length of features to plot and features to plot', (len(featuresToPlot), featuresToPlot))
    hist_params = {'normed': True, 'histtype': 'bar', 'fill': True , 'lw':3, 'alpha' : 0.4}
    sizeArray=int(math.sqrt(len(featuresToPlot))) if math.sqrt(len(featuresToPlot)) % int(math.sqrt(len(featuresToPlot))) == 0 else int(math.sqrt(len(featuresToPlot)))+1
    drawStatErr=True
    residuals=[]
    plt.figure(figsize=(5*sizeArray, 5*sizeArray))
    to_ymax = 10.
    to_ymin = 0.0001
    for n, feature in enumerate(featuresToPlot):
        # add sub plot on our figure
        plt.subplot(sizeArray, sizeArray, n+1)
        # define range for histograms by cutting 1% of data from both ends
        min_value, max_value = np.percentile(data1[feature], [0.0, 99])
        min_value2, max_value2 = np.percentile(data2[feature], [0.0, 99])
        if feature == "gen_mHH" :
            nbin_local = 10*len(masses_all)
            range_local = [masses_all[0]-20, masses_all[len(masses_all)-1]+20]
        else :
            nbin_local = nbin
            range_local = (min(min_value,min_value2),  max(max_value,max_value2))
        if printmin : print ('printing min and max value= ', (min_value, max_value,feature))
        values1, bins, _ = plt.hist(
                                   data1[feature].values,
                                   weights = data1[weights].values.astype(np.float64),
                                   range = range_local,
                                   bins = nbin_local, edgecolor=color1, color=color1,
                                   label = label1, **hist_params
                                   )
        to_ymax = max(values1)
        to_ymin = min(values1)
        if drawStatErr:
            normed = sum(data1[feature].values)
            mid = 0.5*(bins[1:] + bins[:-1])
            err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
            plt.errorbar(mid, values1, yerr=err, fmt='none', color= color1, ecolor= color1, edgecolor=color1, lw=2)
        if len(masses) == 0 : #'gen' not in feature:
            values2, bins, _ = plt.hist(
                                   data2[feature].values,
                                   weights= data2[weights].values.astype(np.float64),
                                   range=range_local,
                                   bins=nbin_local, edgecolor=color2, color=color2,
                                   label=label2, **hist_params
                                   )
            to_ymax2 = max(values2)
            to_ymax = max([to_ymax2, to_ymax])
            to_ymin2 = min(values2)
            to_ymin = max([to_ymin2, to_ymin])
            if drawStatErr :
                normed = sum(data2[feature].values)
                mid = 0.5*(bins[1:] + bins[:-1])
                err=np.sqrt(values2*normed)/normed # denominator is because plot is normalized
                plt.errorbar(mid, values2, yerr=err, fmt='none', color= color2, ecolor= color2, edgecolor=color2, lw=2)
        else :
            hist_params2 = {'normed': True, 'histtype': 'step', 'fill': False , 'lw':3}
            colors_mass = ['m', 'b', 'k', 'r', 'g',  'y', 'c', ]
            for mm, mass in enumerate(masses) :
                values2, bins, _ = plt.hist(
                                       data2.loc[ (data2["gen_mHH"].astype(np.int) == int(mass)), feature].values,
                                       weights = data2.loc[ (data2["gen_mHH"].astype(np.int) == int(mass)), weights].values.astype(np.float64),
                                       range = range_local,
                                       bins = nbin_local, edgecolor=colors_mass[mm], color=colors_mass[mm],
                                       label = label2 + "gen_mHH = " + str(mass), **hist_params2
                                       )
                to_ymax2 = max(values2)
                to_ymax = max([to_ymax2, to_ymax])
                to_ymin2 = min(values2)
                to_ymin = max([to_ymin2, to_ymin])
                if drawStatErr :
                    normed = sum(data2[feature].values)
                    mid = 0.5*(bins[1:] + bins[:-1])
                    err=np.sqrt(values2*normed)/normed # denominator is because plot is normalized
                    plt.errorbar(mid, values2, yerr=err, fmt='none', color= colors_mass[mm], ecolor= colors_mass[mm], edgecolor=colors_mass[mm], lw=2)
        #areaSig = sum(np.diff(bins)*values)
        #print areaBKG, " ",areaBKG2 ," ",areaSig
        if plotResiduals : residuals=residuals+[(plot1[0]-plot2[0])/(plot1[0])]
        plt.ylim(ymin=to_ymin*0.1, ymax=to_ymax*1.2)
        if feature == "avg_dr_jet" : plt.yscale('log')
        else : plt.yscale('linear')
        if n == len(featuresToPlot)-1 : plt.legend(loc='best')
        plt.xlabel(feature)
        #plt.xscale('log')
        #
    plt.ylim(ymin=0)
    plt.savefig(plotname)
    plt.clf()
    if plotResiduals :
        residuals=np.nan_to_num(residuals)
        for n, feature  in enumerate(trainVars(True)):
            (mu, sigma) = norm.fit(residualsSignal[n])
            plt.subplot(8, 8, n+1)
            residualsSignal[n]=np.nan_to_num(residualsSignal[n])
            n, bins, patches = plt.hist(residualsSignal[n], label='Residuals '+label1+'/'+label2)
            # add a 'best fit' line
            y = mlab.normpdf( bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)
            plt.ylim(ymin=0)
            plt.title(feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma))
            print (feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma))
        plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_Signal_fullsim_residuals.pdf")
        plt.clf()
