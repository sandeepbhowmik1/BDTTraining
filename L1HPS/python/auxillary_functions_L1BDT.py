import ROOT
#from rootpy import stl
from ROOT import TLorentzVector, TH2D, TFile
import math 
import pandas as pd
import numpy as np

def constrainValue(value, lowerbound, upperbound):
     if(lowerbound <= upperbound):
          value = max(value, lowerbound)
          value = min(value, upperbound)
          return value
     else:
          raise ValueError("lowerbound ('%f') is larger than upperbound ('%f') !!" % (lowerbound, upperbound))

def getSF_from_TH2(histo2D, x, y):
     xAxis = histo2D.GetXaxis()
     yAxis = histo2D.GetYaxis()
     
     Bin_x = xAxis.FindBin(x)
     Bin_y = yAxis.FindBin(y)

     NBins_x = xAxis.GetNbins()
     NBins_y = yAxis.GetNbins()

     idxBin_x = constrainValue(Bin_x, 1, NBins_x)
     idxBin_y = constrainValue(Bin_y, 1, NBins_y)

     return histo2D.GetBinContent(idxBin_x, idxBin_y)

def MakeDataFrameFlat(channel, data, var_name1, var_name2, label, Target):
     data_copy = data.copy(deep=True) ## Making a deep copy of data ( => separate data and index from data)                                                                                       
     data_copy_small =  data.loc[(data_copy[target]==Target)] ## choose b/w signal and background   

     FileName = "{}/{}_{}_{}_{}.root".format(channel, "Histo2D", var_name1, var_name2, label)
     file = TFile.Open(FileName) 
     canvas = file.Get("c1") ## Extracting the canvas stored in the root file
     histo2D = TH2D()
     histo2D = canvas.GetPrimitive("histo2D")
     #print("No. of entries ", histo2D.GetEntries())

     list_sf_values = [] ## Creating empty lists

     for index, row in data_copy_small.iterrows():
          #print("(X,Y) Values: ", row[var_name1], row[var_name2])
          SF = getSF_from_TH2(histo2D, row[var_name1], row[var_name2])
          Value = 0.
          if(SF == 0.):
               Value = 1.0 ## Such events will have unit weights
          else:
               Value = 1.0/SF
          #print("Scale Factor: ", Value)
          list_sf_values.append(Value)

     nparray_sf_values = np.array(list_sf_values) ## converting list to numpy array      
     data_copy_small["Kin_weight"] = nparray_sf_values       ## Adding numpy array as new column to dataframe
     
     #print data_copy_small
     return data_copy_small



def numpyarrayHisto2DFill(data_X, data_Y, data_wt, histo2D):
    for x,y,w in zip(data_X, data_Y, data_wt):
        #print("x: {}, y: {},  w: {}".format(x, y, w))
        histo2D.Fill(x,y,w)


def MakeHisto2D(channel, data, var_name1, var_name2, label, Target):

    Histo_Dict = {'X_min': 30., 'X_max': 8030., 'nbins_X': 800, 'Y_min': -2.5, 'Y_max': 2.5, 'nbins_Y': 10}


    data_copy = data.copy(deep=True) ## Making a deep copy of data ( => separate data and index from data)
    data_copy_small =  data_copy.loc[(data_copy[target]==Target)] ## choose b/w signal and background
    
    var1_array = np.array(data_copy_small[var_name1].values, dtype=np.float)
    var2_array = np.array(data_copy_small[var_name2].values, dtype=np.float)
    wt_array   = np.array(data_copy_small['MC_weight'].values, dtype=np.float)
    
    N_var1  = len(var1_array)
    N_var2  = len(var2_array)
    N_wt    = len(wt_array)

    # Create a new canvas, and customize it.
    c1 = TCanvas( 'c1', "Canvas", 200, 10, 700, 500)
    #c1.SetFillColor(42)
    #c1.GetFrame().SetFillColor(21)
    c1.GetFrame().SetBorderSize(6)
    c1.GetFrame().SetBorderMode(-1)



    if((N_var1 == N_var2) and (N_var2 == N_wt)):
        PlotTitle = "2D Histogram b/w "+ var_name1 + " and " + var_name2
        histo2D   = TH2D( 'histo2D', PlotTitle, Histo_Dict['nbins_X'], Histo_Dict['X_min'], Histo_Dict['X_max'], Histo_Dict['nbins_Y'], Histo_Dict['Y_min'], Histo_Dict['Y_max'])
        histo2D.GetXaxis().SetTitle(var_name1)
        histo2D.GetYaxis().SetTitle(var_name2)
        numpyarrayHisto2DFill(var1_array, var2_array, wt_array, histo2D)
        histo2D.Draw("COLZ")
        c1.Modified()
        c1.Update()
        FileName = "{}/{}_{}_{}_{}.root".format(channel, "Histo2D", var_name1, var_name2, label)
        c1.SaveAs(FileName)
    else:
        print("Array sizes don't match !!")
        raise ValueError("Array sizes don't match: N_var1 = '%i', N_var2 = '%i', N_wt = '%i'  !!" % (N_var1, N_var2, N_wt))


class RecoTau:
    def __init__(self, pt, eta, phi, charge, decay_mode, isMatched):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.charge = charge
        self.decay_mode = decay_mode
        self.isMatched = isMatched

    def Get4Vector(self):
        recotau_vec = ROOT.Math.RhoEtaPhiVector(self.pt, self.eta, self.phi)
        return recotau_vec

    def GetDecayMode(self):
        return self.decay_mode

    def GetL1Matched(self):
        return self.isMatched

    def SetL1Matched(self, decision):
        self.isMatched = decision

    def GetCharge(self):
        return self.charge


class L1PFTau:
    def __init__(self, pt, eta, phi, Z, iso, target, MC_weight):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.Z = Z
        self.iso = iso
        self.target = target
        self.MC_weight = MC_weight

    def Get4Vector(self):
        l1pftau_vec = ROOT.Math.RhoEtaPhiVector(self.pt, self.eta, self.phi)
        return l1pftau_vec

    def GetPt(self):
        return self.pt

    def GetEta(self):
        return self.eta

    def GetPhi(self):
        return self.phi

    def GetZ(self):
        return self.Z

    def GetIso(self):
        return self.iso

    def GetTarget(self):
        return self.target

    def GetWeight(self):
        return self.MC_weight


class IndexPair:
     def __init__(self, index1, index2, metric):
         self.index1 = index1
         self.index2 = index2
         self.metric = metric

     def GetIndex1(self):
         return self.index1

     def GetIndex2(self):
         return self.index2


     def GetMetric(self):
         return self.metric


def sortBydR(obj):
    return obj.GetMetric()
    


def Counter(isMatchedVector, N):
    counter = 0
    for j in range(isMatchedVector.size()):
        if(isMatchedVector.at(j)): counter += 1
    if(N == 0):
        if(counter > 0): return 1
        else: return 0
    elif(N == -1):
        if(counter > 2): return 1
        else: return 0
    else:
        if(counter == N): return 1
        else: return 0



def MakeL1PFTauCollection(tree, target):
    Pt_cut=30.
    Eta_cut=2.4
    isSignal=False
    if(target == 1): isSignal=True
    CONE_SIZE=0.3

    #list_recoTau = []
    list_L1PFTau = []
    list_L1PFtauMatched_recoGMTau = []
    list_Tau_index = []
    list_recoGMTauMatched_L1PFTau = []
    list_L1PFTau_signal = []
    list_L1PFTau_background = []    

    #MC_weight = ROOT.std.vector('float')()
    genTauPt = ROOT.std.vector('float')()
    genTauEta = ROOT.std.vector('float')()
    genTauPhi = ROOT.std.vector('float')()
    genTauCharge = ROOT.std.vector('int')()
    recoTauPt = ROOT.std.vector('float')()
    recoTauEta = ROOT.std.vector('float')()
    recoTauPhi = ROOT.std.vector('float')()
    recoTauCharge = ROOT.std.vector('int')()
    recoTauDecayMode = ROOT.std.vector('int')()
    recoGMTauPt = ROOT.std.vector('float')()
    recoGMTauEta = ROOT.std.vector('float')()
    recoGMTauPhi = ROOT.std.vector('float')()
    recoGMTauCharge = ROOT.std.vector('int')()
    recoGMTauDecayMode = ROOT.std.vector('int')()
    l1PFTauPt = ROOT.std.vector('float')()
    l1PFTauEta = ROOT.std.vector('float')()
    l1PFTauPhi = ROOT.std.vector('float')()
    l1PFTauCharge = ROOT.std.vector('int')()
    l1PFTauIso = ROOT.std.vector('float')() 
    l1PFTauType = ROOT.std.vector('int')()
    l1PFTauZ = ROOT.std.vector('float')()
    #isMatched = ROOT.std.vector('bool')()
    isGenMatched = ROOT.std.vector('bool')()
    isRecoMatched = ROOT.std.vector('bool')()
    isRecoGMMatched = ROOT.std.vector('bool')()

    #tree.SetBranchAddress('MC_weight', MC_weight)
    tree.SetBranchAddress('genTauPt', genTauPt)
    tree.SetBranchAddress('genTauEta', genTauEta)
    tree.SetBranchAddress('genTauPhi', genTauPhi)
    tree.SetBranchAddress('genTauCharge', genTauCharge)
    tree.SetBranchAddress('recoTauPt', recoTauPt)
    tree.SetBranchAddress('recoTauEta', recoTauEta)
    tree.SetBranchAddress('recoTauPhi', recoTauPhi)
    tree.SetBranchAddress('recoTauCharge', recoTauCharge)
    tree.SetBranchAddress('recoTauDecayMode', recoTauDecayMode)
    tree.SetBranchAddress('recoGMTauPt', recoGMTauPt)
    tree.SetBranchAddress('recoGMTauEta', recoGMTauEta)
    tree.SetBranchAddress('recoGMTauPhi', recoGMTauPhi)
    tree.SetBranchAddress('recoGMTauCharge', recoGMTauCharge)
    tree.SetBranchAddress('recoGMTauDecayMode', recoGMTauDecayMode)
    tree.SetBranchAddress('l1PFTauPt', l1PFTauPt)
    tree.SetBranchAddress('l1PFTauEta', l1PFTauEta)
    tree.SetBranchAddress('l1PFTauPhi', l1PFTauPhi)
    tree.SetBranchAddress('l1PFTauCharge', l1PFTauCharge)
    tree.SetBranchAddress('l1PFTauIso', l1PFTauIso)
    tree.SetBranchAddress('l1PFTauType', l1PFTauType)
    tree.SetBranchAddress('l1PFTauZ', l1PFTauZ)
    #tree.SetBranchAddress('isMatched', isMatched)
    tree.SetBranchAddress('isGenMatched', isGenMatched) ## set to true if genTau is matched to l1pftau (deltaR =0.5)
    tree.SetBranchAddress('isRecoMatched', isRecoMatched) ## set to true if recoTau is matched to l1pftau (deltaR =0.5)
    tree.SetBranchAddress('isRecoGMMatched', isRecoGMMatched) ## set to true if (genMatched) recoTau is matched to l1pftau (deltaR =0.5)

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if(not((genTauPt.size() == genTauEta.size()) and (genTauPhi.size() == genTauEta.size()))): 
            raise ValueError(" genTau branches not of same size !!")

        if(not((recoTauPt.size() == recoTauEta.size()) and (recoTauPhi.size() == recoTauEta.size()))): 
            raise ValueError(" recoTau branches not of same size !!")

        if(not((recoGMTauPt.size() == recoGMTauEta.size()) and (recoGMTauPhi.size() == recoGMTauEta.size()))): 
            raise ValueError(" recoGMTau branches not of same size !!")

        if(not((l1PFTauPt.size() == l1PFTauEta.size()) and (l1PFTauPhi.size() == l1PFTauEta.size()) and (l1PFTauPhi.size() == l1PFTauIso.size()))): 
            raise ValueError(" l1PFTau branches not of same size !!")

        if(not((genTauPt.size() == isGenMatched.size()) and (recoTauPt.size() == isRecoMatched.size()) and (recoGMTauPt.size() == isRecoGMMatched.size()) )):
            raise ValueError(" The Tau branches and the corresponding matching branches not of same size !!")

        #print("tree.RunNumber ", tree.RunNumber)
        #print("tree.lumi ", tree.lumi)
        #print("tree.EventNumber ", tree.EventNumber)
        if(isSignal and (recoGMTauPt.size() == 0)): continue ## Skip signal events where there are no genMatched recoTaus
        
        for k in range(l1PFTauPt.size()): ## loop over l1PFTaus
            if((l1PFTauPt.at(k) > Pt_cut) and (abs(l1PFTauEta.at(k)) < Eta_cut)): ## check for kinematic cuts and matching to "(genMatched) recoPFTau"         
                l1pftau = L1PFTau(l1PFTauPt.at(k), l1PFTauEta.at(k), l1PFTauPhi.at(k), l1PFTauZ.at(k), l1PFTauIso.at(k), target, tree.MC_weight) 
                list_L1PFTau.append(l1pftau)
        #print("L1PFTaus passing pt,eta cuts", len(list_L1PFTau))

        if(isSignal):
            #print("l1PFTauPt.size() ", l1PFTauPt.size())
            #print("recoTauPt.size() ",  recoTauPt.size())
            #print("recoGMTauPt.size() ",  recoGMTauPt.size())
            for i in range(recoGMTauPt.size()): ## loop over (genMatched) recoTaus
                if(recoGMTauPt.at(i) > 0.): 
                    #print("recoGMTauPt.at(i) ",  recoGMTauPt.at(i))
                    recoGMTau = RecoTau(recoGMTauPt.at(i), recoGMTauEta.at(i), recoGMTauPhi.at(i), recoGMTauCharge.at(i), recoGMTauDecayMode.at(i), False) ## (L1PFTau matched) recoGMTau
                    list_L1PFtauMatched_recoGMTau.append(recoGMTau)
                else:
                    continue

            #print("Gen Matched Reco Taus ",  len(list_L1PFtauMatched_recoGMTau))

            for a in range(len(list_L1PFTau)):
                for b in range(len(list_L1PFtauMatched_recoGMTau)):
                    dEta2 = (list_L1PFtauMatched_recoGMTau[b].Get4Vector().Eta() - list_L1PFTau[a].Get4Vector().Eta())**2
                    dPhi2 = (list_L1PFtauMatched_recoGMTau[b].Get4Vector().Phi() - list_L1PFTau[a].Get4Vector().Phi())**2
                    dR = math.sqrt(dEta2 + dPhi2)
                    #print("L1PFTau index %i, RecoTau index %i, dR %f" % (a, b, dR))
                    if(dR < CONE_SIZE):
                        index_pair = IndexPair(a, b, dR)
                        list_Tau_index.append(index_pair)
                    else:    
                        continue
            list_Tau_index.sort(key = sortBydR) ## sort by increasing order of dR        

            first_Matched_L1PFTau_index = 0
            first_Matched_RecoTau_index = 0
            counter = 0

            for c in range(len(list_Tau_index)):
                if(c == 0): 
                    first_Matched_L1PFTau_index = list_Tau_index[0].GetIndex1()
                    first_Matched_RecoTau_index = list_Tau_index[0].GetIndex2()
                    #dRmin = list_Tau_index[0].GetMetric()
                    #print("Selected L1PFTau Pt",  ((list_L1PFTau[first_Matched_L1PFTau_index]).Get4Vector()).Rho())
                    list_recoGMTauMatched_L1PFTau.append(list_L1PFTau[first_Matched_L1PFTau_index])
                    counter += 1
                    continue

                if((list_Tau_index[c].GetIndex1() != first_Matched_L1PFTau_index) and (list_Tau_index[c].GetIndex2() != first_Matched_RecoTau_index)):
                    #print("Sel: L1PFTau index %i, RecoGMTau index %i, dR %f" % (list_Tau_index[0].GetIndex1(), list_Tau_index[0].GetIndex2(), list_Tau_index[0].GetMetric()))
                    #print("Sel: L1PFTau index %i, RecoGMTau index %i, dR %f" % (list_Tau_index[c].GetIndex1(), list_Tau_index[c].GetIndex2(), list_Tau_index[c].GetMetric()))
                    #list_L1PFtauMatched_recoGMTau[first_Matched_RecoTau_index].SetL1Matched(True)
                    #list_L1PFtauMatched_recoGMTau[c].SetL1Matched(True)
                    first_Matched_L1PFTau_index = list_Tau_index[c].GetIndex1()
                    first_Matched_RecoTau_index = list_Tau_index[c].GetIndex2()
                    #dRmin1 = list_Tau_index[c].GetMetric()
                    #print("Selected L1PFTau Pt",  ((list_L1PFTau[first_Matched_L1PFTau_index]).Get4Vector()).Rho())
                    list_recoGMTauMatched_L1PFTau.append(list_L1PFTau[first_Matched_L1PFTau_index])
                    counter += 1
                else:
                    continue
            #print("Number of L1PFTaus matched to recoGMTaus within cone size dR = 0.3 ", counter)
            list_L1PFTau_signal.extend(list_recoGMTauMatched_L1PFTau)
            #print("list_L1PFTau_signal", len(list_L1PFTau_signal))

        else: ## for background just take all L1PFTaus passing pt, eta cuts
            list_L1PFTau_background.extend(list_L1PFTau)

        ## emptying the lists for the next event
        list_L1PFTau *= 0
        list_L1PFtauMatched_recoGMTau *= 0                                                                                                                                                        
        list_Tau_index *= 0                                                                                                                                                                        
        list_recoGMTauMatched_L1PFTau *= 0        
         
    if(isSignal): ## returning (recoGMTau matched) L1PFTaus satifying pt,eta cuts (for all signal events)
            #print("No. of signal L1PFTaus ", len(list_L1PFTau_signal))
            return list_L1PFTau_signal 
    else:         ## returning all L1PFTaus satifying pt,eta cuts (for all background events)
            #print("No. of background L1PFTaus ", len(list_L1PFTau_background))
            return list_L1PFTau_background


def list2df(list_L1PFTau):
    list_l1pftau_pt       = []
    list_l1pftau_eta      = []
    list_l1pftau_phi      = []
    list_l1pftau_z        = []
    list_l1pftau_iso      = []
    list_l1pftau_target   = []
    list_l1pftau_weight   = []

    
    for i in range(len(list_L1PFTau)):
        #print("Eta", list_L1PFTau[i].GetEta())
        list_l1pftau_pt.append(list_L1PFTau[i].GetPt())
        list_l1pftau_eta.append(list_L1PFTau[i].GetEta())
        list_l1pftau_phi.append(list_L1PFTau[i].GetPhi())
        list_l1pftau_z.append(list_L1PFTau[i].GetZ())
        list_l1pftau_iso.append(list_L1PFTau[i].GetIso())
        list_l1pftau_target.append(list_L1PFTau[i].GetTarget())
        list_l1pftau_weight.append(list_L1PFTau[i].GetWeight())
        continue        
    #print("len(list_l1pftau_pt)", len(list_l1pftau_pt))

    ## converting python lists to numpy arrays 
    nparray_l1pftau_pt        = np.array(list_l1pftau_pt)    
    nparray_l1pftau_eta       = np.array(list_l1pftau_eta)
    nparray_l1pftau_phi       = np.array(list_l1pftau_phi)
    nparray_l1pftau_z         = np.array(list_l1pftau_z)    
    nparray_l1pftau_iso       =  np.array(list_l1pftau_iso)    
    nparray_l1pftau_target    =  np.array(list_l1pftau_target)    
    nparray_l1pftau_weight    =  np.array(list_l1pftau_weight)    

    #print("nparray_l1pftau_pt", nparray_l1pftau_pt)
    #print("nparray_l1pftau_eta", nparray_l1pftau_eta)
    #print("nparray_l1pftau_phi", nparray_l1pftau_phi)
    #print("nparray_l1pftau_z", nparray_l1pftau_z)
    #print("nparray_l1pftau_iso", nparray_l1pftau_iso)
    #print("nparray_l1pftau_target", nparray_l1pftau_target)
    #print("nparray_l1pftau_weight", nparray_l1pftau_weight)

    dataframe = pd.DataFrame({'L1PFTauPt': nparray_l1pftau_pt, 
                              'L1PFTauEta': nparray_l1pftau_eta,
                              'L1PFTauPhi': nparray_l1pftau_phi,
                              'L1PFTauZ': nparray_l1pftau_z,
                              'L1PFTauIso': nparray_l1pftau_iso,
                              'target': nparray_l1pftau_target,
                              'MC_weight': nparray_l1pftau_weight})
     
    
    #print("dataframe", dataframe)

    return dataframe
    





        


