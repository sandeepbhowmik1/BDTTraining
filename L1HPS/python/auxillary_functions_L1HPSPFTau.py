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
    def __init__(self, 
                 l1PFTauPt,
                 eta,
                 #l1PFTauEta,  
                 phi, 
                 #l1PFTauPhi, 
                 l1PFTauZ, 
                 l1PFTauIso, 
                 l1PFTauLeadTrackPtOverTauPt, 
                 ChargedIso, 
                 NeutralIso, 
                 ChargedIsoPileup, 
                 NSignalChargedHadrons, 
                 NSignalElectrons, 
                 NSignalPhotons, 
                 NSignalChargedPFCands, 
                 SignalChargeSum, 
                 StripPtOverTauPt, 
                 StripMassOverTauPt, 
                 StripMassOverStripPt, 
                 l1PFTauStripPt, 
                 l1PFTauLeadTrackPt, 
                 l1PFTauVtxIndex, 
                 l1PFTaudz, 
                 l1PFTauSumTrackPtOfVtx, 
                 l1PFTauLeadTrackHoverE, 
                 l1PFTauHoverE, 
                 l1PFTauSignalTrackMass, 
                 l1PFTauNStripElectrons, 
                 l1PFTauNStripPhotons,  
                 l1PFTauDeltaRLeadTrackStrip,
                 target, 
                 MC_weight
                 ):
        self.l1PFTauPt = l1PFTauPt
        self.eta = eta
        #self.l1PFTauEta = l1PFTauEta 
        self.phi = phi
        #self.l1PFTauPhi = l1PFTauPhi
        self.l1PFTauZ = l1PFTauZ
        self.l1PFTauIso = l1PFTauIso
        self.l1PFTauLeadTrackPtOverTauPt = l1PFTauLeadTrackPtOverTauPt
        self.ChargedIso = ChargedIso
        self.NeutralIso = NeutralIso
        self.ChargedIsoPileup = ChargedIsoPileup
        self.NSignalChargedHadrons = NSignalChargedHadrons
        self.NSignalElectrons = NSignalElectrons
        self.NSignalPhotons = NSignalPhotons
        self.NSignalChargedPFCands = NSignalChargedPFCands
        self.SignalChargeSum = SignalChargeSum
        self.StripPtOverTauPt = StripPtOverTauPt
        self.StripMassOverTauPt = StripMassOverTauPt
        self.StripMassOverStripPt = StripMassOverStripPt
        self.l1PFTauStripPt = l1PFTauStripPt
        self.l1PFTauLeadTrackPt = l1PFTauLeadTrackPt
        self.l1PFTauVtxIndex = l1PFTauVtxIndex
        self.l1PFTaudz = l1PFTaudz
        self.l1PFTauSumTrackPtOfVtx = l1PFTauSumTrackPtOfVtx
        self.l1PFTauLeadTrackHoverE = l1PFTauLeadTrackHoverE
        self.l1PFTauHoverE = l1PFTauHoverE
        self.l1PFTauSignalTrackMass = l1PFTauSignalTrackMass
        self.l1PFTauNStripElectrons = l1PFTauNStripElectrons
        self.l1PFTauNStripPhotons = l1PFTauNStripPhotons
        self.l1PFTauDeltaRLeadTrackStrip = l1PFTauDeltaRLeadTrackStrip
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
     genTauPt = ROOT.std.vector('float')()
     genTauEta = ROOT.std.vector('float')()
     genTauPhi = ROOT.std.vector('float')()
     genTauCharge = ROOT.std.vector('int')()
     l1PFTauPt = ROOT.std.vector('float')()
     l1PFTauEta = ROOT.std.vector('float')()
     l1PFTauPhi = ROOT.std.vector('float')()
     l1PFTauCharge = ROOT.std.vector('int')()
     l1PFTauIso = ROOT.std.vector('float')()
     l1PFTauZ = ROOT.std.vector('float')()
     l1PFTauType = ROOT.std.vector('int')()

     l1PFTauLeadTrackPtOverTauPt = ROOT.std.vector('float')()
     l1PFTauChargedIso = ROOT.std.vector('float')()
     l1PFTauNeutralIso = ROOT.std.vector('float')()
     l1PFTauChargedIsoPileup = ROOT.std.vector('float')()
     l1PFTauNSignalChargedHadrons = ROOT.std.vector('float')()
     l1PFTauNSignalElectrons = ROOT.std.vector('float')()
     l1PFTauNSignalPhotons = ROOT.std.vector('float')()
     l1PFTauNSignalChargedPFCands = ROOT.std.vector('float')()
     l1PFTauSignalChargeSum = ROOT.std.vector('float')()
     l1PFTauStripPtOverTauPt = ROOT.std.vector('float')()
     l1PFTauStripMassOverTauPt = ROOT.std.vector('float')()
     l1PFTauStripMassOverStripPt = ROOT.std.vector('float')()
     l1PFTauStripPt = ROOT.std.vector('float')()
     l1PFTauLeadTrackPt = ROOT.std.vector('float')()
     l1PFTauVtxIndex = ROOT.std.vector('int')()
     l1PFTaudz = ROOT.std.vector('float')()
     l1PFTauSumTrackPtOfVtx = ROOT.std.vector('float')()
     l1PFTauLeadTrackHoverE = ROOT.std.vector('float')()
     l1PFTauHoverE = ROOT.std.vector('float')()
     l1PFTauSignalTrackMass = ROOT.std.vector('float')()
     l1PFTauNStripElectrons = ROOT.std.vector('float')()
     l1PFTauNStripPhotons = ROOT.std.vector('float')()
     l1PFTauDeltaRLeadTrackStrip = ROOT.std.vector('float')()

     tree.SetBranchAddress('genTauPt', genTauPt)
     tree.SetBranchAddress('genTauEta', genTauEta)
     tree.SetBranchAddress('genTauPhi', genTauPhi)
     tree.SetBranchAddress('genTauCharge', genTauCharge)
     tree.SetBranchAddress('l1PFTauPt', l1PFTauPt)
     tree.SetBranchAddress('l1PFTauEta', l1PFTauEta)
     tree.SetBranchAddress('l1PFTauPhi', l1PFTauPhi)
     tree.SetBranchAddress('l1PFTauCharge', l1PFTauCharge)
     tree.SetBranchAddress('l1PFTauIso', l1PFTauIso)
     tree.SetBranchAddress('l1PFTauZ', l1PFTauZ)
     tree.SetBranchAddress('l1PFTauType', l1PFTauType)

     tree.SetBranchAddress('l1PFTauLeadTrackPtOverTauPt', l1PFTauLeadTrackPtOverTauPt)
     tree.SetBranchAddress('l1PFTauChargedIso', l1PFTauChargedIso)
     tree.SetBranchAddress('l1PFTauNeutralIso', l1PFTauNeutralIso)
     tree.SetBranchAddress('l1PFTauChargedIsoPileup', l1PFTauChargedIsoPileup)
     tree.SetBranchAddress('l1PFTauNSignalChargedHadrons', l1PFTauNSignalChargedHadrons)
     tree.SetBranchAddress('l1PFTauNSignalElectrons', l1PFTauNSignalElectrons)
     tree.SetBranchAddress('l1PFTauNSignalPhotons', l1PFTauNSignalPhotons)
     tree.SetBranchAddress('l1PFTauNSignalChargedPFCands', l1PFTauNSignalChargedPFCands)
     tree.SetBranchAddress('l1PFTauSignalChargeSum', l1PFTauSignalChargeSum)
     tree.SetBranchAddress('l1PFTauStripPtOverTauPt', l1PFTauStripPtOverTauPt)
     tree.SetBranchAddress('l1PFTauStripMassOverTauPt', l1PFTauStripMassOverTauPt)
     tree.SetBranchAddress('l1PFTauStripMassOverStripPt', l1PFTauStripMassOverStripPt)
     tree.SetBranchAddress('l1PFTauStripPt', l1PFTauStripPt)
     tree.SetBranchAddress('l1PFTauLeadTrackPt', l1PFTauLeadTrackPt)
     tree.SetBranchAddress('l1PFTauVtxIndex', l1PFTauVtxIndex)
     tree.SetBranchAddress('l1PFTaudz', l1PFTaudz)
     tree.SetBranchAddress('l1PFTauSumTrackPtOfVtx', l1PFTauSumTrackPtOfVtx)
     tree.SetBranchAddress('l1PFTauLeadTrackHoverE', l1PFTauLeadTrackHoverE)
     tree.SetBranchAddress('l1PFTauHoverE', l1PFTauHoverE)
     tree.SetBranchAddress('l1PFTauSignalTrackMass', l1PFTauSignalTrackMass)
     tree.SetBranchAddress('l1PFTauNStripElectrons', l1PFTauNStripElectrons)
     tree.SetBranchAddress('l1PFTauNStripPhotons', l1PFTauNStripPhotons)
     tree.SetBranchAddress('l1PFTauDeltaRLeadTrackStrip', l1PFTauDeltaRLeadTrackStrip)

     Pt_cut=20.
     Eta_cut=2.4
     isSignal=False
     if(target == 1): isSignal=True
     CONE_SIZE=0.5

     list_L1PFTau_signal = []
     list_L1PFTau_background = []

     for i in range(tree.GetEntries()):
          tree.GetEntry(i)
          list_L1PFTau = []
          list_genTauMatched_L1PFTau = []

          if(not((genTauPt.size() == genTauEta.size()) and (genTauPhi.size() == genTauEta.size()))): 
               raise ValueError(" genTau branches not of same size !!")
          for j in range(l1PFTauPt.size()): ## loop over l1PFTaus
               if(abs(l1PFTauEta.at(j)) > Eta_cut): ## check for kinematic cuts and matching to "(genMatched) recoPFTau"         
                    continue
               l1pftau = L1PFTau(
                    l1PFTauPt.at(j),
                    l1PFTauEta.at(j),
                    l1PFTauPhi.at(j),
                    l1PFTauZ.at(j),
                    l1PFTauIso.at(j),
                    l1PFTauLeadTrackPtOverTauPt.at(j),
                    l1PFTauChargedIso.at(j),
                    l1PFTauNeutralIso.at(j),
                    l1PFTauChargedIsoPileup.at(j),
                    l1PFTauNSignalChargedHadrons.at(j),
                    l1PFTauNSignalElectrons.at(j),
                    l1PFTauNSignalPhotons.at(j),
                    l1PFTauNSignalChargedPFCands.at(j),
                    l1PFTauSignalChargeSum.at(j),
                    l1PFTauStripPtOverTauPt.at(j),
                    l1PFTauStripMassOverTauPt.at(j),
                    l1PFTauStripMassOverStripPt.at(j),
                    l1PFTauStripPt.at(j),
                    l1PFTauLeadTrackPt.at(j),
                    l1PFTauVtxIndex.at(j),
                    l1PFTaudz.at(j),
                    l1PFTauSumTrackPtOfVtx.at(j),
                    l1PFTauLeadTrackHoverE.at(j),
                    l1PFTauHoverE.at(j),
                    l1PFTauSignalTrackMass.at(j),
                    l1PFTauNStripElectrons.at(j),
                    l1PFTauNStripPhotons.at(j),
                    l1PFTauDeltaRLeadTrackStrip.at(j),
                    target,
                    tree.MC_weight
                    )
               list_L1PFTau.append(l1pftau)
          if(isSignal):
               for k in range(len(list_L1PFTau)):
                    for l in range(genTauPt.size()):
                         if abs(genTauPt.at(l)) < Pt_cut:
                              continue
                         deltaEta2 = (genTauEta.at(l)-list_L1PFTau[k].eta)**2
                         deltaPhi2 = (genTauPhi.at(l)-list_L1PFTau[k].phi)**2
                         DeltaR = math.sqrt(deltaEta2 + deltaPhi2)
                         if (DeltaR < CONE_SIZE):
                              list_genTauMatched_L1PFTau.append(list_L1PFTau[k])
                              break
               list_L1PFTau_signal.extend(list_genTauMatched_L1PFTau)
               #list_L1PFTau_signal.extend(list_L1PFTau)
          else: ## for background just take all L1PFTaus passing pt, eta cuts
               list_L1PFTau_background.extend(list_L1PFTau)

     if(isSignal): ## returning (recoGMTau matched) L1PFTaus satifying pt,eta cuts (for all signal events)
          return list_L1PFTau_signal 
     else:         ## returning all L1PFTaus satifying pt,eta cuts (for all background events)
          return list_L1PFTau_background


def list2df(list_L1PFTau):
    list_l1PFTauPt       = []
    list_l1PFTauEta      = []
    list_l1PFTauPhi      = []
    list_l1PFTauZ        = []
    list_l1PFTauIso      = []
    list_l1PFTauLeadTrackPtOverTauPt      = []
    list_l1PFTauChargedIso      = []
    list_l1PFTauNeutralIso      = []
    list_l1PFTauChargedIsoPileup      = []
    list_l1PFTauNSignalChargedHadrons      = []
    list_l1PFTauNSignalElectrons      = []
    list_l1PFTauNSignalPhotons      = []
    list_l1PFTauNSignalChargedPFCands      = []
    list_l1PFTauSignalChargeSum      = []
    list_l1PFTauStripPtOverTauPt      = []
    list_l1PFTauStripMassOverTauPt      = []
    list_l1PFTauStripMassOverStripPt      = []
    list_l1PFTauStripPt        = []
    list_l1PFTauLeadTrackPt        = []
    list_l1PFTauVtxIndex        = []
    list_l1PFTaudz        = []
    list_l1PFTauSumTrackPtOfVtx        = []
    list_l1PFTauLeadTrackHoverE        = []
    list_l1PFTauHoverE        = []
    list_l1PFTauSignalTrackMass        = []
    list_l1PFTauNStripElectrons        = []
    list_l1PFTauNStripPhotons        = []
    list_l1PFTauDeltaRLeadTrackStrip        = []
    list_target   = []
    list_MC_weight   = []

    
    for i in range(len(list_L1PFTau)):
        list_l1PFTauPt.append(list_L1PFTau[i].l1PFTauPt)
        list_l1PFTauEta.append(list_L1PFTau[i].eta)
        #list_l1PFTauEta.append(list_L1PFTau[i].l1PFTauEta) 
        list_l1PFTauPhi.append(list_L1PFTau[i].phi)
        #list_l1PFTauPhi.append(list_L1PFTau[i].l1PFTauPhi)
        list_l1PFTauZ.append(list_L1PFTau[i].l1PFTauZ)
        list_l1PFTauIso.append(list_L1PFTau[i].l1PFTauIso)
        list_l1PFTauLeadTrackPtOverTauPt.append(list_L1PFTau[i].l1PFTauLeadTrackPtOverTauPt)
        list_l1PFTauChargedIso.append(list_L1PFTau[i].ChargedIso)
        list_l1PFTauNeutralIso.append(list_L1PFTau[i].NeutralIso)
        list_l1PFTauChargedIsoPileup.append(list_L1PFTau[i].ChargedIsoPileup)
        list_l1PFTauNSignalChargedHadrons.append(list_L1PFTau[i].NSignalChargedHadrons)
        list_l1PFTauNSignalElectrons.append(list_L1PFTau[i].NSignalElectrons)
        list_l1PFTauNSignalPhotons.append(list_L1PFTau[i].NSignalPhotons)
        list_l1PFTauNSignalChargedPFCands.append(list_L1PFTau[i].NSignalChargedPFCands)
        list_l1PFTauSignalChargeSum.append(list_L1PFTau[i].SignalChargeSum)
        list_l1PFTauStripPtOverTauPt.append(list_L1PFTau[i].StripPtOverTauPt)
        list_l1PFTauStripMassOverTauPt.append(list_L1PFTau[i].StripMassOverTauPt)
        list_l1PFTauStripMassOverStripPt.append(list_L1PFTau[i].StripMassOverStripPt)
        list_l1PFTauStripPt.append(list_L1PFTau[i].l1PFTauStripPt)
        list_l1PFTauLeadTrackPt.append(list_L1PFTau[i].l1PFTauLeadTrackPt)
        list_l1PFTauVtxIndex.append(list_L1PFTau[i].l1PFTauVtxIndex)
        list_l1PFTaudz.append(list_L1PFTau[i].l1PFTaudz)
        list_l1PFTauSumTrackPtOfVtx.append(list_L1PFTau[i].l1PFTauSumTrackPtOfVtx)
        list_l1PFTauLeadTrackHoverE.append(list_L1PFTau[i].l1PFTauLeadTrackHoverE)
        if list_L1PFTau[i].l1PFTauHoverE < 999.:
             list_l1PFTauHoverE.append(list_L1PFTau[i].l1PFTauHoverE)
        else:
             list_l1PFTauHoverE.append(999.)
        list_l1PFTauSignalTrackMass.append(list_L1PFTau[i].l1PFTauSignalTrackMass)
        list_l1PFTauNStripElectrons.append(list_L1PFTau[i].l1PFTauNStripElectrons)
        list_l1PFTauNStripPhotons.append(list_L1PFTau[i].l1PFTauNStripPhotons)
        list_l1PFTauDeltaRLeadTrackStrip.append(list_L1PFTau[i].l1PFTauDeltaRLeadTrackStrip)
        list_target.append(list_L1PFTau[i].target)
        list_MC_weight.append(list_L1PFTau[i].MC_weight)
        continue        

    ## converting python lists to numpy arrays 
    nparray_l1PFTauPt        = np.array(list_l1PFTauPt)    
    nparray_l1PFTauEta       = np.array(list_l1PFTauEta)
    nparray_l1PFTauPhi       = np.array(list_l1PFTauPhi)
    nparray_l1PFTauZ         = np.array(list_l1PFTauZ)    
    nparray_l1PFTauIso       =  np.array(list_l1PFTauIso)    
    nparray_l1PFTauLeadTrackPtOverTauPt       =  np.array(list_l1PFTauLeadTrackPtOverTauPt)
    nparray_l1PFTauChargedIso       =  np.array(list_l1PFTauChargedIso)
    nparray_l1PFTauNeutralIso       =  np.array(list_l1PFTauNeutralIso)
    nparray_l1PFTauChargedIsoPileup       =  np.array(list_l1PFTauChargedIsoPileup)
    nparray_l1PFTauNSignalChargedHadrons       =  np.array(list_l1PFTauNSignalChargedHadrons)
    nparray_l1PFTauNSignalElectrons       =  np.array(list_l1PFTauNSignalElectrons)
    nparray_l1PFTauNSignalPhotons       =  np.array(list_l1PFTauNSignalPhotons)
    nparray_l1PFTauNSignalChargedPFCands       =  np.array(list_l1PFTauNSignalChargedPFCands)
    nparray_l1PFTauSignalChargeSum       =  np.array(list_l1PFTauSignalChargeSum)
    nparray_l1PFTauStripPtOverTauPt       =  np.array(list_l1PFTauStripPtOverTauPt)
    nparray_l1PFTauStripMassOverTauPt       =  np.array(list_l1PFTauStripMassOverTauPt)
    nparray_l1PFTauStripMassOverStripPt       =  np.array(list_l1PFTauStripMassOverStripPt)
    nparray_l1PFTauStripPt       =  np.array(list_l1PFTauStripPt)
    nparray_l1PFTauLeadTrackPt       =  np.array(list_l1PFTauLeadTrackPt)
    nparray_l1PFTauVtxIndex       =  np.array(list_l1PFTauVtxIndex)
    nparray_l1PFTaudz       =  np.array(list_l1PFTaudz)
    nparray_l1PFTauSumTrackPtOfVtx       =  np.array(list_l1PFTauSumTrackPtOfVtx)
    nparray_l1PFTauLeadTrackHoverE       =  np.array(list_l1PFTauLeadTrackHoverE)
    nparray_l1PFTauHoverE       =  np.array(list_l1PFTauHoverE)
    nparray_l1PFTauSignalTrackMass       =  np.array(list_l1PFTauSignalTrackMass)
    nparray_l1PFTauNStripElectrons       =  np.array(list_l1PFTauNStripElectrons)
    nparray_l1PFTauNStripPhotons       =  np.array(list_l1PFTauNStripPhotons)
    nparray_l1PFTauDeltaRLeadTrackStrip       =  np.array(list_l1PFTauDeltaRLeadTrackStrip)
    nparray_target    =  np.array(list_target)    
    nparray_MC_weight    =  np.array(list_MC_weight)    

    dataframe = pd.DataFrame({'l1PFTauPt': nparray_l1PFTauPt, 
                              'l1PFTauEta': nparray_l1PFTauEta,
                              'l1PFTauPhi': nparray_l1PFTauPhi,
                              'l1PFTauZ': nparray_l1PFTauZ,
                              'l1PFTauIso': nparray_l1PFTauIso,
                              'l1PFTauLeadTrackPtOverTauPt': nparray_l1PFTauLeadTrackPtOverTauPt,
                              'l1PFTauChargedIso': nparray_l1PFTauChargedIso,
                              'l1PFTauNeutralIso': nparray_l1PFTauNeutralIso,
                              'l1PFTauChargedIsoPileup': nparray_l1PFTauChargedIsoPileup,
                              'l1PFTauNSignalChargedHadrons': nparray_l1PFTauNSignalChargedHadrons,
                              'l1PFTauNSignalElectrons': nparray_l1PFTauNSignalElectrons,
                              'l1PFTauNSignalPhotons': nparray_l1PFTauNSignalPhotons,
                              'l1PFTauNSignalChargedPFCands': nparray_l1PFTauNSignalChargedPFCands,
                              'l1PFTauSignalChargeSum': nparray_l1PFTauSignalChargeSum,
                              'l1PFTauStripPtOverTauPt': nparray_l1PFTauStripPtOverTauPt,
                              'l1PFTauStripMassOverTauPt': nparray_l1PFTauStripMassOverTauPt,
                              'l1PFTauStripMassOverStripPt': nparray_l1PFTauStripMassOverStripPt,
                              'l1PFTauStripPt': nparray_l1PFTauStripPt,
                              'l1PFTauLeadTrackPt': nparray_l1PFTauLeadTrackPt,
                              'l1PFTauVtxIndex': nparray_l1PFTauVtxIndex,
                              'l1PFTaudz': nparray_l1PFTaudz,
                              'l1PFTauSumTrackPtOfVtx': nparray_l1PFTauSumTrackPtOfVtx,
                              'l1PFTauLeadTrackHoverE': nparray_l1PFTauLeadTrackHoverE,
                              'l1PFTauHoverE': nparray_l1PFTauHoverE,
                              'l1PFTauSignalTrackMass': nparray_l1PFTauSignalTrackMass,
                              'l1PFTauNStripElectrons': nparray_l1PFTauNStripElectrons,
                              'l1PFTauNStripPhotons': nparray_l1PFTauNStripPhotons,
                              'l1PFTauDeltaRLeadTrackStrip': nparray_l1PFTauDeltaRLeadTrackStrip,
                              'target': nparray_target,
                              'MC_weight': nparray_MC_weight})
     
    
    #print("dataframe", dataframe)

    return dataframe
    





        


