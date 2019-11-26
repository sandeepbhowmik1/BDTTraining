def read_from(channel):
      if channel == "L1HPSPFTau" :
            dirName='L1HPSPFTauAnalyzer'
            treeName='L1PFTauAnalyzer'
            inputPath='/home/sbhowmik/NTuple_Phase2/L1HPSPFTau'
            keys=['GluGluHToTauTau', 'NeutrinoGun']
      else : 
            dirName='L1PFTauAnalyzer'
            treeName='nL1PFTauAnalyzer'
            inputPath='/home/sbhowmik/NTuple_Phase2/L1PFTau'
            keys=['VBFHToTauTau', 'NeutrinoGun']


      output = {
          "dirName" : dirName,  
          "treeName" : treeName,
          "inputPath" : inputPath,
          "keys" : keys,
          }
     
      return output




def trainVars(all, trainvar = None, bdtType="default"):
      if all==True :return [ ## add all variables to be read fron the tree
            'l1PFTauPt', 
            'l1PFTauEta', 
            #'l1PFTauPhi', 
            #'l1PFTauZ', 
            'l1PFTauIso', 
            #'l1PFTauLeadTrackPtOverTauPt', 
            #'l1PFTauChargedIso', 
            'l1PFTauNeutralIso', 
            'l1PFTauChargedIsoPileup', 
            #'l1PFTauNSignalChargedHadrons', 
            #'l1PFTauNSignalElectrons', 
            'l1PFTauNSignalPhotons', 
            #'l1PFTauNSignalChargedPFCands', 
            'l1PFTauSignalChargeSum', 
            #'l1PFTauStripPtOverTauPt', 
            #'l1PFTauStripMassOverTauPt', 
            #'l1PFTauStripMassOverStripPt', 
            'l1PFTauStripPt',
            'l1PFTauLeadTrackPt',
            #'l1PFTauVtxIndex',
            'l1PFTaudz',
            #'l1PFTauSumTrackPtOfVtx',
            'l1PFTauLeadTrackHoverE',
            'l1PFTauHoverE',
            'l1PFTauSignalTrackMass',
            #'l1PFTauNStripElectrons',
            #'l1PFTauNStripPhotons',
            #'l1PFTauDeltaRLeadTrackStrip',
            'MC_weight'
            ]

      if trainvar=="testVars"  and bdtType=="default" and all==False :return [
            'l1PFTauPt',
            'l1PFTauEta',
            #'l1PFTauPhi',
            #'l1PFTauZ',
            'l1PFTauIso',
            #'l1PFTauLeadTrackPtOverTauPt',
            #'l1PFTauChargedIso',
            'l1PFTauNeutralIso',
            'l1PFTauChargedIsoPileup',
            #'l1PFTauNSignalChargedHadrons',
            #'l1PFTauNSignalElectrons',
            'l1PFTauNSignalPhotons',
            #'l1PFTauNSignalChargedPFCands',
            'l1PFTauSignalChargeSum',
            #'l1PFTauStripPtOverTauPt',
            #'l1PFTauStripMassOverTauPt',
            #'l1PFTauStripMassOverStripPt',
            'l1PFTauStripPt',
            'l1PFTauLeadTrackPt',
            #'l1PFTauVtxIndex',
            'l1PFTaudz',
            #'l1PFTauSumTrackPtOfVtx',
            'l1PFTauLeadTrackHoverE',
            'l1PFTauHoverE',
            'l1PFTauSignalTrackMass',
            #'l1PFTauNStripElectrons',
            #'l1PFTauNStripPhotons',
            #'l1PFTauDeltaRLeadTrackStrip'
            ]
