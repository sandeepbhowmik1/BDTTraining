def read_from(channel):
      if channel == "TallinnL1PFTau" :
            dirName='TallinnL1PFTauAnalyzer'
            #treeName='TallinnL1PFTauAnalyzer'
            treeName='L1PFTauAnalyzer'
            inputPath='/home/sbhowmik/NTuple_Phase2/TallinnL1PFTau'
            #keys=['VBFHToTauTau', 'NeutrinoGun']
            keys=['GluGluHToTauTau', 'NeutrinoGun']
      else : ## Although cuurently the same, You can use this to train bdt for say Isobel's algorithm for comapring with your own
            dirName='TallinnL1PFTauAnalyzer'
            treeName='TallinnL1PFTauAnalyzer'
            inputPath='/home/sbhowmik/NTuple_Phase2/TallinnL1PFTau'
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
            'L1PFTauPt', 'L1PFTauEta', 'L1PFTauPhi', 'L1PFTauZ', 'L1PFTauIso', 'MC_weight'
            ]

      if trainvar=="testVars"  and bdtType=="default" and all==False :return [
             "L1PFTauPt", "L1PFTauEta", "L1PFTauPhi", "L1PFTauZ", "L1PFTauIso"
            ]
