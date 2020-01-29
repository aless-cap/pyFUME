from BuildTakagiSugeno import BuildTSFIS 
import numpy as np

class pyFUME(object):
    def __init__(self, datapath, nr_clus, method='Takagi-Sugeno', variable_names=None, **kwargs):
        self.datapath=datapath
        self.nr_clus=nr_clus
        self.method=method
        self.variable_names=variable_names
            
            
        if method=='Takagi-Sugeno' or method=='Sugeno':
            self.FIS = BuildTSFIS(self.datapath, self.nr_clus, self.variable_names, **kwargs)
        else:
            raise Exception ("This modeling technique has not yet been implemented.")

    def get_model(self):
        if self.FIS.model is None:
            print ("ERROR: model was not created correctly, aborting.")
            exit(-1)
        else:
            return self.FIS.model

    def get_RMSE(self):
        if self.FIS.error is None:
            print ("ERROR: RMSE was not calculated correctly, aborting.")
            exit(-1)
        else:
            return self.FIS.RMSE
        

if __name__=='__main__':
    
    RMSE=np.ones([20,1])
    for i in range(0,20):
        FIS = pyFUME(datapath='./Tests for WCCI 2020/Concrete_data.csv', nr_clus=4, method='Takagi-Sugeno')
        RMSE[i]=FIS.FIS.RMSE
    print(RMSE)
    
             