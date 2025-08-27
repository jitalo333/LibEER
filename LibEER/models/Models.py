from models.CoralDgcnn import CoralDgcnn
from models.DGCNN import DGCNN
from models.DannDgcnn import DannDgcnn
# from models.RGNN import RGNN
#from models.RGNN_official import SymSimGCNNet
#from models.EEGNet import EEGNet
#from models.STRNN import STRNN
#from models.GCBNet import GCBNet
from models.DBN import DBN
#from models.TSception import TSception
#from models.SVM import SVM
from models.CDCN import CDCN
#from models.HSLT import HSLT
#from models.ACRNN import ACRNN
#from models.GCBNet_BLS import GCBNet_BLS
#from models.MsMda import MSMDA
#from models.R2GSTNN import R2GSTNN
#from models.BiDANN import BiDANN
#from models.FBSTCNet import PowerAndConneMixedNet
#from models.PRRL import PRRL
#from models.NSAL_DGAT import Domain_adaption_model

Model = {
    'DGCNN': DGCNN,
    'CoralDgcnn': CoralDgcnn,
    'DannDgcnn': DannDgcnn,
    'CDCN': CDCN,
    'DBN': DBN,
}
