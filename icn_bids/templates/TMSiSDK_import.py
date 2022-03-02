import TMSiSDK
from TMSiSDK import *
from TMSiSDK.file_readers import Poly5Reader
file = r"C:\Users\Jonathan\Documents\DATA\PROJECT_BERLIN_Conversion\sourcedata\sub-010\ses-EcogLfpMedOff01\532LO56_MedOff1_Rest_StimOff_1 - 20220207T113556\532LO56_MedOff1_Rest_StimOff_1-20220207T113556.DATA.Poly5"
data = Poly5Reader(file)
data.pick(["ecog", "seeg"]).plot(duration=2,scalings='auto')
plt.show()