import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np



# #matdata = loadmat('/mnt/D/CS/ECG_Classif_MitBih/data/dicts/mitdb106_seg1_DictsKsvdOMP_N64_iter40_L4.mat')
# #D = matdata['DOMP']
# #matdata = loadmat('/mnt/D/CS/ECG_Classif_MitBih/data/dicts/mitdb106_seg1_DictsKsvdPrec_N64_iter40_L4.mat')
# #matdata = loadmat('/mnt/D/CS/ECG_Classif_MitBih/data/dicts/mitdb106_seg1_DictsAnyPrecFrameDiag_NoReplace_N64_iter40_L4.mat')
# matdata = loadmat('/mnt/D/CS/ECG_Classif_MitBih/data/dicts/mitdb106_seg1_DictsFrameDiag_N64_iter40_L4.mat')
# D = matdata['DGLSP']
# U,S,Vt = scipy.linalg.svd(D)
# print('Condition number = {}'.format(S[0]/S[-1]))
# plt.plot(S)
# plt.show()

# Replot singular value spectrum
dictfile = 'dicts/ECGdict{}x{}.npz'.format(512, 512)    
print('Loading dictionary from {}'.format(dictfile))
npzfile = np.load(dictfile)
D = npzfile['arr_0']
[U,S,Vt] = np.linalg.svd(D)
print('Condition number = {}'.format(S[0]/S[-1]))
plt.plot(S)
plt.yscale('log')
#plt.show()
#plt.title('Singular values of the ECG dictionary')
# Plot and save singular values to image file
imagename = dictfile[:-4] + '_spectrum'
plt.savefig(imagename + '.' + 'pdf', bbox_inches='tight')
plt.savefig(imagename + '.' + 'png', bbox_inches='tight')
plt.close()
print("Saved images as {}.png/pdf".format(imagename))
