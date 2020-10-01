import h5py
import numpy as np
#import scipy.io as sio
import spams

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt



# Assuming 1xn matrix
def window_stack(a, length=256, stride=1):
    return np.vstack( a[0,i*stride : i * stride + length ] for i in range(0,(a.shape[1]-length+1)//stride) )

def prepareECGdata(ECGdatafile, length=256, stride=13):
    """
    Prepare ECG data in Numpy format
    """
    with h5py.File('dicts/ECGdata.mat', 'r') as f:
        C1 = [np.array(f[f[f['data'][0,0]][i,0]]) for i in range(f[f['data'][0,0]].shape[0])]
        C2 = [np.array(f[f[f['data'][1,0]][i,0]]) for i in range(f[f['data'][1,0]].shape[0])]
        C3 = [np.array(f[f[f['data'][2,0]][i,0]]) for i in range(f[f['data'][2,0]].shape[0])]
        C4 = [np.array(f[f[f['data'][3,0]][i,0]]) for i in range(f[f['data'][3,0]].shape[0])]

    # Extract segments
    C1mat = np.vstack( window_stack(ECGrecording, length=length, stride=stride) for ECGrecording in C1)
    C2mat = np.vstack( window_stack(ECGrecording, length=length, stride=stride) for ECGrecording in C2)
    C3mat = np.vstack( window_stack(ECGrecording, length=length, stride=stride) for ECGrecording in C3)
    C4mat = np.vstack( window_stack(ECGrecording, length=length, stride=stride) for ECGrecording in C4)

    Call = np.vstack((C1mat, C2mat, C3mat, C4mat)).T
    print("Dataset size = {}x{}".format(Call.shape[0], Call.shape[1]))

    savename = ECGdatafile[:-4] + '_stride{}'.format(stride) + ECGdatafile[-4:]
    np.savez(savename, C1mat, C2mat, C3mat, C4mat, Call)


def generate_dict(ECGdatafile, N, dictfile, iter=-30, pcasize=64):
    """
    Generates an ECG dictionary from available data
    """

    print("Loading data from {}".format(ECGdatafile))
    npzfile = np.load(ECGdatafile)

    C1mat = npzfile['arr_0']
    C2mat = npzfile['arr_1']
    C3mat = npzfile['arr_2']
    C4mat = npzfile['arr_3']
    Call  = npzfile['arr_4']

    if pcasize is not None:
        #U,S,Vt = np.linalg.svd(Call)
        U,E,Ut = np.linalg.svd(Call@Call.T)
        P = U[:,:pcasize]
        Call = P.T @ Call

    print("Dataset size = {}x{}".format(Call.shape[0], Call.shape[1]))

    # Normalize columns
    print('Normalizing the data...')
    for i in range(Call.shape[1]):
        Call[:,i] = Call[:,i] / np.linalg.norm(Call[:,i])

    # Shuffle the columns (Review: is it necessary?)
    #np.random.shuffle(np.transpose(Call))z

    print("Training dictionary of size = {}x{}".format(Call.shape[0], N))
    #D = spams.trainDL(Call, K=N, lambda1=0.2, mode=1, iter=iter)
    D = spams.trainDL(np.asfortranarray(Call), K=N, lambda1=0.01, mode=2, iter=iter, modeD=1, gamma1=0.2)
    print("Created dictionary of size = {}x{}".format(D.shape[0], D.shape[1]))

    # Compute singular values
    [U,S,Vt] = np.linalg.svd(D)
    print('Condition number = {}'.format(S[0]/S[-1]))
    plt.plot(S)
    plt.yscale('log')
    #plt.show()
    plt.title('Singular values of the ECG dictionary')

    # Save dictionary matrix to disk
    dictfile = '{}{}x{}.npz'.format(dictfile, D.shape[0], D.shape[1])
    np.savez(dictfile, D)
    print("Saved dictionary to {}".format(dictfile))

    # Plot and save singular values to image file
    imagename = dictfile[:-4] + '_spectrum'
    plt.savefig(imagename + '.' + 'pdf', bbox_inches='tight')
    plt.savefig(imagename + '.' + 'png', bbox_inches='tight')
    plt.close()
    print("Saved images as {}.png/pdf".format(imagename))

    # Write singular values to file
    textfile = dictfile[:-4] + '_spectrum.txt'   
    with open(textfile, 'w') as f:
        f.write(str(S))
    print("Saved singular values in {}".format(textfile))
    plt.show()


# Main script
if __name__ == "__main__":

    # Run only once to prepare the data in ECGdatafile
    #prepareECGdata('dicts/ECGdata256.npz', stride=13)
    #prepareECGdata('dicts/ECGdata256.npz', stride=25)
    #prepareECGdata('dicts/ECGdata256.npz', stride=3)
    #prepareECGdata('dicts/ECGdata256.npz', stride=7)
    #prepareECGdata('dicts/ECGdata512.npz', length=512, stride=13)

    # Run to generate a dictionary of size 128x128
    # generate_dict('dicts/ECGdata.npz', 256, 'dicts/ECGdict256x256.npz', iter=-300)

    # Run to generate a dictionary of size 256x256
    #generate_dict('dicts/ECGdata256_stride25.npz', 256, 'dicts/ECGdict256x256.npz', iter=-30)
    #generate_dict('dicts/ECGdata256_stride7.npz', 256, pcasize=128, dictfile='dicts/ECGdict', iter=-30)
    generate_dict('dicts/ECGdata512_stride13.npz', 512, pcasize=None, dictfile='dicts/ECGdict', iter=-30)

    # Run to generate a dictionary of size 256x512
    #generate_dict('dicts/ECGdata_stride25.npz', 512, 'dicts/ECGdict256x512.npz', iter=-120)

    print('Done.')