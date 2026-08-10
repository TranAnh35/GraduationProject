#%% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile

from scipy.signal import butter, lfilter, freqz, firwin
from scipy.signal.windows import gaussian
from scipy import signal

from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from sklearn.model_selection import train_test_split
import pywt

#%% Filters ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def butter_lowpass_filter(data, cutoff=0.04, fs=1, order=4):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    data = np.pad(data, (50,50), 'constant',constant_values=(data[0], data[-1]))
    y = lfilter(b, a, data)
    y = y[50+order//2:-50+order//2]
    return y

def LPF_butter(X_train, cutoff=0.04, fs=1, order=4):
    X = []
    for x in X_train: 
        x = butter_lowpass_filter(x, cutoff, fs, order)
        X.append(x)
    X = np.array(X)[:,:,None]
    return X

def LPF(x, cutoff=0.04, order=24):
    # b = firwin(order+1, cutoff, window = ('gaussian', len(x)), pass_zero = False)
    b = firwin(order+1, cutoff, window = 'hamming', pass_zero = "lowpass")
    x = np.pad(x, (50,50), 'constant',constant_values=(x[0], x[-1]))
    y = lfilter(b, 1.0, x)
    y = y[50+order//2:-50+order//2]
    return y

def BPF(x, fl = 0.7, fh = 0.9, order = 24, subtract_mean = True):
    # b = firwin(order+1, [fl, fh], window = ('gaussian', order/5), pass_zero = False)
    b = firwin(order+1, [fl, fh], window = 'hamming', pass_zero = False)
    x = np.pad(x, (50,50), 'constant',constant_values=(x[0], x[-1]))
    if subtract_mean:
        y = lfilter(b, 1.0, x-np.mean(x))
    else:
        y = lfilter(b, 1.0, x)
    y = y[50+order//2:-50+order//2]
    return y

def filter_2D(X, fl=500, fh=1500, fs=100000):
    Y = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            y = BPF(X[i,j], fl/fs, fh/fs, order = 24, subtract_mean = False)
            Y.append(y)
    Y = np.array(Y)
    Y = Y.reshape((X.shape[0],X.shape[1],-1))
    return Y

def LPF_1D(X_train, cutoff=0.04, order=24):
    X = []    
    for x in X_train.squeeze():    
        x = LPF(x, cutoff=cutoff, order=order)
        X.append(x)
    X = np.array(X)
    return X

def LPF_2D(X, cutoff=0.04, order=24):
    Y = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            y = LPF(X[i,j], cutoff, order)
            Y.append(y)
    Y = np.array(Y)
    Y = Y.reshape((X.shape[0],X.shape[1],-1))
    return Y

def filter_1D(X,cutoff=0.04,norm=True):
    X = X - np.mean(X)
    X = LPF_butter(X[None,:], cutoff=cutoff, fs=1, order=2)
    if norm:
        X = norm01(X,axis=(1))
    return X.squeeze()


    
#%% Previos methods ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def diff_method(data,cutoff=0.04,order=2):
    # paper: Pulsed Eddy Current Technology: Characterizing Material Loss with Gap and Lift-off Variations
    # data: N x samples
    # x = norm01(data, axis=(1))
    x = np.diff(data,axis=1)        
    x = LPF_butter(x, cutoff=cutoff, fs=1, order=order).squeeze()
    return x

def norm_method(data,air,ref_id=None,cutoff=0.04,order=2):
    # paper: Reduction of lift-off effects for pulsed eddy current NDT
    # data: N x samples including ref at ref_id
    # data = norm01(data, axis=(1))
    # air = norm01(air, axis=(1))
    x = air - data
    x = LPF_butter(x, cutoff=cutoff, fs=1, order=order).squeeze()
    x = norm01(x, axis=(1))    
    if ref_id is not None:
        x = x - x[ref_id:ref_id+1] # subtract ref
    return x

#%% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def running_avr(x, alpha, c):
    # alpha = 2/(N+1)
    c = alpha*x+ (1-alpha)*c # c will be modified
    y = x - c
    return y, c

def running_avr_2D(x, alpha, c):
    # x = scan_data[2]
    # N = 10
    # alpha = 2/(N+1)
    data_c = np.zeros(x.shape[0])
    y  = []
    for i in range(x.shape[1]):    
        data_ravr, data_c = running_avr(x[:,i], alpha, data_c)
        y.append(data_ravr)
    y = np.array(y).T

    plt.imshow(x, cmap='jet',interpolation='bilinear')
    plt.show()
    plt.imshow(y[:,N:], cmap='jet',interpolation='bilinear')
    plt.show()


    return y

#%% 1D to 2D RASTER SCAN MAPPING ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def reshape_raster(data,sX):
    arr = []
    for i in range(data.shape[0]//sX):
        xi = data[sX*i: sX*(i+1)]
        if i%2 == 1:
            xi = xi[::-1]
        arr.append(xi)
    data = np.array(arr)
    return data

#%% NORM ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def offset(x, axis=(1,2,3)):
    val_mean = np.mean(x,axis = axis, keepdims=True)    
    y = x-val_mean

    return y

def norm01(X_train, X_scan, axis=(1,2,3)):
    val_max = np.max(X_train,axis = axis, keepdims=True)
    val_min = np.min(X_train,axis = axis, keepdims=True)
    X_train = (X_train-val_min)/(val_max-val_min)
    X_scan = (X_train-val_min)/(val_max-val_min)

    return X_train

def standard_norm(X_train, X_scan, axis=0):
    m = np.mean(X_train,axis = axis, keepdims=True)
    st = np.std(X_train,axis = axis, keepdims=True) + 1e-9
    X_train -= m;    X_train /= st  
    X_scan -= m;    X_scan /= st 
    return X_train, X_scan, m, st

def self_minmax(X_train, X_scan, axis=(1,2,3)):
    val_max = np.max(X_train,axis = axis, keepdims=True)
    val_min = np.min(X_train,axis = axis, keepdims=True)
    X_train = (X_train-val_min)/(val_max-val_min)

    val_max = np.max(X_scan,axis = axis, keepdims=True)
    val_min = np.min(X_scan,axis = axis, keepdims=True)
    X_scan = (X_scan-val_min)/(val_max-val_min)

    return X_train, X_scan

def segment(X_train, w=5,s=1):
    N = (len(X_train)-w)//s+1
    X = []
    for i in range(N):
        X.append(X_train[i*s:i*s+w].reshape(-1))
    X = np.array(X)
    X = X[:,:,None]
    return X

def desegment(X,w=5,s=1):
    X2 = []
    for i in range(len(X)):
        X2.append(X[i].reshape(w,-1)[:s])
    X2 = np.array(X2)
    X2 = X2.reshape(-1,X2.shape[-1])
    X2 = np.concatenate([X2, X[i].reshape(w,-1)[s:]])
    return X2

#%% TRAINING DATA ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_csv(path):
    # path = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off2'
    path_list = os.listdir(path)
    files = list(filter(lambda f: f.endswith('.csv') and f.startswith('no'), path_list))
    x_no = []
    for file in files:
        temp = pd.read_csv(os.path.join(path,file),index_col=None, header=None)
        x_no.append(temp)
    x_no = pd.concat(x_no, axis=0, ignore_index=True).to_numpy()

    files = list(filter(lambda f: f.endswith('.csv') and f.startswith('d'), path_list))
    x_c = []
    for file in files:
        temp = pd.read_csv(os.path.join(path,file),index_col=None, header=None)
        x_c.append(temp)
    x_c = pd.concat(x_c, axis=0, ignore_index=True).to_numpy()

    X_train = x_no[:,:,None]
    X_test = x_c[:,:,None]

    return X_train, X_test

#%% SCAN DATA ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def loadscan_tdms(file):
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz30x90.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz150x150v3.tdms'

    tdms_file = TdmsFile.read(file)
    # tdms_file.groups()
    group1 = tdms_file['Freq_Sampling_SizeX_SizeY']
    infor = group1.channels()[0]
    f = infor[:][0]; sampling = infor[:][1]; sX = int(infor[:][2]); sY = int(infor[:][3])
    samples = int(sampling/f)

    group2 = tdms_file['Waveform']
    data = group2.channels()[0][:]
    X_scan = np.reshape(data,(-1,samples))
    
    return X_scan, sX, samples

def read_scan_tdms(file,raster=False):
    data, sX, samples = loadscan_tdms(file)
    if samples > 500:
        samples = samples//4
        idx = np.arange(0,len(data[0]),4)
        data = data[:,idx]
    sY = data.shape[0]//sX

    if raster:
        data = reshape_raster(data,sX)
    return data, samples, sX, sY

def loadscan_tdms_train_test(file):
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz30x90.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz150x150v3.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off_2.0\PECT_collusion_150x150.tdms'

    tdms_file = TdmsFile.read(file)
    # tdms_file.groups()
    group1 = tdms_file['Freq_Sampling_SizeX_SizeY']
    infor = group1.channels()[0]
    f = infor[:][0]; sampling = infor[:][1]; sX = int(infor[:][2]); sY = int(infor[:][3])
    samples = int(sampling/f)

    group2 = tdms_file['Waveform']
    data = group2.channels()[0][:]
    X_scan = np.reshape(data,(-1,samples))
    # ids = 90*sX; idn = 120*sX
    ids = 30*sX; idn = 60*sX
    X_no1 = X_scan[ids:idn,:]
    ids = 90*sX; idn = 120*sX
    X_no2 = X_scan[ids:idn,:]
    ids = 140*sX; idn = 150*sX
    X_no3 = X_scan[ids:idn,:]
    X_no = np.concatenate([X_no2,X_no3])

    X_no = X_no[:,:,None]    
    X_scan = X_scan[:,:,None]

    x = reshape_raster(np.reshape(data,(-1,samples)),sX)
    X_features = np.mean(x[:,:,:50],axis=2)

    # plt.figure()
    # plt.imshow(X_features, cmap='jet',interpolation='bilinear')
    # plt.colorbar()
    # plt.show()

    return X_no, X_scan, X_features, sX, samples

def loadsteel_tdms_train_test(file):
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\Scan steel\step_0.5_Steel_Basic_front_160x110.tdms'
    
    X_scan, sX, samples = loadscan_tdms(file)
    if samples > 500:
        samples = samples//4
        idx = np.arange(0,len(X_scan[0]),4)
        X_scan = X_scan[:,idx]

    ids = 0*sX; idn = 50*sX
    X_no1 = X_scan[ids:idn,:]
    ids = 60*sX; idn = 90*sX
    X_no2 = X_scan[ids:idn,:]
    ids = 120*sX; idn = 140*sX
    X_no3 = X_scan[ids:idn,:]
    ids = 170*sX; idn = 220*sX
    X_no4 = X_scan[ids:idn,:]
    X_no = np.concatenate([X_no1,X_no2,X_no3,X_no4])

    X_no = X_no[:,:,None]    
    X_scan = X_scan[:,:,None]

    x = reshape_raster(X_scan,sX)
    X_features = np.mean(x[:,:,:50],axis=2)

    plt.figure()
    plt.imshow(X_features, cmap='jet',vmax=2.0,vmin=1.0,interpolation='bilinear')
    plt.colorbar()
    plt.show()

    return X_no, X_scan, X_features, sX, samples


def loadscan_tdms_nomal_anomal(file,idsx,idnx):
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz30x90.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz150x150v3.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off_2.0\PECT_collusion_150x150.tdms'

    X_scan, sX, samples = loadscan_tdms(file)
    if samples > 500:
        samples = samples//4
        idx = np.arange(0,len(X_scan[0]),4)
        X_scan = X_scan[:,idx]

    # idsx = [30,90,145]
    # idnx = [60,120,150]
    X_no = [];
    for i in range(len(idsx)):
        ids = idsx[i]*sX; idn = idnx[i]*sX
        x = X_scan[ids:idn,:]
        X_no.append(x)
    X_no = np.concatenate(X_no)

    X_no = X_no[:,:,None]    
    X_scan = X_scan[:,:,None]

    return X_no, X_scan, sX, samples

def loadscan_tdms_nomal_anomal2(file,idsx,idnx):
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz30x90.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off\step_1_f_200_amp_2.0_sz150x150v3.tdms'
    # file = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data\on_off_2.0\PECT_collusion_150x150.tdms'

    X_scan, sX, samples = loadscan_tdms(file)
    if samples > 500:
        samples = samples//4
        idx = np.arange(0,len(X_scan[0]),4)
        X_scan = X_scan[:,idx]
    X_scan -= np.mean(X_scan, axis=1, keepdims=True)

    # idsx = [30,90,145]
    # idnx = [60,120,150]
    X_no = []; X_corrosion = np.copy(X_scan)
    for i in range(len(idsx)):
        ids = idsx[i]*sX; idn = idnx[i]*sX
        x = X_scan[ids:idn,:]
        X_corrosion[ids:idn,:] = 0
        X_no.append(x)
    X_no = np.concatenate(X_no)

    X_no = X_no[:,:,None]    
    X_scan = X_scan[:,:,None]

    return X_no, X_corrosion, X_scan, sX, samples

def load_nomal_anomal2():
    # folder = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data'
    parent_dir = os.path.split(os.getcwd())[0]
    folder = parent_dir + '/Data/Minh/Gaussian/A2024/Corrosion/differential'
    
    idsx_al = [70,185,300]; idnx_al = [130,260,320]
    idsx_steel = [20,65,120,175]; idnx_steel = [45,85,140,200]
    files = []; idsx =[]; idnx =[]


    files.append('/above_side_differential_lf_1mm_amp2.1mA.tdms')
    idsx.append(idsx_al); idnx.append(idnx_al)
    
    files.append('/above_side_differential_lf_2mm_amp2.1mA.tdms')
    idsx.append(idsx_al); idnx.append(idnx_al)
    
    files.append('/above_side_differential_lf_3mm_amp2.1mA.tdms')
    idsx.append(idsx_al); idnx.append(idnx_al)

    X_no = []; X_scan = []; sX = []; samples = []
    for i in range(len(files)):
        file = files[i]; ids = idsx[i]; idn = idnx[i]
        X_no_i, X_scan_i, sX_i, samples_i = loadscan_tdms_nomal_anomal(folder+file,ids,idn)
        # X_scan_i -= X_scan_i[0]
        # X_no_i -= X_no_i[0]
        X_scan_i -= np.mean(X_scan_i, axis=1, keepdims=True)
        X_no_i -= np.mean(X_no_i, axis=1, keepdims=True)
        X_no.append(X_no_i); X_scan.append(X_scan_i); sX.append(sX_i); samples.append(samples_i)    

    return X_no, X_scan, sX, samples


def scan_features_plot(data,sX,segment,method='mean',do_raster=True):
# data is a list of 2D array (N,samples)
    plt.figure(figsize=(5,12))
    for i in range(len(data)):
        plt.subplot(len(data),1,i+1)
        if do_raster:
            x = reshape_raster(data[i],sX)
        else:
            x = data[i]

        # x = data[i].reshape(-1,sX,512,1)        
        if method == 'abs':
            x_features = np.mean(np.abs(x[:,:,segment]),axis=2)
        elif method == 'rms':
            x_features = np.mean(x[:,:,segment]**2,axis=2)**0.5
        elif method == 'std':
            x_features = np.std(x[:,:,segment],axis=2)
        else: # mean
            x_features = np.mean(x[:,:,segment],axis=2)
            
        plt.imshow(x_features[:,:], cmap='jet',interpolation='bilinear')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.show()

def scan_features_plot_hor(data,sX,segment,method='mean',do_raster=True):
# data is a list of 2D array (N,samples)
    plt.figure(figsize=(12,4))
    for i in range(len(data)):
        plt.subplot(1,len(data),i+1)
        if do_raster:
            x = reshape_raster(data[i],sX)
        else:
            x = data[i]

        # x = data[i].reshape(-1,sX,512,1)        
        if method == 'abs':
            x_features = np.mean(np.abs(x[:,:,segment]),axis=2)
        elif method == 'rms':
            x_features = np.mean(x[:,:,segment]**2,axis=2)**0.5
        elif method == 'std':
            x_features = np.std(x[:,:,segment],axis=2)
        else: # mean
            x_features = np.mean(x[:,:,segment],axis=2)
            
        plt.imshow(x_features[:120,:], cmap='jet',interpolation='bilinear')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.show()

def extract_features(data,segment,method='mean',sub_bg_x=False,sub_bg_y=False,plot=False,do_raster=False, sX=1):
    # data = (data)/(data.max())
    if do_raster:
            data = reshape_raster(data,sX)
    if method == 'abs':
        x_features = np.mean(np.abs(data[:,:,segment]),axis=2)
    elif method == 'max':
        x_features = np.max(data[:,:,segment],axis=2)
    elif method == 'rms':
        x_features = np.mean(data[:,:,segment]**2,axis=2)
    elif method == 'std':
        x_features = np.std(data[:,:,segment],axis=2)
    else: # mean
        x_features = np.mean(data[:,:,segment],axis=2)

    if sub_bg_x:
        x_features = x_features - x_features[:,0][:,None]
    if sub_bg_y:
        x_features = x_features - x_features[0,:][None,:]

    if plot:
        plt.figure()
        plt.imshow(x_features, cmap='jet',interpolation='bilinear')
        plt.show()
    return x_features

def plot_img(x_features):
    plt.figure()
    plt.imshow(x_features, cmap='jet',interpolation='bilinear')
    plt.show()


def load_nomal_anomal():
    # folder = 'D:\OneDrive - Phenikaa Univesity\RESEARCH DIRECTIONS\AI NDT\PECT\Data'
    parent_dir = os.path.split(os.getcwd())[0]
    folder = parent_dir + '/Data'
    
    files = []; idsx =[]; idnx =[]

    # files.append('\San 9 Colusiom 30_5_23\step_0.5_AL_basic_front_160x160.tdms')
    # # idsx.append([60,180,300]); idnx.append([140,260,320])
    # idsx.append([70,180]); idnx.append([140,260])

    # files.append('\San 9 Colusiom 30_5_23\step_0.5_AL_basic_behind_160x160.tdms')
    # # idsx.append([60,180,300]); idnx.append([140,260,320])
    # idsx.append([70,180]); idnx.append([140,260])

    files.append('/Scan 9 Colusiom/Colusion_basic_front.tdms')
    # idsx.append([30,90,140]); idnx.append([60,120,150])
    idsx.append([90,140]); idnx.append([120,150])

    files.append('/on_off_2.0/PECT_collusion_150x150.tdms')
    # idsx.append([30,90,140]); idnx.append([60,120,150])
    idsx.append([90,140]); idnx.append([120,150])

    # files.append('\San 9 Colusiom 30_5_23\step_0.5_AL_basic_behind_160x160.tdms')
    # # idsx.append([70,185,300]); idnx.append([130,260,320])
    # idsx.append([185,300]); idnx.append([260,320])
    

    X_no = []; X_scan = []; sX = []; samples = []
    for i in range(len(files)):
        file = files[i]; ids = idsx[i]; idn = idnx[i]
        X_no_i, X_scan_i, sX_i, samples_i = loadscan_tdms_nomal_anomal(folder+file,ids,idn)
        X_no.append(X_no_i); X_scan.append(X_scan_i); sX.append(sX_i); samples.append(samples_i)    

    return X_no, X_scan, sX, samples

#% Data for classification model ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data_cls():
    
    #% SCAN DATA ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    _, X_scan, sX, samples = load_nomal_anomal()

    #% Scan data ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    no_data = 0
    X_scan = X_scan[no_data]
    sX = sX[no_data]
    samples = samples[no_data]

    print('Scan size: ', X_scan.shape)

    #% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    X_org = reshape_raster(X_scan,sX).squeeze()

    #% Separate Corrosion and Background ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # function to return a matrix with points within a circle having center at (px, py) and radius r, the other points are set to zero

    def circle_mask(px, py, r, X):
        # X could be 2D (feature) or 3D (raw)
        mask = X.copy()
        for x in range(px-r, px+r):
            for y in range(py-r, py+r):
                if (x-px)**2 + (y-py)**2 <= r**2:
                    mask[x,y] = 0
        return mask

    # P = [[15,12,8],[15,73,6],[15,133,5],[75,12,5],[75,73,5],[75,133,5]]
    # P = [[15,12,8],[15,73,6],[15,133,5],[75,12,5],[75,73,5],[75,133,5],[134,12,5],[134,73,5],[134,133,5]]
    P = [[75,12,5],[75,73,5],[75,133,5],[134,12,5],[134,73,5],[134,133,5]]

    # Raw pulse
    X_ground = []
    X_corrosion = []

    # a = X_org[:120,:]  
    a = X_org  
    x = a
    for pi in P:
        x = circle_mask(pi[0],pi[1],pi[2],x)
    X_ground = x
    X_corrosion = a-x


    X_ground_feat = extract_features(X_ground,range(250,320),method='mean',sub_bg_x=False,sub_bg_y=False,plot=False)
    X_corrosion_feat = extract_features(X_corrosion,range(250,320),method='mean',sub_bg_x=False,sub_bg_y=False,plot=False)


    plt.figure(figsize=(6, 10))
    plt.subplot(3,1,1)
    plt.imshow(X_ground_feat,cmap='jet',interpolation='bilinear')    
    plt.colorbar()
    plt.title(' ')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,1,2)
    plt.imshow(X_corrosion_feat,cmap='jet',interpolation='bilinear')    
    plt.colorbar()
    plt.title(' ')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,1,3)
    plt.imshow(X_ground_feat + X_corrosion_feat,cmap='jet',interpolation='bilinear')    
    plt.colorbar()
    plt.title(' ')
    plt.xticks([])
    plt.yticks([])
    plt.show()


    # %

    X_ground = X_ground[50:,:]
    Xg = X_ground[X_ground_feat[50:,:]!=0]
    Xc = X_corrosion[X_corrosion_feat!=0]
    yg = np.array([0]*len(Xg))
    yc = np.array([1]*len(Xc))
    X = np.concatenate([Xg,Xc]); X = X[:,:,None]
    y = np.concatenate([yg,yc])


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


#%% Data preprocessing functions ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def reduce_resolution(data, factor=1):
    # Take every other row and column (downsample by factor)
    return data[::1, ::1, ::factor]

def wavelet_denoise(image, wavelet='db4', level=2, threshold_ratio=5):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresh = []
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        sigma = np.median(np.abs(cD)) / 0.6745
        threshold = threshold_ratio * sigma
        cH = pywt.threshold(cH, threshold, mode='soft')
        cV = pywt.threshold(cV, threshold, mode='soft')
        cD = pywt.threshold(cD, threshold, mode='soft')
        coeffs_thresh.append((cH, cV, cD))
    coeffs_denoised = [coeffs[0]] + coeffs_thresh
    return pywt.waverec2(coeffs_denoised, wavelet)

def detrend_1d(image):
    detrended_rows = np.apply_along_axis(detrend, axis=1, arr=image)
    detrended = np.apply_along_axis(detrend, axis=0, arr=detrended_rows)
    return detrended

def normalize_img(img):
    img_min, img_max = np.min(img), np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)

def detrend_2d_surface(image, degree=3):
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)

    poly = PolynomialFeatures(degree)
    XY_poly = poly.fit_transform(XY)

    model = LinearRegression().fit(XY_poly, image.ravel())
    trend_surface = model.predict(XY_poly).reshape(h, w)

    return image - trend_surface

def remove_background_trend(img, method="gaussian", sigma_trend=30, sigma_noise=1, poly_deg=2):
    """
    Remove background trend from a 2D image -> then Filter noise
    
    Parameters:
        img : np.ndarray
            Input image (2D grayscale or 3D RGB).
            Input image (2D grayscale or 3D RGB).
        method : str
            'gaussian' = smooth background with Gaussian filter
            'polyfit'  = fit 2D polynomial surface
        sigma : int
            Gaussian kernel std for smoothing background (if gaussian method).
        poly_deg : int
            Polynomial degree for background fitting (if polyfit method).
    
    Returns:
        detrended : np.ndarray
            Image with background removed.
        background : np.ndarray
            Estimated background.
    """
    img = img.astype(np.float32)
    
    if method == "gaussian":
        # Estimate background with heavy Gaussian blur
        background = gaussian_filter(img, sigma=sigma_trend)
        
    elif method == "polyfit":
        # Fit 2D polynomial surface to the image
        x = np.linspace(0, 1, img.shape[1])
        y = np.linspace(0, 1, img.shape[0])
        X, Y = np.meshgrid(x, y)
        
        Z = img.ravel()
        A = []
        for i in range(poly_deg + 1):
            for j in range(poly_deg + 1 - i):
                A.append((X.ravel()**i) * (Y.ravel()**j))
        A = np.vstack(A).T
        
        coeff, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        background = (A @ coeff).reshape(img.shape)
        
    else:
        raise ValueError("Unknown method: choose 'gaussian' or 'polyfit'")
    
    detrended = img - background

    detrended = gaussian_filter(detrended, sigma=sigma_noise)

    return detrended, background


def load_filter(file_diff):
    """
    Load and filter scan data.

    Args:
        file_diff (str): File path of the differential scan data.

    Returns:
        X_filter_diff (np.ndarray): Filtered differential scan data.
        sX_diff (int): Number of columns in the scan.
        sY_diff (int): Number of rows in the scan.
    """
    data_diff, samples_diff, sX_diff, sY_diff = read_scan_tdms(file_diff, raster=True)
    data_diff = reduce_resolution(data_diff)
    data_diff -= np.mean(data_diff, axis=2, keepdims=True)
    sY_diff = data_diff.shape[0]
    sX_diff = data_diff.shape[1]

    # Apply LPF using LPF_2D
    data_diff = data_diff - data_diff[:, :, :1]
    X_filter_diff = LPF_2D(data_diff, cutoff=0.04, order=24)
    X_filter_diff -= np.mean(X_filter_diff, axis=2, keepdims=True)

    return X_filter_diff, sX_diff, sY_diff, samples_diff


def extract_center_area(img, x, y, size=20):
    """
    Extract a square area of given size around the center (x, y) from img.
    img: 2D numpy array
    x, y: center coordinates
    size: half-size of the square (total size will be 2*size+1)
    Returns: extracted area as numpy array
    """
    x = int(x)
    y = int(y)
    x_min = max(x - size, 0)
    x_max = min(x + size + 1, img.shape[1])
    y_min = max(y - size, 0)
    y_max = min(y + size + 1, img.shape[0])
    return img[x_min:x_max, y_min:y_max]

def convert_stft(X, nperseg=8,nfft = 256, scaling='psd'):
    Z = []
    for i in range(len(X)):
        f, t, z = signal.stft(X[i], 1, nperseg = nperseg, nfft = nfft, scaling = scaling)
        Z.append(np.abs(z))
        if i%10000 == 0:
        # print i with percentage
            print(f" {100*i/len(X)} %")
    Z = np.array(Z)
    return Z