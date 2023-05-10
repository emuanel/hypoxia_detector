import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import find_peaks
from pandas_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import pywt
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import pickle 
from sklearn.metrics import f1_score
BP_TQR_BASE_path = "Dane_do_analiz/BP_TQR_ECG_fs100Hz/BP_TQR_BASE_"
ECG_BASE_path = "Dane_do_analiz/BP_TQR_ECG_fs100Hz/ECG_BASE_"
NIRS_BASE_path = "Dane_do_analiz/NIRS_fs10Hz/NIRS_BASE_"

BP_TQR_BASE = glob.glob(BP_TQR_BASE_path + "*.dat")
ECG_BASE = glob.glob(ECG_BASE_path + "*.dat")
NIRS_BASE = glob.glob(NIRS_BASE_path + "*.dat")

BP_TQR_HYPO_path = "Dane_do_analiz/BP_TQR_ECG_fs100Hz/BP_TQR_HYPO_"
ECG_HYPO_path = "Dane_do_analiz/BP_TQR_ECG_fs100Hz/ECG_HYPO_"
NIRS_HYPO_path= "Dane_do_analiz/NIRS_fs10Hz/NIRS_HYPO_"

BP_TQR_HYPO = glob.glob(BP_TQR_HYPO_path + "*.dat")
ECG_HYPO = glob.glob(ECG_HYPO_path + "*.dat")
NIRS_HYPO = glob.glob(NIRS_HYPO_path + "*.dat")

BP_TQR_ECG_fs = 100
NIRS_fs = 10
t_start = 10*60 
t_stop = 30*60

BP_TQR_ECG_sample_start = t_start * BP_TQR_ECG_fs
BP_TQR_ECG_sample_stop = t_stop * BP_TQR_ECG_fs
BP_TQR_ECG_samples = BP_TQR_ECG_sample_stop-BP_TQR_ECG_sample_start

NIRS_sample_start = t_start * NIRS_fs
NIRS_sample_stop = t_stop * NIRS_fs
NIRS_samples = NIRS_sample_stop - NIRS_sample_start

class_NORM, class_HYPO = (0,1)

data = []
y = []
for i, j, m in zip(BP_TQR_BASE, ECG_BASE, NIRS_BASE):
    BP_TQR = np.loadtxt(i, skiprows = BP_TQR_ECG_sample_start, max_rows = BP_TQR_ECG_samples)
    ECG = np.loadtxt(j, skiprows = BP_TQR_ECG_sample_start, max_rows = BP_TQR_ECG_samples)
    NIRS = np.loadtxt(m, skiprows = NIRS_sample_start, max_rows = NIRS_samples)
    y.append(class_NORM)
    data.append([BP_TQR[:,0], BP_TQR[:,1], ECG, NIRS[:,3], class_NORM])
    
for i, j, m in zip(BP_TQR_HYPO, ECG_HYPO, NIRS_HYPO):
    BP_TQR = np.loadtxt(i, skiprows = BP_TQR_ECG_sample_start, max_rows = BP_TQR_ECG_samples)
    ECG = np.loadtxt(j, skiprows = BP_TQR_ECG_sample_start, max_rows = BP_TQR_ECG_samples)
    NIRS = np.loadtxt(m, skiprows = NIRS_sample_start, max_rows = NIRS_samples)
    y.append(class_HYPO)
    data.append([BP_TQR[:,0], BP_TQR[:,1], ECG, NIRS[:,3], class_HYPO])
    
def plot(ax, signal, signal_type):

    lenght = 1000
    
    if signal_type == "względna zmiana oksyhemoglobiny":
        lenght = 200
        X = np.linspace(0, lenght/10, lenght)
    else:
        X = np.linspace(0, lenght/100, lenght)
    ax.plot(X, signal[0:lenght])
    peaks, _ = find_peaks(-signal[0:lenght], distance=55)
    #ax.plot(peaks, signal[0:lenght][peaks], "or"); 
    ax.set_xlabel('czas [s]', fontsize=12)
    ax.set_title(signal_type, fontsize=14)
        
def remove_trend(signal, window = 1000):
    '''
    Function Using Moving Averages to Smooth Time Series Data

    Parameters
    ----------
    signal : Numpy array
        1d signal to remove trend
    window : int   
        Size of the moving window.
    Returns
    -------
    pandas Series
        signal without trend.

    '''
    average = pd.Series(signal).rolling(window, min_periods = 1).mean()
    return pd.Series(signal)-average  

signal_types = ["ciśnienie krwi", "szerokość przestrzeni podpajęczynówkowej","EKG", "względna zmiana oksyhemoglobiny"]

data_without_trend = data.copy()
for index in range(len(data)):
    data_without_trend[index][1] = remove_trend(data[index][1])
    data_without_trend[index][3] = remove_trend(data[index][3])

for record in data_without_trend[::10]:
    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True) 
    fig.suptitle('Hipoksja', fontsize=16) if record[4] == 1 else fig.suptitle('Normoksja', fontsize=16)
    for ax, signal, signal_type in zip(axs.flat, record, signal_types):
        plot(ax, signal, signal_type)
        
def signal_split(data, window_size=10000, step = 10000):
    '''
    split long signals into small chunks with using moving window

    Parameters
    ----------
    data : list
        list containing records with signals and class.
    window_size : int, optional
        target length of the signals. The default is 10000.
    step : int, optional
        step length for moving window. The default is 10000.

    Returns
    -------
    data_splited : list 
        list containing splited records
    y : TYPE
        list containing classes

    '''
    BP_TQR_ECG_NIRS_fs = (100, 100, 100, 10)
    NIRS_step = int(step*BP_TQR_ECG_NIRS_fs[3]/BP_TQR_ECG_NIRS_fs[2])
    NIRS_window_size = int(window_size*BP_TQR_ECG_NIRS_fs[3]/BP_TQR_ECG_NIRS_fs[2])
    y = []
    data_splited = []
    for record in data:
        start = 0
        stop = window_size
        start_NIRS = 0
        stop_NIRS = NIRS_window_size
        for i in range(int(len(record[0])/step)):
            data_splited.append([record[0][start+i*step:stop+i*step], 
                                record[1][start+i*step:stop+i*step], 
                                record[2][start+i*step:stop+i*step], 
                                record[3][start_NIRS+i*NIRS_step:stop_NIRS+i*NIRS_step], 
                                record[4]])   
            y.append(record[4])
    return data_splited, y
window_size = 10000 
step = 10000
data_more, y = signal_split(data_without_trend, window_size = window_size, step = step)

def cwt_coeffs(X, n_scales, wavelet_name='morl'):
    '''
    Performs a continuous wavelet transform on data, using the wavelet function.

    Parameters
    ----------
    X : list
        list containing splited records.
    n_scales : int
        the scale size - Widths to use for transform..
    wavelet_name : string, optional
        wavelet name. The default is "morl".

    Returns
    -------
    median_comps : TYPE
        numpy array containing the results of continuous wavelet transform.

    '''
    # create range of scales
    widths = np.arange(1, n_scales + 1)
    median_comps = np.empty((0, n_scales),dtype='float32')
    for sample in X:
        signal = sample[1]
        coef, freqs = pywt.cwt(signal, widths, wavelet_name)  
        magn = np.median(np.absolute(coef), axis=1).flatten()[::-1]
        median_comps = np.vstack([median_comps, magn]) 
        # plt.figure(figsize=(15, 10))
        # plt.plot(np.flip(freqs)[:110],magn[:110])
        # plt.show()
        # plt.figure(figsize=(15, 10))
        # plt.plot(magn)   
        # plt.show()
    return median_comps
 
# define the scale size
n_scales = 128
SAS_cwt = cwt_coeffs(data_more, n_scales)

def heart_rate(electrocardiogram, distance_RR = 75, ECG_fs = 100):
    '''
    calculate heart rate from electrocardiogram

    Parameters
    ----------
    electrocardiogram : numpy array
        ECG signal .
    distance_RR : int, optional
        distance between RR waves. The default is 75.
    ECG_fs : int, optional
        frequency sampling. The default is 100.

    Returns
    -------
    float
        heart rate/pulse.

    '''
    peaks, _ = find_peaks(electrocardiogram, distance=distance_RR)
    periods = np.subtract(peaks[1:], peaks[:-1])
    median_period = np.median(periods)
    return (60/(median_period/ECG_fs))

def systolic_blood_pressure(BP_signal, distance_peaks = 45):
    '''
    calculate systolic blood pressure from blood pressure signal

    Parameters
    ----------
    BP_signal : numpy array
        blood pressure signal.
    distance_peaks : int, optional
        distance between peaks. The default is 45.

    Returns
    -------
    SBP : float
        systolic blood pressure.

    '''
    peaks, _ = find_peaks(BP_signal, distance = distance_peaks)
    SBP = np.mean(BP_signal[peaks])
    return SBP

def diastolic_blood_pressure(BP_signal, distance_peaks = 45):
    '''
    calculate diastolic blood pressure from blood pressure signal

    Parameters
    ----------
    BP_signal : numpy array
        blood pressure signal.
    distance_peaks : int, optional
        distance between peaks. The default is 45.

    Returns
    -------
    DBP : float
        diastolic blood pressure.

    '''
    peaks, _ = find_peaks(-BP_signal, distance = distance_peaks)
    DBP = np.mean(BP_signal[peaks])
    
    return DBP

def average_arterial_pressure(BP_signal, distance_peaks = 45):   
    '''
    calculate average arterial pressure from blood pressure signal
    
    BP_min + 0.3333*(BP_max-BP_min) 

    Parameters
    ----------
    BP_signal : numpy array
        blood pressure signal.
    distance_peaks : int, optional
        distance between peaks. The default is 45.
    
    Returns
    -------
    AAP : float
        average arterial pressure.
    
    '''             
    peaks_max, _ = find_peaks(BP_signal, distance = distance_peaks)
    peaks_min, _ = find_peaks(-BP_signal, distance = distance_peaks)
    
    return diastolic_blood_pressure(BP_signal) + 1/3 * (systolic_blood_pressure(BP_signal) - diastolic_blood_pressure(BP_signal))

def subarachnoid_width(SAS_signal, distance_peaks = 55):           
    '''
    calculate mean subarachnoid width from SAS signal
    0.5*(SAS_min+SAS_max) 

    Parameters
    ----------
    SAS_signal : numpy array
        SAS signal.
    distance_peaks : int, optional
        distance between peaks. The default is 55.

    Returns
    -------
    float
        mean subarachnoid width.

    '''                     
    peaks_max, _ = find_peaks(SAS_signal, distance = distance_peaks)
    peaks_min, _ = find_peaks(-SAS_signal, distance = distance_peaks)
    
    #return (np.mean(SAS_signal[peaks_max]) + np.mean(SAS_signal[peaks_min]))/2

    return np.mean(SAS_signal)

def average_oxygenated_haemoglobin(HbO2_signal):  
    '''
    calculate average oxygenated haemoglobin OH2 signal

    Parameters
    ----------
    OH_signal : numpy array
        Oxyhemoglobin signal.

    Returns
    -------
    np.mean(OH_signal) : float
        average oxygenated haemoglobin.
    '''                  
    return np.mean(HbO2_signal)

pulses = []
diastolic_BP = []
systolic_BP = []
average_AP = []
SAS_width = []
average_oxygenated = [] 

for record in data_more:
    pulses.append(heart_rate(record[2]))
    systolic_BP.append(systolic_blood_pressure(record[0]))
    diastolic_BP.append(diastolic_blood_pressure(record[0]))
    average_AP.append(average_arterial_pressure(record[0]))
    SAS_width.append(subarachnoid_width(record[1]))
    average_oxygenated.append(average_oxygenated_haemoglobin(record[3]))
df = pd.concat([pd.DataFrame(pulses),pd.DataFrame(systolic_BP), 
                pd.DataFrame(diastolic_BP), pd.DataFrame(average_AP), 
                pd.DataFrame(SAS_width), pd.DataFrame(average_oxygenated), 
                pd.DataFrame(y)], axis=1)

columns = ["tetno", "skurczowe BP", "rozkurczowe BP", "tetnicze BP", 
           "szerokosc SAS", "srednie HbO2", "hipoksja"]
df.columns = columns
df.to_csv("data1.csv")
extracted_features = df.iloc[:,:-1]
extracted_features2 = pd.concat([pd.DataFrame(SAS_cwt),extracted_features], axis=1)
                                
print(df['hipoksja'].value_counts()[1], " obiektów z hypoxia; ", 
      df['hipoksja'].value_counts()[0], " obiektów z normoxia")
            

# report = ProfileReport(df)
# report.to_file("data.html")



# scaler = StandardScaler()
# extracted_features_scaled = scaler.fit_transform(extracted_features2)

#PCA
def dimensionality_reduction_PCA(data, n_components=10):
    '''
    Dimensionality reduction with using PCA and plot variance(components)

    Parameters
    ----------
    data : numpy array
        array containing data to reduction.
    n_components : int, optional
        Number of components to keep. The default is 10.

    Returns
    -------
    X_new : TYPE
         data with applied the dimensionality reduction 

    '''
    pca1 = PCA()
    X_new = pca1.fit_transform(data)        #Fit the model with X
    explained_variance1 = pca1.explained_variance_ratio_
    
    plt.figure(figsize=(12, 8))
    plt.title('PCA wariancja składowych głównych', fontsize= 20)
    plt.bar(np.arange(0,len(explained_variance1[:10])), explained_variance1[:10], alpha=0.5, align='center')
    plt.ylabel('Wariancja', fontsize= 15)
    plt.xlabel('Główne składowe', fontsize= 15)
    plt.show()
    plt.savefig('PCA')
    pca1 = PCA(n_components=n_components) 
    X_new = pca1.fit_transform(data) #apply the dimensionality reduction to data
    
    with open('pca.pkl', 'wb') as pickle_file:
            pickle.dump(pca1, pickle_file)
    return X_new

n_components = 10
X_new = dimensionality_reduction_PCA(extracted_features2, n_components=n_components)

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

def LogisticRegressionCV_train(trainX, trainY):
    '''
    Exhaustive search over specified parameter values for an Logistic Regression model.

    Parameters
    ----------
    trainX : numpy array
        train data.
    trainY : numpy array
        labels.

    Returns
    -------
    bestModel : model
        best Logistic Regression model.

    '''
    penalty = ['l1', 'l2', 'elasticnet', 'none']
    C = [0.01, 0.1, 1, 100]
    grid = dict(penalty=penalty, C=C)
    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    model = LogisticRegression()
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
    	cv=cvFold, scoring="accuracy")
    searchResults = gridSearch.fit(trainX, trainY)
    bestModel = searchResults.best_estimator_
    return bestModel
    
def SVM_train(trainX, trainY):
    '''
    Exhaustive search over specified parameter values for an SVM model.

    Parameters
    ----------
    trainX : numpy array
        train data.
    trainY : numpy array
        labels.

    Returns
    -------
    bestModel : model
        best SVM model.

    '''

    kernel = ["linear", "rbf", "sigmoid", "poly"]
    tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
    C = [1, 1.5, 2, 2.5, 3]
    grid = dict(kernel=kernel, tol=tolerance, C=C)
    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = SVC()
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
    	cv=cvFold, scoring="accuracy")
    searchResults = gridSearch.fit(trainX, trainY)
    bestModel = searchResults.best_estimator_
    return bestModel
  
def RandomForestClassifier_train(trainX, trainY):
    '''
    Exhaustive search over specified parameter values for an Random Forest model.

    Parameters
    ----------
    trainX : numpy array
        train data.
    trainY : numpy array
        labels.

    Returns
    -------
    bestModel : model
        best Random Forest model.

    '''
    n_estimators = [10,50,100,500]
    criterion = ["gini", "entropy", "log_loss"]
    max_features = ["sqrt", "log2", None]
    grid = dict(n_estimators=n_estimators, criterion=criterion, max_features=max_features)
    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    model = RandomForestClassifier()
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
    	cv=cvFold, scoring="accuracy")
    searchResults = gridSearch.fit(trainX, trainY)
    bestModel = searchResults.best_estimator_
    return bestModel

def XGB_train(trainX, trainY):
    '''
    Exhaustive search over specified parameter values for an XGBoost model.

    Parameters
    ----------
    trainX : numpy array
        train data.
    trainY : numpy array
        labels.

    Returns
    -------
    bestModel : model
        best XGBoost model.

    '''
    n_estimators = [10,50,100,500]
    max_depth = [0, 3, 4, 5]
    grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    model = XGBClassifier(verbosity = 0, silent=True)
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
     	cv=cvFold, scoring="accuracy")
    searchResults = gridSearch.fit(trainX, trainY)
    bestModel = searchResults.best_estimator_
    return bestModel

def build_clf(units1, units2, units3, units4, input_shape):
    '''
    build artificial neural network
    Parameters
    ----------
    units1 : int
        units for first layer.
    units2 : int
        units for second layer.
    units3 : int
        units for third layer.
    units4 : int
        units for fourth layer.
    Returns
    -------
    ann : model
        compiled artificial neural network.

    '''

    # creating the layers of the NN
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape)),
        tf.keras.layers.Dense(units=units1,activation="relu"),
        tf.keras.layers.Dense(units=units2,activation="relu"),
        tf.keras.layers.Dense(units=units3,activation="relu"),
        tf.keras.layers.Dense(units=units4,activation="relu"),
        tf.keras.layers.Dense(units=1,activation="sigmoid")]
        )
    
    ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    return ann

def ANN_train(trainX, trainY):
    '''
    Exhaustive search over specified parameter values for an artificial neural network.
    Parameters
    ----------
    trainX : numpy array
        train data.
    trainY : numpy array
        labels.
    Returns
    -------
    bestModel : model
        best artificial neural network model.

    '''
    epochs=[5,10,20]
    batch_size=[2,8,16,32]
    units = [8,16,32,48,64,128]         
    grid = dict(nb_epoch=epochs, batch_size=batch_size, units1=units,
                units2=units, units3=units, units4=units)
    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    model=KerasClassifier(build_fn=build_clf)
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
     	cv=cvFold, scoring="accuracy")
    searchResults = gridSearch.fit(trainX, trainY)
    bestModel = searchResults.best_estimator_
    return bestModel


reg = LogisticRegressionCV_train(X_train, y_train)
#pickle.dump(reg, open('LogisticRegression.sav', 'wb'))
svm = SVM_train(X_train[:2000], y_train[:2000])
#pickle.dump(svm, open('svm.sav', 'wb'))
rf = RandomForestClassifier_train(X_train, y_train)
#pickle.dump(rf, open('rf.sav', 'wb'))
XGB = XGB_train(X_train, y_train)
#XGB.save_model("XGB.json")
ann = build_clf(32, 32, 32, 32, n_components)
history = ann.fit(X_train, np.array(y_train), epochs=20, batch_size=2)
#ann.save("ann.h5")
print(f"window size: {window_size}, step: {step}")
X_new = dimensionality_reduction_PCA(extracted_features2, n_components=n_components)

print('Logistic regression: \nf1:', f1_score(y_test, reg.predict(X_test)),'\n', classification_report(y_test, reg.predict(X_test), target_names=['normoxia', 'hypoxia']))
print('Logistic svm: \nf1:', f1_score(y_test, svm.predict(X_test)),'\n', classification_report(y_test, svm.predict(X_test), target_names=['normoxia', 'hypoxia']))
print('Logistic random forest: \nf1:', f1_score(y_test, rf.predict(X_test)),'\n', classification_report(y_test, rf.predict(X_test), target_names=['normoxia', 'hypoxia']))
print('Logistic XGboost: \nf1:', f1_score(y_test, XGB.predict(X_test)),'\n', classification_report(y_test, XGB.predict(X_test), target_names=['normoxia', 'hypoxia']))
print('ANN: \nf1:', f1_score(y_test, ((ann.predict(X_test) > 0.5)+0).ravel()),'\n', classification_report(y_test, ((ann.predict(X_test) > 0.5)+0).ravel(), target_names=['normoxia', 'hypoxia']))

# import seaborn as sn
# def CM_plot(conf_matrix):
#     cm_path = "confusion_matrix_rf.png"
#     ax= plt.subplot()
#     plot = sn.heatmap(conf_matrix, annot = True, cmap='Blues', fmt = '.0f')
#     fig = plot.get_figure()
#     ax.set_title("Macierz pomyłek")

#     ax.set_xlabel('Przewidziana klasa');ax.set_ylabel('Prawdziwa klasa'); 
    
#     fig.savefig(cm_path, dpi=100)
#     return cm_path

# CM_plot(confusion_matrix(y_test, rf.predict(X_test)))

# from sklearn import metrics
# plt.figure(0)
# metrics.plot_roc_curve(rf, X_test, y_test) 
# plt.savefig('roc_curve_rf')
# plt.close()
# plt.figure(2131)
# #define metrics
# y_pred_proba = rf.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)

# #create ROC curve
# plt.plot(fpr,tpr,label="AUC="+str(auc))
# plt.ylabel('Czułosc')
# plt.xlabel('Swoistosc')
# plt.legend(loc=4)
# plt.savefig('roc_curve_rf2')