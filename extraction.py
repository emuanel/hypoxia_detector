# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:32:03 2022

@author: jfili
"""
import numpy as np
import pywt

from scipy.signal import find_peaks
class Extraction():
    def cwt_coeffs(signal, n_scales, wavelet_name='morl'):
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
        
        coef, freqs = pywt.cwt(signal, widths, wavelet_name)  
        magn = np.median(np.absolute(coef), axis=1).flatten()[::-1]
        median_comps = np.vstack([median_comps, magn]) 
        return median_comps
     

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
        
        return Extraction.diastolic_blood_pressure(BP_signal) + 1/3 * (Extraction.systolic_blood_pressure(BP_signal) - Extraction.diastolic_blood_pressure(BP_signal))

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
