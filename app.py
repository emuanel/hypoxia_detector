from kivy.config import Config
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '1400')
Config.set('graphics', 'height', '700')
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.image import Image
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
import os
from kivy.uix.popup import Popup
import pandas as pd
import numpy as np
from extraction import Extraction
import pickle as pk
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from kivy_garden.graph import Graph, MeshLinePlot


class tab(TabbedPanel):
    def __init__(self,**kwargs):
        super(tab,self).__init__(**kwargs)
        self.padding = 30
        self.spacing = 5
        self.pos_hint = {'center_x':0.6, 'top':0.73} 
        self.size_hint=(0.7, 0.7)
        self.do_default_tab= False
        self.background_color= [66/255, 230/255, 245/255, 0.5]
        self.th1 = TabbedPanelItem(text='Charts',background_color = [66/255, 230/255, 245/255, 1])
        self.th2 = TabbedPanelItem(text='Features', background_color = [66/255, 230/255, 245/255, 1])
        self.th3 = TabbedPanelItem(text='Detection', background_color = [66/255, 230/255, 245/255, 1])
        self.add_widget(self.th1)
        self.add_widget(self.th2)
        self.add_widget(self.th3)
    
    def show_plots(self, data):
        '''
        Generate plots 

        Parameters
        ----------
        data : DataFrame
            biosignals.

        Returns
        -------
        None.

        '''
        lenght1 = 800
        lenght2 = 80
        X1 = np.linspace(0, lenght1/100, lenght1)
        X2 = np.linspace(0, lenght2/10, lenght2)
        
        params = {'xlabel':'Time [s]', "x_ticks_minor": 1, "x_ticks_major" : 1, 
                  "y_grid_label": True, "x_grid_label": True, "padding": 1,
                  "x_grid": False, "y_grid": False}
        
        signal = data['BP'].values[0:lenght1]
        ymin=int(signal.min()+1)
        ymax=int(signal.max()+1)
        graph1 = Graph(ylabel='BLOOD PRESSURE', xmin=int(X1.min()), xmax=int(X1.max()), 
                       ymin=ymin, ymax=ymax, y_ticks_major=int((ymax-ymin)/5), **params)
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        points = []
        for i,j in zip(X1, signal):
            points.append((i,j))
        plot.points = points
        graph1.add_plot(plot)
        
        signal = data['TQR'].values[0:lenght1]
        ymin=float(round(signal.min()-0.005, 2))
        ymax=float(round(signal.max()+0.005, 2))
        graph2 = Graph(ylabel='SAS WIDTH',xmin=int(X1.min()), xmax=int(X1.max()), 
                       ymin=ymin, ymax=ymax, y_ticks_major=(ymax-ymin)/5, **params)
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        points = []
        for i,j in zip(X1, signal):
            points.append((i,j))
        plot.points = points
        graph2.add_plot(plot)
        
        signal = data['ECG'].values[0:lenght1]
        ymin=float(round(signal.min()-0.005, 2))
        ymax=float(round(signal.max()+0.005, 2))
        graph3 = Graph(ylabel='ECG',xmin=int(X1.min()), xmax=int(X1.max()), 
                       ymin=ymin, ymax=ymax, y_ticks_major=(ymax-ymin)/5, **params)
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        points = []
        for i,j in zip(X1, signal):
            points.append((i,j))
        plot.points = points
        graph3.add_plot(plot)
        
        signal = data['NIRS'].values[0:lenght2]
        ymin=float(round(signal.min()-0.05, 2))
        ymax=float(round(signal.max()+0.05, 2))
        graph4 = Graph(ylabel='OXYGENATION HbO2',xmin=int(X2.min()), xmax=int(X2.max()), 
                       ymin=ymin, ymax=ymax, y_ticks_major=(ymax-ymin)/5, **params)
        plot = MeshLinePlot(color=[1, 0, 0, 1])
        points = []
        for i,j in zip(X2, signal):
            points.append((i,j))
        plot.points = points
        graph4.add_plot(plot)
        
        layout = BoxLayout(padding=10, orientation='horizontal')
        layout1 = BoxLayout(padding=10, orientation='vertical')
        layout2 = BoxLayout(padding=10, orientation='vertical')
        layout1.add_widget(graph1)
        layout1.add_widget(graph2)
        layout2.add_widget(graph3)
        layout2.add_widget(graph4)
        
        layout.add_widget(layout1)
        layout.add_widget(layout2)
        self.th1.add_widget(layout)
    
    def show_features(self, features):
        layout = BoxLayout(padding=10, orientation='vertical')
        l1 = Label(text=f'Pulse: {round(features[0],2)}', font_size = 22)
        l2 = Label(text=f'Systolic blood pressure: {round(features[1],2)}', font_size = 22)
        l3 = Label(text=f'Diastolic blood pressure: {round(features[2],2)}', font_size = 22)
        l4 = Label(text=f'Average arterial pressure: {round(features[3],2)}', font_size = 22)
        l5 = Label(text=f'SAS width: {round(features[4],2)}', font_size = 22)
        l6 = Label(text=f'Average oxygenated: {round(features[5],2)}', font_size = 22)
        
        layout.add_widget(l1)
        layout.add_widget(l2)
        layout.add_widget(l3)
        layout.add_widget(l4)
        layout.add_widget(l5)
        layout.add_widget(l6)
        self.th2.add_widget(layout)
    
    def show_prediction(self, features):
        layout = BoxLayout(padding=10, orientation='vertical')
        
        l1 = Label(text=f'Logistic regression: {"Normoxia" if features[0] < 0.5 else "Hypoxia"}', font_size = 22)
        l2 = Label(text=f'Support vector machine: {"Normoxia" if features[1] < 0.5 else "Hypoxia"}', font_size = 22)
        l3 = Label(text=f'Random forest: {"Normoxia" if features[2] < 0.5 else "Hypoxia"}', font_size = 22)
        l4 = Label(text=f'XGBoost: {"Normoxia" if features[3] < 0.5 else "Hypoxia"}', font_size = 22)
        l5 = Label(text=f'Artificial neural network: {"Normoxia" if features[4] < 0.5 else "Hypoxia"}', font_size = 22)
        
        layout.add_widget(l1)
        layout.add_widget(l2)
        layout.add_widget(l3)
        layout.add_widget(l4)
        layout.add_widget(l5)
        self.th3.add_widget(layout)
        
class left(BoxLayout):
    loadfile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    
    def __init__(self,tab, **kwargs):
        super(left,self).__init__(**kwargs)
        self.tab = tab
        self.padding = 30
        self.orientation = "vertical"
        self.spacing = 20
        self.pos_hint = {'top': 0.73} 
        self.size_hint=(0.25, 0.7)
        self.data_loaded = False
        self.data_extracted = False
        
        btn1 = Button(text='Load signals', on_press=self.read_data, pos_hint ={'center_x':0.5}, background_color = [66/255, 230/255, 245/255, 1])   
        btn2 = Button(text='Plot signals', on_press=self.plot, pos_hint = {'center_x':0.5}, background_color = [66/255, 230/255, 245/255, 1])
        btn3 = Button(text='Extract features', on_press=self.extract, pos_hint = {'center_x':0.5}, background_color = [66/255, 230/255, 245/255, 1])
        btn4 = Button(text='Detect hypoxia', on_press=self.detection, pos_hint = {'center_x':0.5}, background_color = [66/255, 230/255, 245/255, 1])
        btn5 = Button(text='Close Window', on_press=self.close, pos_hint ={'center_x':0.5}, background_color = [66/255, 230/255, 245/255, 1])
        
        #self.add_widget(btn)
        self.add_widget(btn1)
        self.add_widget(btn2)
        self.add_widget(btn3)
        self.add_widget(btn4)
        self.add_widget(btn5)
    
    def read_data(self, obj):
        return self.show_load()
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()   
    def load(self, path, filename):
        self.data = pd.read_csv(filename[0], delimiter = ';')
        self.data_loaded = True
        self.dismiss_popup()
        
    def plot(self, obj):
        if self.data_loaded == True:
            self.tab.show_plots(self.data)
        else:
            content = Button(text='Ok')
            popup = Popup(title='Load signals first',content=content, auto_dismiss=False,size_hint=(None, None),size=(400, 100))
            content.bind(on_press=popup.dismiss)
            popup.open()
            
    def extract(self, obj):
        if self.data_loaded == True:
            n_scales = 128 # define the scale size
            self.SAS_cwt = Extraction.cwt_coeffs(self.data['TQR'] , n_scales)
            self.pulse = Extraction.heart_rate(self.data['ECG'])
            self.systolic_BP = Extraction.systolic_blood_pressure(self.data['BP'])
            self.diastolic_BP = Extraction.diastolic_blood_pressure(self.data['BP'])
            self.average_AP = Extraction.average_arterial_pressure(self.data['BP'])
            self.SAS_width = Extraction.subarachnoid_width(self.data['NIRS'])
            self.average_oxygenated = Extraction.average_oxygenated_haemoglobin(self.data['TQR'])
            self.data_extracted = True
            self.tab.show_features([self.pulse, self.systolic_BP, self.diastolic_BP, 
                                    self.average_AP, self.SAS_width, self.average_oxygenated])
        else:
            content = Button(text='Ok')
            popup = Popup(title='Load signals first',content=content, auto_dismiss=False,size_hint=(None, None),size=(400, 100))
            content.bind(on_press=popup.dismiss)
            popup.open()
            
    def detection(self, obj):
        if self.data_loaded == True and self.data_extracted == True:
            features_time = np.array([self.pulse, self.systolic_BP, self.diastolic_BP, 
                         self.average_AP, self.SAS_width, self.average_oxygenated]).reshape(1, -1)
            features_all = np.concatenate([features_time, self.SAS_cwt],axis=1)
            
            pca = pk.load(open("models/pca.pkl",'rb'))
            features_pca = pca.transform(features_all)
            
            log_reg = pk.load(open('models/LogisticRegression.sav', 'rb'))
            svm = pk.load(open('models/svm.sav', 'rb'))
            rf = pk.load(open('models/rf.sav', 'rb'))
            XGB = XGBClassifier()
            XGB.load_model("models/XGB.json")
            ann = load_model('models/ann.h5')
    
            result_reg = log_reg.predict(features_pca)
            result_svm = svm.predict(features_pca)
            result_rf = rf.predict(features_pca)
            result_XGB = XGB.predict(features_pca)
            result_ann = ann.predict(features_pca)[0]
            self.tab.show_prediction([result_reg, result_svm, result_rf, 
                                    result_XGB, result_ann])
        else:
            content = Button(text='Ok')
            popup = Popup(title='Extract features first',content=content, auto_dismiss=False,size_hint=(None, None),size=(400, 100))
            content.bind(on_press=popup.dismiss)
            popup.open()
        
        
    #load data
    def dismiss_popup(self):
        self._popup.dismiss()
    
    #close window
    def close(self, obj):
        App.get_running_app().stop()
        Window.close()
        os._exit(0)
    
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
            
class logo(Image):
    def __init__(self, source):
        super(logo,self).__init__(source = source)
        self.pos_hint = {'center_x':0.5, 'top': 1} 
        self.size_hint=(0.3, 0.3)
        
class BioApp(App):
    def build(self):
        root=FloatLayout()
        with root.canvas.before:
            Color(13/255, 103/255, 181/255, 1)  # green; colors range from 0-1 not 0-255
            self.rect = Rectangle(size=(11111, 1111))
        im = logo(source ="plots/log.jpeg")
        tab1 = tab()
        root.add_widget(tab1)
        root.add_widget(left(tab1))
        root.add_widget(im)
        
        return root
from kivy.lang import Builder
Builder.load_file('main.kv')   
if __name__ == '__main__':
    BioApp().run()
    
    

