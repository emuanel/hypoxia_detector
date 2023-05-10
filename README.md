# hypoxia_detector
App to detect hypoxia based on 4 signals 

The aim of this study was to develop, implement and evaluate software for the analysis
of biomedical signals recorded under normoxic and hypoxic conditions in healthy people with the
use of machine learning techniques. Four signals were analyzed: blood pressure (BP) recorded
from the finger, electrocardiogram (EKG), relative change in oxyhemoglobin (HbO2) from the right
hemisphere of the brain, and the width of the subarachnoid space (SAS). In addition, the current
state of knowledge was reviewed and described on the basis of the literature, and solutions similar
to the issues raised in the paper were presented. Then, a proprietary solution was designed and
implemented based on the current state of knowledge.
As part of the work, five machine learning classifiers and a simple and intuitive graphical
interface enabling the visualization of biomedical signals, their basic parameters and the results
of algorithms' operation were prepared. The paper describes the method of pre-processing of
biomedical signals and the training method along with the evaluation of several models of
automatic data analysis for the detection of hypoxia in healthy people. The results obtained by
using classical machine learning classifiers were compared: logistic regression, SVM support
vector machine, random forest, XGBoost and MLP multilayer perceptron. Moreover, the analysis
of the influence of cutting biosignals into shorter fragments on the predictive results was
performed and the influence of the selection of the number of principal components in the PCA
technique on the operation of the models was analyzed.
The Python programming language was used to implement the proposed solution. Tests
and experiments have shown that all assumptions and requirements have been fulfilled and a
decent accuracy of machine learning models has been obtained
