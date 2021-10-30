from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import math 
from math import sqrt


def get_mse(actual, predicted):
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss
    

   
def get_accuracy(actual, predicted, threshold):
    correct = 0
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    for i in range(len(actual)):
      if actual[i] == predicted_classes[i]:
        correct += 1
    return correct / float(len(actual)) * 100.0



def pred_to_classes(actual, predicted, threshold):
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    return predicted_classes
    
#precision
def get_tp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tp = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 1 and actual[i] == 1:
       tp += 1
    return tp
    
     
    
def get_fp(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fp = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 1 and actual[i] == 0:
       fp += 1
    return fp


def get_tn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    tn = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 0 and actual[i] == 0:
       tn += 1
    return tn


def get_fn(actual, predicted, threshold):
    predicted_classes = pred_to_classes(actual, predicted, threshold)
    fn = 0
    for i in range(len(predicted_classes)):
      if predicted_classes[i] == 0 and actual[i] == 1:
       fn += 1
    return fn
    
    
#precision = TP/ (TP + FP)    
def precision(actual, predicted, threshold):
    prec = get_tp(actual, predicted, threshold) / (get_tp(actual, predicted, threshold) + get_fp(actual, predicted, threshold))
    return prec
    
    
    
#recall = TP / (TP + FN)   
# sensitivity = recall 
def sensitivity(actual, predicted, threshold):
    sens = get_tp(actual, predicted, threshold)/ (get_tp(actual, predicted, threshold) + get_fn(actual, predicted, threshold))
    return sens
    

    
#Specificity = TN/(TN+FP)    
def specificity(actual, predicted, threshold):     
   spec =  get_tn(actual, predicted, threshold)/ (get_tn(actual, predicted, threshold) + get_fp(actual, predicted, threshold))
   return spec


#f1 score  = 2 / ((1/ precision) + (1/recall))   
def f_score(actual, predicted, threshold):
    f_sc = 2 / ( (1 / precision(actual, predicted, threshold)) + (1/ sensitivity(actual, predicted, threshold)))
    return f_sc

   
#mcc = (TP * TN - FP * FN) / sqrt((TN+FN) * (FP+TP) *(TN+FP) * (FN+TP)) 
def mcc(act, pred, thre):
   tp = get_tp(act, pred, thre) 
   tn = get_tn(act, pred, thre)
   fp = get_fp(act, pred, thre)
   fn = get_fn(act, pred, thre)
   mcc_met = (tp*tn - fp*fn) / (sqrt((tn+fn)*(fp+tp)*(tn+fp)*(fn+tp)))
   return mcc_met
   
   

def auroc(act, pred):
   return roc_auc_score(act, pred)
  

   
def auprc(act, pred):
   return average_precision_score(act, pred)
 
   