# m6ABRP
**A machine learning-based model for predicting YTHDF2 binding regions in mRNAs via sequence-based properties.**
## Dependencies:

     1--sys  

     2--numpy  

     3--sklearn 

     4--joblib  

**m6ABRP is implemented using Python2.7.**  
## model_training.py
**Usage: python model_training.py training_positive_dataset training_negative_dataset model_file scale_file pca_file**  
This script is used to train the m6ABRP tool.  
**Outputs:**    
     
     1--a model file, model.pkl, which can be directly used for prediction. 
     
     2--a normalized file, normalization.pkl, which can be used to normalized the input data. 
     
     3--a pca model file, pca.pkl, which can be used to generate the principal components.  
## model_indepedent_testing.py 
**Usage: python model_indepedent_testing.py test_positive_dataset test_negative_dataset model_file scale_file pca_file**
This script is used to evaluate the performance of m6ABRP on the indepedent testing dataset.
**The model_file, scale_file and pca_file generated in the training process must be involved.**  
**Outputs**
     1--m6ABRP_score.txt, which consists of the prediction score of each sample.
     2--test_label.txt, which includes the label of each sample.
