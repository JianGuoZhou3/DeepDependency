# Prediction of cancer dependencies by deep learning
This site is a source code repository for *in silico* CRISPR/RNAi and deep learning for prediction of cancer vulnerabilities from transcriptome data.

## Table of Contents
* [Requirements](#requirements)
* [*In silico* CRISPR/RNAi](#insilico)
* [Deep learning](#dnnmodel)
   
## <a name="requirements">Requirements</a>
>Required modules for CRISPR/RNAi simulation.

* Python (>= 2.7.12): https://www.python.org/
* NumPy  (>= 1.13.3): http://www.numpy.org/
* Pandas (>= 0.23.1): https://pandas.pydata.org/
* Networkx (>= 2.1) : https://networkx.github.io/

>Required modules for deep learning.

* Scikit-learn (>= 0.19.1): http://scikit-learn.org/stable/index.html
* Theano (>= 0.9.0): http://deeplearning.net/software/theano/
   
## <a name="insilico">*In silico* CRISPR/RNAi</a>
To run *in silico* CRISPR/RNAi, a network file and basal gene expression profile are required.
All data used in this study are available at the links below.
You can also use your own basal expression profiles to run *in silico* CRISPR/RNAi.

#### <a name="download">Download data</a>
* Downlaod regulatory networks
  * Bayesian network *(Nucleic Acids Res. 2015 13;43*(12)*)*  
    ***[Breast](http://143.248.31.34/~omics/data/DeepDependency/Networks/Bayesian_real_breast.tsv)*** 
    ***[Shuffled](http://143.248.31.34/~omics/data/DeepDependency/Networks/Bayesian_shuffled_breast.tsv)*** 
    ***[Inverted](http://143.248.31.34/~omics/data/DeepDependency/Networks/Bayesian_inverted_breast.tsv)*** 
    ***[Liver](http://143.248.31.34/~omics/data/DeepDependency/Networks/Bayesian_liver.tsv)*** 
  * ARACNe network *(Nat Genet. 2005 ;37*(4)*)*  
    ***[Breast](http://143.248.31.34/~omics/data/DeepDependency/Networks/ARACNe_real_breast.tsv)*** 
    ***[Shuffled](http://143.248.31.34/~omics/data/DeepDependency/Networks/ARACNe_shuffled_breast.tsv)*** 
    ***[Inverted](http://143.248.31.34/~omics/data/DeepDependency/Networks/ARACNe_inverted_breast.tsv)*** 
    ***[Liver](http://143.248.31.34/~omics/data/DeepDependency/Networks/ARACNe_liver.tsv)*** 
* Download breast cancer cell line expression profile  
  ***[cell line RNA-seq data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73526)*** *(Cell. 2016 14;164(1-2):293-309)*  
* Downlaod dependency screens of breast cancer cell lines  
  ***[CRISPR-Cas9 screens](https://depmap.org/portal/download/)*** *(Nat Genet. 2017 ;49*(12)*)*  
  ***[CRISPR-Cas9 screens](https://score.depmap.sanger.ac.uk/downloads)*** *(Nature. 2019 ;568*(7753)*)*   
  ***[RNAi screens](https://github.com/neellab/bfg/blob/gh-pages/data/shrna/breast_zgarp.txt.zip?raw=true)*** *(Cell. 2016 14;164*(1-2)*)*   


#### Run *in silico* CRISPR/RNAi

>*in silico* CRISPR/RNAi Bayesian Example

  ```
  cd insilico_CRISPR/Bayesian # or cd insilico_RNAi/Bayesian

  # sample file directory
  mkdir Bayesian_sample
  cd Bayesian_sample

  # Download cellline basal expression
  wget http://143.248.31.34/~omics/data/DeepDependency/sample/Cellline_basal_Bayesian.txt

  # Download network file 
  wget http://143.248.31.34/~omics/data/DeepDependency/Networks/Bayesian_real_breast.tsv
  
  cd ..
  ./run.sh
  ```

>*in silico* CRISPR/RNAi ARACNe Example

  ```
  cd insilico_CRISPR/ARACNe # or cd insilico_RNAi/ARACNe

  # sample file directory
  mkdir ARACNe_sample
  cd ARACNe_sample

  # Download cellline basal expression
  wget http://143.248.31.34/~omics/data/DeepDependency/sample/Cellline_basal_ARACNe.txt

  # Download network file 
  wget http://143.248.31.34/~omics/data/DeepDependency/Networks/ARACNe_real_breast.tsv
  
  cd ..
  ./run.sh
  ```

*In silico* CRISPR/RNAi output ```perturbed_result.txt``` for the example are available in ```output``` directory.
      
## <a name="dnnmodel">Deep learning</a>

#### Create model input
To build a model, the output of *in silico* CRISPR/RNAi should be matched with experimental dependency data.
The whole dependency dataset is available at the link [above](#download).

The dataset is divided into training data (infile.tsv) and test data (testset.tsv). Then, the training dataset is shuffled and split into five datasets.

>Deep learning model input example

  ```
  cd DNN_codes
  cd input

  # Download sample model input
  wget http://143.248.31.34/~omics/data/DeepDependency/sample/Model_input.txt

  # run
  ./run_input.sh
  ```
  
#### Model training

>Deep learning model running

  ```
  cd ../codes
  ./run_train.sh
  ```
  
Edit ```run_train.sh``` and change "MAX_J" and "TOT_J" to assign the maximum nodes for parallel running and total number of model simulation.

Hyperparameters search space can be modified in ```SdA.py``` file.

>Default hyperparameters search space

  ```
  n_layer        = numpy.random.choice([2, 3, 4])
  Hidden         = sorted([numpy.random.choice([100, 500, 1000]) for i in range(n_layer)], reverse=True)
  dropout        = [numpy.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(n_layer+1)]
  pretrain_lr    = numpy.random.choice([0.1, 0.01, 0.001, 0.0001])
  finetune_lr    = numpy.random.choice([0.1, 0.01, 0.001, 0.0001])
  L1_param       = numpy.random.choice([1e-08, 1e-04, 1e-02, 1e-01, 1, 10])
  L2_param       = numpy.random.choice([1e-08, 1e-04, 1e-02, 1e-01, 1, 10])
  mom            = numpy.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5])
  batch_size     = numpy.random.choice([10, 50, 100])
  useRelu        = numpy.random.choice([True, False])
  W_distribution = numpy.random.choice(["norm", "uni"])
  insize         = 1000   #(number of features)
  
  pretrain_epochs = 100
  training_epochs = 10000
  ```

All models are saved in ```./out``` directory. If you want to see the log file and error of each model, please check ```./log``` and ```./err``` directory.

>Cheking a model performance.

  ```
  ./model_AUC.sh
  ```

The best model is moved into ```./out/best_model``` directory and the rest of models are moved into ```./out/models```.
AUC of the best model are plotted at ```Model_AUC.pdf``` in ```out``` directory.
Also, performance of all trained models can be checked at ```summary.txt``` in ```out``` directory.


#### Sample prediction
*In silico* CRISPR/RNAi output for the sample data is subjected to prediction.

>Example of sample data prediction

  ```
  cd ../

  # Download Clincal samples 
  wget http://143.248.31.34/~omics/data/DeepDependency/sample/Clinical_sample.tar.gz
  tar zxvf Clinical_sample.tar.gz 

  cd codes

  # Sample prediction
  ./run_predict.sh
  ```

```*.pred``` files summarize the results of five-fold cross-validation with votes and average prediction score.
The default option is to save the prediction outputs in ```prediction``` directory.


