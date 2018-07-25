# Prediction of cancer dependencies by deep learning
This site is a source code repository for *in silico* RNAi and deep learning for prediction of cancer vulnerabilities from transcriptome data.

## Table of Contents
* [Requirements](#requirements)
* [In silico RNAi](#insilico)
* [Deep learning model](#dnnmodel)
   
## <a name="requirements">Requirements</a>

>Required modules for RNAi simulation.

* Python (>= 2.7.12): https://www.python.org/
* NumPy  (>= 1.13.3): http://www.numpy.org/
* Pandas (>= 0.23.1): https://pandas.pydata.org/
* Networkx (>= 2.1) : https://networkx.github.io/

>Required modules for Deep learning.

* Scikit-learn (>= 0.19.1): http://scikit-learn.org/stable/index.html
* Theano (>= 0.9.0): http://deeplearning.net/software/theano/
   
## <a name="insilico">In silico RNAi</a>
To run *in silico* RNAi, a network file and basal gene expression profile are required.
All data used in this study are available at the links below.
You can also use your own basal expression profiles to run *in silico* RNAi.

#### Download data

* Click here to download ***[networks](http://143.248.31.34/~omics/)***
* Download breast cancer cell line expression and dependency data *(Cell. 2016 Jan 14;164(1-2):293-309)*
  * Click here to download ***[cell line RNA-seq data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73526)***
  * Click here to download ***[dependency screening data](https://github.com/neellab/bfg/blob/gh-pages/data/shrna/breast_zgarp.txt.zip?raw=true)***


#### Run *in silico* RNAi

>Bayesian Example

  ```
  cd insilico_RNAi_Bayesian

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

>ARACNe Example

  ```
  cd insilico_RNAi_ARACNe

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

*In silico* RNAi output ```perturbed_result.txt``` for the example are available in ```output``` directory.
      
## <a name="dnnmodel">Deep learning model</a>

#### Create model input
To build a model, the output of *in silico* RNAi should be matched with experimental dependency data.
The whole dependency dataset is available at the link above.

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
  
#### Model running

>Deep learning model running

  ```
  cd ../codes
  ./run.sh
  ```
  
Edit ```run.sh``` and change "MAX_J" and "TOT_J" to assign the maximum nodes for parallel running and total number of model simulation.

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

#### <a name="predictionoutcome">Prediction</a>
*In silico* RNAi output for the sample data is subjected to prediction.
Select the best performing model.

  ```
  ./run_predict.sh
  ```

The default option is to save prediction outputs in ```result``` directory.


