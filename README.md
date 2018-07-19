# Prediction of cancer dependencies by deep learning
This site is a source code repository for in silico RNAi and deep learning for prediction of cancer vulnerabilities from transcriptome data.

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
* Theano (>= 1.0.2): http://deeplearning.net/software/theano/
   
## <a name="insilico">In silico RNAi</a>

To run in silico RNAi, network file and basal gene expression profile are required.
Networks and expression profiles that were used in this study are available at the link below.

#### Download data

* Click here to download ***[networks](http://143.248.31.34/~omics/)***
* Download breast cancer cell line expression and dependency data *(Cell. 2016 Jan 14;164(1-2):293-309)*
  * Click here to download ***[cell line RNA-seq data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73526)***
  * Click here to download ***[dependency screening data](https://github.com/neellab/bfg/blob/gh-pages/data/shrna/breast_zgarp.txt.zip?raw=true)***
   
You can use your basal expression profile to run in silico RNAi.

#### Run in silico RNAi

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

The default option is to save in silico simulated file ```perturbed_result.txt``` in ```output``` directory.

If you want to run simulation with your data, edit ```run.sh``` file and change directory path to "NETWORK", "BASAL", "OUTDIR" and "RESULT" depending on your data.

      
## <a name="dnnmodel">Deep learning model</a>

#### Create model input

First, "INPUT" file which in silico simulated transcriptome with properly annotated dependecy class should be prepared.
Edit ```run_input.sh``` file and change directory path to "INPUT", and "FILE_IDX" name depending on your files.

  ```
  cd DNN_codes
  cd input
  ```
>Deep learning model input example

  ```
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
  ```

All models are saved in ```./out``` directory. If you want to see log file and error of each model, please check ```./log``` and ```./err``` directory.

#### <a name="predictionoutcome">Prediction</a>
Select the best performing model. You can use that model for prediction.
Also, samples for prediction should be simulated.

Edit ```run_predict.sh``` and change directory path to "Sample" and "Model".

Then run.

  ```
  ./run_predict.sh
  ```

The default option is to save prediction files in ```result``` directory.


