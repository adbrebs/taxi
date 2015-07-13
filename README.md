Winning entry to the Kaggle ECML/PKDD destination competition.

https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i



**Dependencies**

We used the following packages developped at the MILA lab:
â¢  Theano. A general GPU-accelerated python math library, with an interface similar to numpy (see [3, 4]). http://deeplearning.net/software/theano/
â¢  Blocks. A deep-learning and neural network framework for Python based on Theano. https://github.com/mila-udem/blocks
â¢  Fuel. A data pipelining framework for Blocks. https://github.com/mila-udem/fuel 
We also used the scikit-learn Python library for their mean-shift clustering algorithm. numpy, cPickle and h5py are also used at various places.



**Structure**

  Here is a brief description of the Python files in the archive:

  <\itemize>
    <item><verbatim|config/*.py> : configuration files for the different
    models we have experimented with

    The model which gets the best solution is
    <verbatim|mlp_tgtcls_1_cswdtx_alexandre.py>

    <item><verbatim|data/*.py> : files related to the data pipeline:

    <\itemize>
      <item><verbatim|__init__.py> contains some general statistics about the
      data

      <item><verbatim|csv_to_hdf5.py> : convert the CSV data file into an
      HDF5 file usable directly by Fuel

      <item><verbatim|hdf5.py> : utility functions for exploiting the HDF5
      file

      <item><verbatim|init_valid.py> : initializes the HDF5 file for the
      validation set

      <item><verbatim|make_valid_cut.py> : generate a validation set using a
      list of time cuts. Cut lists are stored in Python files in
      <verbatim|data/cuts/> (we used a single cut file)

      <item><verbatim|transformers.py> : Fuel pipeline for transforming the
      training dataset into structures usable by our model
    </itemize>

    <item><strong|<verbatim|data_analysis/*.py>> : scripts for various
    statistical analyses on the dataset

    <\itemize>
      <item><verbatim|cluster_arrival.py> : the script used to generate the
      mean-shift clustering of the destination points, producing the 3392
      target points
    </itemize>

    <item><verbatim|model/*.py> : source code for the various models we tried

    <\itemize>
      <item><verbatim|__init__.py> contains code common to all the models,
      including the code for embedding the metadata

      <item><verbatim|mlp.py> contains code common to all MLP models

      <item><verbatim|dest_mlp_tgtcls.py> containts code for our MLP
      destination prediction model using target points for the output layer
    </itemize>

    <item><verbatim|error.py> contains the functions for calculating the
    error based on the Haversine Distance

    <item><verbatim|ext_saveload.py> contains a Blocks extension for saving
    and reloading the model parameters so that training can be interrupted

    <item><verbatim|ext_test.py> contains a Blocks extension that runs the
    model on the test set and produces an output CSV submission file

    <item><verbatim|train.py> contains the main code for the training and
    testing
  </itemize>
  
  
  **How to reproduce the winning results?**
  
  
    <\enumerate>
    <item>Set the <verbatim|TAXI_PATH> environment variable to the path of
    the folder containing the CSV files.

    <item>Run <verbatim|data/csv_to_hdf5.py> to generate the HDF5 file (which
    is generated in <verbatim|TAXI_PATH>, along the CSV files). This takes
    around 20 minutes on our machines.

    <item>Run <verbatim|data/init_valid.py> to initialize the validation set
    HDF5 file.

    <item>Run <verbatim|data/make_valid_cut.py test_times_0> to generate the
    validation set. This can take a few minutes.

    <item>Run <verbatim|data_analysis/cluster_arrival.py> to generate the
    arrival point clustering. This can take a few minutes.

    <item>Create a folder <verbatim|model_data> and a folder
    <verbatim|output> (next to the training script), which will receive
    respectively a regular save of the model parameters and many submission
    files generated from the model at a regular interval.

    <item>Run <verbatim|./train.py dest_mlp_tgtcls_1_cswdtx_alexandre> to
    train the model. Output solutions are generated in <verbatim|output/>
    every 1000 iterations. Interrupt the model with three consecutive Ctrl+C
    at any times. The training script is set to stop training after 10 000
    000 iterations, but a result file produced after less than 2 000 000
    iterations is already the winning solution. We trained our model on a
    GeForce GTX 680 card and it took about an afternoon to generate the
    winning solution.

    When running the training script, set the following Theano flags
    environment variable to exploit GPU parallelism:

    <verbatim|THEANO_FLAGS=floatX=float32,device=gpu,optimizer=FAST_RUN>

    Theano is only compatible with CUDA, which requires an Nvidia GPU.
    Training on the CPU is also possible but much slower.
  </enumerate>

  
  
  
  
  More information in this pdf: https://github.com/adbrebs/taxi/blob/master/doc/short_report.pdf
  
