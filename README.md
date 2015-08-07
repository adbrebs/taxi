Code of the winning entry to the [Kaggle ECML/PKDD taxi destination competition](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i). Our approach is described in [our paper](http://arxiv.org/abs/1508.00021).

## Dependencies

We used the following packages developped at the MILA lab:

* Theano. A general GPU-accelerated python math library, with an interface similar to numpy (see [3, 4]). See <http://deeplearning.net/software/theano/>
* Blocks. A deep-learning and neural network framework for Python based on Theano. As Blocks evolves very rapidly, we suggest you use commit `1e0aca9171611be4df404129d91a991354e67730`, which we had the code working on. See <https://github.com/mila-udem/blocks>
* Fuel. A data pipelining framework for Blocks. Same that for Blocks, we suggest you use commit `ed725a7ff9f3d080ef882d4ae7e4373c4984f35a`. See <https://github.com/mila-udem/fuel>

We also used the scikit-learn Python library for their mean-shift clustering algorithm. numpy, cPickle and h5py are also used at various places.


## Structure

Here is a brief description of the Python files in the archive:

* `config/*.py`: configuration files for the different models we have experimented with the model which gets the best solution is `mlp_tgtcls_1_cswdtx_alexandre.py`
* `data/*.py` : files related to the data pipeline:
  * `__init__.py` contains some general statistics about the data
  * `csv_to_hdf5.py` : convert the CSV data file into an HDF5 file usable directly by Fuel
  * `hdf5.py` : utility functions for exploiting the HDF5 file
  * `init_valid.py` : initializes the HDF5 file for the validation set
  * `make_valid_cut.py` : generate a validation set using a list of time cuts. Cut lists are stored in Python files in `data/cuts/` (we used a single cut file)
  * `transformers.py` : Fuel pipeline for transforming the training dataset into structures usable by our model
* `data_analysis/*.py` : scripts for various statistical analyses on the dataset
  * `cluster_arrival.py` : the script used to generate the mean-shift clustering of the destination points, producing the 3392 target points
* `model/*.py` : source code for the various models we tried
  * `__init__.py` contains code common to all the models, including the code for embedding the metadata
  * `mlp.py` contains code common to all MLP models
  * `dest_mlp_tgtcls.py` containts code for our MLP destination prediction model using target points for the output layer
* `error.py` contains the functions for calculating the error based on the Haversine Distance
* `ext_saveload.py` contains a Blocks extension for saving and reloading the model parameters so that training can be interrupted
* `ext_test.py` contains a Blocks extension that runs the model on the test set and produces an output CSV submission file
* `train.py` contains the main code for the training and testing
  
## How to reproduce the winning results?

There is an helper script `prepare.sh` which might help you (by performing steps 1-6 and some other checks), but if you encounter an error, the script will re-execute all the steps from the beginning (before the actual training, steps 2, 4 and 5 are quite long).

Note that some script expect the repository to be in your PYTHONPATH (go to the root of the repository and type `export PYTHONPATH="$PWD:$PYTHONPATH"`).
  
1. Set the `TAXI_PATH` environment variable to the path of the folder containing the CSV files.
2. Run `data/csv_to_hdf5.py "$TAXI_PATH" "$TAXI_PATH/data.hdf5"` to generate the HDF5 file (which is generated in `TAXI_PATH`, along the CSV files). This takes around 20 minutes on our machines.
3. Run `data/init_valid.py valid.hdf5` to initialize the validation set HDF5 file.
4. Run `data/make_valid_cut.py test_times_0` to generate the validation set. This can take a few minutes.
5. Run `data_analysis/cluster_arrival.py` to generate the arrival point clustering. This can take a few minutes.
6. Create a folder `model_data` and a folder `output` (next to the training script), which will receive respectively a regular save of the model parameters and many submission files generated from the model at a regular interval.
7. Run `./train.py dest_mlp_tgtcls_1_cswdtx_alexandre` to train the model. Output solutions are generated in `output/` every 1000 iterations. Interrupt the model with three consecutive Ctrl+C at any times. The training script is set to stop training after 10 000 000 iterations, but a result file produced after less than 2 000 000 iterations is already the winning solution. We trained our model on a GeForce GTX 680 card and it took about an afternoon to generate the winning solution.
   When running the training script, set the following Theano flags environment variable to exploit GPU parallelism:
   `THEANO_FLAGS=floatX=float32,device=gpu,optimizer=fast_run`

*More information in [this pdf](https://github.com/adbrebs/taxi/blob/master/doc/short_report.pdf)*
