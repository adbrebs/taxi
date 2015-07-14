<TeXmacs|1.99.2>

<style|generic>

<\body>
  <doc-data|<doc-title|Taxi Destination Prediction Challenge<next-line>Winner
  Team's Report>|<doc-author|<author-data|<\author-affiliation>
    <em|Montréal, July 2015>
  </author-affiliation>>>>

  <center|<tabular*|<tformat|<table|<row|<cell|<name|Alex
  Auvolat>>|<cell|<name|Alexandre de Brébisson>>|<cell|<name|Étienne
  Simon>>>|<row|<cell|ENS Paris>|<cell|Université de Montréal>|<cell|ENS
  Cachan>>|<row|<cell|France>|<cell|Québec,
  Canada>|<cell|France>>|<row|<cell|<verbatim|alexis211@gmail.com>>|<cell|<verbatim|<strong|adbrebs@gmail.com>>>|<cell|<verbatim|esimon@esimon.eu>>>>>>>

  <section|Summary>

  Our model is based on a multi-layer perceptron (MLP). Our MLP model is
  trained by stochastic gradient descent (SGD) on the training trajectories.
  The inputs of our MLP are the 5 first and 5 last positions of the known
  part of the trajectory, as well as embeddings for the context information
  (date, client and taxi identification). \ The embeddings are trained with
  SGD jointly with the MLP parameters. The MLP outputs probabilities for 3392
  target points, and a mean is calculated to get a unique destination point
  as an output. We did no ensembling and did not use any external data.

  <section|Feature Selection/Extraction>

  We used a mean-shift algorithm on the destination points of all the
  training trajectories to extract 3392 classes for the destination point.
  These classes were used as a fixed softmax layer in the MLP architecture.

  We used the embedding method which is common in neural language modeling
  approaches (see [1]) to take the metainformation into account in our model.
  The following embeddings were used (listed with corresponding
  dimensionnality):

  <big-table|<tabular|<tformat|<table|<row|<cell|<tabular|<tformat|<cwith|1|1|1|-1|cell-bborder|1px>|<table|<row|<cell|<strong|Meta-data>>|<cell|<strong|Embedding
  Dimension>>|<cell|<strong|Number of classes>>>|<row|<cell|Unique caller
  number>|<cell|10>|<cell|57125>>|<row|<cell|Unique stand
  number>|<cell|10>|<cell|64>>|<row|<cell|Unique taxi
  number>|<cell|10>|<cell|448>>|<row|<cell|Week of
  year>|<cell|10>|<cell|54>>|<row|<cell|Day of
  week>|<cell|10>|<cell|7>>|<row|<cell|1/4 of hour of the
  day>|<cell|10>|<cell|96>>|<row|<cell|Day type (invalid
  data)>|<cell|10>|<cell|3>>>>>>>>>>|Embeddings and corresponding dimensions
  used by the model>

  The embeddings were first initialized to random variables and were then let
  to evolve freely with SGD along with the other model parameters.

  The geographical data input in the network is a centered and normalized
  version of the GPS data points.

  We did no other preprocessing or feature selection.

  <section|Modelling Techniques and Training>

  Here is a brief description of the model we used:

  <\itemize>
    <item><strong|Input.> The input layer of the MLP is the concatenation of
    the following inputs:

    <\itemize>
      <item>Five first and five last points of the known part of the
      trajectory.

      <item>Embeddings for all the metadata.
    </itemize>

    <item><strong|Hidden layer.> We use a single hidden layer MLP. The hidden
    layer is of size 500, and the activation function is a Rectifier Linear
    Unit (ie <math|f<around*|(|x|)>=max<around*|(|0,x|)>>) [2].

    <item><strong|Output layer.> The output layer predicts a probability
    vector for the 3392 output classes that we obtained with our clustering
    preprocessing step. If <math|\<b-p\>> is the probability vector output by
    our MLP (output by a softmax layer) and <math|c<rsub|i>> is the centroid
    of cluster <math|i>, our prediciton is given by:

    <\eqnarray*>
      <tformat|<table|<row|<cell|<wide|y|^>>|<cell|=>|<cell|<big|sum><rsub|i>p<rsub|i>*c<rsub|i>>>>>
    </eqnarray*>

    Since <math|\<b-p\>> sums to one, this is a valid point on the map.

    <item><strong|Cost.> We directly train using an approximation
    (Equirectangular projection) of the mean Haversine Distance as a cost.

    <item><strong|SGD and optimization.> We used a minibatch size of 200. The
    optimization algorithm is simple SGD with a fixed learning rate of 0.01
    and a momentum of 0.9.

    <item><strong|Validation.> To generate our validation set, we tried to
    create a set that looked like the training set. For that we generated
    ``cuts'' from the training set, i.e. extracted all the taxi rides that
    were occuring at given times. The times we selected for our validation
    set are similar to those of the test set, only one year before:

    <code|1376503200, # 2013-08-14 18:00<next-line>1380616200, # 2013-10-01
    08:30<next-line>1381167900, # 2013-10-07 17:45<next-line>1383364800, #
    2013-11-02 04:00<next-line>1387722600 \ # 2013-12-22 14:30>
  </itemize>

  <big-figure|<image|winning_model.png|577px|276px||>|Illustration of the
  winning model.>

  <section|Code Description>

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

  In the archive we have included only the files listed above, which are the
  strict minimum to reproduce our results. More files for the other models we
  tried are available on GitHub at <hlink|https://github.com/adbrebs/taxi|><hlink||https://github.com/adbrebs/taxi>.

  <section|Dependencies>

  We used the following packages developped at the MILA lab:

  <\itemize>
    <item><strong|Theano.> A general GPU-accelerated python math library,
    with an interface similar to numpy (see [3, 4]).
    <hlink|http://deeplearning.net/software/theano/|>

    <item><strong|Blocks.> A deep-learning and neural network framework for
    Python based on Theano. <hlink|https://github.com/mila-udem/blocks|>

    <item><strong|Fuel.> A data pipelining framework for Blocks.
    <hlink|https://github.com/mila-udem/fuel|>
  </itemize>

  We also used the <verbatim|scikit-learn> Python library for their
  mean-shift clustering algorithm. <verbatim|numpy>, <verbatim|cPickle> and
  <verbatim|h5py> are also used at various places.

  <section|How To Generate The Solution>

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

  <section|Additional Comments and Observations>

  The training examples fed to the model are not full trajectories, since
  that would make no sense, but prefixes of those trajectories that are
  generated on-the-fly by a Fuel transformer, <verbatim|TaxiGenerateSplits>,
  whose code is available in <verbatim|data/transformers.py>. The data
  pipeline is as follows:

  <\itemize>
    <item>Select a random full trajectory from the dataset

    <item>Generate a maximum of 100 prefixes for that trajectory. If the
    trajectory is smaller than 100 data points, generate all possible
    prefixes. Otherwise, chose a random subset of prefixes. Keep the final
    destination somewhere as it is used as a target for the training.

    <item>Take only the 5 first and 5 last points of the trajectory.

    <item>At this points we have a stream of prefixes sucessively taken from
    different trajectories. We create batches of size 200 with the items of
    the previous stream, taken in the order in which they come. The prefixes
    generated from a single trajectory may end up in two sucessive batches,
    or all in a single batch.
  </itemize>

  <section|References>

  <\enumerate>
    <item><label|gs_cit0>Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C.
    (2003). A neural probabilistic language model.
    <with|font-shape|italic|The Journal of Machine Learning Research>,
    <with|font-shape|italic|3>, 1137-1155.

    <item><label|gs_cit0>Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep
    sparse rectifier neural networks. In <with|font-shape|italic|International
    Conference on Artificial Intelligence and Statistics> (pp. 315-323).

    <item><label|gs_cit0>Bergstra, J., Bastien, F., Breuleux, O., Lamblin,
    P., Pascanu, R., Delalleau, O., ... & Bengio, Y. (2011). Theano: Deep
    learning on gpus with python. In <with|font-shape|italic|NIPS 2011,
    BigLearning Workshop, Granada, Spain>.

    <item><label|gs_cit0>Bastien, F., Lamblin, P., Pascanu, R., Bergstra, J.,
    Goodfellow, I., Bergeron, A., ... & Bengio, Y. (2012). Theano: new
    features and speed improvements. <with|font-shape|italic|arXiv preprint
    arXiv:1211.5590>.
  </enumerate>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|true>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|8|?>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|1|1>>
    <associate|auto-4|<tuple|3|1>>
    <associate|auto-5|<tuple|1|2>>
    <associate|auto-6|<tuple|4|3>>
    <associate|auto-7|<tuple|5|3>>
    <associate|auto-8|<tuple|6|4>>
    <associate|auto-9|<tuple|7|4>>
    <associate|firstHeading|<tuple|1|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|gs_cit0|<tuple|4|4>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|table>
      <tuple|normal|Embeddings and corresponding dimensions used by the
      model|<pageref|auto-3>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Summary>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Feature
      Selection/Extraction> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Modelling
      Techniques and Training> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Code
      Description> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Dependencies>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>How
      To Generate The Solution> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|7<space|2spc>Additional
      Comments and Observations> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|8<space|2spc>References>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>