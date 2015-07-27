from model.bidirectional import SegregatedBidirectional


class Model(Initializable):
    @lazy()
    def __init__(self, config, output_dim=2, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.config = config

        self.context_embedder = ContextEmbedder(config)
        
        act = config.rec_activation() if hasattr(config, 'rec_activation') else None
        self.rec = SegregatedBidirectional(LSTM(dim=config.hidden_state_dim, activation=act,
                                                name='recurrent'))

        self.fwd_fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'],
                             prototype=Linear(), name='fwd_fork')
        self.bkwd_fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'],
                              prototype=Linear(), name='bkwd_fork')

        rto_in = config.hidden_state_dim * 2 + sum(x[2] for x in config.dim_embeddings)
        self.rec_to_output = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Identity()], 
                                 dims=[rto_in] + config.dim_hidden + [output_dim])

        self.softmax = Softmax()

        self.sequences = ['latitude', 'latitude_mask', 'longitude']
        self.inputs = self.sequences + self.context_embedder.inputs

        self.children = [ self.context_embedder, self.fwd_fork, self.bkwd_fork,
                          self.rec, self.rec_to_output, self.softmax ]

        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX),
                                     name='classes')

    def _push_allocation_config(self):
        for i, fork in enumerate([self.fwd_fork, self.bkwd_fork]):
            fork.input_dim = 2 * self.config.window_size
            fork.output_dims = [ self.rec.children[i].get_dim(name)
                                 for name in fork.output_names ]

    def _push_initialization_config(self):
        for brick in [self.fwd_fork, self.bkwd_fork, self.rec, self.rec_to_output]:
            brick.weights_init = self.config.weights_init
            brick.biases_init = self.config.biases_init

    def process_outputs(self, outputs):
        return tensor.dot(self.softmax.apply(outputs), self.classes)

    @application(outputs=['destination'])
    def predict(self, latitude, longitude, latitude_mask, **kwargs):
        latitude = (latitude.dimshuffle(1, 0, 2) - data.train_gps_mean[0]) / data.train_gps_std[0]
        longitude = (longitude.dimshuffle(1, 0, 2) - data.train_gps_mean[1]) / data.train_gps_std[1]
        latitude_mask = latitude_mask.T

        rec_in = tensor.concatenate((latitude, longitude), axis=2)

        last_id = tensor.cast(latitude_mask.sum(axis=0) - 1, dtype='int64')

        path = self.rec.apply(merge(self.fwd_fork.apply(rec_in, as_dict=True),
                                    {'mask': latitude_mask}),
                              merge(self.bkwd_fork.apply(rec_in, as_dict=True),
                                    {'mask': latitude_mask}))[0]

        path_representation = (path[0][:, -self.config.hidden_state_dim:],
                               path[last_id - 1, tensor.arange(latitude_mask.shape[1])]
                                   [:, :self.config.hidden_state_dim])

        embeddings = tuple(self.context_embedder.apply(
                        **{k: kwargs[k] for k in self.context_embedder.inputs }))

        inputs = tensor.concatenate(path_representation + embeddings, axis=1)
        outputs = self.rec_to_output.apply(inputs)

        return self.process_outputs(outputs)

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        y_hat = self.predict(**kwargs)
        y = tensor.concatenate((kwargs['destination_latitude'][:, None],
                                kwargs['destination_longitude'][:, None]), axis=1)

        return error.erdist(y_hat, y).mean()

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']



class Stream(object):
    def __init__(self, config):
        self.config = config

    def train(self, req_vars):
        stream = TaxiDataset('train', data.traintest_ds)

        if hasattr(self.config, 'use_cuts_for_training') and self.config.use_cuts_for_training:
            stream = DataStream(stream, iteration_scheme=TaxiTimeCutScheme())
        else:
            stream = DataStream(stream, iteration_scheme=ShuffledExampleScheme(stream.num_examples))

        if not data.tvt:
            valid = TaxiDataset(data.valid_set, data.valid_ds, sources=('trip_id',))
            valid_trips_ids = valid.get_data(None, slice(0, valid.num_examples))[0]
            stream = transformers.TaxiExcludeTrips(stream, valid_trips_ids)

        if hasattr(self.config, 'max_splits'):
            stream = transformers.TaxiGenerateSplits(stream, max_splits=self.config.max_splits)
        elif not data.tvt:
            stream = transformers.add_destination(stream)

        if hasattr(self.config, 'train_max_len'):
            idx = stream.sources.index('latitude')
            def max_len_filter(x):
                return len(x[idx]) <= self.config.train_max_len
            stream = Filter(stream, max_len_filter)

        stream = transformers.TaxiExcludeEmptyTrips(stream)

        stream = transformers.window(stream, config.window_size)
        
        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = transformers.balanced_batch(stream, key='latitude',
                                             batch_size=self.config.batch_size,
                                             batch_sort_size=self.config.batch_sort_size)
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        stream = MultiProcessing(stream)

        return stream

    def valid(self, req_vars):
        stream = TaxiStream(data.valid_set, data.valid_ds)

        stream = transformers.window(stream, config.window_size)

        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = transformers.balanced_batch(stream, key='latitude',
                                             batch_size=self.config.batch_size,
                                             batch_sort_size=self.config.batch_sort_size)
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        stream = MultiProcessing(stream)

        return stream

    def test(self, req_vars):
        stream = TaxiStream('test', data.traintest_ds)

        stream = transformers.window(stream, config.window_size)
        
        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.taxi_remove_test_only_clients(stream)

        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        return stream

    def inputs(self):
        return {'call_type': tensor.bvector('call_type'),
                'origin_call': tensor.ivector('origin_call'),
                'origin_stand': tensor.bvector('origin_stand'),
                'taxi_id': tensor.wvector('taxi_id'),
                'timestamp': tensor.ivector('timestamp'),
                'day_type': tensor.bvector('day_type'),
                'missing_data': tensor.bvector('missing_data'),
                'latitude': tensor.tensor('latitude'),
                'longitude': tensor.tensor('longitude'),
                'latitude_mask': tensor.matrix('latitude_mask'),
                'longitude_mask': tensor.matrix('longitude_mask'),
                'destination_latitude': tensor.vector('destination_latitude'),
                'destination_longitude': tensor.vector('destination_longitude'),
                'travel_time': tensor.ivector('travel_time'),
                'input_time': tensor.ivector('input_time'),
                'week_of_year': tensor.bvector('week_of_year'),
                'day_of_week': tensor.bvector('day_of_week'),
                'qhour_of_day': tensor.bvector('qhour_of_day')}

