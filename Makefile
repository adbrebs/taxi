FILES=config/__init__.py \
	  config/dest_mlp_tgtcls_1_cswdtx_alexandre.py \
	  data/cuts/__init__.py \
	  data/cuts/test_times_0.py \
	  data/__init__.py \
	  data/csv_to_hdf5.py \
	  data/hdf5.py \
	  data/init_valid.py \
	  data/make_valid_cut.py \
	  data/transformers.py \
	  model/__init__.py \
	  model/mlp.py \
	  model/dest_mlp_tgtcls.py \
	  data_analysis/cluster_arrival.py \
	  doc/report.pdf \
	  __init__.py \
	  error.py \
	  ext_saveload.py \
	  ext_test.py \
	  train.py
		
submission.tgz: $(FILES)
	tar czf $@ $(FILES)
