#!/usr/bin/env bash

RESET=`tput sgr0`
BOLD="`tput bold`"
RED="$RESET`tput setaf 1`$BOLD"
GREEN="$RESET`tput setaf 2`"
YELLOW="$RESET`tput setaf 3`"
BLUE="$RESET`tput setaf 4`$BOLD"

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "${YELLOW}This script will prepare the data."
echo "${YELLOW}You should run it from inside the repository."
echo "${YELLOW}You should set the TAXI_PATH variable to where the data downloaded from kaggle is."
echo "${YELLOW}Three data files are needed: ${BOLD}train.csv.zip${YELLOW}, ${BOLD}test.csv.zip${YELLOW} and ${BOLD}metaData_taxistandsID_name_GPSlocation.csv.zip${YELLOW}. They can be found at the following url: ${BOLD}https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data"
if [ ! -e train.py ]; then
    echo "${RED}train.py not found, you are not inside the taxi repository."
    exit 1
fi


echo -e "\n$BLUE# Checking dependencies"

python_import(){
    echo -n "${YELLOW}$1... $RESET"
    if ! python2 -c "import $1; print '${GREEN}version', $1.__version__, '${YELLOW}(we used version $2)'"; then
        echo "${RED}failed, $1 is not installed"
        exit 1
    fi
}

python_import h5py 2.5.0
python_import theano 0.7.0.dev
python_import fuel 0.0.1-ed725a7ff9f3d080ef882d4ae7e4373c4984f35a
python_import blocks 0.0.1-1e0aca9171611be4df404129d91a991354e67730
python_import sklearn 0.16.1


echo -e "\n$BLUE# Checking data"

echo "${YELLOW}TAXI_PATH is set to $TAXI_PATH"

md5_check(){
    echo -n "${YELLOW}md5sum $1... $RESET"
    if [ ! -e "$TAXI_PATH/$1" ]; then
        echo "${RED}file not found, are you sure you set the TAXI_PATH variable correctly?"
        exit 1
    fi
	if command -v md5 >/dev/null 2>&1; then
		md5=`md5 "$TAXI_PATH/$1" | sed -e 's/^.* //'`
	elif command -v md5sum >/dev/null 2>&1; then
		md5=`md5sum "$TAXI_PATH/$1" | sed -e 's/ .*//'`
	else
        echo "${RED} no md5 utility"
		return
	fi
    if [ $md5 = $2 ]; then
        echo "$GREEN$md5 ok"
    else
        echo "$RED$md5 failed"
        exit 1
    fi
}

md5_check train.csv.zip 87a1b75adfde321dc163160b495964e8
md5_check test.csv.zip 47133bf7349cb80cc668fa56af8ce743
md5_check metaData_taxistandsID_name_GPSlocation.csv.zip fecec7286191af868ce8fb208f5c7643


echo -e "\n$BLUE# Extracting data"

zipextract(){
	echo -n "${YELLOW}unziping $1... $RESET"
	unzip -o "$TAXI_PATH/$1" -d "$TAXI_PATH"
	echo "${GREEN}ok"
}

zipextract train.csv.zip
md5_check train.csv 68cc499ac4937a3079ebf69e69e73971

zipextract test.csv.zip
md5_check test.csv f2ceffde9d98e3c49046c7d998308e71

zipextract metaData_taxistandsID_name_GPSlocation.csv.zip

echo -n "${YELLOW}patching error in metadata csv... $RESET"
cat "$TAXI_PATH/metaData_taxistandsID_name_GPSlocation.csv" | sed -e 's/41,Nevogilde,41.163066654-8.67598304213/41,Nevogilde,41.163066654,-8.67598304213/' > "$TAXI_PATH/metaData_taxistandsID_name_GPSlocation.csv.tmp"
mv "$TAXI_PATH/metaData_taxistandsID_name_GPSlocation.csv.tmp" "$TAXI_PATH/metaData_taxistandsID_name_GPSlocation.csv"
echo "${GREEN}ok"

md5_check metaData_taxistandsID_name_GPSlocation.csv 724805b0b1385eb3efc02e8bdfe9c1df


echo -e "\n$BLUE# Conversion of training set to HDF5"
echo "${YELLOW}This might take some time$RESET"
python2 data/csv_to_hdf5.py "$TAXI_PATH" "$TAXI_PATH/data.hdf5"


echo -e "\n$BLUE# Generation of validation set"
echo "${YELLOW}This might take some time$RESET"

echo -n "${YELLOW}initialization... $RESET"
python2 data/init_valid.py
echo "${GREEN}ok"

echo -n "${YELLOW}cutting... $RESET"
python2 data/make_valid_cut.py test_times_0
echo "${GREEN}ok"


echo -e "\n$BLUE# Generation of destination cluster"
echo "${YELLOW}This might take some time$RESET"
echo -n "${YELLOW}generating... $RESET"
python2 data_analysis/cluster_arrival.py
echo "${GREEN}ok"


echo -e "\n$BLUE# Creating output folders"
echo -n "${YELLOW}mkdir model_data... $RESET"; mkdir model_data; echo "${GREEN}ok"
echo -n "${YELLOW}mkdir output... $RESET"; mkdir output; echo "${GREEN}ok"

echo -e "\n$GREEN${BOLD}The data was successfully prepared"
echo "${YELLOW}To train the winning model on gpu, you can now run the following command:"
echo "${YELLOW}THEANO_FLAGS=floatX=float32,device=gpu,optimizer=fast_run python2 train.py dest_mlp_tgtcls_1_cswdtx_alexandre"
