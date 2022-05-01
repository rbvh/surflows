wget -O training_files.tar https://zenodo.org/record/3961917/files/training_files.tar?download=1
wget -O secret_data.tar https://zenodo.org/record/4443151/files/secret_data.tar?download=1
tar -xvf training_files.tar
tar -xvf secret_data.tar
mkdir secret_data
mv chan1 secret_data/
mv chan2a secret_data/
mv chan2b secret_data/
mv chan3 secret_data/