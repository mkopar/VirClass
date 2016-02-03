# Virus Classification

Run load_sequences.py first to download data from genbank and generate the
files needed for the analysis. The script generates two files and stores them
in the cache folder: data1-100.npy and labels1-100.npy.

Then run 5_convolutional_net.py to run the analysis. The script uses load.py to
read the two data files, which get converted into numpy arrays.
