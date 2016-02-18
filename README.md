# Virus Classification

This is a tool for classifying virus samples into virus classes.
It uses convolutional neural network for classifying.
I worked under the mentorship of Mr. Tomaž Curk, PhD, Assistant Professor.

The motivation for this was my Diploma thesis, where my goal was to classify viral sequences into
taxonomic groups by using standard machine learning methods and attributive description of the data.
The results wasn't really successful therefore I wanted to research further.

Now we approach the problem using convolutional neural network on short
sequence reads to classify it in class.

## Data
We assembled the taxonomic structure by collecting data from [NCBI](www.ncbi.nlm.nih.gov) web site.
To clean the data we applied several filtering steps - we exclude all bacterias, unclassified and
unspecified parts. After the taxonomy tree is built we shrink number of classes by excluding
all the list nodes from tree.

slika taksonomije

Then we calculate the number of examples per class we want (it depends on given dataset size)
and randomly sample short (100 nucleotides long) reads from chosen virus sequence.
These two steps are repeated until we reach examples per class.
For more detailed explanation of how the code works please see the documentation in files.

## Neural network
We found the skeleton for our convolutional neural network
[here](https://github.com/newmu/theano-tutorials). The code is explained in
[this](https://www.youtube.com/watch?v=S75EdAcXHKk) tutorial. The architecture
is the same as for convolutional neural networks for pictures, except that in our case
we represent sequences as "pictures" with 1px height.

slika iz deeplearning

Every nucleotide is represented with following vectors:
* `A = [1, 0, 0, 0]`
* `T = [0, 1, 0, 0]`
* `C = [0, 0, 1, 0]`
* `G = [0, 0, 0, 1]`
* `(everything else) = [1, 1, 1, 1]`

`ACG = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].`

As all heights in our code is 1, I will skip the whole shape from now on and
I will talk only about width.

We have reads of length 100 and every nucleotide is represented by 4 integers. That means that
lengths of our input are 400.
We perform convolutions in 3 stages and so we have 3 "blocks" of computation in our code.
Firstly we perform convolution. We then do rectify activation function, perform max pool, add drop out
noise and repeat for next stage.
In `cnet.py` we have following parameters:
- first stage (l1):
    - convolution with 6 nucleotides and 32 filters to learn on raw input, convolution stride is 4
    - max pool stride is 2
- second stage (l2):
    - convolution with 5 nucleotides and 48 filters to learn on output of l1, convolution stride is 1
    - max pool stride is 2
- third stage (l3)
    - convolution with 3 nucleotides and 64 filters to learn on l2, convolution stride is 1
    - max pool stride is 2
- last stage (l4)
    - connects the outputs of n filters to 500 (arbitrary) hidden nodes
    - hidden nodes are then connected to the output nodes

We automatically calculate the number of filters for the fourth stage,
because it is dependent on downscale parameters for each layer.

## Results
Results vary on network parameters. We tried 3 different architectures for dataset with
seed `7970223320302509880` (other parameters are default).

File        | Result  | Execution time
---         | ---     | ---
`cnet.py`   |  | ~24h
`cnet_2.py` |  | ~14h
`cnet_3.py` |  | ~7h


results
zakaj so taki
kaj bi lahko še izboljšali

## Software

Tool consists of 3 python scripts - `load.py`, `load_sequences.py` and `cnet_n.py`
(where n is integer and represents specific neural network architecture).
All code is written in Python 2.7.2. The main script is `cnet_n.py` - you simply run it and it
handles the other two scripts to get the data.

Scripts which are responsible for data create folder cache and media in working directory.
When `load_sequences.py` is run folder cache is created (if it does not exists already).
It stores records from NCBI website to avoid downloading next time we run it.
When `load.py` is run folder media is created (if does not exists already).
Media directory stores data ids and list labels from our dataset.
It also stores train and test data for unique random seeds that have already been built.