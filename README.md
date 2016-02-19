# Virus Classification

This is a tool for classifying virus samples into virus classes.
It uses convolutional neural network for classifying.
I worked under the mentorship of Mr. Tomaž Curk, PhD, Assistant Professor.

The motivation for this was my diploma thesis, where my goal was to classify viral sequences into
taxonomic groups by using standard machine learning methods and attributive description of the data.
The results were not really successful therefore we wanted to research further.

Now we approach the problem using convolutional neural network on short
sequence reads to classify it in class.

## Data
We assembled the taxonomic structure by collecting data from [NCBI](www.ncbi.nlm.nih.gov) web site.
To clean the data we applied several filtering steps - we exclude all bacterias, unclassified and
unspecified parts. After the taxonomy tree is built we shrink classes by removing
all the list nodes from tree (as shown in picture below).

![alt-text](https://github.com/mkopar/Virus-classification-theano/blob/master/taksonomija.png)

Then we calculate the number of examples per class we want (it depends on given data set size)
and randomly sample short (100 nucleotides long) reads from chosen virus sequence.
For more detailed explanation of how the code works please see documentation in files.

## Neural network
We checked a few articles about text classification using neural networks.
We found the skeleton for our convolutional neural network
[here](https://github.com/newmu/theano-tutorials). The code is explained in
[this](https://www.youtube.com/watch?v=S75EdAcXHKk) tutorial. The architecture
is the same as for convolutional neural networks for pictures, except that in our case
we represent sequences as "pictures" with 1px height.

![alt-text](https://github.com/mkopar/Virus-classification-theano/blob/master/mylenet.png)

Source: deeplearning.net

Picture above shows basic idea on how the convolutional neural network works. Our network works similar to this,
only that we do not have a regular picture.

We implemented following encoding for nucleotides:
* `A = [1, 0, 0, 0]`
* `T = [0, 1, 0, 0]`
* `C = [0, 0, 1, 0]`
* `G = [0, 0, 0, 1]`
* `(everything else) = [1, 1, 1, 1]`

For example, sequence `ACG` looks like this:
`ACG = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].`

As all heights in our code is 1, I will skip the whole shape from now on and will talk only about width.

We have reads of length 100 and every nucleotide is represented by 4 integers. That means that
length of our inputs is 400.
We perform convolutions in 3 stages and so we have 3 "blocks" of computation in our code.
Firstly we perform convolution. We then do rectify activation function, perform max pool, add drop out
noise. Those steps are repeated for every stage (except the last one).
In `cnet.py` we have following parameters:
- first stage (l1):
    - we do convolution with 6 nucleotides and 32 filters to learn on raw input, convolution stride is 4, output is of width 95
    - max pool stride is 2, output is of width 47
- second stage (l2):
    - we do convolution with 5 nucleotides and 48 filters to learn on output of l1, convolution stride is 1, output is of width 43
    - max pool stride is 2, output is of width 21
- third stage (l3)
    - we do convolution with 3 nucleotides and 64 filters to learn on l2, convolution stride is 1, output is of width 19
    - max pool stride is 2, output is of width 10
- last stage (l4)
    - we connect the outputs of n filters to 500 (arbitrary) hidden nodes
    - hidden nodes are then connected to the output nodes

The number of filters for the fourth stage is automatically calculated,
because it is dependent on downscale parameters for each layer.

For more detailed info on how the `cnet.py` works please see [here](https://www.youtube.com/watch?v=S75EdAcXHKk).

## Results
We tried 3 different architectures for data set with seed `7970223320302509880` (other parameters are default).
Each architecture is in different file.

File        | Result  | Execution time
---         | ---     | ---
`cnet.py`   |  16.16% | ~26h
`cnet_2.py` |  13.31% | ~14h
`cnet_3.py` |  9.04%  | ~7h

Results vary on the architecture of neural network. In the table above are precision results and estimate execution time
for each file.
Results are not really good, which might be because of our evaluation type. The evaluation very strict -
we want to predict exact class for one read, but one read might be present in multiple classes.
It would be good to check the results for different representation of nucleotides
(we should encode it into pyrimidines and purines instead of every single nucleotide). It would also be good to
check the results with different architecture of neural network (where we exclude one layer or with different
parameters).

## Software

Tool consists of 3 python scripts - `load.py`, `load_sequences.py` and `cnet_n.py`
(where n is integer and represents specific neural network architecture).
All code is written in Python 2.7.2. We used libraries such as numpy, BioPython and Theano.
The main script is `cnet_n.py` - you run it and it handles the other two scripts to get the data.

Scripts which are responsible for data create folders cache and media in working directory.
Folder cache is created by `load_sequences.py` (if it does not exists already).
This folder stores records from NCBI website to avoid downloading every time we run it.
Folder media is created by `load.py` (if does not exists already).
Media directory stores data ids and list labels from our data set, so we do not need to built taxonomy
tree every time we run it. Media directory also stores train and test data for
unique random seeds that have already been built.