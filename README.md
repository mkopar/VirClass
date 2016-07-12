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
The latest version of tool expects file with short reads in fasta format. If you do not provide it,
it will generate the dataset and save it in files for further use.
We assembled the taxonomic structure by collecting data from [NCBI](www.ncbi.nlm.nih.gov) web site.
To clean the data we applied several filtering steps - we exclude all bacterias, unclassified and
unspecified parts. After the taxonomy tree is built we apply tree level filter - by default, we
cut the tree at the 4th level.
That means that final classes for classifying are nodes at 4th level
but data is same as before, just divided into less groups than before.
If we cut the tree at 2nd level, the tree would look something like that:

![alt-text](https://github.com/mkopar/Virus-classification-theano/blob/master/taksonomija_2.png)

After that we just split the genomes into smaller chunks (by default the size is 100) and randomly
sample short reads from chosen virus sequence. The bigger the sample, the longer it would take
to build the dataset. After that we transform letters into numeric representation, where in our
code we use the following transmission dictionary:

* `A : [1, 0, 0, 0]`
* `T : [0, 1, 0, 0]`
* `C : [0, 0, 1, 0]`
* `G : [0, 0, 0, 1]`

In the example above letters, which are not in the transmission dictionary (e.g. 'M', 'Y', ...) will
get the following numeric representation:

* `_ (everything else) : [1, 1, 1, 1]`

If you do not provide transmission dictionary, then every letter will be represented by one bit.

For more detailed explanation of how the code works please see documentation in files.

## Neural network
We checked a few articles about text classification using neural networks.
We found the skeleton for our convolutional neural network
[here](https://github.com/newmu/theano-tutorials). The code is explained in
[this](https://www.youtube.com/watch?v=S75EdAcXHKk) tutorial. The architecture
is the same as for convolutional neural networks for pictures, except that in our case
we represent sequences as "pictures" with 1px height.

![alt-text](https://github.com/mkopar/Virus-classification-theano/blob/master/mylenet.png)

Picture source: deeplearning.net

Picture above shows basic idea on how the convolutional neural network works. Our network works similar to this,
only that we do not have a regular picture.

We implemented following encoding for nucleotides (but the encoding could be different - user just need to change
the transmission dictionary):
* `A = [1, 0, 0, 0]`
* `T = [0, 1, 0, 0]`
* `C = [0, 0, 1, 0]`
* `G = [0, 0, 0, 1]`
* `(everything else) = [1, 1, 1, 1]`

For example, sequence `ACG` looks like this:
`ACG = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].`

As all heights in our code is 1, I will skip the whole shape from now on and will talk only about width.

If we have reads of length 100 and every nucleotide is represented by 4 integers, that means that
length of our inputs are 400.
We perform convolutions in 3 stages and so we have 3 "blocks" of computation in our code.
Firstly we perform convolution. We then do rectify activation function, perform max pool and add drop out
noise. Those steps are repeated for every stage (except the last one).
In `build.py` we have following parameters:
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
because it is dependent on downscale parameters for each layer. Output of the neural net
is a list of probabilities for each class.

At the end of fitting the model, the best model is evaluated and saved for
further usage in `predict.py` module, where we actually perform prediction experiments.

For more detailed info on how the `build.py` works please see [here](https://www.youtube.com/watch?v=S75EdAcXHKk).

## Software

Current version of tool is developed primarily for command line use. This we believe is simplier for users
to perform their own predictions and/or building the model. It is on our TODO list to build some useful
models and push it to repository for users to test the tool by themselves, as building the neural net takes quite a big
amount of time.

Tool consists of 4 python scripts - `load.py`, `load_ncbi.py`, `build.py` and `predict.py`.
All code is written in Python 2.7.2. We used libraries such as numpy, BioPython and Theano.
The main scripts are `build.py` and `predict.py`.

`build.py`:
- `learning data file [OPTIONAL]`
- `sliding window size [OPTIONAL]`

If you provide file with learning data and corresponding class labels, than this file will be used for learning model.
File must be in 'media/' directory. If you do not provide file with learning data, new dataset will be generated.

If you do provide length of read window size, then this window size is used when slicing sequences. Otherwise
default value is used (100).
If we need to generate new datasets, then files `load.py` and `load_ncbi.py` are called
for proper building of the dataset.
Outputs of this file are best model, saved in specific file in 'models/' directory
and (naive) score for the best model.

`predict.py`:
- `prediction data file [MANDATORY]`
- `class labels file [OPTIONAL]`
- `model filename [MANDATORY]`

In order for `predict.py` to work, user must provide prediction data file and model filename he wants to use.
Class labels file is optional because it is only possible to use it when we perform controlled experiments with
prebuild prediction dataset and not with real data. Output of this file is table that shows the presence of
final class labels for every read.

Both main scripts have help if you find troubles with running them.

It is best if user can provide that directories 'media', 'cache' and 'models'
are present and have the right permissions.
Folder 'cache' stores records from NCBI website to avoid downloading it every time we want to run the tool.
Folder 'media' stores datasets for train and test and other data.
Folder 'models' stores best models with few other params for use in prediction.