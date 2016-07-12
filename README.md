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
- `debug mode flag [OPTIONAL]`

If you provide file with learning data and corresponding class labels, than this file will be used for learning model.
File must be in `media/` directory. If you do not provide file with learning data, new dataset will be generated.

If you do provide length of read window size, then this window size is used when slicing sequences. Otherwise
default value is used (100).
If we need to generate new datasets, then files `load.py` and `load_ncbi.py` are called
for proper building of the dataset.
Outputs of this file are best model, saved in specific file in `models/` directory
and (naive) score for the best model.

`predict.py`:
- `prediction data file [MANDATORY]`
- `model filename [MANDATORY]`

In order for `predict.py` to work, user must provide prediction data file and model filename he wants to use.
Prediction data file must be in numeric representation. It must contain some reads (small sequence chunks) which
are translated into numbers.
Output of this file is table that shows the presence of final class labels for whole sample (file) - so it shows
which viruses are detected in the reads.

If you run `build.py` it will generate a few files into `media/` directory and into `models/`.
Log in console will print filenames.

`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 build.py -d`

`...`

`Successfully saved as: media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-trX.fasta.gz`

`Successfully saved as: media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-trY.fasta.gz`

`Successfully saved as: media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-teX.fasta.gz`

`Successfully saved as: media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-teY.fasta.gz`

`Successfully saved as: media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-trteX.fasta.gz`

`Successfully saved as: media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-trteY.fasta.gz`

`...`

`Model saved as: models/best_model_with_params-1468314846.pkl`

After that, you should run `predict.py`:

`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2 -u predict.py media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-teX.fasta.gz models/best_model_with_params-1468314846.pkl -teY media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-teY.fasta.gz`

`raw predicted values:`
`[  28.16671371  28.16671371  28.16671753  28.16673088  28.16672516 28.16678047  28.16675758  28.16679573  28.16674423  28.16673279  28.16673279  28.16672707]`

`weighted and normed predicted values: `
`[0.0014856645242153916, 0.005405241549529766, 0.0037420908103224224, 0.00501187382219525, 0.00026656122686071723, 0.003804265325743079, 0.00022706154490256402, 0.0030329272887467235, 0.005466086596580193, 0.00042061244195690256, 0.000622758247763292, 0.002846273955743739]`

`sorted probabilities:  `
`[(8, 0.16906424299368172), (1, 0.16718232589673507), (3, 0.1550155927385808), (5, 0.11766466302349769), (2, 0.11574162591141302), (7, 0.09380743372189672), (11, 0.08803430812484887), (0, 0.04595111030370201), (10, 0.01926170576807985), (9, 0.013009403132701425), (4, 0.008244650214445804), (6, 0.007022938170417014)]`

`expected classes:  [ 0  9 11]`

Both main scripts have help if you find troubles with running them.

It is best if user can provide that directories `media/`, `cache/` and `models/`
are present and have the right permissions.
Folder 'cache' stores records from NCBI website to avoid downloading it every time we want to run the tool.
Folder 'media' stores datasets for train and test and other data.
Folder 'models' stores best models with few other params for use in prediction.