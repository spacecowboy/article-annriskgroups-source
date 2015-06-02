# Title

and some text

## Experiment files

The source is written in iPython Notebooks, and the easiest way to
read them is using the excellent
(nbviewer)[http://nbviewer.ipython.org/] service. Just click the links
to each script to open them in the viewer directly.

### Scripts with results relevant for the article

- (AnnVariables.ipynb)[http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/AnnVariables.ipynb]

This is the script used to determine suitable parameters for the ANN
and genetic algorithm. Repeated cross-validation is performed where in
each repetition, a single variable is investigated.

- (CrossVal.ipynb)[http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/CrossVal.ipynb]

Cross-validation script which produces figures 1-4.



### Scripts relevant mostly for the development process

- (AnnGroups.ipynb)[http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/AnnGroups.ipynb]

This script loads a data set, trains an ANN Riskgroup ensemble on it,
and plots the resulting grouping. The script is a test/example, and is
not relevant for the results reported.

- CoxGroups.ipynb

This seems to be broken...

- (DataSetHistograms.ipynb)[http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/DataSetHistograms.ipynb]

Only loads the datasets and prints some info and plots some
histograms. No direct relevance to the article results.

### Data sets

Mention something of the data sets

## Dependencies and required software to run the scripts

You will need **GCC-4.7 or higher**. The authors have only run the
code on **Linux** (OpenSuse, Debian, Arch Linux) and while it might be
possible to run the code on other systems (Windows, OS X), we offer no
support or guarantees for them.

The experiments are written in iPython Notebooks, and thus obviously
requires Python. To setup a suitable Python environment it is recommended
to use (Conda)[http://conda.pydata.org/miniconda.html] (with Python 3.4).

Once *Conda* is installed, a new environment named `riskgroups` can be
created as follows:

```
conda create -n riskgroups python=3.4 pip=7.0.1 numpy=1.9.2 scipy=0.15.1 matplotlib=1.4.3 pandas=0.16.1 ipython=3.1.0 ipython-notebook=3.1.0
```

Note that the specific versions of packages have been specified. Newer
versions will most likely work fine, but these have been tested to
work correctly (on *2015-06-02*).

Next you need to activate the environment:

```
source activate riskgroups
```

Final software will now be installed using *pip*.

First is
(Lifelines)[https://github.com/CamDavidsonPilon/lifelines.git], a
survival analysis package for package (used here primarily for
Kaplan-Meier plots).

```
pip install lifelines==0.7.0
```

And last is the software necessary to run the neural networks
experiments. Note that these are installed directly from github and
have the specific commits (versions) specified.

```
pip install git+https://github.com/spacecowboy/jkutils.git@3e5cd26
pip install git+https://github.com/spacecowboy/pyplotthemes.git@9559f9b
pip install git+https://github.com/spacecowboy/pysurvival-ann.git@981fb23
```

You should now be able to view and even re-run the experimental files.

**TODO** Also need pysurvival for Cox and Rpart, but that requires rpy2.

## Viewing the files

The following versions were used to run the experiments (as reported
by `pip freeze`):

```
ipython==2.3.1
Jinja2==2.7.3
jkutils==1.1
lifelines==0.7.0.0
MarkupSafe==0.23
matplotlib==1.4.2
numpy==1.9.1
pandas==0.15.2
py==1.4.26
pyparsing==2.0.1
pyplotthemes==0.1
pysurvival==1.2
pysurvival-ann==0.9
pytest==2.6.4
python-dateutil==2.1
pytz==2014.9
pyzmq==14.5.0
rpy2==2.5.6
scipy==0.15.1
six==1.9.0
tornado==4.0.2
```
