# Finding risk groups by optimizing artificial neural networks on the area under the survival curve using genetic algorithms

This is the data, code, and results behind *Finding risk groups by
  optimizing artificial neural networks on the area under the survival
  curve using genetic algorithms* by Kalderstam J., Edén P., and Ohlsson M.

The data (under the `data` directory) was originally published with the
[survival](http://cran.r-project.org/web/packages/survival/index.html)
package and is published under the LGPL-2 license.

The code and all other files are available under the GPL-3 license
(see
[LICENSE](https://github.com/spacecowboy/article-annriskgroups-source/blob/master/LICENSE)).

## Experiment files

The source is written in iPython Notebooks, and the easiest way to
read them is using the excellent
[nbviewer](http://nbviewer.ipython.org/) service. Just click the links
to each script to open them in the viewer directly.

### Scripts with results relevant for the article

- [AnnVariables.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/AnnVariables.ipynb)

This is the script used to determine suitable parameters for the ANN
and genetic algorithm. Repeated cross-validation is performed where in
each repetition, a single variable is investigated.

- [CrossVal.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/CrossVal.ipynb)

Cross-validation script which produces figures 1-4.

- [DataSetStratification.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/DataSetStratification.ipynb)

Code used to stratify data sets into training and test pieces.

- [RPartVariables.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/RPartVariables.ipynb)

Script to compare different parameters for Rpart.

- [TestData.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/TestData.ipynb)

Here all models are trained on the training data using the decided
parameters, and then predicts their groupings on the test
data. Produces figure 5 and data for table 1.

### Scripts relevant mostly for the development process

- [AnnGroups.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/AnnGroups.ipynb)

This script loads a data set, trains an ANN Riskgroup ensemble on it,
and plots the resulting grouping. The script is a test/example, and is
not relevant for the results reported.

- [DataSetHistograms.ipynb](http://nbviewer.ipython.org/github/spacecowboy/article-annriskgroups-source/blob/master/DataSetHistograms.ipynb)

Only loads the datasets and prints some info and plots some
histograms. No direct relevance to the article results.


### Data sets

The file
[datasets.py](https://github.com/spacecowboy/article-annriskgroups-source/blob/master/datasets.py)
contain helper methods to load the files, fill missing values, and
normalize covariates.

The data sets are stored in the `data` folder. Each data set comes in
two files, *data.csv* and *data_org.csv*. *data_org.csv* is the
original file extracted from `R`'s `survival` package. *data.csv* has
labels which classify each entry as being part of either the training
set or the testing set. Here follows a brief description of each data
set, please see the
[survival package's manual](http://cran.r-project.org/web/packages/survival/survival.pdf)
for more details on each data set.


#### colon

One of the first successful trials of adjuvant chemotherapy
for colon cancer. Consists of 929
patients, 461 (50%) of which were censored before recurrence.  The
target variable is *days until recurrence* and the 11 input
features are: type of treatment, sex, age, obstruction of colon by
tumor, perforation of colon, adherence to nearby organs, number of
lymph nodes with detectable cancer, differentiation of tumor, extent
of local spread, time from surgery to registration (short/long), and
more than 4 positive lymph nodes (yes/no).

#### flchain

A study of the relationship between serum free light chain and
mortality. In total 7871 patients are included with 5705 (72%)
patients censored (still alive at last contact date). The target
variable is *days until death* and the 7 input features are: age, sex,
kappa portion, lambda portion, FLC group, serum creatine, and if
diagnosed with monoclonal gammapothy.

#### nwtco

From the National Wilm's Tumor Study. 4028 patients where 3457 (86%)
are censored before relapse. The target variable is *days to relapse*
and it contains 4 input features: histology from local institution,
histology from central lab, age, and disease stage.


#### pbc

A randomized trial in primary biliary cirrhosis (PBC) of the liver at
the Mayo Clinic. The randomized trial consisted of 312 patients where
187 (60%) were censored. The target variable is *days until death* and
the 17 input features are: type of treatment, age, sex, presence of
ascites, presence of hepatomegaly or enlarged liver, blood vessel
malformations in the skin, presence of edema, serum bilirunbin, serum
cholesterol, serum albumin, urine copper, alkaline phosphotase,
aspartate aminotransferase, triglycerides, platelet count, blood
clotting time, and histologic stage of disease.



#### lung

Originates from the North Central Cancer Treatment Group and consists
of 228 patients with advanced lung cancer where 63 patients (28%) were
censored. The target variable is *survival time in days* and the 7
input features are: age, sex, ECOG performance score, Karnofsky
performance score by physician, Karnofsky performance score by
patient, calories consumed at meals, and weight loss in the last six
months.


## Dependencies and required software to run the scripts

You will need **GCC-4.7 or higher**. The authors have only run the
code on **Linux** (OpenSuse, Debian, Arch Linux) and while it might be
possible to run the code on other systems (Windows, OS X), we offer no
support or guarantees for them.

The experiments are written in iPython Notebooks, and thus obviously
requires Python. To setup a suitable Python environment it is recommended
to use [Conda](http://conda.pydata.org/miniconda.html) (with Python 3.4).

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
[Lifelines](https://github.com/CamDavidsonPilon/lifelines.git), a
survival analysis package for package (used here primarily for
Kaplan-Meier plots).

```
pip install lifelines==0.7.0
```

Second is the software necessary to run the neural network
experiments. Note that these are installed directly from github and
have the specific commits (versions) specified.

```
pip install git+https://github.com/spacecowboy/jkutils.git@3e5cd26
pip install git+https://github.com/spacecowboy/pyplotthemes.git@9559f9b
pip install git+https://github.com/spacecowboy/pysurvival-ann.git@981fb23
```

Final step is to get the python wrappers around `R` to work
properly. This can be often be problematic. To install `R` as a
shared-library, see
[pysurvival](https://github.com/spacecowboy/pysurvival). You also need
the following packages available in in `R`:

- [survival](http://cran.r-project.org/web/packages/survival/index.html)
- [rpart](http://cran.r-project.org/web/packages/rpart/index.html)

Once that is done however, all you should have to do is:

```
pip install rpy2==2.5.6
pip install git+https://github.com/spacecowboy/pysurvival.git@c862d85
```

You should now be able to view and even re-run the experimental
files. Just run `ipython notebook` and open a notebook in the browser
window that starts.
