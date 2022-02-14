# Oger

## TL;DR
Oger is a Python 2 toolbox mainly used for Reservoir Computing neural networks, such as Echo State Networks (ESN) and Liquid State Machines (LSM).

It was developped during the European FP7 Organic project (2009-2012) focused on Reservoir Computing.

The Oger toolbox is no longer maintained.

I provide the original code available with the original license (LGPL).


## Why do I provide the Oger toolbox?
I am not the author of this toolbox, but I participated (as a PhD student) to the european Organic FP7 project which develpped it.
As I previously [developped code](https://github.com/neuronalX/HinautDominey2013_PLoS_ONE) using this toolbox in a study (published as supplementary material of a [PLoS ONE paper](https://doi.org/10.1371/journal.pone.0052946)) I want my code to be still executable today with the original toolbox.

That's why I copy/pasted the version of the Oger code that I was using for my papers (early/mid 2012), which is probably close to the latest stable version.

## Original paper describing the toolbox
The original paper describing Oger is [available here](https://www.researchgate.net/publication/262356589_Oger_Modular_Learning_Architectures_For_Large-Scale_Sequential_Processing).

**Abstract**

*Oger (OrGanic Environment for Reservoir computing) is a Python toolbox for building, training and evaluating modular learning architectures on large data sets. It builds on MDP for its modularity, and adds processing of sequential data sets, gradient descent training, several crossvalidation schemes and parallel parameter optimization methods. Additionally, several learning algorithms are implemented, such as different reservoir implementations (both sigmoid and spiking), ridge regression, conditional restricted Boltzmann machine (CRBM) and others, including GPU accelerated versions. Oger is released under the GNU LGPL.*

## Current

### Repository of Oger no longer maintained
The development version of the Oger repository is (EDIT: was) still available at [Benjamin Schrauwen's bitbucket repository](https://bitbucket.org/benjamin_schrauwen/organic-reservoir-computing-engine/downloads/) (for how long still?). The last update is from April 2012.

### Currently maintained Reservoir Computing toolbox
[Here](https://github.com/reservoirpy/reservoirpy) you can find a new and flexible library on Reservoir Computing (mainly about ESN) which is made with pure Python (no dependencies but numpy/scipy/matplotlib).

### Tutorial
You can find a tutorial here about the [Oger toolbox](http://www.nilsschaetti.ch/2018/01/30/introduction-reservoir-computing-2-oger-toolbox/).

### Extended Oger versions
You can find another extended versions of the toolbox below. I don't know if these extended versions are compatible with codes developped for the original Oger.
- [nschaetti's repository](https://github.com/nschaetti/Oger)
- [tilemmpon's repository](https://github.com/tilemmpon/Extended-OGER)
