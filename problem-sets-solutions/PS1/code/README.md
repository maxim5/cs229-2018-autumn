# CS229 Fall 2018 Problem Set #1


## Setup for Coding Parts

1. Install [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. cd into src, run `conda env create -f environment.yml`
  - This creates a Conda environment called `cs229`
3. Run `source activate cs229`
  - This activates the `cs229` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cs229/bin/python`
  - Select `OK` then `Apply`
5. Browse the code in `linear_model.py`
  - The `LinearModel` class roughly follows the sklearn classifier interface: You must implement a `fit` and a `predict` method for every `LinearModel` subclass you write.
6. Browse the `util.py` file. Notice you have access to methods that do the following tasks:
  - Load a dataset in the CSV format provided in PS1
  - Add an intercept to a dataset (*i.e.,* add a new column of 1s to the design matrix)
  - Plot a dataset and a linear decision boundary. Some plots in PS1 will require modified plotting code, but you can use this as a starting point.
7. Notice the `run.py` file. You should **not** modify the commands in this file, since the autograder expects your code to run with the flags given in `run.py`. Use this script to make sure your code runs without errors.
  - You can run `python run.py` to run all problems, or add a problem number (*e.g.,* `python run.py 1` to run problem 1).
  - When you submit to Gradescope, the autograder will immediately check that your code runs and produces output files of the correct name, output format, and shape.
