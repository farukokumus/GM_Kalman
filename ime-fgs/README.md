ime-fgs
=======
Python implementation of a generic (Forney-style) factor graph toolbox.
Current features include:
* Multivariate, linear Gaussian Message Passing
* Sigma-point methods for treatment of nonlinear transformations
* Expectation Maximization (EM) for parameter estimation
* Kalman filtering and smoothing


Development Workflow
--------------------
All of the following steps are **mandatory** to contribute to the development of this toolbox:
1. Set up a new branch to work on a new feature / refactor the toolbox / fix bugs.
2. Open a Pull Request with `WIP: Your feature description` as title. Now everyone can easily see what you are working on.
3. Work on your changes, until happy with the result.
4. Test your changes. Both integration tests (i.e., examples) and unit tests are desired.
5. Merge the current master branch into your branch to make sure your changes still make sense with the latest version of the toolbox.
6. Convince the admins that your contribution is useful and valid. :)

**Important:** If you're working on a branch for longer than 1-2 weeks, make sure to merge current master changes into your branch regularly! 
There's nothing worse than trying to merge two branches after months of diverging changes in both branches.

Pipeline and jobs (automatic testing)
-------------------------------------
The current pipeline has three jobs at the Test stage:
 - `testing_and_coverage`
 - `code_style`
 - `run_demos`

All jobs of the pipline are run automatically after you push any changes to the server. Especially in case of a failed job, it might be useful to inspect the the output by clicking on the test status icon.

The `testing_and_coverage` job executes all tests in the [ime_fgs/tests/](ime_fgs/tests/) folder and creates a [code coverage](https://en.wikipedia.org/wiki/Code_coverage) report. A fail of this job indicates a bug (either in the toolbox code or in the test code).
The code coverage report includes all source files, except anything inside the [ime_fgs/demos/](ime_fgs/demos/) or [ime_fgs/tests/](ime_fgs/tests/) folder. It is desirable to increase the code coverage. A detailed Code coverage report in the html format can be downloaded as job artifact.

The `code_style` job checks the code style of all python source files in the repository. The check uses [pycodestyle](https://github.com/PyCQA/pycodestyle) and basically checks against the [PEP8](https://www.python.org/dev/peps/pep-0008) rule set, with some exception like the 120 characters per line limit. For details please take a look at the config file. To maintain nice and readable code, this job shouldn't fail. If you have good reasons to violate some of the rules please make appropiate changes to the config file.

The `run_demos` job executes all pythons files in the [ime_fgs/demos/](ime_fgs/demos/) directory to ensure all of them run without any exceptions.

Performance hints
-----------------
There is a big performance boost if you add `-O` (and also `-OO`) to the python interpreter parameters. This removes all assert statements (!) and will especially speed up the message creation. 

Installation hints 
------------------
(relevant only if you're using the toolbox for other projects)

Install a project in *editable mode* for package development (creates `src` directory in current directory)

`pip install --user -e git+https://git.ime.uni-luebeck.de/factor-graphs/ime-fgs.git#egg=ime-fgs`
    
Install package for current user

`pip install --user git+https://git.ime.uni-luebeck.de/factor-graphs/ime-fgs.git`

