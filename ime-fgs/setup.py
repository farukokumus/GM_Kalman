# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from setuptools import setup, find_packages

setup(name='ime-fgs',
      version='0.1',
      description='Python implementation of a generic (Forney-style) factor graph toolbox',
      url=' https://git.ime.uni-luebeck.de/factor-graphs/ime-fgs',
      packages=find_packages(),
      install_requires=['numpy', 'scipy'],
      extras_require={
          'RiemannSums': ["sobol_seq"],
          'Demos': ["matplotlib", "seaborn", "pandas"],
      },
      zip_safe=False)
