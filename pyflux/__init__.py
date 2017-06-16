__version__ = "0.4.15"

# Use of SETUP built-in adapted from scikit-learn's setup utility.
try:
    __PYFLUX_SETUP__
except NameError:
    __PYFLUX_SETUP__ = False

if __PYFLUX_SETUP__:
    sys.stderr.write('Partial import of PyFlux during the build process.\n')
else:
    from . import __check_build
	
from .arma import *
from .var import *
from .ensembles import *
from .families import *
from .gas import *
from .garch import *
from .gpnarx import *
from .inference import *
from .ssm import *
from .covariances import *
from .output import *
from .tests import *
