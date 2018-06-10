# calibrate-line-camera
A python toolbox for calibrating line scanning cameras (1xN pixel images) on a moving platform with a navigation source (like GPS/INS), based on the paper citation given below. Please consider citing our paper if you use the code in your work.

Please see the example notebook "linescan_calibration_example.ipynb" for usage details. The package requires the following python packages to be installed:

- numpy
- sympy
- dill
- lmfit
- emcee
- corner (for plot in example notebook)

The file "wrapper_module_0.so" must be in the working directory (from where you run your python script). It contains the point jacobian function compiled using sympy/Fortran for speed. The jacobian itself was calculated using Matlab, because Python's sympy package was too inefficient for this task. Currently, the package throws an exception if wrapper_module_0.so is not found. It also only works with Python 2. If you wish to try generating the jacobian and compiling from scratch, set the allow_point_jac_compilation flag to True. This will take a very long time or may not work at all, so it is not recommended at this stage.

```
@Article{s17112491,
AUTHOR = {Wendel, Alexander and Underwood, James},
TITLE = {Extrinsic Parameter Calibration for Line Scanning Cameras on Ground Vehicles with Navigation Systems Using a Calibration Pattern},
JOURNAL = {Sensors},
VOLUME = {17},
YEAR = {2017},
NUMBER = {11},
ARTICLE NUMBER = {2491},
URL = {http://www.mdpi.com/1424-8220/17/11/2491},
ISSN = {1424-8220},
DOI = {10.3390/s17112491}
}
```
