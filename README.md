<img src="https://render.githubusercontent.com/render/math?math={\color{BurntOrange}\Huge\textbf{F}}\!{\color{RedOrange}\Huge\textbf{F}}{\color{BrickRed}\Huge\mathbf{S\!A\!S}}">   v1.1

**F**ree-**F**orm inversion for **S**mall-**A**ngle **S**cattering (SAS)

#

`ffsas` is a Python library to invert for free-form distributions of model 
parameters from polydisperse SAS experiment data. It yields the maximum likelihood 
estimation of parameter distributions and the normalized sensitivity of 
each parameter at the maximum likelihood estimator. 
`ffsas` comes with the following features:


* **Generality**: `ffsas` formulates the SAS inverse problem as a 
Green's-tensor-based system, which covers *any* complex SAS models with 
multiple parameters. An arbitrary model can be easily implemented by 
supplying the Green's function for forward modelling, i.e., a function 
that computes the monodisperse intensity on a structured grid of model parameters 
(i.e., the Green's tensor). In short, users only need to care about the 
physics for forward modelling, leaving the inverse problem all to `ffsas`.
Besides, it does not require an initial guess of the parameter
distributions, the scaling factor or the source background. 

* **Efficiency**: through theoretical analysis, the inverse problem is 
simplified as a highly solvable nonlinear programming (NLP) problem 
with a few equality constraints. It is solved by a trust-region method, 
implemented in SciPy as `scipy.optimize.minimize(method='trust-constr')`, 
officially mentioned as "the most versatile constrained minimization algorithm 
implemented in SciPy and the most appropriate for large-scale problems". 
The computations of the Jacobian and Hessian are the most expensive steps 
during solution, for which `ffsas` uses GPU acceleration and file-based
mini-batch computation to significantly reduce runtime and memory requirement. 
Such ideas are borrowed from deep learning and implemented with PyTorch. 

* **Accuracy**: the model parameters and the resulting intensity in a SAS problem 
can span many orders of magnitude, and a good choice of unit system is essential 
to avoid an ill-conditioned inverse problem. `ffsas` *automatically* analyzes 
the scales of data and parameters and determines a proper internal unit system to 
avoid accuracy loss from input to output. Such an internal unit system is hidden 
from users, who can use any external unit system for input and output.

* **Usability**: `ffsas` can be installed with `pip` in one line. Its usage only 
includes four API functions, respectively for 
    - computing the Green's tensor **G** given a model and a parameter space
    - define a **G**-system with this Green's tensor
    - using this **G**-system to compute intensity given parameter 
    distributions (forward modelling)
    - using this **G**-system to invert for parameter distributions given 
    intensity data (inversion)
    
    A built-in logging system produces detailed and readable logs during these processes. 



## Quick Start

To install `ffsas`:

```bash
pip install ffsas
```

Follow the [User Guide](doc/USER-GUIDE.md) to learn the usage.


## Credits
Kuangdai Leng<sup>1</sup>, Steve King<sup>2</sup>, Tim Snow<sup>3</sup>, Sarah Rogers<sup>2</sup>, 
Jeyan Thiyagalingam<sup>1</sup>

<sup>1</sup> Scientific Computing Department, STFC, UK

<sup>2</sup> ISIS Neutron and Muon Source, STFC, UK

<sup>3</sup> Diamond Light Source, STFC, UK

## Funding and Support 
This work was supported by the ISIS Neutron and Muon Source (ISIS) of the Science and Technology Facilities Council through the ISIS-ML funding, and by Wave I of the UKRI Strategic Priorities Fund under the EPSRC grant (EP/T001569/1), particularly the AI for Science theme in that grant and the Alan Turing Institute. We gratefully acknowledge their support.
