# FFSAS  v1.1

**F**ree-**F**orm inversion for **S**mall-**A**ngle **S**cattering (SAS)

#

`ffsas` is a Python library to invert for free-form distributions of model 
parameters in a polydisperse SAS system. It yields the maximum likelihood 
estimator of the parameter distributions and the sensitivity and uncertainty 
of the maximum likelihood estimator. 
`ffsas` comes with the following features:


* **Generality**: `ffsas` formulates the SAS inverse problem as a 
Green's-tensor-based multi-linear map, which covers *any* complex SAS models with 
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
Computation of the Jacobian and Hessian of the χ2 error, as the most expensive 
step during solution, is accelerated by GPU and mini-batch computation. 
The idea is borrowed from deep learning and implemented with PyTorch. 

* **Accuracy**: the model parameters and the resulting intensity in a SAS problem 
can span many orders of magnitude, and a good choice of unit system is essential 
to avoid an ill-conditioned inverse problem. `ffsas` *automatically* analyzes 
the orders of magnitude of data and parameters so as to determine a proper internal 
unit system to avoid accuracy loss. Such an internal unit system is hidden 
from users, who can use an arbitrary unit system for input and output.

* **Usability**: `ffsas` can be installed with `pip` in one line. Its usage only 
includes four APIs respectively for 
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

Follow the [User Guide](https://github.com/stfc-sciml/ffsas/blob/main/doc/USER-GUIDE.md) to learn the usage.


## Citation
* Full paper (open access): [http://doi.org/10.1107/S1600576722006379](http://doi.org/10.1107/S1600576722006379)

* BIBTex:
    ```bib
    @article{Leng:jl5041,
    author = "Leng, Kuangdai and King, Stephen and Snow, Tim and Rogers, Sarah and Markvardsen, Anders and Maheswaran, Satheesh and Thiyagalingam, Jeyan",
    title = "{Parameter inversion of a polydisperse system in small-angle scattering}",
    journal = "Journal of Applied Crystallography",
    year = "2022",
    volume = "55",
    number = "4",
    pages = "966--977",
    month = "Aug",
    doi = {10.1107/S1600576722006379},
    url = {https://doi.org/10.1107/S1600576722006379},
    }
    ```

## Funding and Support 
This work was supported by the ISIS Neutron and Muon Source (ISIS) of the Science and Technology Facilities Council through the ISIS-ML funding, and by Wave I of the UKRI Strategic Priorities Fund under the EPSRC grant (EP/T001569/1), particularly the AI for Science theme in that grant and the Alan Turing Institute. We gratefully acknowledge their support.
