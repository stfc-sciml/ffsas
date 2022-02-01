<details>
<summary><b>Table of contents</b></summary>

* [1 Theory](#1-Theory)
    * [1.1 Forward modelling](#11-Forward-modelling)
    * [1.2 Inversion](#12-Inversion)
    * [1.3 Tasks of `ffsas`](#13-Tasks-of-ffsas)
* [2 Python APIs](#2-Python-APIs)
    * [2.1 The `SASModel` class](#21-The-SASModel-class)
      * [2.1.1 Using a built-in model](#211-Using-a-built-in-model)
      * [2.1.2 Implementing a user-defined model](#212-Implementing-a-user-defined-model)
    * [2.2 The `SASGreensSystem` class](#22-The-SASGreensSystem-class)
      * [2.2.1 Forward modelling](#221-Forward-modelling)
      * [2.2.2 Inversion](#222-Inversion)
* [3 FAQ](#3-FAQ)

</details>


# FFSAS User Guide


This guide starts with a brief theoretical section to help users to understand
the exact problems `ffsas` solves. It then elaborates on the few Python APIs of `ffsas`
for forward modelling and inversion. We recommend the following steps for new
users to learn `ffsas`:

1. Read [1 Theory](#1-Theory) and understand the problems;
2. Learn `ffsas` by following the Jupyter Notebooks in folder [examples](../examples);
3. Use section [2 Python APIs](#2-Python-APIs) and [3 FAQ](#3-FAQ) as a 
reference for applications and extended developments.

Please report questions, suggestions and bugs by opening an issue in 
our [repository](https://github.com/stfc-sciml/ffsas). Enjoy!

---

# 1 Theory

Without loss of generality, we describe the theory of a general polydisperse, 
multi-parameter SAS problem using the "cylinder" model for problem setup.
This is only to avoid some highly abstract tensorial equations required to
describe an arbitrary number of model parameters. None of the statements
or equations in this section are restricted to the physics of the cylinder model.

The cylinder model has four parameters:

* $l$ - length of cylinder;
* $r$ - radius of cylinder;
* $\theta$ - angle between cylinder axis and beam;
* $\phi$ - rotation of cylinder axis about beam.

The intensity observation, denoted $I$,
is a 2D image as a function of the scattering vectors
$(q_x, q_y)$, or $I=I(q_x,q_y)$. 
See [documentation of SASView/SASModels](https://www.sasview.org/docs/user/models/ellipsoid.html)
for more details.

## 1.1 Forward modelling


Assuming monodispersity, the forward modelling process can be formulated as

$$
I(q_x, q_y, l, r, \theta, \phi)=\xi G(q_x, q_y, l, r, \theta, \phi)+b,
\quad\quad\quad\quad(1)
$$


where $\xi$ is some scaling factor,
$b$ the source background
and $G$, which we refer to as **the Green's function**, has the physical meaning of 
the square of the scattering amplitude, or $F^2$ in some literatures, e.g., 
in [documentation of SASView/SASModels](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html).


Now consider polydispersity. Let the parameter distributions be denoted by 
$w^l(l)$,
$w^r(r)$,
$w^\theta(\theta)$ and
$w^\phi(\phi)$,
all summing into 1 (e.g., $\int w^l(l)\,\mathrm{d}l=1$).
The resulting intensity will be the integral of the Green's function over the four distributions:


$$
I(q_x, q_y, l, r, \theta, \phi)\\[.5em]=\xi \iiiint G(q_x, q_y, l, r, \theta, \phi) w^l(l) w^r(r) w^\theta(\theta)w^\phi(\phi) \;\mathrm{d}l\,\mathrm{d}r\,\mathrm{d}\theta\,\mathrm{d}\phi + b.
\quad\quad\quad\quad(2)
$$



Let $q_x$, $q_y$, $l$, $r$, $\theta$ and $\phi$ be discretized respectively by the 1D vectors: 
$\mathbf{q}_x\in \mathbb{R}^{N_{q_x}}$,
$\mathbf{q}_y\in \mathbb{R}^{N_{q_y}}$,
$\mathbf{l}\in \mathbb{R}^{N_{l}}$,
$\mathbf{r}\in \mathbb{R}^{N_{r}}$,
$\boldsymbol{\theta}\in \mathbb{R}^{N_{\theta}}$ and
$\boldsymbol{\phi}\in \mathbb{R}^{N_{\phi}}$. 
Equation (2) can thereby be discretized as


$$
I_{ij}=\xi \sum_{m,n,s,t}  G_{ijmnst} w^l_m w^r_n w^\theta_s w^\phi_t  +b D_{ij},
\quad\quad\quad\quad(3)
$$


where $G_{ijmnst}$, which we refer to as **the Green's tensor**, is the Green's function computed at the
$i$-th $q_x$ in $\mathbf{q}_x$,
$j$-th $q_y$ in $\mathbf{q}_y$,
$m$-th $l$ in $\mathbf{l}$,
$n$-th $r$ in $\mathbf{r}$,
$s$-th $\theta$ in $\boldsymbol{\theta}$ and
$t$-th $\phi$ in $\boldsymbol{\phi}$, and
$w^l_m$ the weight of the $m$-th $l$ in $\mathbf{l}$, summing into one, 
$\sum_m w^l_m=1$ (the same for $w^r_n$, $w^\theta_s$ and $w^\phi_t$), and 
$D_{ij}$ a second-order tensor (matrix) filled with one. 
Clearly, $G_{ijmnst}$ is a tensor of order six, containing 
$N_{q_x}\times N_{q_y}\times N_{l}\times N_{r}\times N_{\theta}\times N_{\phi}$ elements in total.

## 1.2 Inversion

From a SAS experiment, one observes the mean and standard deviation of the intensity,
as denoted respectively by $\mu_{ij}$ and $\sigma_{ij}$. Given the model type and the 
parameter space (which determine $\mathbf{G}$), 
the normalized intensity misfit, denoted $\epsilon_{ij}$, can be written as

$$
\epsilon_{ij}=\dfrac{I_{ij}-\mu_{ij}}{\sigma_{ij}},
\quad\quad\quad\quad(4)
$$

whose $l^2$-norm yields the $\chi^2$ error:

$$
\chi^2=\dfrac{1}{2}\sum_{i,j} \epsilon_{ij}^2.
\quad\quad\quad\quad(5)
$$


Our inverse problem is to find the parameter distributions 
($w^l_m$, $w^r_n$, $w^\theta_s$ and $w^\phi_t$) and the two scalars ($\xi$ and $b$) 
that minimize $\chi^2$, namely,


$$
\begin{aligned}
& \underset{w^l_m, w^r_n, w^\theta_s, w^\phi_t, \xi, b}{\mathrm{minimize}}\chi^2 \\[.5em]
\mathrm{subject\ to} \quad& \sum_m w^l_m=1,\ \sum_n w^r_n=1,\ \sum_s w^\theta_s=1,\ \sum_t w^\phi_t=1;\\[.5em]
& w^l_m \geq 0,\ w^r_n \geq 0,\ w^\theta_s\geq0,\ w^\phi_t\geq0, \quad\forall m,n,s,t.
\end{aligned}
\quad\quad\quad\quad(6)
$$

Here $\chi^2$ is computed through eqs. (5), (4) and (3), with $G_{ijmnst}$, $\mu_{ij}$ and $\sigma_{ij}$ as knowns. 
The minimizer of this constrained nonlinear programming (NLP) problem, as denoted 
(${\hat{w}^l_m}$, ${\hat{w}^r_n}$, ${\hat{w}^\theta_s}$, ${\hat{w}^\phi_t}$, $\hat\xi$, $\hat b$), 
is also known as the **maximum likelihood estimator**. Note that this NLP is difficult to solve because, 
first, it involves mixed equality and inequality constraints and, second, the number of inequality 
constraints is as large as the number of weights; within `ffsas`, we solve another equivalent 
problem with much lower complexity — the math is hidden from users and thus skipped in this guide.

Once the maximum likelihood estimator is obtained, normalized sensitivity analysis can be performed by 

$$
S_{I}=\left.\sum_{J} H_{I J}\dfrac{X_J}{J_{J}}\right|_{\ \mathbf{X}=\mathbf{\hat X}},
\quad\quad\quad\quad(7)
$$


where the vector $\mathbf{X}$ contains all the variables 
(${{w}^l_m}$, ${{w}^r_n}$, ${{w}^\theta_s}$, ${{w}^\phi_t}$, $\xi$, $b$) in a 
flattened manner, and $\mathbf{J}$ and $\mathbf{H}$ respectively denotes the 
Jacobian and Hessian of $\chi^2$ with respect to $\mathbf{X}$, all evaluated at 
the maximum likelihood estimator $\mathbf{\hat X}$. The normalized sensitivity 
$\mathbf{S}$ indicates how sensitively $\chi^2$ responds to a small variation of 
$\mathbf{X}$ in the neighborhood of $\mathbf{\hat X}$. 
The inverse of $\mathbf{S}$ can thus be understood as the relative uncertainty of 
$\mathbf{\hat X}$. As a non-sampling-based method, `ffsas` cannot deliver 
absolute uncertainty (such as variance) of any parameter estimations.

## 1.3 Tasks of `ffsas`

Having established the general theory as above, we can summarize the tasks of `ffsas` as 

1. **the Green's tensor**: compute $\mathbf{G}$ given a model and a parameter space;
2. **forward modelling**: with $\mathbf{G}$ computed, compute the intensity 
$\mathbf{I}$ given ($\mathbf{w}$'s, $\xi$, $b$) based on eq. (3);
3. **inverse problem**: with $\mathbf{G}$ computed, solve ($\mathbf{\hat w}$'s, 
$\hat\xi$, $\hat b$) given the intensity observations **μ** and **σ** based on eq. (6); 
performing sensitivity analysis at ($\mathbf{\hat w}$'s, $\hat\xi$, $\hat b$) based on eq. (7).

---

# 2 Python APIs

`ffsas` features a lightweight framework. The user APIs are provided via two Python classes, 
`SASModel` and `SASGreensSystem`. The `SASModel` class computes the Green's tensor 
$\mathbf{G}$ whereas `SASGreensSystem` implements a $\mathbf{G}$-based system for both
forward modelling and inversion. 


## 2.1 The `SASModel` class


`SASModel` is the base class for all SAS models. 
The only purpose of this class is to compute the Green's tensor. 
Any concrete (usable) model, such as "cylinder", must be implemented as a 
derived class of `SASModel`. Currently only three built-in models are provided, 
`Sphere`, `Ellipsoid` and `Cylinder`, and the `Cylinder` model is also implemented in 
[examples/cylinder_user_defined.ipynb](../examples/cylinder_user_defined.ipynb) as an 
example of user-defined models. We do not intend to implement many models, but motivated 
to provide easy interfaces for users to implement their own models.

In this section, we first show how to compute the Green's tensor with a built-in model and 
then how to implement a user-defined model.

### 2.1.1 Using a built-in model

Users only call one method to compute the Green's tensor $\mathbf{G}$: 
`SASModel.compute_G_mini_batch()`. Below is a complete example
using the build-in model `Ellipsoid`, extracted from 
[examples/ellipsoid.ipynb](../examples/ellipsoid.ipynb):

```python
# import torch and model
import torch
from ffsas.models import Ellipsoid

# list of q-vectors
qx = torch.linspace(-.25, .25, 32)
qy = torch.linspace(-.25, .25, 30)
q_list = [qx, qy]

# dict of parameters
par_dict = {
    'rp': torch.linspace(200., 600., 18),
    're': torch.linspace(50., 90., 17),
    'theta': torch.deg2rad(torch.linspace(5., 60., 16)),
    'phi': torch.deg2rad(torch.linspace(150., 240., 15))
}

# dict of constants
const_dict = {'drho': 1.}

# compute G
G = Ellipsoid.compute_G_mini_batch(q_list, par_dict, const_dict,
                                   fixed_par_weights=None,
                                   G_file=None, batch_size=None, device='cpu',
                                   log_file=None, log_screen=True)
```


`SASModel.compute_G_mini_batch()` takes six arguments, as explained below:

* `q_list`: `list` of $q$-vectors. If the intensity observation is a series, 
such as for the `Sphere` model, `q_list` will contain one $q$-vector as a 1D `torch.Tensor`; 
if the intensity observation is an image, such as for the `Ellipsoid` model, `q_list` 
will contain two $q$-vectors, i.e., `q_list=[qx, qy]`, with `qx` and `qy` being both 
1D `torch.Tensor`'s. In the above example, `qx` and `qy` both range from -0.25 to 0.25, 
with respectively 32 and 30 elements (so the intensity image has a size of $32\times 30$).
    
    > 📗 NOTE: **The $q$-vectors can be arbitrarily sampled**; e.g., one may sample $q$ in log-scale or use a higher resolution across a feature-rich region in the series or image. The $q$-vectors are even not required to be sorted in an ascending or descending order. Besides, `ffsas` supports any high $q$-dimensions (i.e., `len[q_list] >= 2`), but high-dimensional intensity data currently seem unavailable from SAS experiments.
    
    
* `par_dict`: a `dict` of model parameters. 
Each parameter must be a 1D `torch.Tensor` with the key indicating the name of the parameter. 
The required keys can be found by calling

    ```python
    par_keys = Ellipsoid.get_par_keys_G()
    ```

    For example, `Sphere.get_par_keys_G()` will return `['r']`, and `Ellipsoid.get_par_keys_G()` will return 
    `['rp', 're', 'theta', 'phi']`. **The order of the keys in this list also determines the 
    dimensions of the parameters in the Green's tensor**. In the above example, 
    the shape of `G` will be `[32, 30, 18, 17, 16, 15]` or `[len(qx), len(qy), len(rp), len(re), len(theta), len(phi)]`.
    
    > 📗 NOTE: Similar to a $q$-vector, **the parameters can be arbitrarily sampled**. To fix a parameter (i.e., assuming monodispersity in this dimension), simply use a `torch.Tensor` with one element; e.g., using `par_dict['rp'] = torch.tensor([100.])` will fix the polar radius at 100.

* `const_dict`: a `dict` of constants. 
Each parameter can be a `torch.Tensor` of any order, a scalar or any other types such as 
boolean or string flags. Different from the parameters in `par_dict`, 
the constants in `const_dict` are not considered as variables in the inverse problem.

* `fixed_par_weights`: a `dict` of fixed parameters and their weights;
fixing a parameter means that this parameter will not be considered as a variable
in inversion, with its weights supplied by the user via `fixed_par_weights`.
The weights must be a 1D `torch.Tensor` with the key indicating the 
name of the fixed parameter.

* `G_file`, `batch_size` and `device`: these three arguments set up the hardware and 
algorithmic optimizations for computing the Green's tensor $\mathbf{G}$. 
For a multi-parameter problem, $\mathbf{G}$ can be large in size and the successive 
inner product in eq. (3) can be computationally expensive. 
Taking `Ellipsoid` for example, if the vectors of 
$q_x$, $q_y$, $r_p$, $r_e$, $\theta$ and $\phi$ all have a length of 50, 
the number of elements in $\mathbf{G}$ will grow to $50^6$, 
requiring 125 GB for storage. For a large-scale problem like this, `ffsas` allows 
users to adopt **file-based mini-batch computation with GPU acceleration** to overcome 
memory insufficiency and reduce computing time — common practice in deep learning. 
The computing infrastructure is shown below:

    ![workflow](https://i.ibb.co/7bHSnGK/workflow.png)

    - `G_file`: the storage for $\mathbf{G}$; if `G_file=None`, $\mathbf{G}$ will be 
    computed and stored in memory, returned as a `torch.Tensor`; otherwise, 
    $\mathbf{G}$ will be written to a HDF5 file named `G_file`, 
    with `G_file` returned as a placeholder. 
    - `batch_size`: the batch size along each $q$-dimension; e.g., 
    when `batch_size=4`, one mini-batch will have a shape of `[4, 4, 18, 17, 16, 15]` 
    for the above `Ellipsoid` example; use `batch_size=None` to compute entire $\mathbf{G}$ 
    in one batch.
    - `device`: the computing device, which can be `'cpu'` or `'cuda'` 
    (or `'cuda:0'`, `'cuda:1'`, ...).

* `log_file` and `log_screen`: `ffsas` has a powerful logging system. 
The argument `log_file` specifies a file to save the logs 
and `log_screen` controls whether to print the logs to `sys.stdout`.


### 2.1.2 Implementing a user-defined model


Easy interfaces are provided for incorporating a user-defined model, 
regardless of its physical complexity. Users only implement the physics, 
or the Green's function, whereas the hardware and algorithmic optimizations are 
handled internally by `ffsas`.

Take the "cylinder" model for example. Below is the complete code to 
implement a `Cylinder` class, 
copied from [examples/cylinder_user_defined.ipynb](../examples/cylinder_user_defined.ipynb):

```python
# import base class
from ffsas.models import SASModel

# import torch
import torch

# import math
import math
from scipy.special import j1

# Cylinder class
class Cylinder(SASModel):
    @classmethod
    def compute_G(cls, q_list, par_dict, const_dict, V=None):
        # get parameters
        qx, qy = q_list[0], q_list[1]
        l, r = par_dict['l'], par_dict['r']
        theta, phi = par_dict['theta'], par_dict['phi']
        drho = const_dict['drho']

        # compute volume
        if V is None:
            V = cls.compute_V(par_dict)

        #############
        # Compute G #
        #############

        # step 1: rotate q
        sin_theta = torch.sin(theta)
        r31 = torch.outer(sin_theta, torch.cos(phi))
        r32 = torch.outer(sin_theta, torch.sin(phi))
        qc = (qx[:, None, None] * r31[None, :, :])[:, None, :, :] + \
             (qy[:, None, None] * r32[None, :, :])[None, :, :, :]
        qa = torch.sqrt(torch.clip(
            (qx ** 2)[:, None, None, None] +
            (qy ** 2)[None, :, None, None] - qc ** 2, min=0.))

        # step 2: qa * r, qc * l
        qa_r = torch.moveaxis(qa[:, :, :, :, None] *
                              r[None, None, None, None, :], 4, 2)
        qc_l = torch.moveaxis(qc[:, :, :, :, None] *
                              l[None, None, None, None, :], 4, 2) * .5

        # step 3: shape factor
        sin_qc_l = torch.nan_to_num(2. * torch.sin(qc_l) / qc_l,
                                    nan=1., posinf=1., neginf=1.)
        # NOTE: scipy.special.j1() must be called on cpu
        j1_qa_r = torch.tensor(j1(qa_r.to('cpu').numpy()), device=qa_r.device)
        j1_qa_r = torch.nan_to_num(j1_qa_r / qa_r, nan=1., posinf=1., neginf=1.)
        # shape factor
        shape_factor = sin_qc_l[:, :, :, None, :, :] * j1_qa_r[:, :, None, :, :, :]

        # step 4: G
        G = (drho * shape_factor * V[None, None, :, :, None, None]) ** 2
        return G

    @classmethod
    def get_par_keys_G(cls):
        return ['l', 'r', 'theta', 'phi']

    @classmethod
    def compute_V(cls, par_dict):
        l, r = par_dict['l'], par_dict['r']
        return math.pi * l[:, None] * r[None, :] ** 2

    @classmethod
    def get_par_keys_V(cls):
        return ['l', 'r']
```

Note that there is no method called `compute_G_mini_batch()` in the above code. 
This is the key for understanding. The method `compute_G()` is an abstract method 
of the base class `SASModel`, which must be implemented for each derived class 
(by users) to embody the physics, such as above. In contrast, `compute_G_mini_batch()` 
is a method of the base class, which calls `compute_G()` of a derived class 
*for each mini-batch on the device*, taking care of all computational optimizations 
that are hidden from users. In short, users implement `compute_G()` but use 
`compute_G_mini_batch()`.


The method `get_par_keys_G()` returns the names of the parameters used for computing 
$\mathbf{G}$, ordered as they appear in the dimensions of $\mathbf{G}$. The rest two 
methods, `compute_V()` and `get_par_keys_V()`, are for volume computation, usually trivial.
The volume tensor can be useful for pre- and post-processing, such as transforming 
the scaling factor, see [3 FAQ](#3-FAQ).


> 📗 NOTE: To ensure high computational efficiency, **it is crucial that all tensorial operations be coded "collectively" using PyTorch functions or operators** (most of them are similar to their NumPy counterparts) — simply put, **avoid `for` loops!** For example, to compute $V=\pi l r^2$ with $l$ and $r$ being vectors and the resulting $V$ a second-order tensor or matrix:

```python
# this is 👍
V = math.pi * l[:, None] * r[None, :] ** 2

# this is also 👍
V = math.pi * torch.outer(l, r ** 2)

# this is 👎︎
V = torch.zeros((l.size(0), r.size(0)), device=l.device)
for i, l_val in enumerate(l):
    V[i, :] = math.pi * l_val * r ** 2
    
# this is 👎︎👎︎👎︎
V = torch.zeros((l.size(0), r.size(0)), device=l.device)
for i, l_val in enumerate(l):
    for j, r_val in enumerate(r):
        V[i, j] = math.pi * l_val * r_val ** 2
```
    
    

## 2.2 The `SASGreensSystem` class


Once the Green's tensor $\mathbf{G}$ has been computed, we can create a 
$\mathbf{G}$-based system as an object of the `SASGreensSystem` class:

```python
# create a G-based system for SAS
g_sys = SASGreensSystem(G, par_keys, batch_size=None, device='cpu', 
                        log_file=None, log_screen=True)
```

The arguments are explained below:

* `G`: the Green's tensor computed by `SASModel.compute_G_mini_batch()`, 
either a `torch.Tensor` or a filename, depending on the `G_file` argument 
passed to `SASModel.compute_G_mini_batch()`.
* `par_keys`: the `list` of parameter keys returned by 
`SASModel.get_par_keys_G()`; this argument informs the system about the 
parameter names so that it can report the inverse results in a more user-friendly style;
fixed parameters passed by `fixed_par_weights` to 
`SASModel.compute_G_mini_batch()` must be excluded from this list. 
* `batch_size` and `device`: they have the same meanings as they were in 
`SASModel.compute_G_mini_batch()` but not necessarily have the same values as used for 
`SASModel.compute_G_mini_batch()`; see the previous figure of computational infrastructure. 
* `log_file` and `log_screen`: `ffsas` has a powerful logging system. 
The argument `log_file` specifies a file to save the logs 
and `log_screen` controls whether to print the logs to `sys.stdout`.

> 📗 NOTE: It is important to understand that, at this stage of creating a `SASGreensSystem` object, we are no longer concerned with any physics about SAS — all physics has been turned into the numbers in `G`.  



### 2.2.1 Forward modelling

With the $\mathbf{G}$-based system established, forward modelling based on eq. (3) 
becomes straightforward using the method `SASGreensSystem.compute_intensity()`:

```python
# compute intensity given the weights, ξ and b
intensity = g_sys.compute_intensity(w_dict, xi, b)
```

where `w_dict` is a `dict` containing the weights (distributions) of the model parameters, 
each a 1D `torch.Tensor` keyed by its parameter name 
(recall that we have sent `par_keys` to create a `SASGreensSystem` object, 
so it knows the dimension of each parameter in `G`). 

> 📗 NOTE: `compute_intensity()` does NOT normalize the weights to make them sum into 1.



### 2.2.2 Inversion

The inverse problem in eq. (6) is solved by the method 
`SASGreensSystem.solve_inverse()`, given the mean (μ) and standard deviation (σ) 
of an intensity observation:

```python
# solve inverse problem given mu and sigma
inverse_result = g_sys.solve_inverse(mu, sigma, nu_mu=.0, nu_sigma=1.,
                                     w_dict_init=None, xi_init=None, b_init=None,
                                     auto_scaling=True, maxiter=1000, verbose=1,
                                     trust_options=None, save_iter=None)
```

The arguments are explained below:
* `mu` and `sigma`: the mean and standard deviation of the intensity observation. 
In the case where `sigma` is unavailable, such as for a synthetic test, we suggest 
users to use `mu` to take its place. This is because the intensity in SAS usually 
spans many orders of magnitude, and without the normalization by `sigma` in eq. (4), 
the inversion can be too dominated by the brightest parts in the intensity image.
* `nu_mu` and `nu_sigma`: another two scalars introduced to the inverse problem: 
$\nu_\mu$ and $\nu_\sigma$. In `ffsas`, an extended version of eq. (4) is used:

    $$
    \epsilon_{ij}=\dfrac{I_{ij}-\mu_{ij}}{\mu_{ij}^{\nu_\mu} \sigma_{ij}^{\nu_\sigma}}.
    \quad\quad\quad\quad(8)
    $$

    In other words, $\nu_\mu$ and $\nu_\sigma$ change the normalization term for 
    misfit calculation. By default, $\nu_\mu=0$ and $\nu_\sigma=1$, so eq. (8) is 
    the same as eq. (4), suitable for most applications. We have found so far two 
    situations where $\nu_\mu$ and $\nu_\sigma$ can be useful: 
    1. when an ultra-low resolution is used for a parameter distribution, tuning 
    $\nu_\mu$ and $\nu_\sigma$ can avoid some drifting artifacts (rarely seen); 
    2. when the pattern of $\sigma_{ij}$ appears too irregular (e.g., the uncertainty 
    seems too large at some $q$ values), one may use $\mu_{ij}$ to regularize the 
    normalization term so as to obtain a more reasonable inverse result, e.g., 
    $\nu_\mu=\nu_\sigma=0.5$.
    
> 📗 NOTE: we do not require $\nu_\mu+\nu_\sigma=1$, but theoretically this is required to make $\epsilon_{ij}$ and $\chi^2$ dimensionless.

* `w_dict_init`, `xi_init` and `b_init`: the initial guesses of the variables. Users normally
do not need to provide them.

* `auto_scaling`: users should always use `auto_scaling=True` for accuracy preservation. 
How it works is explained in the question about "unit system" in [3 FAQ](#3-FAQ).

* `maxiter`, `verbose`, `trust_options`: 
arguments sent to [scipy.optimize.minimize(method='trust-constr')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html#optimize-minimize-trustconstr).

* `save_iter`: save the inverse results every `save_iter` iterations so that one can examine the convergence 
history of the parameter distributions; see [examples/Ellipsoid.ipynb](../examples/Ellipsoid.ipynb) for example.
Examining the convergence history is a good practice for all optimization problems. 

`SASGreensSystem.solve_inverse()` returns a `dict` containing the inverse results, including:

* `inverse_result['w_dict']`: MLE of $\mathbf{w}$'s;
* `inverse_result['xi']`: MLE of $\xi$;
* `inverse_result['b']`: MLE of $b$;
* `inverse_result['sens_w_dict']`: normalized sensitivity of $\mathbf{w}$'s;
* `inverse_result['sens_xi']`: normalized sensitivity of $\xi$;
* `inverse_result['sens_b']`: normalized sensitivity of $b$;
* `inverse_result['I']`: fitted intensity $\mathbf{I}$;
* `inverse_result['wct']`: wall-clock time to solution;
* `inverse_result['opt_res']`: a [scipy.optimize.OptimizeResult](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult) 
object returned by [scipy.optimize.minimize(method='trust-constr')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html#optimize-minimize-trustconstr), 
containing rich information about the solution (objective, Jacobian, Hessian, constraint violations, ...);
* `inverse_result['saved_res']`: `list` of `dict`'s, each `dict` containing all 
the above items at every `save_iter` iterations.







---

# 3 FAQ

* **What is the unit system of `ffsas`**?

    `ffsas` uses an arbitrary unit system. In other words, users can adopt any unit system, 
    and they should be responsible for inferring the units of their output based on the 
    units of their input. 
    
    Some libraries such as 
    [SASView/SASModels](http://www.sasview.org/docs/user/models/sphere.html) impose a 
    specific unit system. They benefit from that all input parameters will have an order 
    of magnitude close to 1 (i.e., not too big or too small) for typical applications, 
    and so to avoid any terms 
    being overwhelmed by numerical errors during inversion. Still, a fixed unit system 
    may have two caveats: first, it may not be always suitable for all source types and 
    models and, second, it may introduce some hidden constants to the equations 
    (e.g., a factor of $10^{-4}$ for 
    [SASView/SASModels](http://www.sasview.org/docs/user/models/sphere.html)). 
    
    For an inverse problem, how to preserve accuracy for an arbitrary unit system can be a 
    challenge. In `ffsas`, we develop a technique called **auto scaling**. 
    `ffsas` automatically analyzes the orders of magnitude of the data and the Green's tensor 
    and determines an internal unit system under which the Hessian of $\chi^2$ never gets 
    ill-conditioned. This internal unit system is completely hidden from users; namely, 
    from a user's perspective, using nanometer or meter as the length unit will lead to the 
    same accuracy.
    
    
* **Why the intensity is not normalized by the average volume as 
[SASView/SASModels](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html) does? 
What is the relation between `scale` in [SASView/SASModels](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html) 
and $\xi$ in `ffsas`?**

    The average volume is a scalar, unable to change the pattern of $\epsilon_{ij}$, 
    so it can be multiplied with `scale` to form another scaling factor, 
    which is $\xi$ in `ffsas`. 
    The relation between `scale` in 
    [SASView/SASModels](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html) 
    and $\xi$ in `ffsas` is
    
    $$
    \xi = 10^{-4}\times\dfrac{\mathrm{scale}}{V_\mathrm{ave}},
    \quad\quad\quad\quad(9)
    $$

    where the average volume $V_\mathrm{ave}$ can be easily computed using the $\mathbf{V}$ tensor 
    returned by `SASModel.compute_V()` and the relevant parameter weights. 
    One can use `SASModel.compute_average_V(par_dict, w_dict)` for this task,
    where `par_dict` contains the parameter grids and `w_dict` the parameter weights. 
    The factor $10^{-4}$ comes from the unit system of 
    [SASView/SASModels](http://www.sasview.org/docs/user/models/sphere.html); 
    without this factor, `ffsas` will still yield the same weights for model parameters 
    (as explained in the previous question), but with the intensity unit changed from 
    $\mathrm{cm}^{-1}$ to $\mathrm{(100\ m)}^{-1}$.
    
    

* **I am more used to `numpy` than `torch`. How do I use `ffsas`?**

    `ffsas` is completely based on `torch` for easy GPU utilization. 
    Conversions between `torch.Tensor` and `numpy.ndarray` are however 
    extremely easy with many `torch` APIs such as 
    `torch.tensor()`, `torch.from_numpy()` and `torch.Tensor.numpy()`. 
    
    There is only one place users should pay special attention. 
    The implementation of a user-defined model may involve some special functions, 
    such as the Bessel function `J1` for `Cylinder`, which may be unavailable from `torch`. 
    In that case, one must send the operand `torch.Tensor` to CPU, 
    convert it to `numpy.ndarry`, do the computation with `numpy` or `scipy`, 
    convert the result back to `torch.Tensor` and send it to GPU. 
    For example, in our `Cylinder` class, all these steps are encompassed by the following line: 
    ```python
    # need to compute j1(qa_r) with scipy.special.j1
    j1_qa_r = torch.tensor(j1(qa_r.to('cpu').numpy()), device=qa_r.device)
    ```
    
* **Can I use `float32` to save CPU/GPU memory?**
    
    Yes. To be consistent with `scipy.optimize.minimize(method='trust-constr')`,
    `ffsas` uses `float64` by default. To switch to `float32`, simply do
    (see [examples/cylinder_user_defined.ipynb](../examples/cylinder_user_defined.ipynb) for example):
    ```python
    import torch
    import ffsas
    ffsas.set_torch_dtype(torch.float32)
    
    from ffsas.models import Sphere
    from ffsas.system import SASGreensSystem
    ```
    > 📗 NOTE: `ffsas.set_torch_dtype(torch.float32)` must be called before importing from `ffsas.models` and `ffsas.system`. Also note that `ffsas.set_torch_dtype()` will reset the default date type of PyTorch.

    


---

The authors thank [latex.codecogs.com](https://latex.codecogs.com/) for real-time 
rendering of the $\text{\LaTeX}$ equations.
