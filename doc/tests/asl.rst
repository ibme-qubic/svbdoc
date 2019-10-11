Tests using Arterial Spin Labelling model
=========================================

This model implements a basic resting-state ASL kinetic model for PASL
and pCASL acquisitions. The model parameters are :math:`f_{tiss}`, the
relative perfusion and :math:`\delta t` the transit time of the 
blood from the labelling plane to the voxel.

Time points are divided into two categories:

*During bolus* is defined as :math:`\delta t < t <= \tau + \delta t`

*Post bolus* is defined as :math:`t > \tau + \delta t`

Here :math:`\tau` is the bolus duration. The model output is zero for pre-bolus 
time points.

The following rate constant is defined:

:math:`\frac{1}{T_{1app}} = \frac{1}{(1 / T_1 + f_{calib} / \lambda)}`

:math:`\lambda` is the tissue/blood partition coefficient of water which we take to 
be 0.9. :math:`f_{calib}` is the calibrated CBF which typically we do not do not have 
access to (since we are inferring relative CBF) so we use a typical value of 0.01 :math:`s^{-1}`.

CASL model
----------

During bolus
~~~~~~~~~~~~

:math:`M(t) = 2 f_{tiss} T_{1app} \exp{(\frac{-\delta t}{T_{1b}})} (1 - \exp{(-\frac{(t - \delta t)}{T_{1app}})})`

Post bolus
~~~~~~~~~~

:math:`M(t) = 2 f_{tiss} T_{1app} \exp{(-\frac{\delta t}{T_{1b}})} \exp{(-\frac{(t - \tau - \delta t)}{T_{1app}})} (1 - \exp{(-\frac{\tau}{T_{1app}})})`

PASL model
----------

:math:`r = \frac{1}{T_{1app}} - \frac{1}{T_{1b}}`

:math:`f = 2\exp{(-\frac{t}{T_{1app}})}`

During bolus
~~~~~~~~~~~~

:math:`M(t) = f_{tiss} \frac{f}{r} (\exp{(rt)} - \exp{(r\delta t)})`

Post bolus
~~~~~~~~~~
    
:math:`M(t) = f_{tiss} \frac{f}{r} (\exp{(r(\delta t + \tau))} - \exp{(r\delta t)})`

The time points in evaluating an ASL model are the :math:`T_i` values, which may be expressed
as the sum of the bolus duration :math:`\tau` and a post-labelling delay time. For 2D acquisitions
they may be further modified by the additional time delay in acquiring each slice.

Test data
---------

The test data used is a pCASL acquisition with :math:`\tau = 1.8s` and six post-labelling
delays of 0.25, 0.5, 0.75, 1.0, 1.25 and 1.5s. The acquisition was 2D with an additional
time delay of 0.0452s per slice. 8 repeats of the full set of PLDs was obtained.

The test data was fitted in two ways. One method was to average over the repeats
and fit the model to the repeat-free data. The other is to fit the model to the whole
data including repeats. Naturally this involves a larger data size and hence the possibility
of a mini-batch approach to the optimization.

As with the biexponential test case, we focus initially on learning rate and sample size
and subsequently on mini-batch processing as an optimization technique.

Learning rate and sample size
-----------------------------

The best cost achieved is shown below by learning rate and sample size
for the full and averaged data. In these tests mini-batch processing was not used,
and the analytic calculation of the KL divergence was used.

.. image:: /images/asl/best_cost_lr_ss_cov.png
    :alt: Best cost by learning rate and sample size with covariance

.. image:: /images/asl/best_cost_lr_ss_nocov.png
    :alt: Best cost by learning rate and sample size without covariance

The pattern is similar to that obtained using a biexpoential model, supporting
learning rates of between 0.1 and 0.05. It is noticeable that the full
data set is tolerant of smaller sample sizes than in the biexponential case with
a size of 5 giving a solution close to the optimal. By contrast the averaged
data requires a larger sample size which may offset the advantage of the smaller
data set.

We can see this by plotting convergence time to within fixed tolerances of the 
best overall cost at various sample sizes using a learning rate of 0.1:

.. image:: /images/asl/conv_speed_ss_cov.png
    :alt: Convergence speed by sample size with covariance

.. image:: /images/asl/conv_speed_ss_nocov.png
    :alt: Convergence speed by sample size with covariance

To attain a result within 2% of the best overall cost when inferring covariance, 
the time required with the averaged data set (with a sample size of 20) is almost 
the same as that required on the much larger full data set (with a sample size of 5).

With the full data it is notable that very small sample sizes (2-5) can be used
successfully and accelerate convergence by a factor of 2-5 compared to the 
largest sample used (20).

The picture is similar for the runs without covariance however slightly smaller
samples sizes are better tolerated in all cases.

Use of mini-batch processing to accelerate convergence
------------------------------------------------------

The use of mini-batch processing is limited for the averaged data as we only
have 6 time points. However for the 48-point full data we can see if faster
convergence can be attained with the use of batch processing. 

Since the data set consists of 8 repeats of 6 delay times, a natural idea would
be to use a batch size of 6, meaning that each batch contains a single repeat
of all the delay times. We use this value as one possibility, but also consider
using other batch sizes that do not necessarily correspond to the structure
of the data.

These tests used a sample size of 5, which 
was found to be generally optimal for the repeated data set, which is the one we
are more interested in here.

First we want to determine whether mini-batch processing still attains the
optimal cost found without the use of batches:

.. image:: /images/asl/best_cost_lr_bs_cov.png
    :alt: Best cost achieved by batch size and learning rate

.. image:: /images/asl/best_cost_lr_bs_nocov.png
    :alt: Best cost achieved by batch size and learning rate
  
As in the biexponential case, smaller batch sizes work better at lower 
learning rates, and worse at higher learning rates.
The previously identified optimum learning rate of 0.1 is tolerant of
a range of batch sizes, however. It is noticeable that smaller batch sizes can
attain lower cost than processing the full data set, i.e. the increased
gradient noise enables a slightly more optimal solution.

However our main motivation for batch processing is to accelerate 
convergence, so fixing a learning rate of 0.1 we can plot the time
required to achieve various levels of convergence:

.. image:: /images/asl/conv_speed_bs_cov.png
    :alt: Convergence speed by sample size with covariance

.. image:: /images/asl/conv_speed_bs_nocov.png
    :alt: Convergence speed by sample size with covariance

We can see that use of smaller batch sizes can accelerate convergence by 
a factor of 2-3. With covariance, the optimal size is between 12 and 18
whereas without covariance all batch sizes of 18 or smaller are comparable.

Parameter recovery
------------------

.. image:: /images/asl/conv_ss_ftiss_nocov.png
    :alt: Convergence of ftiss parameter by sample size (without covariance)

.. image:: /images/asl/conv_ss_ftiss_cov.png
    :alt: Convergence by ftiss parameter sample size (with covariance)

.. image:: /images/asl/conv_ss_delttiss_nocov.png
    :alt: Convergence of delttiss parameter by sample size (without covariance)

.. image:: /images/asl/conv_ss_delttiss_cov.png
    :alt: Convergence by delttiss parameter sample size (with covariance)
