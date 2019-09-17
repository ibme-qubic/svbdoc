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

Convergence by learning rate
----------------------------

The convergence of mean cost by learning rate is shown below with and without
covariance:

.. image:: /images/asl/conv_lr_nocov.png
    :alt: Convergence by learning rate (without covariance)

.. image:: /images/asl/conv_lr_cov.png
    :alt: Convergence by learning rate (with covariance)

The pattern is closely similar to that obtained using a biexpoential model
although the convergence here is generally 'cleaner'. Up to 0.25, higher learning
rates minimise the cost faster, however a learning rate of 0.25 becomes unstable
with covariance as the minimum is approached. This suggests the use of a 'quenching'
method (investigated later). Note that these plots show convergence by runtime not
epochs as this is more relevant to an end user. These plots used a posterior
sample size of 20 and no mini-batch processing.

Lower learning rates attain eventually attain a better minimum cost, with the 
optimimum learning rate of 0.1 - 0.05 over most sample sizes, although the 
repeated data tolerates higher learning rates at large sample sizes.

.. image:: /images/asl/best_cost_lr_ss_nocov.png
    :alt: Best cost achieved in 500 epochs by learning rate (without covariance)

.. image:: /images/asl/best_cost_lr_ss_cov.png
    :alt: Best cost achieved in 500 epochs by learning rate (with covariance)

Convergence by batch size
-------------------------

These tests used a fixed learning rate of 0.1 and sample size of 20. These
tests are most relevant to the repeated data set since it has a large enough
number of volumes to make batch processing worthwhile.

.. image:: /images/asl/conv_bs_nocov.png
    :alt: Convergence by batch size (without covariance)

.. image:: /images/asl/conv_bs_cov.png
    :alt: Convergence by batch size (with covariance)

Mini batch processing is associated with faster convergence with the best
convergence achieved with a batch size of 9 which compares well with the 
optimal batch size of 10 estimated in the biexponential tests. Interestingly
processing the mean data with a batch size of 5 rather than using the full
set of 6 repeats also gives faster convergence. However with covariance
enabled we get instability with smaller batch sizes as convergence approaches
the minimum.

.. image:: /images/asl/best_cost_lr_bs_nocov.png
    :alt: Best cost achieved in 500 epochs by batch size (without covariance)

.. image:: /images/asl/best_cost_lr_bs_cov.png
    :alt: Best cost achieved in 500 epochs by batch size (with covariance)

The best cost achieved was largely independent of the batch size for the 
optimal learning rates between 0.1 and 0.05. However small batch sizes 
converge better at lower learning rates than large batch sizes and worse
at higher learning rates, consistent with the results of the biexponential tests.

Convergence by posterior sample size
------------------------------------

These tests used a fixed learning rate of 0.1 and no mini-batch processing.

.. image:: /images/asl/conv_ss_nocov.png
    :alt: Convergence by sample size (without covariance)

.. image:: /images/asl/conv_ss_cov.png
    :alt: Convergence by sample size (with covariance)

Smaller sample sizes converge faster (as would be expected since the 
computational demands are essentially proportional to the sample size)
however a larger sample is needed to get the best cost (this is less
evident when not inferring covariance). Note however that the cost
with sample size is not necessarily strictly variational so we cannot
immediately conclude that a lower cost is better.

Similar results are obtained when we use a mini-batch approach for
the repeated data, with a batch size of 6, however we again see 
instability in the cost as convergence is approached when inferring
covariance.

.. image:: /images/asl/conv_ss_bs_6_nocov.png
    :alt: Convergence by sample size with batch size of 6 (without covariance)

.. image:: /images/asl/conv_ss_bs_6_cov.png
    :alt: Convergence by sample size with batch size of 6 (with covariance)

Since we cannot directly rely on the variational principle here we can 
also look at variation of parameter values with sample size. These plots
are remarkably dull and suggest that the lower free energy achieved at 
higher sample sizes does not necessarily translate into significant 
differences in the posterior parameter distributions.
 
.. image:: /images/asl/conv_ss_ftiss_nocov.png
    :alt: Convergence of ftiss parameter by sample size (without covariance)

.. image:: /images/asl/conv_ss_ftiss_cov.png
    :alt: Convergence by ftiss parameter sample size (with covariance)

.. image:: /images/asl/conv_ss_delttiss_nocov.png
    :alt: Convergence of delttiss parameter by sample size (without covariance)

.. image:: /images/asl/conv_ss_delttiss_cov.png
    :alt: Convergence by delttiss parameter sample size (with covariance)
