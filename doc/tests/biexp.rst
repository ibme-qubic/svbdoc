Tests using Biexponential model
===============================

The biexponential model outputs a sum of exponentials:

.. math::
    M(t) = A_1 \exp{(-R_1 t)} + A_2 \exp{(-R_2 t)}

The model parameters are the amplitudes :math:`A_1`, :math:`A_2`
and the decay rates :math:`R_1` and :math:`R_2`.

Although the model is straightforward it can be challenging as
an inference problem as the effect of the two decay rates on the
output is nonlinear and can be difficult to distinguish in the
presence of noise.

Test data
---------

For testing purposes we define the ground truth parameters as:

 - :math:`A_1=10`
 - :math:`A_2=10`
 - :math:`R_1=1`
 - :math:`R_2=10`

The variables within the test data are:

 - The level of noise present. For this test data we use Gaussian
   noise with a standard deviation of 1.0.
 - The number of time points generated. We generate data sets with
   10, 20, 50 and 100 time points (in each case the value of :math:`t`
   ranges from 0 to 5 so only the data resolution changes in each case)

An example timeseries with these parameters is show below (100 time points,
ground truth overlaid on noisy data):

.. image:: /images/biexp/sample_timeseries.png
    :alt: Sample timeseries

1000 timeseries instances were generated and used for each test.

One issue with the biexponential model is that there are always two 
equivalent solutions obtained by exchanging :math:`A_1, R_1` with 
:math:`A_2, R_2`. To prevent this from confusing reports of mean
parameter values, we normalize the results of each run such that
in each voxel :math:`A_1, R_1` is the exponential with the lower
rate.

Test variables
--------------

The key variables in the algorithm are:

 - The size of the sample taken from the posterior when set independently
   of the batch size
 - The learning rate

These are respectively the main adjustable parameter of the stochastic VB method and
the key parameter of the optimization.

Additional factors we explore are:

 - Whether covariance between parameters is modelled. The output posterior
   distribution can either be modelled as a full multivariate Gaussian
   with covariance matrix, or we can constrain the covariance matrix
   to be diagonal so there is no correlation between parameter values.
 - The use of mini-batch processing where time points are divided into chunks
   each of which is used to determine an optimization step
 - The prior distribution of the parameter
 - The initial posterior distribution of the parameters
 - The use of the numerical (sample-based) calculation of the KL
   divergence on the posterior vs the analytic solution (possible 
   in this case only because both prior and posterior are represented
   by a multivariate normal distribution).

The main measure of convergence is the cost function (the negative of the free
energy in the VB formalism). The variational principle guarantees that lower
cost indicates a posterior which is closer to the exact Bayesian solution.

When investigating convergence speed across a range of
parameters we first find the minimum cost achieved by any combination of
parameters and identify that with the 'best' cost achievable. We can then 
compare the runtime required for different combinations to achieve within
a percentage error of this 'best' cost. This avoids the problem of 'fast'
optimizations which do not actually get close to the best solution. We consider
runtime rather than number of epochs as this is the measure of most interest
to an end user, and when using mini-batch processing an epoch might encompass
multiple optimization steps.

Learning rate and sample size
-----------------------------

The learning rate determines the size of optimization steps made by the
gradient optimizer and can be a difficult variable to select. Too high
and the optimizer may repeatedly overshoot the minima and never actually
converge, too low and convergence may simply be too slow. In many machine
learning problems the learning rate is determined by trial and error however
in our case we do not have this luxury as we need to be able to converge
the model fitting on any unseen data without user intervention.

The sample size is used to estimate the integrals in the calculation of
the cost function, so we would expect that a certain minimum size would
be required for a good result. The smaller the sample, the more the
resulting cost gradients are affected by the random sample selection
which may lead to a noisier optimisation process that may not converge
at all. On the other hand, larger sample sizes will take longer to 
calculate the mean cost giving potentially slower real-time convergence.

The best cost achieved is shown below by learning rate and sample size
number of time points. In these tests mini-batch processing was not used,
and the analytic calculation of the KL divergence was used.

.. image:: /images/biexp/best_cost_lr_ss_cov.png
    :alt: Best cost by learning rate and sample size with covariance

.. image:: /images/biexp/best_cost_lr_ss_nocov.png
    :alt: Best cost by learning rate and sample size without covariance

A number of observations can be made from these results:

 - Excessively high and low learning rates do not achieve the best cost -
   there is a 'sweet spot' between approximately 0.25 and 0.05 where close to the 
   minimum can be achieved reliably
 - Very small sample sizes do not achieve close to the minimum cost
 - Large sample sizes achieve minimum cost more reliably over a range of learning
   rates
 - A learning rate of 0.05 gives the best compromise between reliable convergence
   at a range of sample sizes
 - A sample size of around 10 is the minimum which achieves close to the best cost
   except for the 10 timepoint data where at least 20 appears to be required
 - The picture is essentially the same with and without covariance although without
   covariance we can tolerate higher learning rates better and smaller sample
   sizes - 10 for the NT=10 data, 5 otherwise seems adequate.
   
Our main motivation for wanting to minimise the sample size is convergence speed.
The plots below show the runtime required to achieve within set percentage factors
of the *overall* best cost (i.e. not just the best cost for that particular
set of parameters). In these plots we used a learning rate of 0.05.

.. image:: /images/biexp/conv_speed_ss_cov.png
    :alt: Convergence speed by sample size with covariance

.. image:: /images/biexp/conv_speed_ss_nocov.png
    :alt: Convergence speed by sample size with covariance

The general pattern that large sample sizes give reliable but slow convergence
is clear. Lower sample sizes are faster to compute but do not always achieve 
the best cost. Interestingly, very small sample sizes can be slower even 
when they do converge. 

These results generally support sample sizes of 10-20 with larger numbers of
timepoints, 20-50 with smaller. They also suggest an optimization whereby
sample size is increased during the optimization to achieve a fast convergence
to an approximate solution, followed by slower more careful steps to 
attain the minimum cost.

The picture is similar for the runs without covariance however slightly smaller
samples seem to be optimal for the data sets with larger number of timepoints.

Use of mini-batch processing to accelerate convergence
------------------------------------------------------

Optimization of the cost function proceeds by 'epochs' which consists
of a single pass through all of the data. Batch processing consists
of dividing the data into smaller batches and performing multiple
iterations of the optimization - one for each batch - during an epoch.
Processing the data in batch is a commonly used method to accelerate
convergence and works because updates to the parameters occurs multiple
times during each epoch. The optimization steps are 'noisier' because
they are based on less training samples and this helps to avoid 
converging onto local minima. Of course if the batch size is too small
the optimization may become so noisy that convergence does not occur
at all.

The plots below show the best cost achieved at various batch sizes and
learning rates using a sample size of 20. We are concerned here with
whether mini-batch processing is still able to achieve the minimum
cost.

.. image:: /images/biexp/best_cost_lr_bs_cov.png
    :alt: Best cost achieved by batch size and learning rate

.. image:: /images/biexp/best_cost_lr_bs_nocov.png
    :alt: Best cost achieved by batch size and learning rate
    
It is clear that there is an interaction between batch processing and
learning rate - as expected the 'noisier' optimization steps from a mini-batch
work better at lower learning rates, and worse at higher learning rates.
The previously identified optimum learning rate of 0.05 is tolerant of
a range of batch sizes, however.

The aim of batch processing is to accelerate convergence, so we
now need to look at whether smaller batch sizes do indeed achieve this.

.. image:: /images/biexp/conv_speed_bs_cov.png
    :alt: Convergence speed by sample size with covariance

.. image:: /images/biexp/conv_speed_bs_nocov.png
    :alt: Convergence speed by sample size with covariance

With fewer timepoints the data is necessarily limited since we cannot have
batch sizes larger than the data size. However it is clear that convergence
can be accelerated by the use of a mini-batch approach. A batch size of 10
seems to be a good overally compromise and with 50 or 100 timepoints can lead
to convergence within 2% of best cost that is a factor of 2-3 time faster 
than obtained by processing the entire data set in each step. For lower
numbers of timepoints the improvement is less significant but at any rate
not harmful.

There is no real difference in conclusions with and without inferred 
covariance.

Parameter recovery
------------------

So far we have just investigated convergence through the cost function,
however ultimately it is the parameter values that we are interested in.
These plots show the voxelwise distribution of the parameters with and
without covariance by number of timepoints. The 'VB' plots show the
equivalent distributions using conventional Variational Bayes algorithm
(based on the Fabber_ tool - note that this algorithm always infers 
covariance between parameters):

.. image:: /images/biexp/amp1_method.png
    :alt: Parameter recovery for amp1

.. image:: /images/biexp/amp2_method.png
    :alt: Parameter recovery for amp2

.. image:: /images/biexp/r1_method.png
    :alt: Parameter recovery for r1

.. image:: /images/biexp/r2_method.png
    :alt: Parameter recovery for r2

.. image:: /images/biexp/noise_method.png
    :alt: Parameter recovery for noise

Parameter recovery with fewer data points tends to lead to a solution where
the first exponential has higher amplitude and decay rate, while the second
has lower amplitude and sometimes very high decay rate. Essentially here
there isn't really enough information in the time course to separate the
two exponential contributions to the signal and the model is tending
towards a single-exponential solution. Implementation of an ARD method would
be useful here in allowing the model to explicitly determine that the
second exponential is not helpful in modelling the data and drop it out
from the inference.

For larger numbers of time points parameter recovery is good. Outliers are
consistently within reasonable bounds of the true solution, particularly
for NT=100.

In comparison with conventional variational Bayes, the distributions are 
similar for the higher numbers of timepoints where the
bi-exponential solution is found by both methods. Both methods also struggle
to recover the bi-exponential property with 20 or fewer timepoints, although
the direction of the errors is different in each case.

.. _Fabber: https://fabber_core.readthedocs.io/

.. note::
    Since the biexponential model has two identical solutions obtained by
    exchanging the amplitude and rate parameters for the exponentials,
    we have normalized the data here by identifying `r1` and `amp1` with
    the exponential having the lower decay rate

Comparison with conventional Variational Bayes
----------------------------------------------

Effect of prior and initial posterior
-------------------------------------

The following combinations of prior and posterior were used. An informative
prior was set with a mean equal to the true parameter value and a standard
deviation of 2.0. Non-informative priors were set with a mean of 1 and a
standard deviation of 1e6 for all parameters.

Non-informative initial posteriors were set equal to the non-informative
prior. Informative posteriors were set with a standard deviation of 2.0
and a mean which either matched or did not match the true parameter value as
described below. In addition, an option in the model enabled the initial 
posterior mean for the amplitude parameters to be initialised from the data.

+----------------+----------------------------------------------------------------------+
|Code            |Description                                                           |
+----------------+----------------------------------------------------------------------+
|``i_i``         |Informative prior, informative posterior initialised with mean values |
|                |equal to 1.0 for all parameters                                       |
+----------------+----------------------------------------------------------------------+
|``i_i_init``    |Informative prior, informative posterior initialised with true values |
|                |of the decay rates and with amplitude initialised from the data       |
+----------------+----------------------------------------------------------------------+
|``i_i_true``    |Informative prior, informative posterior initialised with true values |
+----------------+----------------------------------------------------------------------+
|``i_i_wrong``   |Informative prior, informative posterior initialised with mean values |
|                |of 1.0 for the decay rate and 100.0 for the amplitudes (i.e. very far |
|                |from the true values)                                                 |
+----------------+----------------------------------------------------------------------+
|``i_ni``        |Informative prior, non-informative posterior                          |
+----------------+----------------------------------------------------------------------+
|``i_ni_init``   |Informative prior, non-informative posterior with amplitude           |
|                |initialised from the data                                             |
+----------------+----------------------------------------------------------------------+
|``ni_i``        |Non-informative prior, informative posterior initialised with mean    |
|                |values equal to 1.0 for all parameters                                |
+----------------+----------------------------------------------------------------------+
|``ni_i_init``   |Non-informative prior, informative posterior initialised with true    |
|                |values of the decay rates and with amplitude initialised from the data|
+----------------+----------------------------------------------------------------------+
|``ni_i_true``   |Non-informative prior, informative posterior initialised with true    |
|                |values                                                                |
+----------------+----------------------------------------------------------------------+
|``ni_i_wrong``  |Non-informative prior, informative posterior initialised with mean    |
|                |values of 1.0 for the decay rate and 100.0 for the amplitudes (i.e.   |
|                |very far from the true values)                                        |
+----------------+----------------------------------------------------------------------+
|``ni_ni``       |Non-informative prior, non-informative posterior                      |
+----------------+----------------------------------------------------------------------+
|``ni_ni_init``  |Non-informative prior, non-informative posterior with amplitude       |
|                |initialised from the data                                             |
+----------------+----------------------------------------------------------------------+

.. image:: /images/biexp/prior_post.png
    :alt: Best cost achieved by prior and posterior combinations

These results show that in terms of absolute convergence there is no significant 
difference between the choice of prior and posterior. Note that the absolute cost
achieved can be different between the informative and non-informative priors as 
expected. The exception is the cases where a *non-informative* initial posterior is
used - these cases do not achieve convergence.

The explanation for this lies in the fact that components of the cost are dependent
on a sample drawn from the posterior. In the case of a non-informative posterior 
samples of realistic sizes cannot be large enough to be representative and different
samples may contain widely varying contents. Such samples cannot reliably 
direct the optimisation to minimise the cost function because the calculated cost 
(and its gradients) are dominated by random variation in the values contained within
the sample.

By contrast if the posterior is informative - even if it is far from the best solution
- different moderately-size random samples are all likely to provide a reasonable representation
of that distribution. The optimisation will therefore be directed to minimse the cost
more reliably since it is less dependent on the particular values that happened
to be included in the sample.

We conclude that the initial posterior must be informative even if it is a long way 
from the true solution.

The ``_analytic`` and ``_num`` plots are identical apart from using the analytic
or the numerical solution to the KL divergence between two MVNs. The similarity between these results
suggests that the numerical solution should be sufficient
in cases where the prior and posterior cannot be represented as two MVN distributions.

The ``_corr`` and ``__nocorr`` plots were generated with and without a full posterior
covariance matrix. In this case we see little difference between the two.

It is reassuring that the cost can converge under a wide variety of prior and posterior
assumptions, however it is also useful to consider the effect of these variables
on speed of convergence. The results below illustrate this:

.. image:: /images/biexp/prior_post_conv_speed.png
    :alt: Best cost achieved by prior and posterior combinations

This plot shows the epoch at which each voxel converged (to with 5% of its final values).
The box plot show the median and IQR, while the circles show slow-converging outliers.
For the reasons given above, non-informative posterior test cases were excluded from
this plot.

It is clear that the main impact on convergence speed is the initial posterior. 
Where it is far from the true values (``i_wrong``) convergence is slowest. However
this problem is much less obvious when the priors are informative as in this case the
'wrong' posterior values generate high latent cost as they are far from the 'true'
prior values. This quickly guides the optimisation to the correct solution. Initialisation of the
posterior from the data (where there is a reasonable method for doing this) is
therefore recommended to improve convergence speed.

Numerical vs analytic evaluation of the KL divergence
-----------------------------------------------------

In the results above we have used the analytic result for the KL divergence of two
multivariate Gaussian distributions. In general where the posterior is not 
constrained to this distribution we need to use a numerical evaluation which involves
the posterior sample. So it is useful to assess the effect of forcing the
numerical method in this case, particularly in combination with variation in
the sample size.

.. image:: /images/biexp/best_cost_ss_num_cov.png
    :alt: Best cost achieved by analytic and numerical solution

.. image:: /images/biexp/best_cost_ss_num_nocov.png
    :alt: Best cost achieved by analytic and numerical solution

The absolute values of the free energy cannot be compared directly since 
some constant terms in the analytic solution are dropped from the calculation.
For this reason the plots above have been normalized by subtracting the mean
cost in each case. The resulting convergence properties with sample size
are closely similar indicating that the numerical solution is a viable
alternative to the analytic method where the latter cannot be used.

Inference of covariance
-----------------------

The effect of inferring covariance or not has been shown throughout
these tests. In general the effect is that convergence is more
challenging with covariance as would be expected with the increased
parameter space, and instabilities caused by small batch or sample
sizes, or large learning rates, are exacerbated by the inclusion
of covariance. It's worth mentioning that the symmetry of the 
biexponential model would expect to generate significant parameter
covariances.

A strategy of initially optimizing without covariance, and then 
restarting the optimization with the covariance parameters included
is an obvious way to address this.
