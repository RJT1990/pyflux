Families
==================================

Introduction
----------

PyFlux uses a unified family API that can be used for specifying model measurement densities as well as the priors on latent variables in the model. This guide shows the various distributions available and their uses.

Family Guidebook
----------

.. py:class:: Cauchy(loc, scale, transform)

   **Cauchy Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   loc                  float                              Location parameter for Cauchy family
   scale                float                              Scale for Cauchy family
   transform            string                             Whether to transform lmd (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for priors and model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.Cauchy())

   .. code-block:: python

      model.adjust_prior(0, pf.Cauchy(0,1))

.. py:class:: Exponential(lmd, transform)

   **Exponential Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   lmd                  float                              Rate parameter for the Exponential family
   transform            string                             Whether to transform lmd (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.Exponential())

.. py:class:: Flat(lmd, transform)

   **Flat Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   transform            string                             Whether to transform the parameter
   ==================   ===============================    ======================================

   This class can be used as an non-informative prior distribution.

   .. code-block:: python

      model.adjust_prior(0, pf.Flat())

.. py:class:: InverseGamma(alpha, beta, transform)

   **Inverse Gamma Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   alpha                float                              Alpha parameter for the IGamma family
   beta                 float                              Beta parameter for the IGamma family
   transform            string                             Whether to transform the parameter
   ==================   ===============================    ======================================

   This class can be used as a prior distribution.

   .. code-block:: python

      model.adjust_prior(0, pf.InverseGamma(1,1))

.. py:class:: InverseWishart(v, Psi, transform)

   **Inverse Wishart Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   v                    float                              v parameter for the family
   Psi                  float                              Psi covariance matrix for the family
   transform            string                             Whether to transform the parameter
   ==================   ===============================    ======================================

   This class can be used as a prior distribution.

   .. code-block:: python

      my_covariance_prior = np.eye(3)
      model.adjust_prior(0, pf.InverseWishart(3, my_covariance_prior))

.. py:class:: Laplace(loc, scale, transform)

   **Laplace Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   loc                  float                              Location parameter for Laplace family
   scale                float                              Scale for Laplace family
   transform            string                             Whether to transform loc (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for priors and model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.Laplace())

   .. code-block:: python

      model.adjust_prior(0, pf.Laplace(0,1))

.. py:class:: Normal(mu, sigma, transform)

   **Normal Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   mu                   float                              Location parameter for Normal family
   sigma                float                              Standard deviation for Normal family
   transform            string                             Whether to transform mu (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for priors and model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.Normal())

   .. code-block:: python

      model.adjust_prior(0, pf.Normal(0,1))

.. py:class:: Poisson(lmd, transform)

   **Poisson Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   lmd                  float                              Rate parameter for the Poisson family
   transform            string                             Whether to transform mu (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.Poisson())

.. py:class:: t(loc, scale, df, transform)

   **Student-t Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   loc                  float                              Location parameter for t family
   scale                float                              Standard deviation for t family
   df                   float                              Degrees of freedom for t family
   transform            string                             Whether to transform mu (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for priors and model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.t())

   .. code-block:: python

      model.adjust_prior(0, pf.t(0, 1, 3))

.. py:class:: Skewt(loc, scale, df, gamma, transform)

   **Skewed Student-t Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   loc                  float                              Location parameter for t family
   scale                float                              Standard deviation for t family
   df                   float                              Degrees of freedom for t family
   gamma                float                              Skewness parameter for t family
   transform            string                             Whether to transform mu (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used for priors and model measurement densities. For example:

   .. code-block:: python

      model = pf.ARIMA(ar=1,ma=0,data=my_data, family=pf.Skewt())

   .. code-block:: python

      model.adjust_prior(0, pf.Skewt(0, 1, 3, 0.9))

.. py:class:: TruncatedNormal(mu, sigma, lower, upper, transform)

   **Truncated Normal Family**

   ==================   ===============================    ======================================
   Parameter            Type                                Description
   ==================   ===============================    ======================================
   mu                   float                              Location parameter for TNormal family
   sigma                float                              Standard deviation for TNormal family
   lower                float                              Lower limit for the truncation
   upper                float                              Upper limit for the truncation
   transform            string                             Whether to transform mu (e.g. 'exp')
   ==================   ===============================    ======================================

   This class can be used as a prior. For example:

   .. code-block:: python

      model.adjust_prior(0, pf.TruncatedNormal(0, 1, lower=0.0, upper=1.0))



