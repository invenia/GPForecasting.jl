# GPForecasting F.A.Q.


- [What is the problem we are using a GP to solve?](#what-is-the-problem-we-are-using-a-gp-to-solve)
- [How are our GP models different from the EmpiricalForecaster?](#how-are-our-gp-models-different-from-the-empiricalforecaster)
- [Why would we like to use GPs?](#why-are-we-using-gps)
- [What are the disadvantages of using GPs](#what-are-the-disadvantages-of-using-gps)
- [How complex are GPs as a statistical model?](#how-complex-are-gps-as-a-statistical-model)
- [What are the risks of overfitting?](#what-are-the-risks-of-overfitting)
- [How long does the model take to run?](#why-does-it-need-more-memory-than-the-empiricalforecaster)
- [Why does it need more memory to run?](#how-long-does-the-model-take-to-run)

## What is the problem we are using a GP to solve?

Our aim is to forecast the difference between the day-ahead (DA) price and real-time (RT) price of a megawatt hour (MWh) at each “node” in an energy grid. This difference is a number (in units of $/MWh). If we have 10 nodes, we are trying to forecast 10 numbers per
timestamp.

We make these predictions so that we can use them as part of a decision making process. For that it is helpful to know how confident we are in our predictions. We represent our confidence with probabilities. In particular all of our Forecasters return a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) over the prices at the nodes.

For example, at a given timestamp, for our 10 nodes we may return a [Gaussian distribution](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2019/notes/w2e_multivariate_gaussian.html) `N(mean, cov)` with `mean` a 10-dimensional vector giving the the price we expect to see at each node, and `cov` being the [covariance](https://en.wikipedia.org/wiki/Covariance) which represents the uncertainty we have about the forecasts, including information about how we expect the price at one node to be related to the prices at the other nodes.

Both the EmpiricalForecaster and GPForecasters return Gaussian distributions when the `predict` function is called, but other methods may return different types of distributions. For more details on the modelling aspect, see [these notes](Modelnotes.md).

## How are our GP models different from the EmpiricalForecaster?

The [EmpiricalForecaster](https://gitlab.invenia.ca/invenia/BaselineForecasters.jl) (EF) treats each node separately when computing the means, and then after the fact adds a covariance based on historical data. Our [Gaussian Process](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2019/notes/w5a_gaussian_processes.html) (GP) [model (called OLMM)](https://invenia.slack.com/archives/C0255M3U9/p1570592334011800), imposes a “low-rank structure” over the grid: we assume that the behaviour of all nodes is described by a small number (much less than the number of nodes) of “latent factors” that are never directly observed. In other words, [some unobserved phenomenon governs the behaviour of (possibly) multiple parts of the system](https://en.wikipedia.org/wiki/Latent_variable). This fact is reflected in the total number of [degrees of freedom](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)) that the model has (lower for the GPs than for the EmpiricalForecaster).

## Why are we using GPs?

There is more than one way to look at this. One of the advantages is that they are fully [Bayesian](https://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/) models, so they provide a simple way of representing our uncertainty about our forecasts. In practice, that means that the predictive model is “regularised” by our prior assumptions, that is, it doesn't simply rely on the data, as a non-Bayesian model would, we also get to specify some beliefs we have about how things work before we have seen the data. The makes Bayesian models more robust towards data-related issues. The uncertainties are built-in to the model and can be well calibrated.

Among the Bayesian approaches, [GPs are some of the simplest ones](https://www.ritchievink.com/blog/2019/02/01/an-intuitive-introduction-to-gaussian-processes/), offering exact, closed-form solutions in general (and in the case of the “OLMM” model we use in particular).

Another important advantage of adopting models like GPs is that they are “principled”: they provide a framework that can be used to build upon to construct more complex models. For instance, adding extra features to the EmpiricalForecaster is non-trivial, whereas for GPs it is very simple. We can also do other things, like insert input data transformations as an integral part of the model, or use automatic relevance determination for features. There is a vast literature on GPs which can be leveraged to improve our models.

## What are the disadvantages of using GPs?

In general, [GPs don't scale well with the size of the dataset](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2019/notes/w5b_gaussian_process_kernels.html#computation-cost-and-limitations). While one of the advantages of GPs is that they are simple---they’re largely just a bunch of matrix operations---the downside is they involve inverting matrices, which requires both a lot of computation and a lot of memory.

That means that, for some GPs, the memory usage blows up to the point of making the model computationally intractable. There are approximations to get around this, but that adds some extra complexity. How much complexity is added depends on which kind of problem we are solving and how. Approximation methods range from simple implementation of sparse GPs (which we have in GPForecasting.jl) to potentially elaborate [variational inference](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2019/notes/w9a_variational_kl.html) schemes which involve solving optimisation problems and are generally [harder to apply](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2019/notes/w9a_variational_kl.html#overview-of-gaussian-approximations).

Despite GPs being a simple model when compared to deep learning approaches, when compared to the EmpiricalForecaster they are definitely more complex. This makes it possible for us to keep improving our GP models, but it also increases the number of model parameters, making it harder to find good configurations. In an analogy, compare the task of fitting a least squares line against building a multilayer neural network. While the latter is much more powerful, it is much harder to configure.

GPs may be a bit more opaque than the EmpiricalForecaster for people who have not studied them. In particular GPs involve a lot of matrix algebra, and several mathematical “tricks” are included in the GPForecasting.jl package to compute things more efficiently. This allows us to use better models, but also leads to a specific kind of complex “maths-y” code, which we try to mitigate by still having high standards for the code (being clean, modular, commented etc.) and with [documentation](https://research.pages.invenia.ca/GPForecasting.jl/).

## How complex are GPs as a statistical model?

While the theory behind these models may seem opaque to people from different fields, [GPs are not particularly complex models. Rather, they are recognised for their simplicity](https://www.cs.toronto.edu/~hinton/csc2515/notes/gp_slides_fall08.pdf). As the name suggests they heavily rely on the Gaussian distribution, which we know a lot about, and allows us to simplify “fitting a model” down to the much simpler task of “do some Matrix algebra”. Moreover, [GPs can be linked with methods as ubiquitous as (kernelised) linear regression](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2019/notes/w5b_gaussian_process_kernels.html#bayesian-linear-regression-as-a-gp). The machine learning literature is replete with considerably more complex models, such as CNNs, RNNs, GANs, VAEs, etc.

Usually, the more complex the model the harder it is to tune properly and make it work satisfactorily. Given the complexity of the data we deal with, the simplicity of the model represents a strength.

## What are the risks of overfitting?

Any statistical model can overfit under unfavourable conditions, and GPs are no different. However, [the Bayesian nature of the model should make it much more robust to the data](https://medium.com/neuralspace/how-bayesian-methods-embody-occams-razor-43f3d0253137) than, for example, Neural Networks. Moreover, in our specific construction, we restrict the degrees of freedom of the model by enforcing a low-rank structure on it, which increases its robustness towards overfitting.

Moreover, our GP model has exact, closed-form, inference. That means we eliminate several sources of computational complexity and numerical instability that would be present in other models.

In the worst case scenario in which our GPs completely overfit the training data, a property of the model is that the predictions will automatically “fallback” to the prior model, which assumes deltas equal to the historical means. It is important to notice that this is built-in to the model, and not something added _ad hoc_.

## Why does it need more memory than the EmpiricalForecaster?

GPs are naturally memory-hungry, due to the need to invert large matrices. However, our main memory consumption comes from the computation graph Nabla.jl builds during the automatic differentiation of the code. This should be heavily decreased when we adopt Zygote.jl. It is important to keep in mind the distinction between the intrinsic memory requirements of GP-based models and those stemming from a given Automatic Differentiation (AD) framework. Profiling in Julia is not the easiest task, so a detailed analysis of this is not currently available, however, there are some superficial discussions [here](https://gitlab.invenia.ca/research/GPForecasting.jl/issues/31#note_70280) and [here](https://drive.google.com/open?id=1AdUq88jj7lDB_iJgK22dwVJVNVClxIRw).

## How long does the model take to run?

This is highly dependent on which settings we are using and for which grid we are running. The current settings we are using for MISO lead to approximately 10 minutes for the model to go from building the features to constructing the predictive distribution. This is bound to decrease once we change the automatic differentiation package we are using from Nabla.jl with Zygote.jl.
