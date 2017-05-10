# Early Stopping Algorithm for Experiment
This repository contains simulation scripts to evaluate different statistical approaches for optional stopping of the experiment(A/B testing in particular).

## Simulation

Results of Bayes factors are saved in a `csv` as a column vector, where the 
inner loop is the number of days and the outer loop is the number of simulations.

## Frequentist

- Wald's SPRT test based on testing a single point hypothesis against another 
point hypothesis, possible workaround:

>  Gary Lordon. 2-sprt’s and the modified kiefer-weiss problem of minimizing an expected sample size. The Annals of Statistics, 4:281–291, 1976.

- Based on the z-score and a spending function that distributes the false 
positive rate over time 
	- Pocock 
	- O'Brien-Fleming (more conservative)
	- alpha spending function ftp://maia-2.biostat.wisc.edu/pub/chappell/641/papers/paper35.pdf

> http://www.aarondefazio.com/tangentially/?p=83

> Review: Sebille et al. Sequential methods and group sequential designs for comparative clinical trials. 2003

## Bayesian

## Open questions
- While controlling the false positive rate, one should also check the power of alternative methods.
- Compare with the outcome of the `BayesFactor` package.

## Literature
- blog post

> Defazio HOW TO DO A/B TESTING WITH EARLY STOPPING CORRECTLY? 2016

- SD density ratio 

> Wagenmakers et al. Bayesian hypothesis testing for psychologists: A tutorial on the Savage–Dickey method. Cog Psych 2010 

- How to choose the prior; JZS prior (Cauchy on effect size, Jeffreys on variance); Analytical solution to BF
	
> Rouder et al. Bayesian t tests for accepting and rejecting the null hypothesis. Psych Bulletin & Review 2009

- overview of NHST/Bayesian parameter estimation/Bayes factor

> Kruschke Bayesian assessment of null values via parameter estimation and model comparison. Perspectives on Psych Sci 2011

## Misc
- Make libraries in the virtualenv available in the jupyter notebook

> http://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs

- Bayesian A/B testing tutorial using `Stan`

> https://rpubs.com/rasmusab/exercise_2_bayesian_ab_testing

> http://m-clark.github.io/docs/IntroBayes.html

> http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/IntroToStan_basics_workflow.ipynb

- HMC

> http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html

- General overview of frequentist vs. Bayesian, null hypothesis testing vs. parameter estimation

> http://www.valuewalk.com/wp-content/uploads/2015/05/SSRN-id2606016.pdf

- Review of NHST, GS and BF

> Schoenbrodt et al. Sequential Hypothesis Testing With Bayes Factors: Efficiently Testing Mean Differences

- Running too many processes in parallel on `slurm` is problematic because 
unpickling/compiling of the `Stan` model file automatically fills up `/tmp` and 
thus halts the whole program. The current solution is to change the environment
variable `TMPDIR` to some local directory which does not have a space limit, and
limit the number of CPUs to 5 at the same time.

- Statistical advice for A/B Testing

> http://sl8r000.github.io/ab_testing_statistics/

- Most winning A/B test results Are Illusory

> http://www.qubit.com/sites/default/files/pdf/mostwinningabtestresultsareillusory_0.pdf

- Warning signs in experimental design and interpretation

> http://norvig.com/experiment-design.html

- How to do A/B testing with early stopping correctly

> http://www.aarondefazio.com/tangentially/?p=83

- A/B tests at Airbnb

> https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7


## Command to run simulation
For example:

```python3.6 simulate.py -c 4 -f bayes_factor -k normal_shifted -m /Users/shuang/PersWorkSpace/ABTestEarlyStopping/normal_kpi_template.stan --distribution cauchy --scale 1```