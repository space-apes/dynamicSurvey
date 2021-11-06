"""
NAME: mathtools.py
AUTHOR: Brian Reza Smith
INSTITUTION: California State University Dominguez Hills
YEAR: 2021
DESCRIPTION:
	definition of assorted math functions used in implementation
	of RBM machine learning model

"""
from math import exp, pi, sqrt

def logistic(lnOdds):
        """
        converts logOdds to probability
                log odds of independent events is product of them

        @param float lnOdds
                natural log of ratio of probability event happens / probability it does not happen

        @return float probability
                probability of outcome such that 0 <= probability <= 1
        """
        return 1.0 / (1.0 + (exp(lnOdds)**-1) )


def univariateGaussianCurve(m, sd, x):
	""" calculates value on gaussian curve given mean, standard deviation, x-value 

	@param float m mean value of distribution
	@param float sd standard deviation of distribution
	@param float x value for sample 

	@return float y value on curve, given parameters
	"""
	return (1/(sqrt(2*pi*sd**2))) * exp(-(x-m)**2/(2*sd**2))

