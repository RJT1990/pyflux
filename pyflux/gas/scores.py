from math import exp
import numpy as np

def lik_score(rv,mean,var,df,dist):
	if dist == "Laplace":
		return (rv-mean)/float(var*np.abs(rv-mean))

	elif dist == "Normal":
		if (rv-mean) == 0:
			return 0.0
		else:
			return (rv-mean)

	elif dist == "Poisson":
		if float(mean) == 0:
			return 0.0
		else:
			return float(rv-mean)/float(mean)

	elif dist == "Exponential":
		return 1.0 - mean*rv

	elif dist == 't':
		return (rv-mean)/float(1+(((rv-mean)**2)/(df*exp(-2.0*var))))

	elif dist == "Beta-t":
		try:
			return (((var+1.0)*(rv**2))/float(var*exp(mean) + (rv**2))) - 1.0
		except:
			return -1.0
