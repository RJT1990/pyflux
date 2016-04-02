from math import exp

def lik_score(rv,mean,var,dist):
	if dist == "Laplace":
		if float(var*abs(rv-mean)) == 0:
			return 0
		else:
			return float(rv-mean)/float(var*abs(rv-mean))

	elif dist == "Normal":
		if (rv-mean) == 0:
			return 0
		else:
			return (rv-mean)

	elif dist == "Poisson":
		if float(mean) == 0:
			return 0
		else:
			return float(rv-mean)/float(mean)

	elif dist == "Exponential":
		return 1.0 - mean*rv

	elif dist == "Beta-t":
		"""
		return (((var+1.0)*(rv**2.0))/float(var*exp(mean) + (rv**2))) - 1.0
		"""
		
		try:
			return (((var+1.0)*(rv**2))/float(var*exp(mean) + (rv**2))) - 1.0
		except:
			return -1.0
