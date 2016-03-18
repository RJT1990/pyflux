import scipy.stats as ss

def find_p_value(z):
	p_value = 0
	if z>=0:
		p_value += (1-ss.norm.cdf(z,loc=0,scale=1))
		p_value += (ss.norm.cdf(-z,loc=0,scale=1))
	else:
		p_value += (1-ss.norm.cdf(-z,loc=0,scale=1))
		p_value += (ss.norm.cdf(z,loc=0,scale=1))
	return p_value