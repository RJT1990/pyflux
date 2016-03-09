
def lik_score(rv,mean,var,dist):
	if dist == "Laplace":
		if float(var*abs(rv-mean)) == 0:
			return 0
		else:
			return float(rv-mean)/float(var*abs(rv-mean))