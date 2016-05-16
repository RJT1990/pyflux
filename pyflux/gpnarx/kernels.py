import numpy as np
import sys
if sys.version_info < (3,):
    range = xrange

class SquaredExponential(object):

	def __init__(self, X, l, tau):

		self.X = X.transpose()
		self.Xo = X
		self.l = l
		self.tau = tau

	def K(self):
		K = np.zeros((self.X.shape[0],self.X.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,self.X.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-0.5*np.sum(np.power(self.X[i] - self.X[j],2))/np.power(self.l,2))

		K = K + np.identity(K.shape[0])*(10**-10)

		return K

	def Kstar(self,Xstar):
		K = np.zeros((self.X.shape[0],Xstar.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-0.5*np.sum(np.power(self.X[i] - Xstar[j],2))/np.power(self.l,2))

		return K

	def Kstarstar(self,Xstar):
		K = np.zeros((Xstar.shape[0],Xstar.shape[0]))

		for i in range(0,Xstar.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-0.5*np.sum(np.power(Xstar[i] - Xstar[j],2))/np.power(self.l,2))

		return K

class ARD(object):

	def __init__(self, X, l, tau):

		self.X = X.transpose()
		self.Xo = X
		self.l = l
		self.tau = tau

	def K(self):
		K = np.zeros((self.X.shape[0],self.X.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,self.X.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-0.5*np.sum(np.power(self.X[i] - self.X[j],2)/np.power(self.l,2)))

		K = K + np.identity(K.shape[0])*(10**-10)

		return K

	def Kstar(self,Xstar):
		K = np.zeros((self.X.shape[0],Xstar.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-0.5*np.sum(np.power(self.X[i] - self.X[j],2)/np.power(self.l,2)))

		return K

	def Kstarstar(self,Xstar):
		K = np.zeros((Xstar.shape[0],Xstar.shape[0]))

		for i in range(0,Xstar.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-0.5*np.sum(np.power(self.X[i] - self.X[j],2)/np.power(self.l,2)))

		return K


class RationalQuadratic(object):

	def __init__(self, X, l, a, tau):

		self.X = X.transpose()
		self.Xo = X
		self.l = l
		self.a = a
		self.tau = tau

	def K(self):
		K = np.zeros((self.X.shape[0],self.X.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,self.X.shape[0]):
				K[i,j] = self.tau*(1.0 + 0.5*np.sum(np.power(self.X[i] - self.X[j],2))/((self.l**2)*self.a))**-self.a

		K = K + np.identity(K.shape[0])*(10**-10)

		return K

	def Kstar(self,Xstar):
		K = np.zeros((self.X.shape[0],Xstar.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,Xstar.shape[0]):
				K[i,j] = self.tau*(1.0 + 0.5*np.sum(np.power(self.X[i] - Xstar[j],2))/((self.l**2)*self.a))**-self.a

		return K

	def Kstarstar(self,Xstar):
		K = np.zeros((Xstar.shape[0],Xstar.shape[0]))

		for i in range(0,Xstar.shape[0]):
			for j in range(0,Xstar.shape[0]):
				K[i,j] = self.tau*(1.0 + 0.5*np.sum(np.power(Xstar[i] - Xstar[j],2))/((self.l**2)*self.a))**-self.a

		return K


class OrnsteinUhlenbeck(object):

	def __init__(self, X, l, tau):

		self.X = X.transpose()
		self.Xo = X
		self.l = l
		self.tau = tau

	def K(self):
		K = np.zeros((self.X.shape[0],self.X.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,self.X.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-np.sum(np.abs(self.X[i] - self.X[j]))/self.l)

		K = K + np.identity(K.shape[0])*(10**-10)

		return K

	def Kstar(self,Xstar):
		K = np.zeros((self.X.shape[0],Xstar.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-np.sum(np.abs(self.X[i] - Xstar[j]))/self.l)

		return K

	def Kstarstar(self,Xstar):
		K = np.zeros((Xstar.shape[0],Xstar.shape[0]))

		for i in range(0,Xstar.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-np.sum(np.abs(Xstar[i] - Xstar[j]))/self.l)

		return K

class Periodic(object):

	def __init__(self, X, l, tau):

		self.X = X.transpose()
		self.Xo = X
		self.l = l
		self.tau = tau

	def K(self):
		K = np.zeros((self.X.shape[0],self.X.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,self.X.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-np.sum(np.power(-2.0*np.sin(self.X[i] - self.X[j]),2))/self.l)

		K = K + np.identity(K.shape[0])*(10**-10)

		return K

	def Kstar(self,Xstar):
		K = np.zeros((self.X.shape[0],Xstar.shape[0]))

		for i in range(0,self.X.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-np.sum(np.power(-2.0*np.sin(self.X[i] - self.X[j]),2))/self.l)

		return K

	def Kstarstar(self,Xstar):
		K = np.zeros((Xstar.shape[0],Xstar.shape[0]))

		for i in range(0,Xstar.shape[0]):
			for j in range(0,Xstar.shape[0]):
				# Due to Toeplitz structure
				if i == j:
					K[i,j] = self.tau
				else:
					K[i,j] = self.tau*np.exp(-np.sum(np.power(-2.0*np.sin(self.X[i] - self.X[j]),2))/self.l)

		return K

