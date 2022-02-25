import sklearn as sk
import numpy as np
import scipy as sp
import statsmodels.api as sm
import pylab as plt
from matplotlib.colors import ListedColormap

#src: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
class LinearRegression(sk.linear_model.LinearRegression):

    def __init__(self,*args,**kwargs):
        # *args is the list of arguments that might go into the LinearRegression object
        # that we don't know about and don't want to have to deal with. Similarly, **kwargs
        # is a dictionary of key words and values that might also need to go into the orginal
        # LinearRegression object. We put *args and **kwargs so that we don't have to look
        # these up and write them down explicitly here. Nice and easy.

        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False

        super(LinearRegression,self).__init__(*args,**kwargs)

    # Adding in t-statistics for the coefficients.
    def fit(self,x,y):
        # This takes in numpy arrays (not matrices). Also assumes you are leaving out the column
        # of constants.

        # Not totally sure what 'super' does here and why you redefine self...
        self = super(LinearRegression, self).fit(x,y)
        n, k = x.shape
        yHat = np.matrix(self.predict(x)).T

        # Change X and Y into numpy matricies. x also has a column of ones added to it.
        x = np.hstack((np.ones((n,1)),np.matrix(x)))
        y = np.matrix(y).T

        # Degrees of freedom.
        df = float(n-k-1)

        # Sample variance.
        sse = np.sum(np.square(yHat - y),axis=0)
        self.sampleVariance = sse/df

        # Sample variance for x.
        self.sampleVarianceX = x.T*x

        # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
        self.covarianceMatrix = sp.linalg.sqrtm(self.sampleVariance[0,0]*self.sampleVarianceX.I)

        # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
        self.se = self.covarianceMatrix.diagonal()[1:]

        # T statistic for each beta.
        self.betasTStat = np.zeros(len(self.se))
        for i in range(len(self.se)):
            print(self.coef_.shape)
            self.betasTStat[i] = self.coef_[0]/self.se[i]

        # P-value for each beta. This is a two sided t-test, since the betas can be
        # positive or negative.
        self.betasPValue = 1 - sp.stats.t.cdf(abs(self.betasTStat),df)

def make_plots(num_networks, syncs, maxes, means, times, nat_freqs):
    x = np.ones((num_networks, 3))
    y = np.ones((num_networks, 3))

    x[:, 0:3] = (1, 0, 0)
    y[:, 0:3] = (0, 1, 0)
    c = np.linspace(0, 1, num_networks)[:, None]
    gradient = x + (y - x) * c #make a color gradient for datapoints based on the timeslice used to make the network

    new_cmap = ListedColormap(gradient, name = 'time of network')


    cmap = plt.cm.viridis
    color = [cmap(float(i + 1) / float(num_networks)) for i in range(num_networks)]
    norm = None
    mapping = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    syncs_reshaped = syncs.reshape(-1,1)
    X = sm.add_constant(syncs_reshaped)
    est = sm.OLS(maxes, X)
    est2 = est.fit()

    print(est2.summary())
    dummy_x = np.array([-0.2, 0.4, 0.65]).reshape(-1,1) #dummy input used for plotting line of best-fit
    dummy_x = sm.add_constant(dummy_x)
    line = est2.predict(dummy_x)
    print(line, "LINE")
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.scatter(syncs, maxes, c = color, label = 'max', s= 25)
    plt.rcParams.update({'font.size': 16})
    plt.plot(dummy_x, line, color = 'orange')
    plt.xlabel('Synchronizability')
    plt.ylabel('Max Order Parameter')
    plt.xlim([-0.1,0.65])
    plt.colorbar(mapping)
    plt.tight_layout()
    #plt.legend()

    est = sm.OLS(times, X)
    est2 = est.fit()

    print(est2.summary())
    line = est2.predict(dummy_x)
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.scatter(syncs, times, c = color, label = 'time', s=25)
    plt.xlabel('Synchronizability')
    plt.ylabel('Time to Max Order Param.')
    plt.xlim([-0.1,0.65])
    plt.colorbar(mapping)
    plt.tight_layout()

    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.scatter(syncs, times, c=color, label='time', s=25)
    plt.plot(dummy_x, line, color = 'orange')
    plt.xlabel('Synchronizability')
    plt.ylabel('Time to Max Order Param.')
    plt.xlim([-0.1,0.65])
    plt.colorbar(mapping)
    plt.tight_layout()
    #plt.legend()

    est = sm.OLS(means, X)
    est2 = est.fit()

    print(est2.summary())
    line = est2.predict(dummy_x)
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.scatter(syncs, means, c = color, label = 'mean', s = 25)
    plt.rcParams.update({'font.size': 16})
    plt.plot(dummy_x, line, color='orange')
    plt.xlabel('Synchronizability')
    plt.ylabel('Mean Order Parameter')
    plt.xlim([-0.1,0.65])
    plt.colorbar(mapping)
    plt.tight_layout()
    #plt.legend()

    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.hist(nat_freqs, bins = 20)
    plt.show()