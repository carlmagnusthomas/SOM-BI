

# set the colormap and centre the colorbar
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def fldmean_xr(xr_da):
    """
    Calculate the area averaged mean from an xarray DataArray with coordinates time, latitude, longitude
    will need to be adapted if data is not 3d e.g. if there are other levels
    """
    lat_str, lon_str = "latitude", "longitude"
    import numpy as np
    import xarray as xr
    #print(lat_str)
    
    #multiply the ataarray by the relevant weights using cos(latitude)
    xr_da_latcorr = (xr_da*np.cos(np.radians(xr_da[lat_str])))    
    #calculate the weighted mean by summing across all the weighted values and dividing by the total sum of all weights appliced
    if xr.ufuncs.isnan(xr_da[0]).sum() > 0:
        print("Masked")
        #calculating the weighted mean for a masked array
        isnan = xr.ufuncs.isnan(xr_da[0])
        xr_da_latcorr = (xr_da*np.cos(np.radians(xr_da[lat_str])))
        weights_sum = ((isnan==0)*np.cos(np.radians(xr_da[lat_str]))).sum()
        xr_da_fldmean_latcorr = xr_da_latcorr.sum(axis=1, skipna=True).sum(axis=1, skipna=True)/weights_sum
    
    
    if len(xr_da.shape) == 3:
        xr_da_fldmean_latcorr = xr_da_latcorr.sum(axis=1).sum(axis=1)/(np.cos(np.radians(xr_da_latcorr[lat_str])).sum()*xr_da_latcorr[lon_str].shape[0])
    if len(xr_da.shape) == 2:
        xr_da_fldmean_latcorr = xr_da_latcorr.sum(axis=1).sum(axis=0)/(np.cos(np.radians(xr_da_latcorr[lat_str])).sum()*xr_da_latcorr[lon_str].shape[0])        
    return xr_da_fldmean_latcorr


def get_cols_for_mdls_comb(mdls_arr_sin_len, darkness_factor, colormap_str):
    """
    Obtain a set of colours for a given number of models under investigation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.cm as cm
    class MplColorHelper:
        def __init__(self, cmap_name, start_val, stop_val):
            self.cmap_name = cmap_name
            self.cmap = plt.get_cmap(cmap_name)
            self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
            self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        def get_rgb(self, val):
            return self.scalarMap.to_rgba(val)    
    
    
    #identify the number of emsemble members this model has
    ens_count = mdls_arr_sin_len#len(np.unique(mdls_arr_sin))
    COL = MplColorHelper(colormap_str, 0, ens_count)
    bounds = np.arange(0,ens_count,1)
    mdl_cols=COL.get_rgb(bounds)[:,:3] # this removes the alpha values    
    #from the primary colours for each model, create different shades and add these to a new array
    mdl_ens_cols = np.zeros((ens_count,3))
    mdl_ens_cols=COL.get_rgb(bounds)[:,:3]*darkness_factor
    return mdl_ens_cols


def lag1corr(X): 
    """Compute the lag-1 autocorrelation coefficient for time series X"""
    n = len(X)
    meanx = np.mean(X)
    num = 0
    den = 0
    for i in range(n-1):
        num += (X[i+1] - meanx) * (X[i] - meanx)
    for i in range(n): 
        den += (X[i] - meanx)**2
    r = num/den # lag-1 correlation coefficient
    return r

def ESS(X): 
    """X is a time series
    Output is the equivalent sample size as suggested by Zwiers and von Storch
    """
    n = len(X)
    ## compute the sample lag-1 correlation coefficient
    r = lag1corr(X)
    n_e = n*(1-r)/(1+r)
    # n_e = n*(1-abs(r))/(1+abs(r))  # use the absolute calue of lag-1 correlation coefficient
    # estimated equivalent sample size
    if n_e <= 2: 
        eess = 2
    elif 2<n_e<=n: 
        eess = n_e
    else: 
        eess = n
    return eess

def RSS(y, yhat): 
    """y are the true values of response variable
    yhat are the predicted values of response variable
    The output is the residual sum of squares"""
    N = len(y)
    rss = 0
    for i in range(N): 
        rss += (y[i]-yhat[i])**2
    return rss

def lmtrendtest(x, y): 
    import scipy
    """ x is the time, y is the observation
    Output: test statistic, p-value using standard Normal, p-value using student t 
    for the slop of linear least squares regression"""
    
    # fit a linear regression
    m, c, r, p, err = scipy.stats.linregress(x,y) # slope, intercept, correlation, pvalue, std error
    # make a design matrix
    N = len(y)
    x0 = np.ones(N)
    X = np.column_stack((x0,x)) # design matrix with two columns, intercept & time
    # calculate the test statistic
    a = np.linalg.inv(np.matmul(X.transpose(),X))[1,1]
    ess = ESS(y) # equivalent sample size
    yhat = [c+i*m for i in x]
    rss = RSS(y, yhat) # residual sum of squares
    t = m / (np.sqrt(a*rss/(ess-2)))
    # p-value using standard normal (two tail test)
    p_value_normal = scipy.stats.norm.sf(abs(t))*2
    # p-value using student t (two tail)
    p_value_student = stats.t.sf(np.abs(t), ess-2)*2
    # lag1 autocorrelation coefficient
    rho = lag1corr(y)
    return p_value_student #rho, p, 
