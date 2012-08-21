import pysb.bng
import numpy 
import sympy 
import ctypes
import csv
import scipy.interpolate
from pysb.integrate import odesolve

def annealrun():
    pass

def compare_data(xparray, simarray, axispairlist, vardata=False):
    """
    Compares two arrays of different size and returns the mean square
    difference between them with an objective function:
    
                            1
     obj(t, params) = -------------(S_sim(t,params)-S_exp(t))^2
                      2*sigma_exp^2
        
    This function uses the experimental X axis as the unit to re-grid both arrays.
    
    xparray: array of experimental data. The format of this array is
    (X, Y1, Y2, Y3...) for arrays without variance data or (X, Y1,
    sigma1, Y2, Sigma2, Y3, sigma3...) for arrays with variance
    data. If variance is included in this array it should contain a
    variance value for each time point.
    
    simarray: array of simulation data returned from
    pysb.integrate.odesolve.
    
    axispairlist: a list of tuple pairs that indicates which
    experimental axis corresponds with which simulation
    axis. E.g. [(2,1), (5,2), (8,3)] would indicate that the second
    axis in the experimental array and the first axis in the
    simulation array should be compared, the fifth axis in the
    experimental array and the second axis in the simulation array
    should be compared, and the eighth axis in the experimental array
    and the third axis in the simulation array should be compared.
    
    vardata: Whether variance data for the experiments is included in
    the xparray. If vardata is set to *True*, the axis following the
    "data" axis will be considered the variance axis. If it is set to
    *False* a coefficient of variation (sigma/mean) of .25 will be
    assumed (See Chen, Gaudet reference for a justification of this
    value). 
    
    Note: the x-axis (time) is assumed to be roughly the same for both
    the experimental data and the simulation data. The shortest time
    from the experiment or simulations will be taken as reference to
    regrid the data. The regridding is done using a b-spline
    interpolation.
    """
    
    # FIXME: This prob should figure out the overlap of the two arrays
    # and get a spline of the overlap. For now just assume the
    # simarray domain is bigger than the xparray.
    
    ipsimarray = numpy.zeros(xparray.shape[1])
    objout = 0
   
    for i in range(len(xspairlist)):
        # some error checking
        assert type(xspairlist[i]) is tuple
        assert len(xspairlist[i]) == 2
        
        xparrayaxis = xspairlist[i][0]
        simarrayaxis = xspairlist[i][1]

        # get the spline representation of the curve
        tck = scipy.interpolate.splrep(simarray[0], simarray[simarrayaxis])
        # evaluate the b-spline and get the interpolated values at the xp x-axis values
        ipsimarray = scipy.interpolate.splev(xparray[0], tck) #xp x-coordinate values to extract from y splines
        
        # calculate the objective function
        diffarray = ipsimarray - xparray[xparrayaxis]
        diffsqarray = diffarray * diffarray

        if vardata is True:
            xparrayvar = xparray[xparrayaxis+1] # variance data provided in xparray in next column
        else:
            # assume a default variance
            xparrayvar = numpy.ones(xparray.shape[1])
            xparrayvar = xparray[xparrayaxis]*.25 #assume a coeff of variation of .25 = sigma/mean (from Chen, Gaudet...)
            # FIXME: Remove any zeros in the variance array to avoid infs... this is a cheap fix
            for i in range(0, len(xparrayvar)):
                if xparrayvar[i] == 0:
                    xparrayvar[i] = 1

        xparrayvar = xparrayvar*2.0
        #numpy.seterr(divide='ignore')
        objarray = diffsqarray / xparrayvar

        # check for inf in objarray, they creep up when there are near zero or zero values in xparrayvar
        for i in range(len(objarray)):
            if numpy.isinf(objarray[i]) or numpy.isnan(objarray[i]):
                #print "CORRECTING NAN OR INF. IN ARRAY"
                # print objarray
                objarray[i] = 1e-100 #zero enough

        objout += objarray.sum()

    print "OBJOUT(total):", objout
    return objout

def logparambounds(model, omag=1, useparams=[], usemag=None, initparams=[], initmag=None):
    """
    DOCUMENTATION HERE
    """
    params = []
    for i in model.params:
        params.append(i.value)
    params= numpy.asarray(params)

    ub = numpy.zeros(len(params))
    lb = numpy.zeros(len(params))
    # set upper/lower bounds for generic problem
    for i in range(len(params)):
        if i in useparams and i not in initparams:
            ub[i] = params[i] * pow(10,usemag)
            lb[i] = params[i] / pow(10,usemag)
        elif i in initparams:
            ub[i] = params[i] * pow(10,initmag)
            lb[i] = params[i] / pow(10,initmag)
        else:
            ub[i] = params[i] * pow(10, omag)
            lb[i] = params[i] / pow(10, omag)
    return lb, ub

def linparambounds(model, fact=.25, useparams=[], usefact=None):
    """
    DOCUMENTATION HERE
    """
    params = []
    for i in model.params:
        params.append(i.value)
    params= numpy.asarray(params)

    ub = numpy.zeros(len(params))
    lb = numpy.zeros(len(params))
    # set upper/lower bounds for generic problem
    for i in range(len(params)):
        if i in useparams:
            ub[i] = params[i] + (params[i] * fact)
            lb[i] = params[i] - (params[i] * fact)
        else:
            ub[i] = params[i] + (params[i] * usefact)
            lb[i] = params[i] - (params[i] * usefact)
    lb[numpy.where(lower<0.)] = 0.0 #make sure we don't go negative on parameters...
    return lb, ub

def mapprms(nums01, lb, ub, scaletype="log"):
    """
    given an upper bound(ub), lower bound(lb), and a sample between zero and one (zosample)
    return a set of parameters within the lb, ub range. 
    nums01: array of numbers between zero and 1
    lb: array of lower bound for each parameter
    ub: arary of upper bound for each parameter
    """
    params = numpy.zeros_like(nums01)
    if scaletype == "log":
        params = lb*(ub/lb)**nums01 # map the [0..1] array to values sampled over their omags
    elif scaletype == "lin":
        params = (nums01*(ub-lb)) + lb
    return params

def tenninetycomp(outlistnorm, arglist, xpsamples=1.0):
    """
    Determine Td and Ts. Td calculated at time when signal goes up to 10%.
    Ts calculated as signal(90%) - signal(10%). Then a chi-square is calculated.
    outlistnorm: the outlist from anneal_odesolve
    arglist: simaxis, Tdxp, varTdxp, Tsxp, varTsxp
    xpsamples
    """
    xarr = outlistnorm[0] #this assumes the first column of the array is time
    yarr = outlistnorm[arglist[0]] #the argument passed should be the axis
    Tdxp = arglist[1]
    varTdxp = arglist[2]
    Tsxp = arglist[3]
    varTsxp = arglist[4]
    
    # make a B-spine representation of the xarr and yarr
    tck = scipy.interpolate.splrep(xarr, yarr)
    t, c, k = tck
    tenpt = numpy.max(yarr) * .1 # the ten percent point in y-axis
    ntypt = numpy.max(yarr) * .9 # the 90 percent point in y-axis
    #lower the spline at the abcissa
    xten = scipy.interpolate.sproot((t, c-tenpt, k))[0]
    xnty = scipy.interpolate.sproot((t, c-ntypt, k))[0]

    #now compare w the input data, Td, and Ts
    Tdsim = xten #the Td is the point where the signal crosses 10%; should be the midpoint???
    Tssim = xnty - xten
    
    # calculate chi-sq as
    # 
    #            1                           1
    # obj = ----------(Tdsim - Tdxp)^2 + --------(Tssim - Tsxp)^2
    #       2*var_Tdxp                   2*var_Td 
    #
    obj = ((1./varTdxp) * (Tdsim - Tdxp)**2.) + ((1./varTsxp) * (Tssim - Tsxp)**2.)
    #obj *= xpsamples
    
    print "OBJOUT-10-90:(%g,%g):%g"%(Tdsim, Tssim, obj)

    return obj    

def annealfxn(zoparams, time, model, xpdata, xspairlist, lb, ub, tn = [], scaletype="log", norm=True, vardata=False, fileobj=None):
    """
    Feeder function for scipy.optimize.anneal
    zoparams: the parameters in the range [0,1) to be sampled
    time: the time scale for the simulation
    model: a PySB model object
    envlist: an environment list for the sundials integrator
    xpdata: experimental data
    xspairlist: the pairlist of the correspondence of experimental and simulation outputs
    lb: lower bound for parameters
    ub: upper bound for parameters
    tn: list of values for ten-ninety fits
    scaletype: log, linear,etc to convert zoparams to real params b/w lb and ub. default "log"
    norm: normalization on. default true
    vardata: variance data available. default "false"
    fileobj: file object to write data output. default "None"
    """

    # convert of linear values from [0,1) to desired sampling distrib
    paramarr = mapprms(zoparams, lb, ub, scaletype="log")

    # eliminate values outside the boundaries, i.e. those outside [0,1)
    if numpy.greater_equal(paramarr, lb).all() and numpy.less_equal(paramarr, ub).all():
        print "integrating... "

        # assign paramarr values to model.parameters this assumes that
        # assume the zoparams list order corresponds to the model.parameters list order
        for i,j in enumerate(model.parameters):
            j.value = paramarr[i]
        
        #update reltol/abstol here?
        outlist = odesolve(model, time)

        # normalized data needs a bit more tweaking before objfxn calculation
        if norm is True:
            print "Normalizing data"
            datamax = numpy.max(outlist[0], axis = 1)
            datamin = numpy.min(outlist[0], axis = 1)
            outlistnorm = ((outlist[0].T - datamin)/(datamax-datamin)).T
            # xpdata[0] should be time, get from original array
            outlistnorm[0] = outlist[0][0].copy()
            # xpdata here should be normalized
            objout = compare_data(xpdata, outlistnorm, xspairlist, vardata)
            if tn:
                tn = tenninetycomp(outlistnorm, tn, len(xpdata[0]))
                objout += tn 
            print "NORM objout TOT:", objout
        else:
            objout = compare_data(xpdata, outlist[0], xspairlist, vardata)
            if tn:
                tn = tenninetycomp(outlist[0], tn)
                objout += tn 
            print "objout TOT:", objout
    else:
        print "======> VALUE OUT OF BOUNDS NOTED"
        temp = numpy.where((numpy.logical_and(numpy.greater_equal(paramarr, lb), numpy.less_equal(paramarr, ub)) * 1) == 0)
        for i in temp:
            print "======>",i,"\n======", paramarr[i],"\n", zoparams[i],"\n"
        objout = 1.0e300 # the largest FP in python is 1.0e308, otherwise it is just Inf

    return objout


