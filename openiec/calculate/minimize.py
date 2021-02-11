"""
Resolve the interfacial equilibrium condition in Kaptay’s models.
"""

from openiec.utils.makemultigrid import makemultigrid
from scipy.optimize import minimize
from math import fabs
import numpy as np
from functools import reduce


def SearchEquilibrium(objectfunction, limit, dx, bulkX=None):
    """
    Search initial values of the nonlinear optimization.

    Parameters
    -----------
    objectfunction: function
        The function of calculating partial interfacial energies of components.
    limit: list
        The composition range of the searched interfacial composition.
        The composition range usually dosen't exceed maximum composition range of two-phase region.
    dx: float
        The step of searching initial interfacial equilibrium composition.
    """
    xs = makemultigrid(len(dx), [int(1.0 / dxi) for dxi in dx])

    pn, xn = len(xs), len(xs[0])
    for i in range(pn):
        p = xs[i]
        for j in range(xn):
            xs[i][j] = (limit[j][1] - limit[j][0]) * p[j] + limit[j][0]

    cons = lambda ys: reduce(lambda x, y: x + y, ys) < 1.0
    tmp = [each for each in xs if cons(each)]
    if (bulkX != None):
        def cons2(x):
            sum1=0.0
            sum2=0.0
            for i in range(xn):
                sum1 += (x[i]-bulkX[i][0])**2
                sum2 += (x[i]-bulkX[i][1])**2
            return min(sum1,sum2)>1E-4
        # avoid values too close to the bulk phase compositions
        tmp = [each for each in tmp if cons2(each)]
    # remove duplicates (in a very inefficient way...)
    xs = [tmp.pop(0), ]
    for value in tmp:
        if (np.linalg.norm(np.array(xs[-1]) - np.array(value), np.inf) < 1E-5):
            continue
        xs.append(value)
    #print(len(xs))
    #print(xs)

    vs = [objectfunction(x) for x in xs]

    index = np.argmin(vs)

    return {"index": index, "x": xs[index], "vmin": vs[index]}

def SearchEquilibriumAlongLine(objectfunction, limit, dx):
    alpha = np.linspace(0.0, 1.0, int(1.0/dx))
    pn, xn = len(alpha), len(limit)
    xs = []
    for i in range(pn):
        p = alpha[i]
        xs.append([ (limit[j][1] - limit[j][0]) * p + limit[j][0] for j in range(xn) ])
    #print(xs)
    vs = [objectfunction(x) for x in xs]
    index = np.argmin(vs)
    return {"index": index, "x": xs[index], "vmin": vs[index]}

def ComputeEquilibrium(objectfunction, x0, method="Nelder-Mead", tol=1e-10):
    """
    Optimize searched initial values of the nonlinear optimization.
    This program uses the scipy.optimize package. For more imformation visit https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html.

    Parameters
    -----------
    objectfunction: function
        The function of calculating partial interfacial energies of components.
    x0: list
        Searched interfacial equilibrium composition.
    method: str or callable
        Minimization algorithm.
        Default minimization algorithm in this program is "Nelder-Mead".
    tol : float
        Tolerance for termination. For detailed control, use solver-specific options.
    """

    res = minimize(objectfunction, x0, method=method, tol=tol, options={'disp': True})

    return res.x

