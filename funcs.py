def ExRK(inits, funcs, h, steps, method='', tab=[]):
    
    '''
    *Explicit Runge-Kutta Methods*
    
    
    This code is capable of solving any system of first order ODEs (IVPs).
    
    In case you have an Nth-order ODE, by introducing the new variables Z1=Y, Z2=(d/dx)Y, ... and ZN=(d^N/dX^N)Y 
    simply convert it to N first-order ODEs.
    
    You merely have to follow this syntax:


    The general form of such a system is as follows.

    dZ1/dx = f1(x, [Z1, Z2, ..., ZN]); Z1(x=x0) = Z10
    dZ2/dx = f2(x, [Z1, Z2, ..., ZN]); Z2(x=x0) = Z20
    .
    .
    .
    dZN/dx = fN(x, [Z1, Z2, ..., ZN]); ZN(x=x0) = ZN0

    Store the initial values in the following manner.

    inits = [x0, [Z10, Z20, ..., ZN0]]


    For example, the equation of motion of a damped oscillation is
    
    (d^2/dt^2)X = -(k/m)X - (c/m)(d/dt)X.
    
    If Z1 = Y and Z2 = (d/dx)Y (x = t & Y = X),
    
    dZ1/dx = Z2 and dZ2/dx = -(k/m)Z1 - (c/m)Z2,
    
    which should be written as
    
    def f1(x, zees):
        return zees[1]
    def f2(x, zees):
        return -(k/m)*zees[0] - (c/m)*zees[1]


    Arguments passed to 'ExRK':
    
    funcs = [f1, ..., fN]
    h := step size
    steps := number of steps
    method = 'Euler', 'Midpoint', 'Heun' or 'RK4' (predefined tableaus)
    tab = [[a_1, ..., a_n], [[b21], [b31, b32], ..., [bn1, ..., bnn-1]], [c_1, ..., c_n]]

    'tab' refers to Butcher tableaux of explicit Runge-Kutta methods:
    
    a_1 #
    a_2 # b21
    a_3 # b31 b32
    .   # .
    .   # .
    .   # .
    a_n # bn1 bn2 . . . bnn-1
    ###########################
        # c_1 c_2 c_3 . . . c_n
        #
    '''
    
    if method=='Euler':
        tab = [[0], [], [1]]
        
    elif method=='Midpoint':
        tab = [[0, 0.5], [[0.5]], [0, 1]]
        
    elif method=='Heun':
        tab = [[0, 1], [[1]], [0.5, 0.5]]
        
    elif method=='RK4':
        tab = [[0, 0.5, 0.5, 1], [[0.5], [0, 0.5], [0, 0, 1]], [1/6, 1/3, 1/3, 1/6]]

    m = len(funcs)  # number of first-order equations ought to be solved
    n = len(tab[0])  # number of tangents (points) used in estimating the derivative
        
    data = [[inits[0]]*(steps+1), [[inits[1][i]]*(steps+1) for i in range(m)]]  # for storing the solutions
    
    for i in range(steps):  # the main loop (solving the equations)
        
        data[0][i+1] = data[0][i] + h  # updating the independent variable
        
        kays = [[0]*m]*n  # calculating all the tangents (Ks) needed and storing them in 'kays'
        for j in range(n):
            for k in range(m):
                kays[j][k] = funcs[k](
                    data[0][i] + h*tab[0][j], 
                    [data[1][l][i] + h*sum([tab[1][j-1][o]*kays[o][l] for o in range(j)]) for l in range(m)])
        
        for j in range(m):  # updating the dependent variables
            data[1][j][i+1] = data[1][j][i] + h*sum([tab[2][k]*kays[k][j] for k in range(n)])
        
    return data

###

def RKF45(inits, n, func, e=1e-10):
    
    x = [inits[0]]*(n+1)
    y = [inits[1]]*(n+1)
    
    coeffs = [[0, 2/9, 1/3, 3/4, 1, 5/6], 
              [[], [2/9], [1/12, 1/4], [69/128, -243/128, 135/64], [-17/12, 27/4, -27/5, 16/15], 
                        [65/432, -5/16, 13/16, 4/27, 5/144]], 
              [47/450, 0, 12/25, 32/225, 1/30, 6/25], [-1/150, 0, 3/100, -16/75, -1/20, 6/25]]
    
    kays = [0]*6
    
    h = 1e-3
    
    for i in range(n):

        while True:
        
            for j in range(6):
                kays[j] = func(x[i] + coeffs[0][j]*h, 
                                y[i] + sum([coeffs[1][j][l]*kays[l]*h for l in range(j)]))
            
            error = abs(h*sum([coeffs[3][j]*kays[j] for j in range(6)]))
            
            if error>e:
                h = 0.9*h*(e/error)**0.2
            else:
                break
            
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h*sum([coeffs[2][j]*kays[j] for j in range(6)])
        
        h = 0.9*h*(e/error)**0.2
        
    return x, y