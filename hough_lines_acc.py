import numpy as np


def hough_lines_acc(BW, rhoStep=1, thetaRes=1):
    # Compute Hough accumulator array for finding lines.
    #
    # BW: Binary (black and white) image containing edge pixels
    # RhoResolution (optional): Difference between successive rho values,
    # in pixels
    # Theta (optional): Vector of theta values to use, in degrees
    #
    # Please see the Matlab documentation for hough():
    # http://www.mathworks.com/help/images/ref/hough.html
    # Your code should imitate the Matlab implementation.
    #
    # Pay close attention to the coordinate system specified in the assignment.
    # Note: Rows of H should correspond to values of rho,
    # columns those of theta.

    # Parse input arguments

    # TODO: Your code here
    maxD = ((BW.shape[0]-1)**2 + (BW.shape[1]-1)**2)**0.5
    RhoSize = 2*int(np.ceil(maxD/rhoStep))+1
    diagonal = int(rhoStep*np.ceil(maxD/rhoStep))

    theta = (np.arange(-90, 89+thetaRes, thetaRes))*np.pi/180
    H = np.zeros([RhoSize, theta.shape[0]])

    YX = np.argwhere(BW > 0)
    for yx in YX:
        y = yx[0]
        x = yx[1]
        if(BW[y, x] > 0):
            d = x*np.cos(theta)+y*np.sin(theta)
            dcon = np.round((d+diagonal)/rhoStep).astype(np.int64)
            for index, val in enumerate(theta):
                H[dcon[index], index] = H[dcon[index], index] + 1
    Rho = np.arange(-1*diagonal, diagonal+rhoStep, rhoStep)
    theta = theta*180/np.pi
    return H, Rho, theta
