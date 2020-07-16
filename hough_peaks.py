import numpy as np


def hough_peaks(H, Npeaks=1, Threshold=0, NHoodSize=[0, 0]):
    # Find peaks in a Hough accumulator array.
    #
    # Threshold (optional): Threshold at which values of H
    #   are considered to be peaks
    # NHoodSize (optional): Size of the suppression neighborhood, [M N]
    #
    # Please see the Matlab documentation for houghpeaks():
    # http://www.mathworks.com/help/images/ref/houghpeaks.html
    # Your code should imitate the matlab implementation.

    # Parse input arguments
    # addOptional(p, 'numpeaks', 1, @isnumeric);
    # addParameter(p, 'Threshold', 0.5 * max(H(:)));
    # addParameter(p, 'NHoodSize', floor(size(H) / 100.0) * 2 + 1)
    # odd values >= size(H)/50
    # parse(p, varargin{:});

    # TODO: Your code here
    if(Threshold == 0):
        Threshold = 0.5*np.max(H)
    if(NHoodSize == [0, 0]):
        NHoodSize = np.array(H.shape)/50
        NHoodSize = np.maximum(2*np.ceil(NHoodSize/2)+1, 1)
    else:
        NHoodSize = np.array(NHoodSize)
    ReturnPar = []
    H = np.where(H < Threshold, 0, H)

    if(Npeaks == 0 or not np.any(H)):
        return ReturnPar

    for i in range(0, Npeaks):
        index = np.unravel_index(np.argmax(H, axis=None), H.shape)
        ReturnPar.append(index)
        # Size of the suppression neighborhood
        ylow = int(np.maximum(index[0]-NHoodSize[0], 0))
        yhigh = int(np.minimum(index[0]+NHoodSize[0]+1, H.shape[0]))
        xlow = int(np.maximum(index[1]-NHoodSize[1], 0))
        xhigh = int(np.minimum(index[1]+NHoodSize[1]+1, H.shape[1]))
        H[ylow:yhigh, xlow:xhigh] = 0

        # If No further Non-zeor rows break
        if(not np.any(H)):
            break

    return ReturnPar
