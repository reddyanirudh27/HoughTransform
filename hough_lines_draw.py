import math
import cv2 as cv2
import numpy as np


def hough_lines_draw(img,  peaks, rho, theta):
    # Draw lines found in an image using Hough transform.
    #
    # img: Image on top of which to draw lines
    # outfile: Output image filename to save plot as
    # peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    # rho: Vector of rho values, in pixels
    # theta: Vector of theta values, in degrees

    # TODO: Your code here

    # imgout = np.zeros([img.shape[0], img.shape[1], 3])
    # imgout[:, :, 0] = img
    # imgout[:, :, 1] = img
    # imgout[:, :, 2] = img

    for peak in peaks:
        d0, angle = rho[peak[0]], theta[peak[1]]
        a = math.cos(angle*math.pi/180)
        b = math.sin(angle*math.pi/180)
        x0 = a * d0
        y0 = b * d0
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    return(img)
