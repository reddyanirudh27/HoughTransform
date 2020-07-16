import math
import cv2 as cv2


def hough_lines_draw(img,  peaks, rho, theta):

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
