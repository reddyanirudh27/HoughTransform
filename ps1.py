# ps1
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hough_lines_acc
import hough_peaks
import hough_lines_draw
import timeit


def Sobel(img, depth=cv2.CV_64F, diffx=1, diffy=1, size=3,
          border=cv2.BORDER_REPLICATE):
    img_edgesx = cv2.Sobel(img, ddepth=depth, dx=diffx, dy=0,
                           ksize=size,
                           borderType=border)
    img_edgesy = cv2.Sobel(img, ddepth=depth, dx=0, dy=diffy,
                           ksize=size,
                           borderType=border)
    img_edges = np.sqrt(np.square(img_edgesx) +
                        np.square(img_edgesy))

    return img_edges


# Problem 1 and 2
img = cv2.imread('input\\ps1-input0.png', 0)  # already grayscale
img = np.array(img, dtype=np.float64)


img_edges = Sobel(img)
img_edges = np.where(img_edges < 10, 0, 255).astype(np.uint8)
cv2.imwrite("output\\ps1-1-a-1.png", img_edges)

start = timeit.default_timer()

H, Rho, theta = hough_lines_acc.hough_lines_acc(img_edges, 1, 0.25)
Peaks = hough_peaks.hough_peaks(H, 10)
Hsave = cv2.cvtColor(H.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for peak in Peaks:
    x0 = peak[1]
    y0 = peak[0]
    cv2.circle(Hsave, (x0, y0), 6, (0, 255, 0), -1)
cv2.imwrite("output\\ps1-2-a-1.png", Hsave)

imgin = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
imgout = hough_lines_draw.hough_lines_draw(imgin, Peaks, Rho, theta)
cv2.imwrite("output\\ps1-2-b-1.png", imgout)
print(f"Hough calculation time {timeit.default_timer()-start}")


# Problem 3 and 4
imgnoise = cv2.imread('input\\ps1-input0-noise.png', 0)
imgnoise = np.array(imgnoise, dtype=np.float64)
img_gaus = cv2.GaussianBlur(imgnoise, ksize=(25, 25),
                            sigmaX=3, sigmaY=3,
                            borderType=cv2.BORDER_REPLICATE)
img_edges = Sobel(img_gaus)
img_edges = np.where(img_edges < 25, 0, 255).astype(np.uint8)
cv2.imwrite("output\\ps1-3-a-1.png", img_edges)

H, Rho, theta = hough_lines_acc.hough_lines_acc(img_edges, 1.1, 2.1)
Peaks = hough_peaks.hough_peaks(H, 10, 230, [25, 25])
Hsave = cv2.cvtColor(H.astype(np.uint8), cv2.COLOR_GRAY2BGR)
for peak in Peaks:
    x0 = peak[1]
    y0 = peak[0]
    cv2.circle(Hsave, (x0, y0), 6, (0, 255, 0), -1)
cv2.imwrite("output\\ps1-3-b-1.png", Hsave)


imgin = cv2.cvtColor(imgnoise.astype(np.uint8), cv2.COLOR_GRAY2BGR)
imgout = hough_lines_draw.hough_lines_draw(imgin, Peaks, Rho, theta)
cv2.imwrite("output\\ps1-3-c-1.png", imgout)

fig, (ax1, ax2) = plt.subplots(1, 2)
Peaks = np.array(Peaks)
ax1.imshow(img_edges, cmap='gray')
ax2.imshow(imgout, cmap='gray')
plt.show()
# cv2.namedWindow('output', cv2.WINDOW_NORMAL)
# cv2.imshow('output', imgout)
# cv2.waitKey(0)
# plt.gca().set_xticks((np.linspace(-90, 90, 180)))
# ax.set_xticks(np.linspace(np.min(theta), np.max(theta), 10))
# ax.set_yticks(np.linspace(np.min(Rho), np.max(Rho), 10))


# img_edges
# imwrite(img_edges, fullfile('output', 'ps1-1-a-1.png'))

# 2-a
# [H, theta, rho] = hough_lines_acc(img_edges)  # defined in hough_lines_acc.m
# TODO: Plot/show accumulator array H, save as output/ps1-2-a-1.png

# 2-b
# peaks = hough_peaks(H, 10)  # defined in hough_peaks.m
# TODO: Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png

# TODO: Rest of your code here
