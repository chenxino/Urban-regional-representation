import cv2 as cv
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time


pd.options.mode.chained_assignment = None

#简单二值化，自己设置阈值
def Two(image):
    w,h = image.shape
    size = (w,h)
    iTwo = image
    for i in range(w):
        for j in range(h):
            if image[i,j]>100:
                iTwo[i,j] = 1
            else:
                iTwo[i,j] = 0
    return iTwo


# 细化
def neighbours(x, y, image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape  # x for rows, y for columns
        for x in range(1, rows - 1):  # No. of  rows
            for y in range(1, columns - 1):  # No. of columns
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


# 膨胀细化
def map_segment(map_src, out_src):
    # img = cv.imread('E:/Program Files/jupyter_code/first/images/shang1.png',0)
    img = cv.imread(map_src, 0)
    print(type(img))
    iTwo2 = Two(img)

    #     cv.imshow("iTwo", iTwo2*255)
    #     cv.waitKey(0)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv.dilate(iTwo2, kernel, iterations=1)
    dilated_src = os.path.join(os.path.dirname(map_src), 'dilated.png')
    cv.imwrite('out/dilated.png', dilated * 255)
    img_after = cv.imread('out/dilated.png', 0)
    img_after = zhangSuen(dilated)

    cv.imwrite(out_src, img_after * 255)
    return img_after * 255

# 连通区域划分
from skimage import measure
from skimage import filters


def binarize(img):
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)



def ccl(map_src, out_src, save2disk=False):
    img = cv.imread(str(map_src), 0)
    blobs =  binarize(img)

    # kernel = np.ones((10,10), np.float32) / 25
    blobs = blobs / 255
    # all_labels = measure.label(blobs, connectivity=2)
    blobs_labels = measure.label(blobs, neighbors=8, connectivity=1, background=0)
    #connectivity=1采用四联通
    # array_dir = map_src.parent.joinpath('region_labeled.csv')
    np.savetxt('region_labeled.csv', blobs_labels, fmt='%d')

    if save2disk:
        plt.figure('ccl')
        # plt.imshow(blobs_labels, cmap='nipy_spectral')
        plt.axis('off')
        # plt.show(block=False)
        plt.imsave(out_src, blobs_labels, cmap='Spectral')
        time.sleep(3)
        plt.close()
    return blobs_labels

# def main():
#     map_segment('E:\Program Files\jupyter_code\code\images\shanghai3w.png')
#     ccl('E:\Program Files\jupyter_code\code\out\thinning.png',True)

if __name__ == '__main__':
    code='YrEE'
    print(111111)
    img = cv.imread('E://webapp//blogproject//media//out//%s//1.png '% code, 0)
    # img = cv.imread('E://webapp//blogproject//ccl.png ', 0)
    print(type(img))

    map_segment("E://webapp//blogproject//media//out//%s//1.png" % code, "E://webapp//blogproject//media//out//%s//thinning.png" % code)
    print(222222)
    # ccl("media/out/%s/thinning.png" % code,"media/out/%s/ccl.png" % code, True)
    # print(3333333)