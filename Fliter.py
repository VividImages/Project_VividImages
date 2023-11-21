import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

fliterList = ['Summer,Reminiscence']

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

#@ 功能：给图片加上“夏天的味道”滤镜
#@ 参数： imgPath      图片路径列表
#         fliterName   
#
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    res = cv2.merge((blue_channel, green_channel, red_channel ))
    return res

def Reminiscence(img):
    #获取图像行和列

    rows, cols = img.shape[:2]
    #新建目标图像
    res = np.zeros((rows, cols, 3), dtype="uint8")
    #图像怀旧特效
    for i in range(rows):
        for j in range(cols):
            B = 0.272*img[i,j][2] + 0.534*img[i,j][1] + 0.131*img[i,j][0]
            G = 0.349*img[i,j][2] + 0.686*img[i,j][1] + 0.168*img[i,j][0]
            R = 0.393*img[i,j][2] + 0.769*img[i,j][1] + 0.189*img[i,j][0]
            if B>255:
                B = 255
            if G>255:
                G = 255
            if R>255:
                R = 255

    res[i,j] = np.uint8((B, G, R))
    return res



#@ 功能：对图片列表进行批量滤镜处理
#@ 参数： imgPath      图片路径列表
#         fliterName   
# 注意：滤镜函数只负责处理单张照片，遍历照片在本函数中实现
def Fliter(imgPath,fliterName):
    outputImgList = []
    if fliterName == "Summer":
        for imgName in imgPath:
            img = cv2.imread(imgName)
            res = Summer(img)
            outputImgList.append(res)
    elif fliterName == "Reminiscence":
        for imgName in imgPath:
            img = cv2.imread(imgName)
            res = Reminiscence(img)
            outputImgList.append(res)  
    elif fliterName == "":
        pass
    else:
        pass
    
    
    return outputImgList