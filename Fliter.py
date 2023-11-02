import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

fliterList = ['Summer']

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


#@ 功能：对图片列表进行批量滤镜处理
#@ 参数： imgPath      图片路径列表
#         fliterName   
#
def Fliter(imgPath,fliterName):
    outputImgList = []
    if fliterName == "Summer":
        for imgName in imgPath:
            img = cv2.imread(imgName)
            res = Summer(img)
            outputImgList.append(res)
        
    else:
        pass
    
    
    return outputImgList