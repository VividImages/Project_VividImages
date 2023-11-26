import cv2
import numpy as np

def watermark_lean(img):
    h, w = img.shape[0], img.shape[1]
    mark = np.zeros(img.shape[:2], np.uint8)  # 黑色背景
    for i in range(h//100):
        cv2.putText(mark, "vividImages", (w//2,70+200*i), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 4)
    MAR = cv2.getRotationMatrix2D((w//2,h//2), 45, 1.0)  # 旋转 45 度
    grayMark = cv2.warpAffine(mark, MAR, (w,h))  # 旋转变换，默认为黑色填充
    markC3 = cv2.merge([grayMark, grayMark, grayMark])
    imgMark = cv2.addWeighted(img, 1, markC3, 0.5, 0)  # 加权加法图像融合
    return imgMark

def watermark(imgPath,markName):
    outputImgList = []
    if markName == "watermark_lean":
        for imgName in imgPath:
            img = cv2.imread(imgName)
            res = watermark_lean(img)
        outputImgList.append(res)
        
    return outputImgList