from types import SimpleNamespace
import cv2
import numpy as np
import os
from Constant import *
import random

def get_matcher(args):
    try_cuda = args.try_cuda
    matcher_type = args.matcher
    if args.match_conf is None:
        if args.features == 'orb':
            match_conf = 0.3
        else:
            match_conf = 0.65
    else:
        match_conf = args.match_conf
    range_width = args.rangewidth
    if matcher_type == "affine":
        matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    elif range_width == -1:
        matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    else:
        pass

    return matcher

def get_compensator(args):
    expos_comp_type = EXPOS_COMP_CHOICES[args.expos_comp]
    expos_comp_nr_feeds = args.expos_comp_nr_feeds
    expos_comp_block_size = args.expos_comp_block_size

    if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)

    elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )

    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator

def calWarpPoint(x,y,w,h,K,R,warper,corners,sizes,args):
    t_x, t_y = cv2.detail.resultRoi(corners, sizes)[:2]
    # Create arrays with the coordinates of the rectangle's edges
    top_edge = np.array([(i, y) for i in range(x, x + w, args.dye_step)])
    right_edge = np.array([(x + w, i) for i in range(y, y + h, args.dye_step)])
    bottom_edge = np.array([(i, y + h) for i in range(x + w, x, -args.dye_step)])
    left_edge = np.array([(x, i) for i in range(y + h, y, -args.dye_step)])
    # Concatenate the edges into a single array
    edge_points = np.concatenate((top_edge, right_edge, bottom_edge, left_edge), axis=0).tolist()
    # Warp the edge points
    warped_points = np.array([warper.warpPoint(pt, K, R) for pt in edge_points])
    # Subtract the translation values and convert to integers
    warped_points -= np.array([t_x, t_y])
    #warped_points = warped_points.astype(int)
    return warped_points


def main(img_names):
    args = {
        "try_cuda": True,
        "work_megapix": 0.6,
        "features": "orb",
        "matcher": "homography",
        "estimator": "homography",
        "match_conf": None,
        "conf_thresh": 1.0,
        "ba": "ray",
        "ba_refine_mask": "xxxxx",
        "wave_correct": "horiz",
        "save_graph":None,
        "warp": "spherical", #投影方式
        "seam_megapix": 0.1,
        "seam": "dp_color",
        "compose_megapix": 3,
        "expos_comp": "gain_blocks",
        "expos_comp_nr_feeds": 1,
        "expos_comp_nr_filtering": 2,
        "expos_comp_block_size": 32,
        "blend": "multiband",
        "blend_strength": 5,
        "output": "time_test.jpg",
        "timelapse": None,
        "rangewidth": -1,
        "dye_color":False,
        "dye_color_target":False,
        "dye_step":4
    }
    args = SimpleNamespace(**args)
    #img_names = [path + '/' + folder for folder in os.listdir(path)]
    #img_names.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    work_megapix = args.work_megapix
    seam_megapix = args.seam_megapix
    compose_megapix = args.compose_megapix
    conf_thresh = args.conf_thresh
    ba_refine_mask = args.ba_refine_mask
    wave_correct = WAVE_CORRECT_CHOICES[args.wave_correct]
    if args.save_graph is None:
        save_graph = False
    else:
        save_graph = True
    warp_type = args.warp
    blend_type = args.blend
    blend_strength = args.blend_strength
    result_name = args.output
    if args.timelapse is not None:
        timelapse = True
        if args.timelapse == "as_is":
            timelapse_type = cv.detail.Timelapser_AS_IS
        elif args.timelapse == "crop":
            timelapse_type = cv.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            exit()
    else:
        timelapse = False
    finder = FEATURES_FIND_CHOICES[args.features]()
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    for name in img_names:
        full_img = cv.imread(cv.samples.findFile(name))
        if full_img is None:
            print("Cannot read image ", name)
            exit()
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        images.append(img)

    matcher = get_matcher(args)
    p = matcher.apply2(features)
    matcher.collectGarbage()

    if save_graph:
        with open(args.save_graph, 'w') as fh:
            fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))

    indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    for i in range(len(indices)):
        img_names_subset.append(img_names[indices[i]])
        img_subset.append(images[indices[i]])
        full_img_sizes_subset.append(full_img_sizes[indices[i]])
    images = img_subset
    img_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(img_names)
    if num_images < 2:
        print("Need more images")
        exit()

    estimator = ESTIMATOR_CHOICES[args.estimator]()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        #cam.R 即为图像的单应性矩阵
        cam.R = cam.R.astype(np.float32)
#----------- 估算参数
    adjuster = BA_COST_CHOICES[args.ba]()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    #print(focals)
    focals.sort()
    #取焦距的中位数
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        #---------------
    if wave_correct is not None:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
            #波形校正
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        #单应性矩阵变换 图像旋转矩阵R 相机参数矩阵K corner 为左上角和右上角的坐标
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())
        #mask_up即为二值图像 mask_wp.get()也为图片的轮廓 p为角点坐标--2*图片个数

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)
#块增益补充
    compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

#缝隙估计
    seam_finder = SEAM_FIND_CHOICES[args.seam]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
#-----变换后即为接缝
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    timelapser = None

    for idx, name in enumerate(img_names):
        full_img = cv.imread(name)
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            # 处理相机旋转造成的扭曲
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                      int(round(full_img_sizes[i][1] * compose_scale)))
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                # 缝合区获得 corner为角点 size为原图尺寸
                corners.append(roi[0:2])
                sizes.append(roi[2:4])

        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img
        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)

        #image_warped 为原图变换照片
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)

        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        #mask 为二值后原版照片
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)

        #块增益补偿
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        #cv2.imwrite("./Result/mask.jpg",mask_warped)
        #膨胀
        dilated_mask = cv.dilate(masks_warped[idx], None)
        #seam_mask即为拼缝隙
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        #cv2.imwrite("./Result/diat.jpg", seam_mask)
        # 二进制的与运算，1&1 == 1
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        #cv2.imwrite("./Result/and.jpg", mask_warped)
#---------------------------------融合
        if blender is None and not timelapse:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)

            # noinspection PyArgumentList
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)

            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int64))
            # 羽化融合
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)

        elif timelapser is None and timelapse:
            timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        # 延时处理
        if timelapse:
            ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, ma_tones, corners[idx])

            pos_s = img_names[idx].rfind("/")
            if pos_s == -1:
                fixed_file_name = "fixed_" + img_names[idx]
            else:
                fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv.imwrite(fixed_file_name, timelapser.getDst())
        else:
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        cv2.imwrite("./result.jpg",result)
        return result
    if args.dye_color:
        dots=[[] for _ in range(len(img_names))]
        dots_target = [[] for _ in range(len(img_names))]#生成目标合集
        for idx,name in enumerate(img_names):
            img=cv2.resize(cv2.imread(name),None,fx=compose_scale,fy=compose_scale)
            #读取对应文件
            K=cameras[idx].K().astype(np.float32)
            R=cameras[idx].R
            w, h=img.shape[1],img.shape[0]
            dots[idx]=calWarpPoint(0,0,w,h,K,R,warper,corners,sizes,args)
        result=cv2.imread("./result.jpg")
        color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(dots))]
        for cor_index, rax in enumerate(dots):
            pts = [np.array(rax, np.int32)]
            cv2.polylines(result, pts, isClosed=True, color=color[cor_index], thickness=4)
            if args.dye_color_target:
                for target_index, target_rax in enumerate(dots_target[cor_index]):
                    target_pts = [np.array(target_rax, np.int32)]
                    cv2.polylines(result, target_pts, isClosed=True, color=color[cor_index], thickness=1)
        cv2.imwrite("./dye_result.jpg", result)

# if __name__ == '__main__':
#     import time
#     start = time.time()
#     path=r"./images"
#     main(path)
#     end = time.time()
#     print("时间消耗:%d"%(end - start))

def imgStitch(fileNames):
    res = main(fileNames)
    return res