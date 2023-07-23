import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time

def decompose_essential_matrix(E,K,pts1,pts2):    #基本矩阵分解
    [U, D, V] = np.linalg.svd(E)
    diag_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    new_E = U @ diag_arr @ V
    [U, D, V] = np.linalg.svd(new_E)
    Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = - U @ Y @ V
    R2 = - U @ Y.T @ V
    t = U[:, 2].reshape(3, 1)
    R_mat = np.array([R1, R1, R2, R2])
    T_mat = np.array([t, -t, t, -t])
    P1 = np.zeros((3, 4))
    P1[:, :3] = np.eye(3)
    P1 = K @ P1
    print(R1, "\n", R2)
    for i in range(4):
        P2 = np.concatenate((R_mat[i], T_mat[i]), axis=1)
        P2 = K @ P2
        world_pts = cv2.triangulatePoints(P1, P2, pts1, pts2)
        X, Y, Z = world_pts[:3, :] / world_pts[3, :]
        Z_ = R_mat[i][2, 0] * X + R_mat[i][2, 1] * Y + R_mat[i][2, 2] * Z + T_mat[i][2]
        print(len(np.where(Z < 0)[0]), len(np.where(Z_ < 0)[0]))
        if len(np.where(Z < 0)[0]) == 0:
            R = R_mat[i]
            t = T_mat[i]
            break
    return R,t

def drawlines(img1,img2,lines,pts1,pts2):
    r,c,ch = img1.shape   # 可能是 shape 是外极线校正（平行校正），代表外形形状的校准
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[0]),5,color,-1)  # circle - 圆
        img2 = cv2.circle(img2,tuple(pt2[0]),5,color,-1)
    return img1,img2

def isfunctionValue(i):
    min_disparity = -1  # 最小视差值#
    max_disparity = 159  # 最大视差值
    num_disparity = max_disparity - min_disparity  # 视差相差的值来表示梯度
    SADWindowSize = 5  # SAD 算法的窗口尺寸大小  -- 使用于左右视图的视差的立体匹配
    uniqueness = 5  # 唯一性，独特性，选项组
    speckle_windows_size = 5  # 斑纹窗口大小
    speckle_range = 5  # 斑纹范围
    P1 = 8 * 3 * SADWindowSize ** 2  # * - 表示乘法    ** -- 表示乘方    例如：2**5 = 2^5 = 32
    P2 = 32 * 3 * SADWindowSize ** 2
    # Defining the Parameter for stereoSGBM  - 为 stereoSGBM 定义参数

    # 钢琴图像读取和相机的校准参数
    if i == 1:
        print("当前正在显示第一幅 Piano 3D 图，关闭 3D 图后显示下一幅 Jar 3D 图")
        imgL = cv2.imread('.\dataset\A1_Piano_L.png')
        imgR = cv2.imread('.\dataset\A1_Piano_R.png')
        KL = np.array([[2826.171, 0, 1292.2],
                       [0, 2826.171, 965.806],
                       [0, 0, 1]])
        KR = np.array([[2826.171, 0, 1415.97],
                       [0, 2826.171, 965.806],
                       [0, 0, 1]])

        b = 178.089  # Baseline  --  钢琴图片的基线
        doffs = 123.77/25.4
    # 罐子图像读取和相机的校准参数
    elif i==2:
        print("当前正在显示第二幅 Jar 3D 图，关闭 3D 图后显示下一幅 Motorcycle 3D 图")
        imgL = cv2.imread('.\dataset\A2_Jar_L.png')
        imgR = cv2.imread('.\dataset\A2_Jar_R.png')
        KL = np.array([[7242.753, 0, 1079.538],
                   [0, 7242.753, 1018.846],
                   [0, 0, 1]])
        KR = np.array([[7242.753, 0, 1588.865],
                   [0, 7242.753, 1018.846],
                   [0, 0, 1]])
        b = 379.965  # Baseline  --  相机基线罐子
        doffs = 509.327/25.4
# 摩托车图像读取和相机的校准参数
    elif i==3:
         print("当前正在显示第三幅 Motorcycle 3D 图，关闭 3D 图后结束程序")
         imgL = cv2.imread('.\dataset\A3_Motorcycle_L.png')  # 读取左边的图片
         imgR = cv2.imread('.\dataset\A3_Motorcycle_R.png')  # 读取右边的图片
         KL = np.array([[3997.684, 0, 1176.728],
                   [0, 3997.684, 1011.728],
                   [0, 0, 1]])
         KR = np.array([[3997.684, 0, 1307.839],
                   [0, 3997.684, 1011.728],
                   [0, 0, 1]])
         b = 193.001  # Baseline  --  相机基线摩托车
         doffs = 124.343/25.4 #- - 偏移
    else:
         i = 0

    dist_coeff = None  # dist 多项式系数
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) # 左边图像灰度变换
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) # 右边图像灰度变换
    image_size = grayL.shape  # 获取图像的尺寸大小

    #cv2.imshow('grayL',grayL)  # 左右视图视差滤波后的图像
    #cv2.imshow('grayR',grayR)  # 左右视图视差滤波后的图像

    # 可以选择 SURF 或者 SIRF 特征匹配法
    isSiftOrSurf = 1  # 1 -- SIFT   2 -- SURF
    if isSiftOrSurf == 1:
        #(1) SIFT --Scale Invariant Feature Transform -- 尺度不变特征变换
        sift = cv2.xfeatures2d.SIFT_create()   # 使用 cv2.xfeatures2d.SIFT_create() 实例化 sift 函数
        kp1, desc1 = sift.detectAndCompute(grayL, None)  # 计算出左边图像的关键点和 sift 特征向量 -- kp1（关键点）   sesc1（sift 特征向量）
        kp2, desc2 = sift.detectAndCompute(grayR, None)  # 计算出右边图像的关键点和 sift 特征向量 -- kp2（关键点）   sesc2（sift 特征向量）
        print("SIFT 特征匹配法下运行的结果：")  # SIFT 特征匹配法
    elif isSiftOrSurf == 2:
        #(2) SURF -- Speeded Up Robust Features -- 特征计算 --加速稳健特征
        surf = cv2.xfeatures2d.SURF_create()
        kp1, desc1 = surf.detectAndCompute(grayL, None)
        kp2, desc2 = surf.detectAndCompute(grayR, None)
        print("SURF 特征匹配法下运行的结果：")  # SURF 特征匹配法
    else:
        isSiftOrSurf = 0

    #（1） Flann特征匹配
    start = time.clock()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    isflann_matches = flann.knnMatch(desc1, desc2, k=2)
    print("Flann 匹配的点数：", len(isflann_matches))  # 匹配优化后的点数
    goodMatch = []

    for m, n in isflann_matches:       # 匹配优化
        if m.distance < 0.7 * n.distance:
            goodMatch.append(m)
    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    print("Flann 法匹配成功的点数：", len(goodMatch))
    #flann_img_out = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, goodMatch[:20], None, flags=2)
    flann_time = (time.clock() - start)
    print("Flann 法匹配结束的时刻:", '% 4f' % (flann_time * 1000))

    #（2） BFmatcher with default parms
    # BFMatcher -- Brute Force Matcher （蛮力匹配器）
    # BFMatcher.match（）-- 返回最佳匹配
    # BFMatcher.knnMatch（）-- 返回k个最佳匹配，其中k由用户指定
    # 对knn匹配的特征做判断，满足条件的定义为一个好的匹配点
    # 此处也可以借助RANSAC筛选
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2) # 针对左右视图的蛮力匹配，并返回 k 个最佳匹配
    #print("kp1 = {} \n t = {}",matches)
    print("BF 法匹配成功的点数：", len(matches)) # 匹配优化后的点数
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 需满足定义的条件
            good.append(m)  # append - 添加，追加

    good = sorted(good, key=lambda x: x.distance) # sorted - 排序
    #good = np.expand_dims(good, 1)
    #bf_img_out = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, goodMatch[:20], None, flags=2)
    # print(good) # 打印依据给定条件匹配成功的点
    # import numpy
    # c = numpy.array(good)
    print("BF 法匹配成功的点数：", len(good))
    bf_time = (time.clock() - start)
    print("BF 法匹配结束的时刻:", '% 4f' % (bf_time * 1000))

    #cv2.imshow('flannmatch', flann_img_out)  # 展示图片
    #cv2.imshow('bfmatch', bf_img_out)  # 展示图片


    # 获取匹配点在原图像和目标图像中的的位置
    # kp1 -- 原图像的特征点
    # m.queryIdx -- 匹配点在原图像特征点中的索引
    # .pt -- 特征点的坐标
    # reshape -- 矩阵变维
    # reshape(-1, 1, 2) -- -1 表示列数自动计算
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # reshape - 矩阵变维
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 获取变换矩阵，采用 RANSAC 算法
    # mask -- 输出，极外几何描述如下 [p_2; 1]^T F [p_1; 1] = 0
    # prob -- 概率
    # threshold -- 阈值
    # E -- 本征矩阵 F -- 基本矩阵，可使用findFundamentalMat或stereoRectify 进行估计
    E, mask = cv2.findEssentialMat(pts1, pts2, KL, method=cv2.FM_RANSAC, prob=0.99,
                                   threshold=0.4, mask=None)
    # 我们只选择内层点
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # R,t = decompose_essential_matrix(E,KL,pts1,pts2)
    # pts1 和 pts2 依据 E（本征矩阵） 进行图像恢复
    points,R,t,mask = cv2.recoverPose(E,pts1,pts2,R = None,t = None,mask = None)

    # np.linalg.inv() -- 矩阵求逆
    K_inv = np.linalg.inv(KL)

    # @是一个装饰器，针对函数，起调用传参的作用。
    # 有修饰和被修饰的区别，‘@function'作为一个装饰器，用来修饰紧跟着的函数（可以是另一个装饰器，也可以是函数定义）。
    F = K_inv.T @ E @ K_inv
    print("R = {} \n t = {}".format(R,t))

    # 找到相应的点在右图像(第二图像)和画它的线在左图像
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)

    R1,R2,P1,P2= cv2.stereoRectify(KL,None,KL,None,(image_size[1],image_size[0]),R,t,flags = cv2.CALIB_ZERO_DISPARITY)[:4]

    #print(R1 @ R2.T) # these gives the rotation between the two camera
    mapx1,mapy1 = cv2.initUndistortRectifyMap(KL,None,R1,P1,(image_size[1],image_size[0]),cv2.CV_16SC2)
    mapx2,mapy2 = cv2.initUndistortRectifyMap(KL,None,R2,P2,(image_size[1],image_size[0]),cv2.CV_16SC2)
    print("shape = ",mapx1.shape,mapy1.shape)

    rectified_imgL = cv2.remap(imgL,mapx1,mapy1,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    rectified_imgR = cv2.remap(imgR,mapx2,mapy2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3) # -1 表示行数自动计算
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3) # -1 表示行数自动计算

    # cv2.pyrDown() 从一个高分辨率大尺寸的图像向上构建一个金字塔（尺寸变小，分辨率降低）
    rectified_imgL = cv2.pyrDown(rectified_imgL) # rectified -- 纠正
    rectified_imgR = cv2.pyrDown(rectified_imgR)

    #cv2.imshow('rectified_imgL',rectified_imgL)  # 左右视图视差滤波后的图像
    #cv2.imshow('rectified_imgR',rectified_imgR)  # 左右视图视差滤波后的图像

    # SGBM 视差的立体匹配器
    # minDisparity: 最小视差值。通常我们期望这里是0，但当校正算法移动图像时，有时需要设置。
    # numDisparities: 最大视差值，必须大于0，定义视差边界。
    # blockSize: 匹配块的块大小。推荐使用[3-11]，推荐使用奇数，因为奇数大小的块有一个中心。
    # P1 和 P2: 负责平滑图像，规则是P2>P1。
    # disp12MaxDiff: 视差计算的最大像素差。
    # preFilterCap：过滤前使用的值。在块匹配之前，计算图像x轴的一个导数，并用于检查边界[-prefiltercap, prefiltercap]。其余的值用Birchfield-Tomasi代价函数处理。
    # uniquenessRatio: 经过成本函数计算，此值用于比较。建议取值范围[5-15]。
    # speckleWindowSize: 过滤删除大的值，得到一个更平滑的图像。建议取值范围[50-200]。
    # speckleRange: 使用领域检查视差得到一个平滑的图像。如果你决定尝试，我建议1或2。小心，这个值会乘以16！OpenCV会这样做，所以你不需要自己去乘。
    left_matcher = cv2.StereoSGBM_create(minDisparity=min_disparity,numDisparities=num_disparity,blockSize=SADWindowSize
                                         ,P1=8*3*SADWindowSize**2,P2=32*3*SADWindowSize**2,uniquenessRatio=uniqueness,disp12MaxDiff=2,
                                          speckleWindowSize=speckle_windows_size,speckleRange=speckle_range)

    # 左图视差
    left_disparity = left_matcher.compute(rectified_imgL,rectified_imgR)
    # 右边匹配器
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # 右图视差
    right_disparity = right_matcher.compute(rectified_imgR,rectified_imgL)
    #print("左视差图点数：", len(left_disparity))
    #print("右视差图点数：", len(right_disparity))
    #cv2.imshow('left_disparity',left_disparity)  # 左右视图视差滤波后的图像
    #cv2.imshow('right_disparity',right_disparity)  # 左右视图视差滤波后的图像

    # wls filtering -- wls 滤波器  -- 创建滤波器
    # wls -- 加权最小二乘滤波
    sigma = 1.5
    lambda_ = 8000
    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(lambda_)  # wls 设置参数 lambda_ = 8000
    wls.setSigmaColor(sigma) # wls 设置参数 sigma = 1.5
    # wls -- 加权最小二乘滤波
    filtered_disparity = wls.filter(left_disparity,rectified_imgL,disparity_map_right = right_disparity)
    cv2.filterSpeckles(filtered_disparity,0,400,max_disparity-5) # 过滤器斑点
    _,filtered_disparity = cv2.threshold(filtered_disparity,0,max_disparity*16,cv2.THRESH_TOZERO)
    filtered_disparity = (filtered_disparity/16).astype(np.uint8)

    print("lws 滤波视差点行：", len(filtered_disparity))
    print("lws 滤波视差点列：",len(filtered_disparity[0]))
    #print("点云图像点列：", len(points[0]))
    #print("右视差图点数：", len(right_disparity))
    #cv2.imshow('filter_3', filtered_disparity)  # 左右视图视差滤波后的图像
    #cv2.imwrite("wls_disparity_3.png", filtered_disparity)  # 将视差图存储下来
    depth_map = KL[0, 0] * b / (filtered_disparity + doffs)  # 深度信息计算公式

    depth_map = depth_map.astype('uint16')
    #cv2.imshow('depth map_3', depth_map)  # 显示深度图
    #cv2.imwrite("depth_map3.png",depth_map) # 将视差图存储下来

    if i == 1:
        # cv2.imshow('grayL',grayL)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/Piano_grayL.png", grayL)  # 将视差图存储下来
        # cv2.imshow('grayR',grayR)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/Piano_grayR.png", grayR)  # 将视差图存储下来
        #cv2.imshow('filter_1', filtered_disparity)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/wls_disparity_Piano.png", filtered_disparity)  # 将视差图存储下来
        #depth_map = KL[0, 0] * b / (filtered_disparity)  # 深度信息计算公式
        #depth_map = depth_map.astype('uint16')
        #cv2.imshow('depth map_1',depth_map)  # 显示深度图
        cv2.imwrite("./Result/depth_map_Piano.png",depth_map) # 将视差图存储下来
    elif i == 2:
        # cv2.imshow('grayL',grayL)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/Jar_grayL.png", grayL)  # 将视差图存储下来
        # cv2.imshow('grayR',grayR)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/Jar_grayR.png", grayR)  # 将视差图存储下来
        #cv2.imshow('filter_2', filtered_disparity)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/wls_disparity_Jar.png", filtered_disparity)  # 将视差图存储下来
        #depth_map = KL[0, 0] * b / (filtered_disparity)  # 深度信息计算公式
        #depth_map = depth_map.astype('uint16')
        #cv2.imshow('depth map_2', depth_map)  # 显示深度图
        cv2.imwrite("./Result/depth_map_Jar.png",depth_map) # 将视差图存储下来
    elif i == 3:
        # cv2.imshow('grayL',grayL)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/Motorcycle_grayL.png", grayL)  # 将视差图存储下来
        # cv2.imshow('grayR',grayR)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/Motorcycle_grayR.png", grayR)  # 将视差图存储下来
        #cv2.imshow('filter_3', filtered_disparity)  # 左右视图视差滤波后的图像
        cv2.imwrite("./Result/wls_disparity_Motorcycle.png", filtered_disparity)  # 将视差图存储下来
        #depth_map = KL[0, 0] * b / (filtered_disparity)  # 深度信息计算公式
        #depth_map = depth_map.astype('uint16')
        #cv2.imshow('depth map_3', depth_map)  # 显示深度图
        cv2.imwrite("./Result/depth_map_Motorcycle.png",depth_map) # 将视差图存储下来
    else:
        i = 0

    # Reprojection matrix -- 重新投影矩阵
    Q = np.float32([[1,0,0,-KL[0,2]],
                    [0,1,0,-KL[1,2]],
                    [0,0,0,KL[0,0]],
                    [0,0,-1/b,(KL[0,2]-KR[0,2])/b]])

    points = cv2.reprojectImageTo3D(filtered_disparity,Q)  # 图像的 3D 点云坐标


    #print("3D 图像点列：", len(points[1]))

    points = points.reshape(-1,3) # -1 表示行数自动计算
    color = rectified_imgL.reshape(-1,3) # -1 表示行数自动计算
    color = np.flip(color,axis = 1)/255
    xyzrbg = np.concatenate((points,color),axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrbg[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrbg[:,3:])

    if i == 1:
        o3d.io.write_point_cloud('./Result/A3D_data_Pinao.ply', pcd)  # o3d 写入 3d 数据
        o3d.visualization.draw_geometries([pcd]) # o3d 可视化绘制几何图形
    elif i == 2:
        o3d.io.write_point_cloud('./Result/A3D_data_Jar.ply', pcd)  # o3d 写入 3d 数据
        o3d.visualization.draw_geometries([pcd]) # o3d 可视化绘制几何图形
    elif i == 3:
        o3d.io.write_point_cloud('./Result/A3D_data_Motorcycle.ply', pcd)  # o3d 写入 3d 数据
        o3d.visualization.draw_geometries([pcd]) # o3d 可视化绘制几何图形
    else:
        i = 0
    i = i+1
    #cv2.waitKey()是一个键盘绑定函数。它的时间量度是毫秒ms。函数会等待（n）里面的n毫秒，看是否有键盘输入。若有键盘输入，则返回按键的ASCII值。没有键盘输入，则返回-1.一般设置为0，他将无线等待键盘的输入。
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 删除窗口，（）里不指定任何参数，则删除所有窗口，删除特定的窗口，往（）输入特定的窗口值。
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    current_number = 1
    while current_number <= 3:
        isfunctionValue(current_number)
        current_number = current_number+1
        #print("current_number = ", current_number)




