代码功能：左右视图的 2D 图片，通过双目相机校准后的参数来实现 3D 视图

代码使用工具和版本
软件版本：PyCharm Community Edition 2021.2.3
python 库版本：Python3.6（py36）

实验数据集来源：Middlebury College
数据网址：https://vision.middlebury.edu/stereo/data/scenes2014/

代码执行数据来源：.\dataset\（3组数据）
A1_Piano_L.png     A2_Jar_L.png      A3_Motorcycle_L.png    -- 左图
A1_Piano_R.png     A2_Jar_R.png     A3_Motorcycle_R.png   -- 右图
calib_piano.txt        calib_jar.txt        calib_motorcycle.txt      -- 校准参数

代码执行输出结果：.\Result\（3组数据）
wls_disparity_Piano.png     wls_disparity_Jar.png    wls_disparity_Motorcycle.png  -- 视差图
depth_map_Piano.png       depth_map_Jar.png       depth_map_Motorcycle.png   -- 深度图
A3D_data_Pinao.ply           A3D_data_Jar.ply           A3D_data_Motorcycle.ply      -- 3D 点云数据
etc..

代码执行匹配效率和匹配优良见  print 打印

 
代码执行顺序：按照 Piano - Jar - Motorcycle 的顺序显示 3D 图，（注意：关掉一个 3D 视图后才会显示下一个视图）
代码打印 Process finished with exit code 0 表示执行结束


