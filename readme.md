
# Binocular-3D-reconstruction
Mainly used to compare SIFT and SURF in 3D reconstruction. Includes image preprocessing, feature extraction and matching, parallax and depth information, 3D reconstruction.

Function of the code: 2D pictures for left and right view, 3D view by parameters calibrated by binocular camera

Code using tools and version
Software version: PyCharm Community Edition 2021.2.3
python library version: Python 3.6 (py36)

Experimental dataset source: Middlebury College
Data URL: https://vision.middlebury.edu/stereo/data/scenes2014/

Code execution data source: . \dataset\ (3 sets of data)
A1_Piano_L.png A2_Jar_L.png A3_Motorcycle_L.png -- left image
A1_Piano_R.png A2_Jar_R.png A3_Motorcycle_R.png -- right figure
calib_piano.txt calib_jar.txt calib_motorcycle.txt -- calibration parameters

Code execution output: . \Result\ (3 sets of data)
wls_disparity_Piano.png wls_disparity_Jar.png wls_disparity_Motorcycle.png -- parallax map
depth_map_Piano.png depth_map_Jar.png depth_map_Motorcycle.png -- Depth Map
A3D_data_Pinao.ply A3D_data_Jar.ply A3D_data_Motorcycle.ply -- 3D Point Cloud Data
etc..

See print for code execution matching efficiency and good matching.

 
Code execution order: display 3D images in the order Piano - Jar - Motorcycle, (note: switch off one 3D view before displaying the next one).
Code print Process finished with exit code 0 indicates the end of execution.

See video for actual demo results
