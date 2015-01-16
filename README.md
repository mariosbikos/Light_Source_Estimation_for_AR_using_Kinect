# Light_Source_Estimation_for_AR_using_Kinect
Developed algorithms to estimate the position of a Point Light Source in a scene using Kinect and Point Cloud Library based on code developed by by Zhongshi (Sam) Jiang and Shayan Rezvankhah
Instead of PCL's viewer I used OpenGL for rendering.

Steps:

* Record Point cloud using OpenNI Grabber from Kinect
* Calculate Normals of the point cloud
* Colod-based region growing segmentation
* Select specific segments for further processing through a UI
* Estimate the position of Light-Source
* Render the light source as a sphere in the scene using data retrieved from previous step
