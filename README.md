This reposity present an approach to build 2D occupancy grid maps with Lidar data

### Pipeline for creating an evidential occupancy grid from Lidar data
#### 1 - Ground filtering method
#### 2 - Creation of a Local evidential occupancy grid using Lidar data (KITTI dataset: velodyne) 
#### 3 - Transformation of the Occupancy Grid
#### 4 - Fusion the grids with the different combination rules ( Conjunctive Fusion, Disjunctive Fusion, Dempster Fusion, Yager Fusion) 
#### 5 - Decision making using Pignistic Probability
#### 6 - Validation of results with ground truth

<p align="center">   
  <img src="https://github.com/Mboubaker/Lidar_Evidential_occupancy_grid_mapping-/assets/97898968/eebb9268-7b59-41e6-a8a3-2385d4625331.png?raw=true" alt="Sublime's custom image"/>
       
</p>
<p align="center">                                  
Figure : Pipeline for creating an evidential occupancy grid from Lidar data




### 1 - Ground filtering method
Lidar point cloud filtering is the process of separating ground points from non-ground points and is a particularly important part of point cloud data processing.


#### 1.1  Grid based filtering algorithm

In this method:
1- we use a grid-based filtering approach to divide the point cloud into cells based on the grid_size parameter.
2- For each grid cell, we check if it contains mostly ground points based on the number of points (min_points) and height variation (max_height_diff) in the cell.
3- Points in cells that meet the criteria are labeled as ground points, while points in other cells are labeled as non-ground points.
4- The function uses scikit-learn's KD-tree data structure to efficiently perform the nearest neighbor (KNN) search for each point.
5- It then iterates through each single grid cell, calculates the height variation and assigns labels accordingly.

KDTree is used in the given context to perform a quick nearest neighbor search.
The K-nearest neighbors (KNN) algorithm is used to find the nearest neighbors of a given point in a point cloud.



<p align="center">   
  <img src="https://github.com/Mboubaker/Lidar_Evidential_occupancy_grid_mapping-/assets/97898968/29f3cf16-1d84-41fa-a155-d32d58063653.png?raw=true" alt="Sublime's custom image"/>
       
</p>
<p align="center">                                  
Figure :  Filtering of points on the ground, In red: points on the ground,  In green: Points not on the ground
  
#### 1.2 Filtering ground points based on One-Class SVM unsupervised learning technique

SVM “support vector machines” is a supervised machine learning algorithm used for classification and regression tasks.
The main goal of SVM in this case is to find the optimal hyperplane that separates different classes while maximizing the margin between the closest data points of each class.


<p align="center">   
  <img src="https://github.com/Mboubaker/Lidar_Evidential_occupancy_grid_mapping-/assets/97898968/249815bb-ca87-4a30-9b54-1bd7a8261302.png?raw=true" alt="Sublime's custom image"/>
       
</p>
<p align="center">                                  
Figure :  Filtering of points on the ground, In red: points on the ground,  In green: Points not on the ground
