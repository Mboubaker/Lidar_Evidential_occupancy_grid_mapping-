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

The fuction look like the following:


    def filter_ground_points_grid_knn(point_cloud_file, grid_size=2, min_points=1, max_height_diff=0.2):
    # Load the point cloud data from the .bin file
    point_cloud = np.load(point_cloud_file).reshape(-1, 4)
    #point_cloud = point_cloud[point_cloud[:,2]<=0,:]
    # Extract the X, Y, and Z coordinates from the point cloud data
    points_xyz = point_cloud[:, :3]

    # Build a KD-tree for fast nearest neighbor search
    kdtree = KDTree(points_xyz)

    # Calculate the grid indices for each point
    grid_indices = np.floor(points_xyz[:, :2] / grid_size).astype(int)

    # Initialize the labels for the point cloud
    labels = np.zeros(point_cloud.shape[0])

    # Iterate through each grid cell
    for grid_index in np.unique(grid_indices, axis=0):
        # Get the points in the current grid cell
        grid_points = points_xyz[np.all(grid_indices == grid_index, axis=1)]

        # Check if the number of points in the cell is above the threshold
        if grid_points.shape[0] >= min_points:
            # Calculate the height variation within the grid cell
            height_diff = np.max(grid_points[:, 2]) - np.min(grid_points[:, 2])

            # Check if the height variation is below the threshold
            if height_diff <= max_height_diff:
                # Mark the points in the grid cell as ground points
                grid_indices_in_cell = np.where(np.all(grid_indices == grid_index, axis=1))[0]
                labels[grid_indices_in_cell] = 1

    # Separate the ground and non-ground points based on the labels
    ground_points = point_cloud[labels == 1]
    non_ground_points = point_cloud[labels == 0]
    lidar_raw = non_ground_points[:, :3]
    print('ffff',lidar_raw.shape)
    print(lidar_raw)
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground points in red color
    ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='r', marker='.')

    # Plot non-ground points in green color
    ax.scatter(lidar_raw[:, 0], lidar_raw[:, 1], lidar_raw[:, 2], c='g', marker='.')

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

<p align="center">   
  <img src="https://github.com/Mboubaker/Lidar_Evidential_occupancy_grid_mapping-/assets/97898968/29f3cf16-1d84-41fa-a155-d32d58063653.png?raw=true" alt="Sublime's custom image"/>
       
</p>
<p align="center">                                  
Figure :  Filtering of points on the ground, In red: points on the ground,  In green: Points not on the ground
  
#### 1.2 Filtering ground points based on One-Class SVM unsupervised learning technique

SVM “support vector machines” is a supervised machine learning algorithm used for classification and regression tasks.
The main goal of SVM in this case is to find the optimal hyperplane that separates different classes while maximizing the margin between the closest data points of each class.

The fuction look like the following:

    def filter_ground_points_svm(point_cloud_file):
    # Load the point cloud data from the .bin file
    point_cloud = np.fromfile(point_cloud_file, dtype=np.float32).reshape(-1, 4)
    point_cloud = point_cloud[point_cloud[:,2]<=8,:]
    # Extract the X, Y, and Z coordinates from the point cloud data
    points_xyz = point_cloud[:, :3]
    #points_xyz = points_xyz[points_xyz[:,2]<=0,:]
    # Apply OneClassSVM for outlier detection
    model = OneClassSVM(kernel='linear')
    model.fit(points_xyz)

    # Predict the labels for the entire point cloud
    labels = model.predict(points_xyz)

    # Separate the ground and non-ground points based on the predicted labels
    ground_points = point_cloud[labels == 1]
    non_ground_points = point_cloud[labels == -1]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground points in red color
    ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='r', marker='.')

    # Plot non-ground points in green color
    ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2], c='g', marker='.')

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()




<p align="center">   
  <img src="https://github.com/Mboubaker/Lidar_Evidential_occupancy_grid_mapping-/assets/97898968/249815bb-ca87-4a30-9b54-1bd7a8261302.png?raw=true" alt="Sublime's custom image"/>
       
</p>
<p align="center">                                  
Figure :  Filtering of points on the ground, In red: points on the ground,  In green: Points not on the ground



### 2 - Creation of a Local evidential occupancy grid using Lidar data (KITTI dataset: velodyne) 

To download the dataset : https://www.cvlibs.net/datasets/kitti/raw_data.php

### 3 - Transformation of the Occupancy Grid

Before the grid is updated again at the next time step, it must be shifted according to the movement of the Robot. Therefore, we need to find the next pose of the vehicle.

The fuction look like the following:


    def shift_pose_dgm(dgm, init, fin):
    dgm_o = dgm.copy()
    theta = init[2] 
    rot_m = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    trs_m = np.array([[init[0]],[init[1]]])
    point = np.array(fin[:2]).reshape((-1,1))
    point_1 = (point - trs_m)
    point_2 = np.dot(rot_m,-point_1)
    delta_theta = (fin[2] - init[2])
    delta = np.array([point_2[1,0]/RESOLUTION,point_2[0,0]/RESOLUTION,0])
    M = np.array([[1,0,delta[0]],[0,1,-delta[1]]])
    dst = cv2.warpAffine(dgm_o,M,(dgm_o.shape[1],dgm_o.shape[0]),borderValue=0.5)
    M = cv2.getRotationMatrix2D((dgm_o.shape[1]/2+0.5,dgm_o.shape[0]/2+0.5),delta_theta,1)
    dst = cv2.warpAffine(dst,M,(dgm_o.shape[1],dgm_o.shape[0]),borderValue=0.5)
    return dst

  
<p align="center">   
  <img src="https://github.com/Mboubaker/Lidar_Evidential_occupancy_grid_mapping-/assets/97898968/8b535954-9e7a-4c45-a8fe-45956dde3e97.gif?raw=true" alt="Sublime's custom image"/>
       
</p>
<p align="center">                                  
Figure :  Transformation of the Occupancy Grid



### 4 - Fusion the grids with the different combination rules
To fuse mass functions, there are several operators. Each operator has particular properties and is used in certain cases:

- Conjunctive fusion All sources are reliable

- Disjunctive fusion At least one source is reliable but without knowing which one.

- Dempster fusion The standardized version of conjunctive fusion


#### 4.1 Conjunctive fusion

The fuction look like the following:

      def update_dgm_conj(prior_dgm,new_dgm):
      ###Calculate conflicting mass
      conflict_mass = 0
      ###Calculate free mass
      free_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,2])
      free_mass = np.add(free_mass,np.multiply(prior_dgm[:,:,2],new_dgm[:,:,0]))
      free_mass = np.add(free_mass,np.multiply(prior_dgm[:,:,2],new_dgm[:,:,2]))
      free_mass = np.divide(free_mass,1-conflict_mass)
      ###Calculate occupied mass
      occ_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,1])
      occ_mass = np.add(occ_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,0]))
      occ_mass = np.add(occ_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,1]))
      occ_mass = np.divide(occ_mass,1-conflict_mass)
      ###Calculate unknown mass
      unknown_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,0])
      unknown_mass = np.divide(unknown_mass,1-conflict_mass)
      updated_dgm1 = np.stack((unknown_mass,occ_mass,free_mass),axis=2)
      return updated_dgm1,conflict_mass

#### 4.2 Dempster fusion

The fuction look like the following:

      def update_dgm(prior_dgm,new_dgm):
      ### Calculate conflicting mass
      conflict_mass = np.multiply(prior_dgm[:,:,2],new_dgm[:,:,1])
      conflict_mass = np.add(conflict_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,2]))
      #conflict_mass = 0
      ### Calculate free mass
      free_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,2])
      free_mass = np.add(free_mass,np.multiply(prior_dgm[:,:,2],new_dgm[:,:,0]))
      free_mass = np.add(free_mass,np.multiply(prior_dgm[:,:,2],new_dgm[:,:,2]))
      free_mass = np.divide(free_mass,1-conflict_mass)

      ### Calculate occupied mass
      occ_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,1])
      occ_mass = np.add(occ_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,0]))
      occ_mass = np.add(occ_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,1]))
      occ_mass = np.divide(occ_mass,1-conflict_mass)

      ### Calculate unknown mass
      unknown_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,0])
      unknown_mass = np.divide(unknown_mass,1-conflict_mass)

      updated_dgm1 = np.stack((unknown_mass,occ_mass,free_mass),axis=2)
  
      return updated_dgm1,conflict_mass
