// TH Huang

#include <array>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iostream>
using std::cout;
using std::endl;

using pcl::PointXYZI;
using pcl::PointXYZ;


inline double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}


inline float squared_distance(PointXYZI& p1, PointXYZI& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return dx * dx + dy * dy + dz * dz;
}


inline float point_depth(PointXYZI& p) {
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}


inline Eigen::Quaternionf euler_to_quaternion(const float roll, const float pitch, const float yaw) {
  Eigen::Quaternionf q = Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX())
                       * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
                       * Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ());                         
  return q;                                
}


void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                        float &ox, float &oy, float &oz)
{
  float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
  ox = -asin(srx);

  float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
               + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
  float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
               - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

  float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
               + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
  float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
               - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}



class LidarOdometry {
    ros::NodeHandle& node;
    ros::Publisher pub_edge_points_last;
    ros::Publisher pub_planar_points_last;
    ros::Publisher pub_point_cloud_reprojected;
    ros::Publisher pub_lidar_odometry;
    tf::TransformBroadcaster tf_broadcaster;
    std::string    camera_frame_id;
    std::string    camera_init_frame_id;
    std::string    camera_odom_frame_id;
    pcl::PointCloud<PointXYZI>::Ptr  point_cloud;
    pcl::PointCloud<PointXYZI>::Ptr  edge_points_sharp;
    pcl::PointCloud<PointXYZI>::Ptr  edge_points_less_sharp;
    pcl::PointCloud<PointXYZI>::Ptr  planar_points_flat;
    pcl::PointCloud<PointXYZI>::Ptr  planar_points_less_flat;
    pcl::PointCloud<PointXYZ>::Ptr   imu_trans;
    pcl::PointCloud<PointXYZI>::Ptr  edge_points_last;
    pcl::PointCloud<PointXYZI>::Ptr  planar_points_last;
    pcl::PointCloud<PointXYZI>::Ptr  constrain_points;
    pcl::PointCloud<PointXYZI>::Ptr  constrain_parameters;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree_edge_points_last;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree_planar_points_last;
    std::array<float, 6> transform;
    std::array<float, 6> transform_sum;
    bool   system_initialized;
    bool   new_point_cloud;
    bool   new_edge_points_sharp;
    bool   new_edge_points_less_sharp;
    bool   new_planar_points_flat;
    bool   new_planar_points_less_flat;
    bool   new_imu_trans;
    int    num_edge_points_last;
    int    num_planar_points_last;
    int    frame_count;
    double time_point_cloud;
    double time_edge_points_sharp;
    double time_edge_points_less_sharp;
    double time_planar_points_flat;
    double time_planar_points_less_flat;
    double time_imu_trans;
    const float scan_period;
    const int   num_skip_frame;
    const int   min_edge_point_threshold;
    const int   min_planar_point_threshold;
    const int   max_lm_iteration;
    const int   transformation_recalculate_iteration;
    const float nearest_neighbor_cutoff;
    const float tolerance_rotation;
    const float tolerance_translation;

public:
    LidarOdometry(ros::NodeHandle& node) 
        : node(node),
          pub_edge_points_last(node.advertise<sensor_msgs::PointCloud2>("/edge_points_last", 2)),  // TODO: remap
          pub_planar_points_last(node.advertise<sensor_msgs::PointCloud2>("/planar_points_last", 2)),  // TODO: remap
          pub_point_cloud_reprojected(node.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_reprojected", 2)),  // TODO: remap
          pub_lidar_odometry(node.advertise<nav_msgs::Odometry>("/lidar_odom_to_init", 5)),  // TODO: remap
          camera_frame_id("/camera"),
          camera_init_frame_id("/camera_init"),
          camera_odom_frame_id("/camera_odom"),
          point_cloud(new pcl::PointCloud<PointXYZI>()),
          edge_points_sharp(new pcl::PointCloud<PointXYZI>()),
          edge_points_less_sharp(new pcl::PointCloud<PointXYZI>()),
          planar_points_flat(new pcl::PointCloud<PointXYZI>()),
          planar_points_less_flat(new pcl::PointCloud<PointXYZI>()),
          imu_trans(new pcl::PointCloud<PointXYZ>()),
          edge_points_last(new pcl::PointCloud<PointXYZI>()),
          planar_points_last(new pcl::PointCloud<PointXYZI>()),
          constrain_points(new pcl::PointCloud<PointXYZI>()),
          constrain_parameters(new pcl::PointCloud<PointXYZI>()),
          kdtree_edge_points_last(new pcl::KdTreeFLANN<PointXYZI>()),
          kdtree_planar_points_last(new pcl::KdTreeFLANN<PointXYZI>()),
          system_initialized(false),
          scan_period(0.1),  // Inverse of the scan rate which is 10 Hz (Velodyne VLP-16)
          new_point_cloud(false),
          new_edge_points_sharp(false),
          new_edge_points_less_sharp(false),
          new_planar_points_flat(false),
          new_planar_points_less_flat(false),
          new_imu_trans(false),
          num_edge_points_last(0),
          num_planar_points_last(0),
          num_skip_frame(1),
          frame_count(0),
          min_edge_point_threshold(10),
          min_planar_point_threshold(100),
          max_lm_iteration(25),
          transformation_recalculate_iteration(5),
          nearest_neighbor_cutoff(25),
          tolerance_rotation(0.1),
          tolerance_translation(0.1)
        { 
          transform.fill(0);
          transform_sum.fill(0); 
        }
    
    void spin();
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg);
    void edge_points_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_sharp_msg);
    void edge_points_less_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_less_sharp_msg);
    void planar_points_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_flat_msg);
    void planar_points_less_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_less_flat_msg);
    void imu_trans_callback(const sensor_msgs::PointCloud2ConstPtr& imu_trans_msg);

private:    
    LidarOdometry();
    void reproject_to_start(const PointXYZI& pi, PointXYZI& po);
    void reproject_to_end(const PointXYZI& pi, PointXYZI& po);
    bool new_data_received();
    void reset();
    void initialize();
    void process();
};


void LidarOdometry::point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg) {
    // Record timestamp
    time_point_cloud = point_cloud_msg->header.stamp.toSec();
    
    // Preprocess point cloud
    point_cloud->clear();
    pcl::fromROSMsg(*point_cloud_msg, *point_cloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*point_cloud, *point_cloud, indices);
    
    // Update the flag
    new_point_cloud = true;
}


void LidarOdometry::edge_points_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_sharp_msg) {
    // Record timestamp
    time_edge_points_sharp = edge_points_sharp_msg->header.stamp.toSec();
    
    // Preprocess point cloud
    edge_points_sharp->clear();
    pcl::fromROSMsg(*edge_points_sharp_msg, *edge_points_sharp);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*edge_points_sharp, *edge_points_sharp, indices);
    
    // Update the flag
    new_edge_points_sharp = true;
}


void LidarOdometry::edge_points_less_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_less_sharp_msg) {
    // Record timestamp
    time_edge_points_less_sharp = edge_points_less_sharp_msg->header.stamp.toSec();
    
    // Preprocess point cloud
    edge_points_less_sharp->clear();
    pcl::fromROSMsg(*edge_points_less_sharp_msg, *edge_points_less_sharp);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*edge_points_less_sharp, *edge_points_less_sharp, indices);
    
    // Update the flag
    new_edge_points_less_sharp = true;
}


void LidarOdometry::planar_points_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_flat_msg) {
    // Record timestamp
    time_planar_points_flat = planar_points_flat_msg->header.stamp.toSec();
    
    // Preprocess point cloud
    planar_points_flat->clear();
    pcl::fromROSMsg(*planar_points_flat_msg, *planar_points_flat);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*planar_points_flat, *planar_points_flat, indices);
    
    // Update the flag
    new_planar_points_flat = true;
}


void LidarOdometry::planar_points_less_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_less_flat_msg) {
    // Record timestamp
    time_planar_points_less_flat = planar_points_less_flat_msg->header.stamp.toSec();
    
    // Preprocess point cloud
    planar_points_less_flat->clear();
    pcl::fromROSMsg(*planar_points_less_flat_msg, *planar_points_less_flat);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*planar_points_less_flat, *planar_points_less_flat, indices);
    
    // Update the flag
    new_planar_points_less_flat = true;
}


void LidarOdometry::imu_trans_callback(const sensor_msgs::PointCloud2ConstPtr& imu_trans_msg) {
    // Record timestamp
    time_imu_trans = imu_trans_msg->header.stamp.toSec();
    
    // Update the flag
    new_imu_trans = true;
}


void LidarOdometry::reproject_to_start(const PointXYZI& pi, PointXYZI& po) {
    // Get the reletive time
    float rel_time = (1 / scan_period) * (pi.intensity - int(pi.intensity));

    // Get the transformation information
    float rx = rel_time * transform[0];
    float ry = rel_time * transform[1];
    float rz = rel_time * transform[2];
    float tx = rel_time * transform[3];
    float ty = rel_time * transform[4];
    float tz = rel_time * transform[5];

    // Define transformation matrix
    Eigen::Quaternionf q = euler_to_quaternion(rx, ry, rz);
    Eigen::Isometry3f transformation_matrix = Eigen::Isometry3f::Identity();  
    transformation_matrix.rotate(q);
    transformation_matrix.pretranslate(Eigen::Vector3f(tx, ty, tz));

    // Reproject to the start of the sweep
    Eigen::Vector3f point_in(pi.x, pi.y, pi.z);
    Eigen::Vector3f point_out = transformation_matrix.inverse() * point_in;
    
    // Assign to the output
    po.x = point_out[0];
    po.y = point_out[1];
    po.z = point_out[2];
    po.intensity = pi.intensity;
}


void LidarOdometry::reproject_to_end(const PointXYZI& pi, PointXYZI& po) {
    // Get the reletive time
    float rel_time = (1 / scan_period) * (pi.intensity - int(pi.intensity));

    // Get the transformation information
    float rx = rel_time * transform[0];
    float ry = rel_time * transform[1];
    float rz = rel_time * transform[2];
    float tx = rel_time * transform[3];
    float ty = rel_time * transform[4];
    float tz = rel_time * transform[5];

  float x1 = cos(rz) * (pi.x - tx) + sin(rz) * (pi.y - ty);
  float y1 = -sin(rz) * (pi.x - tx) + cos(rz) * (pi.y - ty);
  float z1 = (pi.z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2;

  rx = transform[0];
  ry = transform[1];
  rz = transform[2];
  tx = transform[3];
  ty = transform[4];
  tz = transform[5];

  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;

  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

    po.x = x6;
    po.y = y6;
    po.z = z6;
    po.intensity = int(pi.intensity);
}


bool LidarOdometry::new_data_received() {
    // TODO: refactor
    return new_point_cloud &&
           new_edge_points_sharp &&
           new_edge_points_less_sharp &&
           new_planar_points_flat &&
           new_planar_points_less_flat &&
           new_imu_trans &&
           fabs(time_point_cloud - time_edge_points_sharp) < 1e-7 &&
           fabs(time_point_cloud - time_edge_points_less_sharp) < 1e-7 &&
           fabs(time_point_cloud - time_planar_points_flat) < 1e-7 &&
           fabs(time_point_cloud - time_planar_points_less_flat) < 1e-7 &&
           fabs(time_point_cloud - time_imu_trans) < 1e-7;
}


void LidarOdometry::reset() {
    new_point_cloud = false;
    new_edge_points_sharp = false;
    new_edge_points_less_sharp = false;
    new_planar_points_flat = false;
    new_planar_points_less_flat = false;
    new_imu_trans = false;
}


void LidarOdometry::initialize() {
    // Update reference feature points with pointer swap
    edge_points_last.swap(edge_points_less_sharp);
    planar_points_last.swap(planar_points_less_flat);
    
    // Update reference feature point sizes
    num_edge_points_last = edge_points_last->points.size();
    num_planar_points_last = planar_points_last->points.size();

    // Store point clonds in KDTree for fast range-search
    kdtree_edge_points_last->setInputCloud(edge_points_last);
    kdtree_planar_points_last->setInputCloud(planar_points_last);

    // Publish last edge points
    sensor_msgs::PointCloud2 edge_points_last_msg;
    pcl::toROSMsg(*edge_points_last, edge_points_last_msg);
    edge_points_last_msg.header.stamp = ros::Time().fromSec(time_edge_points_less_sharp);
    edge_points_last_msg.header.frame_id = camera_frame_id;
    pub_edge_points_last.publish(edge_points_last_msg);

    // Publish last planar points
    sensor_msgs::PointCloud2 planar_points_last_msg;
    pcl::toROSMsg(*planar_points_last, planar_points_last_msg);
    planar_points_last_msg.header.stamp = ros::Time().fromSec(time_planar_points_less_flat);
    planar_points_last_msg.header.frame_id = camera_frame_id;
    pub_planar_points_last.publish(planar_points_last_msg);
} 


void LidarOdometry::process() {
    ////////////////
    // Preprocess //
    ////////////////
    
    if (!new_data_received()) {
        // Wait for new data to arrive
        return;
    }

    // Reset flags
    reset();

    // Initialize ther system at the first time data are received
    if (!system_initialized) {
        initialize();
        system_initialized = true;
        return;
    }

    // Skip if the amount of feature points is not enough
    if (num_edge_points_last <= min_edge_point_threshold || num_planar_points_last <= min_planar_point_threshold) {
        return;
    }
    
    /////////////////////////
    // Odometry Estimation //
    /////////////////////////

    // Calculate feature size
    int num_edge_points_sharp = edge_points_sharp->points.size();
    int num_planar_points_flat = planar_points_flat->points.size();
       
    // Define helper variables
    std::vector<int> point_search_id;
    std::vector<float> point_search_square_distance;
    
    // Create containers for searched points [j], [l], and [m]
    std::array<int, 80000> edge_point_j_id;
    std::array<int, 80000> edge_point_l_id;
    std::array<int, 80000> planar_point_j_id;
    std::array<int, 80000> planar_point_l_id;
    std::array<int, 80000> planar_point_m_id;
    
    // L-M optimization
    for (int iter_count = 0; iter_count < max_lm_iteration; iter_count++) {

        // Define variables for selected points [i], [j], [m]
        PointXYZI point_i, point_j, point_l, point_m;
    
        
        // Clear containers for new constrains
        constrain_points->clear();
        constrain_parameters->clear();

        // Process edge points
        for (int i = 0; i < num_edge_points_sharp; i++) {
        
            // Reproject the selected point [i] to the start of the sweep
            reproject_to_start(edge_points_sharp->points[i], point_i);

            // Recalculate the transformation after transformation_recalculate_iteration iterations
            if (iter_count % transformation_recalculate_iteration == 0) {
                
                // Initialize
                // TODO: refactor
                edge_point_j_id[i] = -1;
                edge_point_l_id[i] = -1;

                // Find the nearest neighbor [j]
                kdtree_edge_points_last->nearestKSearch(point_i, 1, point_search_id, point_search_square_distance);

                // Find the nearest neighbor of [i] in the two consecutive scans to the scan of [j] as [l]
                if (point_search_square_distance[0] < nearest_neighbor_cutoff) {
                    
                    // Update [j]
                    edge_point_j_id[i] = point_search_id[0];
                    int point_j_scan_id = int(edge_points_last->points[edge_point_j_id[i]].intensity);

                    // Define helper variables
                    float point_squared_distance;
                    float min_point_l_squared_distance = nearest_neighbor_cutoff;

                    // Find closest neighbor of [i] in the upper consecutive scan to the scan of [j]
                    for (int p = edge_point_j_id[i] + 1; p < num_edge_points_last; p++) {
                        // Verify whether the point is in the upper consecutive scan to the scan of [j]
                        if (int(edge_points_last->points[p].intensity) > point_j_scan_id + 2) {
                            break;
                        } else if (int(edge_points_last->points[p].intensity) != point_j_scan_id + 2) {
                            continue;
                        }

                        // Calculate squared distance between [i] and the candidate point
                        point_squared_distance = squared_distance(edge_points_last->points[p], point_i);
                        if (point_squared_distance < min_point_l_squared_distance) {
                            min_point_l_squared_distance = point_squared_distance;
                            edge_point_l_id[i] = p;
                        }
                    }  // for (int p = edge_point_j_id[i] + 1; p < num_edge_points_last; p++)
                    
                    // Find closest neighbor of [i] in the lower consecutive scan to the scan of [j]
                    for (int p = edge_point_j_id[i] - 1; p >= 0; p--) {
                        
                        // Verify whether the point is in the lower consecutive scan to the scan of [j]
                        if (int(edge_points_last->points[p].intensity) < point_j_scan_id - 2) {
                            break;
                        } else if (int(edge_points_last->points[p].intensity) != point_j_scan_id - 2) {
                            continue;
                        }

                        // Calculate squared distance between [i] and the candidate point
                        point_squared_distance = squared_distance(edge_points_last->points[p], point_i);
                        if (point_squared_distance < min_point_l_squared_distance) {
                            min_point_l_squared_distance = point_squared_distance;
                            edge_point_l_id[i] = p;
                        }
                    }  // for (int p = edge_point_j_id[i] - 1; p >= 0; p--)

                }  // if (point_search_square_distance[0] < nearest_neighbor_cutoff)

            }  // if (iter_count % transformation_recalculate_iteration == 0)


            // Calculate the distance from [i] to line (jl) if [l] is found
            if (edge_point_l_id[i] >= 0) {

                // Get searched points
                point_j = edge_points_last->points[edge_point_j_id[i]];
                point_l = edge_points_last->points[edge_point_l_id[i]];

                // Get vectors
                Eigen::Vector3f v_ij(point_j.x - point_i.x, point_j.y - point_i.y, point_j.z - point_i.z);    
                Eigen::Vector3f v_il(point_l.x - point_i.x, point_l.y - point_i.y, point_l.z - point_i.z);    
                Eigen::Vector3f v_jl(point_l.x - point_j.x, point_l.y - point_j.y, point_l.z - point_j.z);    

                // Calculate distance
                float area = v_ij.cross(v_il).norm();
                float dist_jl = v_jl.norm();
                float dist = area / dist_jl;

                // Partial derivative (for the construction of the Jacobian matirx)
                float dist_dxi = -(  v_jl[1] * (v_ij[0]*v_il[1] - v_il[0]*v_ij[1]) 
                                   + v_jl[2] * (v_ij[0]*v_il[2] - v_il[0]*v_ij[2]) ) / area / dist_jl;
                float dist_dyi =  (  v_jl[0] * (v_ij[0]*v_il[1] - v_il[0]*v_ij[1]) 
                                   - v_jl[2] * (v_ij[1]*v_il[2] - v_il[1]*v_ij[2]) ) / area / dist_jl;
                float dist_dzi =  (  v_jl[0] * (v_ij[0]*v_il[2] - v_il[0]*v_ij[2]) 
                                   + v_jl[1] * (v_ij[1]*v_il[2] - v_il[1]*v_ij[2]) ) / area / dist_jl;
                
                // Define weight
                // TODO: remove hardcoded parameters
                float s = (iter_count < 5) ? 1 : 1 - 1.8 * dist;

                // Add weight to the constrain
                PointXYZI param;
                param.x = s * dist_dxi;
                param.y = s * dist_dyi;
                param.z = s * dist_dzi;
                param.intensity = s * dist;

                // Concatenate constrains
                // TODO: remove hardcoded parameters
                // TODO: dist > 0 
                if (s > 0.1 && dist != 0) {
                    //constrain_points->push_back(point_i);  // TODO: use edge_points_sharp->points[i] or point_i?
                    constrain_points->push_back(edge_points_sharp->points[i]);
                    constrain_parameters->push_back(param);
                }
            } // if (edge_point_l_id[i] > 0)

        }  // for (int i = 0; i < num_edge_points_sharp; i++)
         
        // Process planar points
        for (int i = 0; i < num_planar_points_flat; i++) {
            
            // Reproject the selected point [i] to the start of the sweep
            reproject_to_start(planar_points_flat->points[i], point_i);
            
            // Recalculate the transformation after transformation_recalculate_iteration iterations
            if (iter_count % transformation_recalculate_iteration == 0) {
               
                // Initialize
                // TODO: refactor
                planar_point_j_id[i] = -1;
                planar_point_l_id[i] = -1;
                planar_point_m_id[i] = -1;

                // Find the nearest neighbor [j]
                kdtree_planar_points_last->nearestKSearch(point_i, 1, point_search_id, point_search_square_distance);

                // Find the nearest neighbor of [i] in the two consecutive scans to the scan of [j] as [m], 
                // and find the nearest neighbor of [i] in the same scan of [j] as [l]
                if (point_search_square_distance[0] < nearest_neighbor_cutoff) {
                    
                    // Update [j]
                    planar_point_j_id[i] = point_search_id[0];
                    int point_j_scan_id = int(planar_points_last->points[planar_point_j_id[i]].intensity);

                    // Define helper variables
                    float point_squared_distance;
                    float min_point_l_squared_distance = nearest_neighbor_cutoff;
                    float min_point_m_squared_distance = nearest_neighbor_cutoff;

                    // Find closest neighbor of [i] in the upper consecutive scan to the scan of [j]
                    for (int p = planar_point_j_id[i] + 1; p < num_planar_points_last; p++) {
                        // Verify whether the point is in the upper consecutive scan to the scan of [j]
                        if (int(planar_points_last->points[p].intensity) > point_j_scan_id + 2) {
                            break;
                        }
                        
                        // Calculate squared distance between [i] and the candidate point
                        if (int(planar_points_last->points[p].intensity) == point_j_scan_id) {
                            // Find [l]
                            point_squared_distance = squared_distance(planar_points_last->points[p], point_i);
                            if (point_squared_distance < min_point_l_squared_distance) {
                                min_point_l_squared_distance = point_squared_distance;
                                planar_point_l_id[i] = p;
                            }
                        } else if (int(planar_points_last->points[p].intensity) == point_j_scan_id + 2) {
                            // Find [m]
                            point_squared_distance = squared_distance(planar_points_last->points[p], point_i);
                            if (point_squared_distance < min_point_m_squared_distance) {
                                min_point_m_squared_distance = point_squared_distance;
                                planar_point_m_id[i] = p;
                            }
                        }
                    }  // for (int p = planar_point_j_id[i] + 1; p < num_planar_points_last; p++)
                    
                    // Find closest neighbor of [i] in the lower consecutive scan to the scan of [j]
                    for (int p = planar_point_j_id[i] - 1; p >= 0; p--) {
            
                        // Verify whether the point is in the lower consecutive scan to the scan of [j]
                        if (int(planar_points_last->points[p].intensity) < point_j_scan_id - 2) {
                            break;
                        }

                        // Calculate squared distance between [i] and the candidate point
                        if (int(planar_points_last->points[p].intensity) == point_j_scan_id) {
                            // Find [l]
                            point_squared_distance = squared_distance(planar_points_last->points[p], point_i);
                            if (point_squared_distance < min_point_l_squared_distance) {
                                min_point_l_squared_distance = point_squared_distance;
                                planar_point_l_id[i] = p;
                            }
                        } else if (int(planar_points_last->points[p].intensity) == point_j_scan_id - 2) {
                            // Find [m]
                            point_squared_distance = squared_distance(planar_points_last->points[p], point_i);
                            if (point_squared_distance < min_point_m_squared_distance) {
                                min_point_m_squared_distance = point_squared_distance;
                                planar_point_m_id[i] = p;
                            }
                        }
                    }  // for (int p = planar_point_j_id[i] - 1; p >= 0; p--)

                }  // if (point_search_square_distance[0] < nearest_neighbor_cutoff)
           
            }  // if (iter_count % transformation_recalculate_iteration == 0)

            // Calculate the distance from [i] to surface (jlm) if [l] and [m] are found
            if (planar_point_l_id[i] >= 0 && planar_point_m_id[i] >= 0) {
                
                // Get searched points
                point_j = planar_points_last->points[planar_point_j_id[i]];
                point_l = planar_points_last->points[planar_point_l_id[i]];
                point_m = planar_points_last->points[planar_point_m_id[i]];
                
                // Get vectors
                Eigen::Vector3f v_jl(point_l.x - point_j.x, point_l.y - point_j.y, point_l.z - point_j.z);    
                Eigen::Vector3f v_jm(point_m.x - point_j.x, point_m.y - point_j.y, point_m.z - point_j.z);
                Eigen::Vector3f v_ji(point_i.x - point_j.x, point_i.y - point_j.y, point_i.z - point_j.z);
                
                // Helper variables
                Eigen::Vector3f v_cross = v_jl.cross(v_jm);

                // Calculate the area
                float area = v_cross.norm();

                // Calculate the distance
                Eigen::Vector3f v_normal = v_cross / area;
                float dist = v_normal.dot(v_ji);
                
                // Define weight
                // TODO: remove hardcoded parameters
                float s = (iter_count < 5) ? 1 : 1 - 1.8 * dist / point_depth(point_i);

                // Add weight to the constrain
                PointXYZI param;
                param.x = s * v_normal[0];
                param.y = s * v_normal[1];
                param.z = s * v_normal[2];
                param.intensity = s * dist;

                // Concatenate constrains
                // TODO: remove hardcoded parameters
                // TODO: dist > 0
                if (s > 0.1 && dist != 0) {
                    //constrain_points->push_back(point_i);  // TODO: use planar_points_flat->points[i] or point_i?
                    constrain_points->push_back(planar_points_flat->points[i]);
                    constrain_parameters->push_back(param);
                }
            
            }  // if (planar_point_l_id[i] >= 0 && planar_point_m_id[i] >= 0)

        }  // for (int i = 0; i < num_planar_points_flat; i++)
       
        // Calculate the number of constrains
        int num_constrains = constrain_points->points.size();
        if (num_constrains < 10) {
            continue;
        }

        // Helper matrices
        Eigen::Matrix<float, Eigen::Dynamic, 6> mat_A(num_constrains, 6);
        Eigen::Matrix<float, 6, Eigen::Dynamic> mat_At(6, num_constrains);
        Eigen::Matrix<float, 6, 6>              mat_AtA;
        Eigen::VectorXf                         mat_B(num_constrains);
        Eigen::Matrix<float, 6, 1>              mat_AtB;
        Eigen::Matrix<float, 6, 1>              mat_X;

        // Build the Jacobian matrix
        for (int i = 0; i < num_constrains; i++) {
        
            const PointXYZI& point = constrain_points->points[i];
            const PointXYZI& param = constrain_parameters->points[i];

            float s = 1;

            float srx = sin(s * transform[0]);
            float crx = cos(s * transform[0]);
            float sry = sin(s * transform[1]);
            float cry = cos(s * transform[1]);
            float srz = sin(s * transform[2]);
            float crz = cos(s * transform[2]);
            float tx = s * transform[3];
            float ty = s * transform[4];
            float tz = s * transform[5];

            float arx = (-s * crx*sry*srz*point.x + s * crx*crz*sry*point.y + s * srx*sry*point.z
                        + s * tx*crx*sry*srz      - s * ty*crx*crz*sry      - s * tz*srx*sry)     * param.x
                      + ( s * srx*srz*point.x     - s * crz*srx*point.y     + s * crx*point.z
                        + s * ty*crz*srx          - s * tz*crx              - s * tx*srx*srz)     * param.y
                      + ( s * crx*cry*srz*point.x - s * crx*cry*crz*point.y - s * cry*srx*point.z
                        + s * tz*cry*srx          + s * ty*crx*cry*crz      - s * tx*crx*cry*srz) * param.z;

            float ary = ( (-s * crz*sry     - s * cry*srx*srz) * point.x
                        + ( s * cry*crz*srx - s * sry*srz)     * point.y 
                        -   s * crx*cry                        * point.z
                        + tx * (s*crz*sry + s * cry*srx*srz) 
                        + ty * (s*sry*srz - s * cry*crz*srx)
                        + s * tz*crx*cry                                 ) * param.x
                      + ( ( s * cry*crz     - s * srx*sry*srz) * point.x
                        + ( s * cry*srz     + s * crz*srx*sry) * point.y 
                        -   s * crx*sry                        * point.z
                        + s  * tz*crx*sry 
                        - ty * (s*cry*srz + s * crz*srx*sry)
                        - tx * (s*cry*crz - s * srx*sry*srz)             ) * param.z;

            float arz = ( (-s * cry*srz - s * crz*srx*sry)*point.x 
                        + ( s * cry*crz - s * srx*sry*srz)*point.y
                        + tx * (s*cry*srz + s * crz*srx*sry) 
                        - ty * (s*cry*crz - s * srx*sry*srz)             ) * param.x
                      + (-s * crx*crz*point.x 
                        - s * crx*srz*point.y
                        + s * ty*crx*srz 
                        + s * tx*crx*crz                                 ) * param.y
                      + ( (s * cry*crz*srx - s * sry*srz)      * point.x 
                        + (s * crz*sry     + s * cry*srx*srz)  * point.y
                        + tx * (s*sry*srz  - s * cry*crz*srx)
                        - ty * (s*crz*sry  + s * cry*srx*srz)            ) * param.z;

            float atx = -s * (cry*crz - srx * sry*srz) * param.x 
                      +  s *  crx*srz                  * param.y
                      -  s * (crz*sry + cry * srx*srz) * param.z;

            float aty = -s * (cry*srz + crz * srx*sry) * param.x 
                      -  s *  crx*crz                  * param.y
                      -  s * (sry*srz - cry * crz*srx) * param.z;

            float atz = s * crx*sry * param.x 
                      - s * srx     * param.y 
                      - s * crx*cry * param.z;

            float d2 = param.intensity;

            mat_A(i, 0) = arx;
            mat_A(i, 1) = ary;
            mat_A(i, 2) = arz;
            mat_A(i, 3) = atx;
            mat_A(i, 4) = aty;
            mat_A(i, 5) = atz;
            mat_B(i, 0) = -0.05 * d2;  // TODO: remove the hardcoded parameter
        }  // for (int i = 0; i < num_constrains; i++)
        
        // Solve
        mat_At = mat_A.transpose();
        mat_AtA = mat_At * mat_A;
        mat_AtB = mat_At * mat_B;
        mat_X = mat_AtA.colPivHouseholderQr().solve(mat_AtB);
        
        // Update transformation
        transform[0] += mat_X(0, 0);
        transform[1] += mat_X(1, 0);
        transform[2] += mat_X(2, 0);
        transform[3] += mat_X(3, 0);
        transform[4] += mat_X(4, 0);
        transform[5] += mat_X(5, 0);

        // Check whether the value of the transformation is finit
        for (auto& ele : transform) {
            if (!pcl_isfinite(ele)) {
                ele = 0;
            }
        }

        // Calculate rotation difference norm translation difference norm
        float dr_norm = Eigen::Vector3f(transform[0], transform[1], transform[2]).norm();
        float dt_norm = Eigen::Vector3f(transform[3], transform[4], transform[5]).norm();


        // Convergece check
        // TODO: refactor
        if (dr_norm * 180 / M_PI < tolerance_rotation && dt_norm * 100 < tolerance_translation) {
            break;
        }

    }  // for (int iter_count = 0; iter_count < max_lm_iteration; iter_count++)

    // Reproject point clouds to the time at the end of the sweep
    for (auto& p : edge_points_less_sharp->points) {
        reproject_to_end(p, p);
    }
    for (auto& p : planar_points_less_flat->points) {
        reproject_to_end(p, p);
    }

    // Update reference feature points with pointer swap
    edge_points_last.swap(edge_points_less_sharp);
    planar_points_last.swap(planar_points_less_flat);
    
    // Update reference feature point sizes
    num_edge_points_last = edge_points_last->points.size();
    num_planar_points_last = planar_points_last->points.size();

    // Store point clonds in KDTree for fast range-search
    if (num_edge_points_last > min_edge_point_threshold && num_planar_points_last > min_planar_point_threshold) {
        kdtree_edge_points_last->setInputCloud(edge_points_last);
        kdtree_planar_points_last->setInputCloud(planar_points_last);
    }

    // Frame Transformation
    // TODO: from zxy to xyz
    // TODO: refactor
////////////////////////////////////////////////////////////////////////////////////////////////////
    float rx, ry, rz, tx, ty, tz;
    AccumulateRotation(transform_sum[0], transform_sum[1], transform_sum[2], 
                       -transform[0], -transform[1] * 1.05, -transform[2], rx, ry, rz);
    
    
    float x1 = cos(rz) * transform[3] 
             - sin(rz) * transform[4];
    float y1 = sin(rz) * transform[3]
             + cos(rz) * transform[4];
    float z1 = transform[5] * 1.05;

    float x2 = x1;
    float y2 = cos(rx) * y1 - sin(rx) * z1;
    float z2 = sin(rx) * y1 + cos(rx) * z1;

    tx = transform_sum[3] - (cos(ry) * x2 + sin(ry) * z2);
    ty = transform_sum[4] - y2;
    tz = transform_sum[5] - (-sin(ry) * x2 + cos(ry) * z2);

    transform_sum[0] = rx;
    transform_sum[1] = ry;
    transform_sum[2] = rz;
    transform_sum[3] = tx;
    transform_sum[4] = ty;
    transform_sum[5] = tz;
    
    // To quaternion
    // TODO: from zxy to xyz
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(rz, -rx, -ry);
////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Broadcast transformation
    // TODO: from zxy to xyz
    tf::StampedTransform lidar_odometry_trans;
    lidar_odometry_trans.stamp_ = ros::Time().fromSec(time_point_cloud);
    lidar_odometry_trans.frame_id_ = camera_init_frame_id;
    lidar_odometry_trans.child_frame_id_ = camera_odom_frame_id;
    lidar_odometry_trans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    lidar_odometry_trans.setOrigin(tf::Vector3(tx, ty, tz));
    tf_broadcaster.sendTransform(lidar_odometry_trans);
    
    // Just for testing
    // TODO: remove this
    lidar_odometry_trans.child_frame_id_ = camera_frame_id;  //TODO: use camera_odom_frame_id
    tf_broadcaster.sendTransform(lidar_odometry_trans);

    // Publish lidar odometry
    // TODO: from zxy to xyz
    nav_msgs::Odometry lidar_odometry_msg;
    lidar_odometry_msg.header.stamp = ros::Time().fromSec(time_point_cloud);
    lidar_odometry_msg.header.frame_id = camera_init_frame_id;
    lidar_odometry_msg.child_frame_id = camera_odom_frame_id;
    lidar_odometry_msg.pose.pose.orientation.x = -geoQuat.y;
    lidar_odometry_msg.pose.pose.orientation.y = -geoQuat.z;
    lidar_odometry_msg.pose.pose.orientation.z = geoQuat.x;
    lidar_odometry_msg.pose.pose.orientation.w = geoQuat.w;
    lidar_odometry_msg.pose.pose.position.x = tx;
    lidar_odometry_msg.pose.pose.position.y = ty;
    lidar_odometry_msg.pose.pose.position.z = tz;
    pub_lidar_odometry.publish(lidar_odometry_msg);

    // Skip frames
    if (frame_count % num_skip_frame == 0) {
         
        // Reproject the point cloud to the time at the end of the sweep
        for (auto& p : point_cloud->points) {
            reproject_to_end(p, p);
        }
        
        // Publish the reprojected full resolution point cloud message
        sensor_msgs::PointCloud2 point_cloud_reprojected_msg;
        pcl::toROSMsg(*point_cloud, point_cloud_reprojected_msg);
        point_cloud_reprojected_msg.header.stamp = ros::Time().fromSec(time_point_cloud);
        point_cloud_reprojected_msg.header.frame_id = camera_frame_id;  
        pub_point_cloud_reprojected.publish(point_cloud_reprojected_msg);
   
        // Publish the reprojected less sharp edge point cloud message
        sensor_msgs::PointCloud2 edge_points_last_msg;
        pcl::toROSMsg(*edge_points_last, edge_points_last_msg);
        edge_points_last_msg.header.stamp = ros::Time().fromSec(time_point_cloud);
        edge_points_last_msg.header.frame_id = camera_frame_id;
        pub_edge_points_last.publish(edge_points_last_msg);
   
        // Publish the reprojected less flat planar point cloud message
        sensor_msgs::PointCloud2 planar_points_last_msg;
        pcl::toROSMsg(*planar_points_last, planar_points_last_msg);
        planar_points_last_msg.header.stamp = ros::Time().fromSec(time_point_cloud);
        planar_points_last_msg.header.frame_id = camera_frame_id;
        pub_planar_points_last.publish(planar_points_last_msg);
    }
    
    // Update frame count
    frame_count++;

    // Debug
    ROS_INFO("XD");  

}


void LidarOdometry::spin() {
    ros::Rate ros_rate(100);
    
    // loop until shutdown TODO: use shutdown hook
    while (ros::ok()) {
        ros::spinOnce();

        // Try processing new data
        process();

        ros_rate.sleep();
    }
}


int main(int argc, char** argv) {
    
    ros::init(argc, argv, "lidar_odometry");
    ros::NodeHandle node;

    LidarOdometry lidar_odometry(node);

    /////////////////
    // Subscribers //
    /////////////////

    ros::Subscriber sub_point_cloud = node.subscribe("/velodyne_cloud_denoised", 2, 
                                                     &LidarOdometry::point_cloud_callback,
                                                     &lidar_odometry);
    ros::Subscriber sub_edge_points_sharp = node.subscribe("/edge_points_sharp", 2, 
                                                     &LidarOdometry::edge_points_sharp_callback,
                                                     &lidar_odometry);
    ros::Subscriber sub_edge_points_less_sharp = node.subscribe("/edge_points_less_sharp", 2, 
                                                     &LidarOdometry::edge_points_less_sharp_callback,
                                                     &lidar_odometry);
    ros::Subscriber sub_planar_points_flat = node.subscribe("/planar_points_flat", 2, 
                                                     &LidarOdometry::planar_points_flat_callback,
                                                     &lidar_odometry);
    ros::Subscriber sub_planar_points_less_flat = node.subscribe("/planar_points_less_flat", 2, 
                                                     &LidarOdometry::planar_points_less_flat_callback,
                                                     &lidar_odometry);
    ros::Subscriber sub_imu_trans = node.subscribe("/imu_trans", 5, 
                                                     &LidarOdometry::imu_trans_callback,
                                                     &lidar_odometry);

    lidar_odometry.spin();

    return 0;
}
