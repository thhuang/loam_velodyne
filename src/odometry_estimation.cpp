// TH Huang

#include <array>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>

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


class OdometryEstimator {
    ros::NodeHandle& node;
    ros::Publisher pub_edge_points_last;
    ros::Publisher pub_planar_points_last;
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

public:
    OdometryEstimator(ros::NodeHandle& node) 
        : node(node),
          pub_edge_points_last(node.advertise<sensor_msgs::PointCloud2>("/edge_points_last", 2)),
          pub_planar_points_last(node.advertise<sensor_msgs::PointCloud2>("/planar_points_last", 2)),
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
          min_edge_point_threshold(10),
          min_planar_point_threshold(100),
          max_lm_iteration(25),
          transformation_recalculate_iteration(5),
          nearest_neighbor_cutoff(25)
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
    OdometryEstimator();
    void reproject_to_start(const PointXYZI& pi, PointXYZI& po);
    bool new_data_received();
    void reset();
    void initialize();
    void process();
};


void OdometryEstimator::point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg) {
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


void OdometryEstimator::edge_points_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_sharp_msg) {
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


void OdometryEstimator::edge_points_less_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_less_sharp_msg) {
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


void OdometryEstimator::planar_points_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_flat_msg) {
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


void OdometryEstimator::planar_points_less_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_less_flat_msg) {
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


void OdometryEstimator::imu_trans_callback(const sensor_msgs::PointCloud2ConstPtr& imu_trans_msg) {
    // Record timestamp
    time_imu_trans = imu_trans_msg->header.stamp.toSec();
    
    // Update the flag
    new_imu_trans = true;
}


void OdometryEstimator::reproject_to_start(const PointXYZI& pi, PointXYZI& po) {
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


bool OdometryEstimator::new_data_received() {
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


void OdometryEstimator::reset() {
    new_point_cloud = false;
    new_edge_points_sharp = false;
    new_edge_points_less_sharp = false;
    new_planar_points_flat = false;
    new_planar_points_less_flat = false;
    new_imu_trans = false;
}


void OdometryEstimator::initialize() {
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


void OdometryEstimator::process() {
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
    if (num_edge_points_last < min_edge_point_threshold || num_planar_points_last < min_planar_point_threshold) {
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
    
    // L-M optimization
    for (int iter_count = 0; iter_count < max_lm_iteration; iter_count++) {

        // Define variables for the selected points [i], [j], [m]
        PointXYZI point_i, point_j, point_l, point_m;
        
        // Clear containers for new constrains
        constrain_points->clear();
        constrain_parameters->clear();

        // Process edge points
        for (int i = 0; i < num_edge_points_sharp; i++) {
            
            // Reproject the selected point [i] to the start of the sweep
            reproject_to_start(edge_points_sharp->points[i], point_i);
            
            // Create containers for neighbor points [j] and [l]
            std::array<int, 80000> point_j_id;
            std::array<int, 80000> point_l_id;
            point_j_id.fill(-1);
            point_l_id.fill(-1);

            // Recalculate the transformation after transformation_recalculate_iteration iterations
            if (iter_count % transformation_recalculate_iteration == 0) {
                
                // Find the nearest neighbor [j]
                kdtree_edge_points_last->nearestKSearch(point_i, 1, point_search_id, point_search_square_distance);
                point_j_id[i] = point_search_id[0];
                int point_j_square_distance = point_search_square_distance[0];
                int point_j_scan_id = int(edge_points_last->points[point_j_id[i]].intensity);

                // Find the nearest neighbor of [i] in the two consecutive scans to the scan of [j] as [l]
                if (point_j_square_distance < nearest_neighbor_cutoff) {
                    
                    // Define helper variables
                    float point_squared_distance;
                    float min_point_l_squared_distance = nearest_neighbor_cutoff;

                    // Find closest neighbor of [i] in the upper consecutive scan to the scan of [j]
                    for (int p = point_j_id[i] + 1; p < num_edge_points_last; p++) {
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
                            point_l_id[i] = p;
                        }
                    }  // for (int p = point_j_id[i] + 1; p < num_edge_points_last; p++)
                    
                    // Find closest neighbor of [i] in the lower consecutive scan to the scan of [j]
                    for (int p = point_j_id[i] - 1; p >= 0; p--) {
                        
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
                            point_l_id[i] = p;
                        }
                    }  // for (int p = point_j_id[i] - 1; p >= 0; p--)

                }  // if (point_j_square_distance < nearest_neighbor_cutoff)

            }  // if (iter_count % transformation_recalculate_iteration == 0)


            // Calculate the distance from [i] to line (jl) if [l] is found
            if (point_l_id[i] >= 0) {

                // Get searched points
                point_j = edge_points_last->points[point_j_id[i]];
                point_l = edge_points_last->points[point_l_id[i]];

                // Get vectors
                Eigen::Vector3f v_ij(point_j.x - point_i.x, point_j.y - point_i.y, point_j.z - point_i.z);    
                Eigen::Vector3f v_il(point_l.x - point_i.x, point_l.y - point_i.y, point_l.z - point_i.z);    
                Eigen::Vector3f v_jl(point_l.x - point_j.x, point_l.y - point_j.y, point_l.z - point_j.z);    

                // Calculate distance
                float area = v_ij.cross(v_il).norm();
                float dist = area / v_jl.norm();

                // Partial derivative (for the construction of the Jacobian matirx)
                float dist_dxi = -(  v_jl[1] * (v_ij[0]*v_il[1] - v_il[0]*v_ij[1]) 
                                   + v_jl[2] * (v_ij[0]*v_il[2] - v_il[0]*v_ij[2]) ) / area / dist;
                float dist_dyi =  (  v_jl[0] * (v_ij[0]*v_il[1] - v_il[0]*v_ij[1]) 
                                   - v_jl[2] * (v_ij[1]*v_il[2] - v_il[1]*v_ij[2]) ) / area / dist;
                float dist_dzi =  (  v_jl[0] * (v_ij[0]*v_il[2] - v_il[0]*v_ij[2]) 
                                   + v_jl[1] * (v_ij[1]*v_il[2] - v_il[1]*v_ij[2]) ) / area / dist;
                
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
                if (s > 0.1 && dist > 0) {
                    constrain_points->push_back(point_i);  // TODO: use edge_points_sharp->points[i] or not?
                    constrain_parameters->push_back(param);
                }
            } // if (point_l_id[i] > 0)

        }  // for (int i = 0; i < num_edge_points_sharp; i++)
        
        // Process planar points
        for (int i = 0; i < num_planar_points_flat; i++) {
            
            // Reproject the selected point [i] to the start of the sweep
            reproject_to_start(planar_points_flat->points[i], point_i);
            
            // Create containers for neighbor points [j], [l], and [m]
            std::array<int, 80000> point_j_id;
            std::array<int, 80000> point_l_id;
            std::array<int, 80000> point_m_id;
            point_j_id.fill(-1);
            point_l_id.fill(-1);
            point_m_id.fill(-1);

            // Recalculate the transformation after transformation_recalculate_iteration iterations
            if (iter_count % transformation_recalculate_iteration == 0) {
                
                // Find the nearest neighbor [j]
                kdtree_planar_points_last->nearestKSearch(point_i, 1, point_search_id, point_search_square_distance);
                point_j_id[i] = point_search_id[0];
                int point_j_square_distance = point_search_square_distance[0];
                int point_j_scan_id = int(planar_points_last->points[point_j_id[i]].intensity);

                // Find the nearest neighbor of [i] in the two consecutive scans to the scan of [j] as [m], 
                // and find the nearest neighbor of [i] in the same scan of [j] as [l]
                if (point_j_square_distance < nearest_neighbor_cutoff) {
                    
                    // Define helper variables
                    float point_squared_distance;
                    float min_point_l_squared_distance = nearest_neighbor_cutoff;
                    float min_point_m_squared_distance = nearest_neighbor_cutoff;

                    // Find closest neighbor of [i] in the upper consecutive scan to the scan of [j]
                    for (int p = point_j_id[i] + 1; p < num_planar_points_last; p++) {
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
                                point_l_id[i] = p;
                            }
                        } else if (int(planar_points_last->points[p].intensity) == point_j_scan_id + 2) {
                            // Find [m]
                            point_squared_distance = squared_distance(planar_points_last->points[p], point_i);
                            if (point_squared_distance < min_point_m_squared_distance) {
                                min_point_m_squared_distance = point_squared_distance;
                                point_m_id[i] = p;
                            }
                        }
                    }  // for (int p = point_j_id[i] + 1; p < num_planar_points_last; p++)
                    
                    // Find closest neighbor of [i] in the lower consecutive scan to the scan of [j]
                    for (int p = point_j_id[i] - 1; p >= 0; p--) {
            
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
                                point_l_id[i] = p;
                            }
                        } else if (int(planar_points_last->points[p].intensity) == point_j_scan_id - 2) {
                            // Find [m]
                            point_squared_distance = squared_distance(planar_points_last->points[p], point_i);
                            if (point_squared_distance < min_point_m_squared_distance) {
                                min_point_m_squared_distance = point_squared_distance;
                                point_m_id[i] = p;
                            }
                        }
                    }  // for (int p = point_j_id[i] - 1; p >= 0; p--)

                }  // if (point_j_square_distance < nearest_neighbor_cutoff)
            
            }  // if (iter_count % transformation_recalculate_iteration == 0)

            // Calculate the distance from [i] to surface (jlm) if [l] and [m] are found
            if (point_l_id[i] >= 0 && point_m_id[i] >= 0) {
                
                // Get searched points
                point_j = planar_points_last->points[point_j_id[i]];
                point_l = planar_points_last->points[point_l_id[i]];
                point_m = planar_points_last->points[point_m_id[i]];
                
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
                if (s > 0.1 && dist > 0) {
                    constrain_points->push_back(point_i);  // TODO: use planar_points_flat->points[i] or not?
                    constrain_parameters->push_back(param);
                }
            
            }  // if (point_l_id[i] >= 0 && point_m_id[i] >= 0)

        }  // for (int i = 0; i < num_planar_points_flat; i++)
       
        // Calculate the number of constrains
        int num_constrains = constrain_points->points.size();
        if (num_constrains < 10) {
            continue;
        }




    }  // for (int iter_count = 0; iter_count < max_lm_iteration; iter_count++)

    // Publish result

    // Debug
    ROS_INFO("XD");  

}


void OdometryEstimator::spin() {
    int frame_count = num_skip_frame;
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
    
    ros::init(argc, argv, "odometry_estimation");
    ros::NodeHandle node;

    OdometryEstimator odometry_estimator(node);

    /////////////////
    // Subscribers //
    /////////////////

    ros::Subscriber sub_point_cloud = node.subscribe("/velodyne_cloud_2", 2, 
                                                     &OdometryEstimator::point_cloud_callback,
                                                     &odometry_estimator);
    ros::Subscriber sub_edge_points_sharp = node.subscribe("/edge_points_sharp", 2, 
                                                     &OdometryEstimator::edge_points_sharp_callback,
                                                     &odometry_estimator);
    ros::Subscriber sub_edge_points_less_sharp = node.subscribe("/edge_points_less_sharp", 2, 
                                                     &OdometryEstimator::edge_points_less_sharp_callback,
                                                     &odometry_estimator);
    ros::Subscriber sub_planar_points_flat = node.subscribe("/planar_points_flat", 2, 
                                                     &OdometryEstimator::planar_points_flat_callback,
                                                     &odometry_estimator);
    ros::Subscriber sub_planar_points_less_flat = node.subscribe("/planar_points_less_flat", 2, 
                                                     &OdometryEstimator::planar_points_less_flat_callback,
                                                     &odometry_estimator);
    ros::Subscriber sub_imu_trans = node.subscribe("/imu_trans", 5, 
                                                     &OdometryEstimator::imu_trans_callback,
                                                     &odometry_estimator);

    odometry_estimator.spin();

    return 0;
}
