// TH Huang

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>


using pcl::PointXYZI;
using pcl::PointXYZ;


inline double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}


inline double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
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


class DataRegistrar {
    ros::NodeHandle& node;
    ros::Publisher pub_point_cloud;
    ros::Publisher pub_edge_points_sharp;
    ros::Publisher pub_edge_points_less_sharp;
    ros::Publisher pub_planar_points_flat;
    ros::Publisher pub_planar_points_less_flat;
    ros::Publisher pub_imu_trans;
    std::string    frame_id;
    bool           system_initiated;
    unsigned       system_init_count;
    const unsigned system_delay;
    const unsigned num_scans;
    const float    scan_period;
    const int      curvature_area; 
    const int      cloud_partition;
    const float    threshold_curvature;
    const float    threshold_parallel;
    const float    threshold_occluded;
    const int      edge_sharp_num;
    const int      edge_less_sharp_num;
    const int      planar_flat_num;
    const int      exclude_neighbor_num;
    const float    exclude_neighbor_cutoff;
    const float    filter_leaf_size;
    std::array<float, 80000> cloud_curvature;
    std::array<int,   80000> cloud_sorted_id;
    std::array<bool,  80000> cloud_avoid;
    std::array<int,   80000> cloud_label;  // 2: sharp, 1: less sharp, 0: not defined, -1: flat (TODO: use enum)

public:
    DataRegistrar(ros::NodeHandle& node) 
        : node(node),
          pub_point_cloud(node.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_denoised", 2)),
          pub_edge_points_sharp(node.advertise<sensor_msgs::PointCloud2>("/edge_points_sharp", 2)),
          pub_edge_points_less_sharp(node.advertise<sensor_msgs::PointCloud2>("/edge_points_less_sharp", 2)),
          pub_planar_points_flat(node.advertise<sensor_msgs::PointCloud2>("/planar_points_flat", 2)),
          pub_planar_points_less_flat(node.advertise<sensor_msgs::PointCloud2>("/planar_points_less_flat", 2)),
          pub_imu_trans(node.advertise<sensor_msgs::PointCloud2>("/imu_trans", 5)),
          frame_id("/camera"),
          system_init_count(0), 
          system_delay(20), 
          system_initiated(false), 
          num_scans(16),  // Number of channel (Velodyne VLP-16)
          scan_period(0.1),  // Inverse of the scan rate which is 10 Hz (Velodyne VLP-16)
          curvature_area(5),  // Index range for calculating the curvature of a point
          cloud_partition(6),  // Divide the point cloud into cloud_partition region for feature extraction
          threshold_curvature(0.1),
          threshold_parallel(0.0002),
          threshold_occluded(0.1),
          edge_sharp_num(2),
          edge_less_sharp_num(20),
          planar_flat_num(4),
          exclude_neighbor_num(5),
          exclude_neighbor_cutoff(0.05),
          filter_leaf_size(0.2)
        {}
        
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg);
    void imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg);

private:
    DataRegistrar();
};


void DataRegistrar::point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg) {
 
    ////////////////
    // Initialize //
    ////////////////

    // Delay
    if (!system_initiated) {
        system_init_count++;
        if (system_init_count >= system_delay) {
            system_initiated = true;
        }
        return;
    }

    ////////////////
    // Preprocess //
    ////////////////

    // Parse data from PointCloud2
    double time_stamp = point_cloud_msg->header.stamp.toSec();
    pcl::PointCloud<PointXYZ> point_cloud_in;
    pcl::fromROSMsg(*point_cloud_msg, point_cloud_in);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(point_cloud_in, point_cloud_in, indices);
    int cloud_size = point_cloud_in.points.size();

    // Find starting orientation
    float start_ori = -M_PI;
    for (auto iter = point_cloud_in.points.cbegin(); iter != point_cloud_in.points.cend(); iter++) {
        float angle = atan(iter->z / sqrt(iter->x * iter->x + iter->y * iter->y));
        if (!std::isnan(angle)) {
            start_ori = -atan2(iter->y, iter->x);
            break;
        }
    }
    
    // Find ending orientation
    float end_ori = M_PI;
    for (auto iter = point_cloud_in.points.crbegin(); iter != point_cloud_in.points.crend(); iter++) {
        float angle = atan(iter->z / sqrt(iter->x * iter->x + iter->y * iter->y));
        if (!std::isnan(angle)) {
            end_ori = -atan2(iter->y, iter->x) + 2 * M_PI;
            break;
        }
    }
    
    // Remap to correct ending orientation
    if (end_ori - start_ori > 3 * M_PI) {
        end_ori -= 2 * M_PI;
    } else if (end_ori - start_ori < M_PI) {
        end_ori += 2 * M_PI;
    }

    bool half_passed = false;
    int cloud_size_denoised = cloud_size;
    PointXYZI point;
    std::vector<pcl::PointCloud<PointXYZI> > point_cloud_scans(num_scans);
    for (const auto& p : point_cloud_in.points) {
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;

        /** 
         * Define scan_id based on its vertical angle
         * TODO: refactor
         * id:  0, rounded_angle: -15, angle: -15
         * id:  1, rounded_angle:   1, angle:   1
         * id:  2, rounded_angle: -13, angle: -13
         * id:  3, rounded_angle:   3, angle:   3
         * id:  4, rounded_angle: -11, angle: -11
         * id:  5, rounded_angle:   5, angle:   5
         * id:  6, rounded_angle:  -9, angle:  -9
         * id:  7, rounded_angle:   7, angle:   7
         * id:  8, rounded_angle:  -7, angle:  -7
         * id:  9, rounded_angle:   9, angle:   9
         * id: 10, rounded_angle:  -5, angle:  -5
         * id: 11, rounded_angle:  11, angle:  11
         * id: 12, rounded_angle:  -3, angle:  -3
         * id: 13, rounded_angle:  13, angle:  13
         * id: 14, rounded_angle:  -1, angle:  -1
         * id: 15, rounded_angle:  15, angle:  15
         */
        float rounded_angle = round(rad2deg(atan(point.z / sqrt(point.x * point.x + point.y * point.y))));
        int scan_id;
        if (rounded_angle > 0) {
            scan_id = rounded_angle;
        } else {
            scan_id = rounded_angle + (num_scans - 1);
        }
        // Denoise: remove points far from others
        if (scan_id > (num_scans - 1) || num_scans < 0 ) {
            cloud_size_denoised--;
            continue;
        }

        // Calculate horizontal angle
        float ori = -atan2(point.y, point.x);
        
        // Remap into [start_ori, end_ori]
        if (!half_passed) {
            if (ori < start_ori) {
                ori += 2 * M_PI;
            } else if (ori > start_ori + 2 * M_PI) {
                ori -= 2 * M_PI;
            }

            if (ori - start_ori > M_PI) {
                half_passed = true;
            }
        } else {
            ori += 2 * M_PI;
            if (ori < end_ori - 2 * M_PI) {
                ori += 2 * M_PI;
            } else if (ori > end_ori) {
                ori -= 2 * M_PI;
            } 
        }

        // Use intensity to store scan_id (decimal part) and dt (fractional part) for convenient
        float rel_time = (ori - start_ori) / (end_ori - start_ori);
        float dt = scan_period * rel_time;
        point.intensity = scan_id + dt;

        // TODO: combine with IMU data 
        
        // Classify a point by its scan_id 
        point_cloud_scans[scan_id].push_back(point);

    }  // for (const auto& p : point_cloud_in.points)
    
    // Change cloude_size to the number of points without noise
    cloud_size = cloud_size_denoised;

    // Construct a new point cloud with sorted points and record the starting and ending index for each scan
    std::vector<int> scan_start_id(1, curvature_area);
    std::vector<int> scan_end_id;
    unsigned accumulate_size;
    pcl::PointCloud<PointXYZI>::Ptr point_cloud(new pcl::PointCloud<PointXYZI>());
    for (auto iter = point_cloud_scans.begin(); iter != point_cloud_scans.end(); iter++) {
        *point_cloud += *iter;
        accumulate_size = point_cloud->points.size();
        scan_start_id.push_back(accumulate_size + curvature_area);
        scan_end_id.push_back(accumulate_size - curvature_area);
    }
    scan_start_id.erase(scan_start_id.end() - 1);
 
    ////////////////////////
    // Feature extraction //
    ////////////////////////

    // Calculate curvature
    for (int i = curvature_area; i < cloud_size - curvature_area; i++) {
        float dx = -2 * curvature_area * point_cloud->points[i].x;
        float dy = -2 * curvature_area * point_cloud->points[i].y;
        float dz = -2 * curvature_area * point_cloud->points[i].z;
        for (int j = 0; j < curvature_area; j++) {
            dx += point_cloud->points[i + j + 1].x + point_cloud->points[i - j - 1].x;
            dy += point_cloud->points[i + j + 1].y + point_cloud->points[i - j - 1].y;
            dz += point_cloud->points[i + j + 1].z + point_cloud->points[i - j - 1].z;
        }
        cloud_curvature[i] = dx * dx + dy * dy + dz * dz;
        cloud_sorted_id[i] = i;
        cloud_avoid[i] = false;
        cloud_label[i] = 0;
    }

    // Exclude points with bad quality
    for (int i = curvature_area; i < cloud_size - curvature_area - 1; i++) {        
        // Calculate the squared distance from point i to the origin
        float dist2 = point_cloud->points[i].x * point_cloud->points[i].x
                    + point_cloud->points[i].y * point_cloud->points[i].y
                    + point_cloud->points[i].z * point_cloud->points[i].z;
        
        // Calculate the squared distance to the neighbor point at the positive direction
        float dist2_pos = squared_distance(point_cloud->points[i], point_cloud->points[i + 1]);

        // Calculate the squared distance to the neighbor point at the negative direction
        float dist2_neg = squared_distance(point_cloud->points[i], point_cloud->points[i - 1]);     

        // Exclude points on a surface patch which is roughly parallel to the laser beam
        if (dist2_pos > threshold_parallel * dist2 && dist2_neg > threshold_parallel * dist2) {
            cloud_avoid[i] = true;
        }

        // Exclude points on the boundary of an occluded region
        if (dist2_pos > threshold_occluded) {
            float depth_1 = sqrt(dist2);

            // Calculate the distance from point i+1 to the origin
            float depth_2 = point_depth(point_cloud->points[i + 1]);

            if (depth_1 > depth_2) {
                // Define distance from point i+1 to the auxiliary point
                float dx_aux = point_cloud->points[i + 1].x - point_cloud->points[i].x * depth_2 / depth_1;
                float dy_aux = point_cloud->points[i + 1].y - point_cloud->points[i].y * depth_2 / depth_1;
                float dz_aux = point_cloud->points[i + 1].z - point_cloud->points[i].z * depth_2 / depth_1;
                float dist_aux = sqrt(dx_aux * dx_aux + dy_aux * dy_aux + dz_aux * dz_aux);
                
                // If the auxiliary distance is below the threshold
                if (dist_aux / depth_2 < threshold_occluded) {
                    // Avoid points behind the obstacle (negative side)
                    for (int j = 0; j < curvature_area + 1; j++) {
                        cloud_avoid[i - j] = true;
                    }
                }
            } else {  // depth_1 <= depth_2
                // Define distance from point i to the auxiliary point
                float dx_aux = point_cloud->points[i].x - point_cloud->points[i + 1].x * depth_1 / depth_2;
                float dy_aux = point_cloud->points[i].y - point_cloud->points[i + 1].y * depth_1 / depth_2;
                float dz_aux = point_cloud->points[i].z - point_cloud->points[i + 1].z * depth_1 / depth_2;
                float dist_aux = sqrt(dx_aux * dx_aux + dy_aux * dy_aux + dz_aux * dz_aux);

                // If the auxiliary distance is below the threshold
                if (dist_aux / depth_1 < threshold_occluded) {
                    // Avoid points behind the obstacle (positive side)
                    for (int j = 0; j < curvature_area + 1; j++) {
                        cloud_avoid[i + j + 1] = true;
                    }
                }
            }
        }

    }  // for (int i = curvature_area; i < cloud_size - curvature_area - 1; i++)

    // Find feature points
    pcl::PointCloud<PointXYZI> edge_points_sharp;
    pcl::PointCloud<PointXYZI> edge_points_less_sharp;
    pcl::PointCloud<PointXYZI> planar_points_flat;
    pcl::PointCloud<PointXYZI> planar_points_less_flat;

    for (int scan = 0; scan < num_scans; scan++) {
        // Container for downsampling use
        pcl::PointCloud<PointXYZI>::Ptr planar_points_less_flat_scan(new pcl::PointCloud<PointXYZI>);

        for (int i = 0; i < cloud_partition; i++) {
            // Divide the point cloud into cloud_partition region with index [sp, ep)
            const unsigned sp = (scan_start_id[scan] * (cloud_partition - i)     + scan_end_id[scan] * i) 
                              / cloud_partition;
            const unsigned ep = (scan_start_id[scan] * (cloud_partition - 1 - i) + scan_end_id[scan] * (i + 1)) 
                              / cloud_partition;

            // Make iterators
            const auto iter_cbegin  = cloud_sorted_id.begin() + sp;
            const auto iter_cend    = cloud_sorted_id.begin() + ep;
            const auto iter_crbegin = std::make_reverse_iterator(iter_cend);  // C++14
            const auto iter_crend   = std::make_reverse_iterator(iter_cbegin);  // C++14
            // const std::reverse_iterator<std::array<int, 80000>::iterator> iter_crbegin(iter_cend);  // C++11
            // const std::reverse_iterator<std::array<int, 80000>::iterator> iter_crend(iter_cbegin);  // C++11

            // Sort the index according to the value of the curvature
            std::sort(cloud_sorted_id.begin() + sp, 
                      cloud_sorted_id.begin() + ep,
                      [&](int id1, int id2) { return cloud_curvature[id1] < cloud_curvature[id2]; });

            // Find edge feature points
            int edge_picked_num = 0;
            for (auto id_iter = iter_crbegin; id_iter != iter_crend; id_iter++) {
                if (!cloud_avoid[*id_iter] && cloud_curvature[*id_iter] > threshold_curvature) {
                    // Add edge feature points to the container
                    edge_picked_num++;
                    if (edge_picked_num <= edge_sharp_num) {
                        // Top edge_sharp_num largest curvature points
                        cloud_label[*id_iter] = 2;
                        edge_points_sharp.push_back(point_cloud->points[*id_iter]);
                        edge_points_less_sharp.push_back(point_cloud->points[*id_iter]);
                    } else if (edge_picked_num <= edge_less_sharp_num) {
                        // Top edge_less_sharp_num largest curvature points
                        cloud_label[*id_iter] = 1;
                        edge_points_less_sharp.push_back(point_cloud->points[*id_iter]);
                    } else {
                        break;
                    }

                    // TODO: refactor (DRY!)
                    // Exclude neighbor points within exclude_neighbor_num and exclude_neighbor_cutoff
                    cloud_avoid[*id_iter] = true;
                    for (int j = 0; j < exclude_neighbor_num; j++) {
                        PointXYZI p1 = point_cloud->points[*id_iter + j];
                        PointXYZI p2 = point_cloud->points[*id_iter + j + 1];
                        if (squared_distance(p1, p2) > exclude_neighbor_cutoff) {
                            break;
                        }
                        cloud_avoid[*id_iter + j + 1] = true;
                    }
                    for (int j = 0; j < exclude_neighbor_num; j++) {
                        PointXYZI p1 = point_cloud->points[*id_iter - j];
                        PointXYZI p2 = point_cloud->points[*id_iter - j - 1];
                        if (squared_distance(p1, p2) > exclude_neighbor_cutoff) {
                            break;
                        }
                        cloud_avoid[*id_iter - j - 1] = true;
                    }
                } 
            } // for (auto id_iter = iter_crbegin; id_iter != iter_crend; id_iter++)

            // Find planar feature points
            int planar_picked_num = 0;
            for (auto id_iter = iter_cbegin; id_iter != iter_cend; id_iter++) {
                if (!cloud_avoid[*id_iter] && cloud_curvature[*id_iter] < threshold_curvature) {
                    // Add planar feature points to the container
                    planar_picked_num++;
                    cloud_label[*id_iter] = -1;
                    planar_points_flat.push_back(point_cloud->points[*id_iter]);

                    if (planar_picked_num >= planar_flat_num) {
                        break;
                    }

                    // TODO: refactor (DRY!)
                    // Exclude neighbor points within exclude_neighbor_num and exclude_neighbor_cutoff
                    cloud_avoid[*id_iter] = true;
                    for (int j = 0; j < exclude_neighbor_num; j++) {
                        PointXYZI p1 = point_cloud->points[*id_iter + j];
                        PointXYZI p2 = point_cloud->points[*id_iter + j + 1];
                        if (squared_distance(p1, p2) > exclude_neighbor_cutoff) {
                            break;
                        }
                        cloud_avoid[*id_iter + j + 1] = true;
                    }
                    for (int j = 0; j < exclude_neighbor_num; j++) {
                        PointXYZI p1 = point_cloud->points[*id_iter - j];
                        PointXYZI p2 = point_cloud->points[*id_iter - j - 1];
                        if (squared_distance(p1, p2) > exclude_neighbor_cutoff) {
                            break;
                        }
                        cloud_avoid[*id_iter - j - 1] = true;
                    }
                }
            }  // for (auto id_iter = iter_cbegin; id_iter != iter_cend; id_iter++)
            
            // Add points which is not in the edge to the container
            for (auto id_iter = iter_cbegin; id_iter != iter_cend; id_iter++) {
                if (cloud_label[*id_iter] <= 0) {
                   planar_points_less_flat_scan->push_back(point_cloud->points[*id_iter]);
                }
            }
        }  // for (int i = 0; i < num_scans; i++)
            
        // Downsampling with a VoxelGrid filter
        pcl::PointCloud<PointXYZI> planar_points_less_flat_scan_filtered;
        pcl::VoxelGrid<PointXYZI> voxel_grid_filter;
        voxel_grid_filter.setInputCloud(planar_points_less_flat_scan);
        voxel_grid_filter.setLeafSize(filter_leaf_size, filter_leaf_size, filter_leaf_size);
        voxel_grid_filter.filter(planar_points_less_flat_scan_filtered);

        // Add to container
        planar_points_less_flat += planar_points_less_flat_scan_filtered;

    }  // for (int scan = 0; scan < num_scans; scan++)

    // Publish denoised point cloud message
    sensor_msgs::PointCloud2 point_cloud_out_msg;
    pcl::toROSMsg(*point_cloud, point_cloud_out_msg);
    point_cloud_out_msg.header.stamp = point_cloud_msg->header.stamp;
    point_cloud_out_msg.header.frame_id = frame_id;
    pub_point_cloud.publish(point_cloud_out_msg);
   
    // Publish sharp edge point cloud message
    sensor_msgs::PointCloud2 edge_points_sharp_msg;
    pcl::toROSMsg(edge_points_sharp, edge_points_sharp_msg);
    edge_points_sharp_msg.header.stamp = point_cloud_msg->header.stamp;
    edge_points_sharp_msg.header.frame_id = frame_id;
    pub_edge_points_sharp.publish(edge_points_sharp_msg);
    
    // Publish less sharp edge point cloud message
    sensor_msgs::PointCloud2 edge_points_less_sharp_msg;
    pcl::toROSMsg(edge_points_less_sharp, edge_points_less_sharp_msg);
    edge_points_less_sharp_msg.header.stamp = point_cloud_msg->header.stamp;
    edge_points_less_sharp_msg.header.frame_id = frame_id;
    pub_edge_points_less_sharp.publish(edge_points_less_sharp_msg);

    // Publish flat surface point cloud
    sensor_msgs::PointCloud2 planar_points_flat_msg;
    pcl::toROSMsg(planar_points_flat, planar_points_flat_msg);
    planar_points_flat_msg.header.stamp = point_cloud_msg->header.stamp;
    planar_points_flat_msg.header.frame_id = frame_id;
    pub_planar_points_flat.publish(planar_points_flat_msg);

    // Publish less flat surface point cloud
    sensor_msgs::PointCloud2 planar_points_less_flat_msg;
    pcl::toROSMsg(planar_points_less_flat, planar_points_less_flat_msg);
    planar_points_less_flat_msg.header.stamp = point_cloud_msg->header.stamp;
    planar_points_less_flat_msg.header.frame_id = frame_id;
    pub_planar_points_less_flat.publish(planar_points_less_flat_msg);
    

    // Publish IMU trans ???
    // TODO: finish this
    pcl::PointCloud<PointXYZ> imu_trans(4, 1);
    imu_trans[0].x = 0;
    imu_trans[0].y = 0;
    imu_trans[0].z = 0;
    imu_trans[1].x = 0;
    imu_trans[1].y = 0;
    imu_trans[1].z = 0;
    imu_trans[2].x = 0;
    imu_trans[2].y = 0;
    imu_trans[2].z = 0;
    imu_trans[3].x = 0;
    imu_trans[3].y = 0;
    imu_trans[3].z = 0;
    sensor_msgs::PointCloud2 imu_trans_msg;
    pcl::toROSMsg(imu_trans, imu_trans_msg);
    imu_trans_msg.header.stamp = point_cloud_msg->header.stamp;
    imu_trans_msg.header.frame_id = frame_id;
    pub_imu_trans.publish(imu_trans_msg);
}


void DataRegistrar::imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
    //ROS_INFO("imu_handler");
}


int main(int argc, char** argv) {
    
    ros::init(argc, argv, "data_registration");
    ros::NodeHandle node;

    DataRegistrar data_registrar(node);

    /////////////////
    // Subscribers //
    /////////////////

    ros::Subscriber sub_point_cloud = node.subscribe("/velodyne_points", 2, 
                                                     &DataRegistrar::point_cloud_callback,
                                                     &data_registrar);
    ros::Subscriber sub_imu = node.subscribe("/imu/data", 50, 
                                             &DataRegistrar::imu_callback,
                                             &data_registrar);
    
    ros::spin();

    return 0;
}
