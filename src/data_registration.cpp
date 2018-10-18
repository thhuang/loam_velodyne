// TH Huang

#include <vector>
#include <set>
#include <sstream>

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <iostream>
using std::cout;
using std::endl;


typedef pcl::PointXYZI PointType;


inline double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}


inline double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}


class DataRegistrar {
    bool            system_initiated;
    unsigned        system_init_count;
    unsigned        system_delay;
    const unsigned  num_scans;
    const float     scan_period;
    const float     M_2PI;
public:
    DataRegistrar() 
        : system_init_count(0), 
          system_delay(20), 
          system_initiated(false), 
          num_scans(16),  // Number of channel (Velodyne VLP-16)
          scan_period(0.1),  // Inverse of the scan rate which is 10 Hz (Velodyne VLP-16)
          M_2PI(2 * M_PI)
        {}
        
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg);
    void imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg);
};


void DataRegistrar::point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg) {
    
    // Delay
    if (!system_initiated) {
        system_init_count++;
        if (system_init_count >= system_delay) {
            system_initiated = true;
        }
        return;
    }

    std::vector<int> scan_start_ind(num_scans, 0);
    std::vector<int> scan_end_ind(num_scans, 0);
  
    // Parse data from PointCloud2
    double time_stamp = point_cloud_msg->header.stamp.toSec();
    pcl::PointCloud<pcl::PointXYZ> point_cloud_in;
    pcl::fromROSMsg(*point_cloud_msg, point_cloud_in);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(point_cloud_in, point_cloud_in, indices);
    int cloud_size = point_cloud_in.points.size();

    // Find starting orientation
    float start_ori = -M_PI;
    for (int i = 0; i < cloud_size; i++) {
        float x = point_cloud_in.points[i].x;
        float y = point_cloud_in.points[i].y;
        float z = point_cloud_in.points[i].z;
        float angle = atan(z / sqrt(x*x + y*y));
        if (!std::isnan(angle)) {
            start_ori = -atan2(y, x);
            break;
        }
    }
    
    // Find ending orientation
    float end_ori = M_PI;
    for (int i = cloud_size - 1; i >= 0; i--) {
        float x = point_cloud_in.points[i].x;
        float y = point_cloud_in.points[i].y;
        float z = point_cloud_in.points[i].z;
        float angle = atan(z / sqrt(x*x + y*y));
        if (!std::isnan(angle)) {
            end_ori = -atan2(y, x) + 2 * M_PI;
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
    PointType point;
    std::vector<pcl::PointCloud<PointType> > point_cloud_scans(num_scans);
    for (int i = 0; i < cloud_size; i++) {
        point.x = point_cloud_in.points[i].x;
        point.y = point_cloud_in.points[i].y;
        point.z = point_cloud_in.points[i].z;

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
        float rounded_angle = round(rad2deg(atan(point.z / sqrt(point.x*point.x + point.y*point.y))));
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
        
    }  // for (int i = 0; i < cloud_size; i++)
    
    // Change cloude_size to the number of points without noise
    cloud_size = cloud_size_denoised;

    // TODO: Feature extraction

    // Debug
    cout << endl << endl << endl;
    

}


void DataRegistrar::imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
    //ROS_INFO("imu_handler");
}


int main(int argc, char** argv) {
    
    ros::init(argc, argv, "data_registration");
    ros::NodeHandle node;

    DataRegistrar data_registrar;

    ros::Subscriber sub_point_cloud = node.subscribe("/velodyne_points", 2, 
                                                     &DataRegistrar::point_cloud_callback,
                                                     &data_registrar);
    ros::Subscriber sub_imu = node.subscribe("/imu/data", 50, 
                                             &DataRegistrar::imu_callback,
                                             &data_registrar);

    ros::spin();

    return 0;
}
