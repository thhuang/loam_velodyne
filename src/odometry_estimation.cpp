// TH Huang

#include <vector>

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
    pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree_edge_points_last;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree_planar_points_last;
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
          max_lm_iteration(25)
        {}
    
    void spin();
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg);
    void edge_points_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_sharp_msg);
    void edge_points_less_sharp_callback(const sensor_msgs::PointCloud2ConstPtr& edge_points_less_sharp_msg);
    void planar_points_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_flat_msg);
    void planar_points_less_flat_callback(const sensor_msgs::PointCloud2ConstPtr& planar_points_less_flat_msg);
    void imu_trans_callback(const sensor_msgs::PointCloud2ConstPtr& imu_trans_msg);

private:    
    OdometryEstimator();
    void reproject_to_start(const PointXYZI& point_in, PointXYZI& point_out);
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


void OdometryEstimator::reproject_to_start(const PointXYZI& point_in, PointXYZI& point_out) {
    // Get the reletive time
    float rel_time = (1 / scan_period) * (point_in.intensity - int(point_in.intensity));

    
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

    int num_edge_points_sharp = edge_points_sharp->points.size();
    int num_planar_points_flat = planar_points_flat->points.size();
        
    // L-M optimization
    for (int iter_count = 0; iter_count < max_lm_iteration; iter_count++) {
        // Process edge points
       

        // Process planar points


    }


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
