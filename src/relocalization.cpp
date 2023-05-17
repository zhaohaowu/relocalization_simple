#include <bits/stdc++.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <opencv/cv.h>
#include <ceres/ceres.h>
#include <lidar_factor.hpp>
#include <tf/transform_broadcaster.h>
using namespace std;

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)
struct smoothness_t{ 
    float value;
    size_t ind;
};
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class Relocalization{
    ros::NodeHandle n;
    ros::Publisher pub_lidar_odom;
    ros::Publisher pub_initial_odom;
    ros::Publisher pub_corner_cloud;
    ros::Publisher pub_surf_cloud;
    ros::Publisher pub_global_corner_cloud;
    ros::Publisher pub_global_surf_cloud;
    ros::Publisher pub_local_corner_cloud;
    ros::Publisher pub_local_surf_cloud;
    ros::Subscriber sub_cloud;
    ros::Subscriber sub_initial_odom;
    ros::Subscriber sub_initialpose;
    tf::TransformBroadcaster tf_pub;
    tf::StampedTransform tf_msg;
    deque<sensor_msgs::PointCloud2::ConstPtr> cloud_ptr_queue;
    mutex cloud_mutex;
    pcl::PointCloud<VelodynePointXYZIRT>::Ptr cur_pcl_ptr = boost::make_shared<pcl::PointCloud<VelodynePointXYZIRT>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr extracted_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr surf_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_corner_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_surf_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_corner_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_surf_pcl_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::VoxelGrid<pcl::PointXYZI> down_filter;
    int N_SCAN = 16;
    int Horizon_SCAN = 1800;
    std_msgs::Header cur_header;
    Eigen::Matrix4f first_pose = Eigen::Matrix4f::Identity();;
    Eigen::Matrix4f pre_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f result_pose = Eigen::Matrix4f::Identity();
    int first_flag = 0;
    vector<smoothness_t> curve_point;
    vector<int> label;
    vector<int> neigh_picked;
    vector<float> distance;
    vector<int> col_index;
    vector<int> start_ring;
    vector<int> end_ring;
    cv::Mat rangeMat;
    pcl::CropBox<pcl::PointXYZI> box_filter;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_corner_map_ = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZI>>();
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_surf_map_ = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZI>>();
public:
    Relocalization(){
        sub_cloud = n.subscribe("velodyne_points", 10, &Relocalization::cloudCallback, this);
        sub_initial_odom = n.subscribe("initial_odom", 10, &Relocalization::initialCallback, this);
        sub_initialpose = n.subscribe("/initialpose", 10, &Relocalization::initialposeCallback, this);
        pub_corner_cloud = n.advertise<sensor_msgs::PointCloud2>("corner_cloud", 10);
        pub_surf_cloud = n.advertise<sensor_msgs::PointCloud2>("surf_cloud", 10);
        pub_global_corner_cloud = n.advertise<sensor_msgs::PointCloud2>("global_corner_cloud", 10);
        pub_global_surf_cloud = n.advertise<sensor_msgs::PointCloud2>("global_surf_cloud", 10);
        pub_local_corner_cloud = n.advertise<sensor_msgs::PointCloud2>("local_corner_cloud", 10);
        pub_local_surf_cloud = n.advertise<sensor_msgs::PointCloud2>("local_surf_cloud", 10);
        pub_lidar_odom = n.advertise<nav_msgs::Odometry>("lidar_odom", 10);
        pub_initial_odom = n.advertise<nav_msgs::Odometry>("initial_odom", 10);
        init();
    }
    void init() {
        down_filter.setLeafSize(0.4, 0.4, 0.4);
        curve_point.resize(N_SCAN*Horizon_SCAN);
        label.resize(N_SCAN*Horizon_SCAN);
        neigh_picked.resize(N_SCAN*Horizon_SCAN);
        distance.resize(N_SCAN*Horizon_SCAN);
        col_index.resize(N_SCAN*Horizon_SCAN);
        start_ring.resize(N_SCAN*Horizon_SCAN);
        end_ring.resize(N_SCAN*Horizon_SCAN);
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        char abs_path[PATH_MAX];
        realpath(__FILE__, abs_path);
        std::string dirpath(abs_path);
        dirpath = dirpath.substr(0, dirpath.find_last_of("/"));
        dirpath = dirpath.substr(0, dirpath.find_last_of("/"));
        string sharp_map_path_ = dirpath + "/data_park/sharp_cloud_map.pcd";
        string flat_map_path_ = dirpath + "/data_park/flat_cloud_map.pcd";
        pcl::io::loadPCDFile(sharp_map_path_, *global_corner_pcl_ptr);
        pcl::io::loadPCDFile(flat_map_path_, *global_surf_pcl_ptr);
    }
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr msg_in){
        lock_guard<mutex> lock(cloud_mutex);
        cloud_ptr_queue.push_back(msg_in);
    }
    void initialCallback(const nav_msgs::Odometry::ConstPtr msg_in) {
        if (first_flag == 0) {
            Eigen::Quaternionf q = Eigen::Quaternionf(  msg_in->pose.pose.orientation.w, msg_in->pose.pose.orientation.x,
                                                        msg_in->pose.pose.orientation.y, msg_in->pose.pose.orientation.z);
            first_pose.block<3,3>(0,0) = q.matrix();
            first_pose(0,3) = msg_in->pose.pose.position.x;
            first_pose(1,3) = msg_in->pose.pose.position.y;
            first_pose(2,3) = msg_in->pose.pose.position.z;
            first_flag = 1;
        }
    }
    void initialposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr msg_in) {
        nav_msgs::Odometry msg_out;
        msg_out.header.frame_id = "map";
        msg_out.header.stamp = msg_in->header.stamp;
        msg_out.pose.pose.position.x = msg_in->pose.pose.position.x;
        msg_out.pose.pose.position.y = msg_in->pose.pose.position.y;
        msg_out.pose.pose.position.z = msg_in->pose.pose.position.z;
        msg_out.pose.pose.orientation.x = msg_in->pose.pose.orientation.x;
        msg_out.pose.pose.orientation.y = msg_in->pose.pose.orientation.y;
        msg_out.pose.pose.orientation.z = msg_in->pose.pose.orientation.z;
        msg_out.pose.pose.orientation.w = msg_in->pose.pose.orientation.w;
        pub_initial_odom.publish(msg_out);
    }
    void run(){
        sensor_msgs::PointCloud2 global_corner_msg;
        pcl::toROSMsg(*global_corner_pcl_ptr, global_corner_msg);
        global_corner_msg.header.frame_id = "map";
        global_corner_msg.header.stamp = ros::Time::now();
        pub_global_corner_cloud.publish(global_corner_msg);

        sensor_msgs::PointCloud2 global_surf_msg;
        pcl::toROSMsg(*global_surf_pcl_ptr, global_surf_msg);
        global_surf_msg.header.frame_id = "map";
        global_surf_msg.header.stamp = ros::Time::now();
        pub_global_surf_cloud.publish(global_surf_msg);
        
        lock_guard<mutex> lock(cloud_mutex);
        if(cloud_ptr_queue.empty()) return;
        cur_header = cloud_ptr_queue.front()->header;
        pcl::fromROSMsg(*cloud_ptr_queue.front(), *cur_pcl_ptr);
        cloud_ptr_queue.pop_front();
        extract(cur_pcl_ptr);
        feature(extracted_pcl_ptr);
        match(corner_pcl_ptr, surf_pcl_ptr);
        freeMemory();
    }
    void extract(const pcl::PointCloud<VelodynePointXYZIRT>::Ptr pcl_ptr) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        tmp_ptr->points.resize(N_SCAN*Horizon_SCAN);
        int cloud_size = pcl_ptr->points.size();
        for (int i = 0; i < cloud_size; i++) {
            int row = pcl_ptr->points[i].ring;
            float angle = atan2(pcl_ptr->points[i].y, pcl_ptr->points[i].x) * 180 / M_PI;
            int col = round(angle/0.2) + 900;//0.2表示每一列为0.2°，1800列为360°
            if (rangeMat.at<float>(row, col) != FLT_MAX)
                continue;
            rangeMat.at<float>(row, col) = sqrt(pcl_ptr->points[i].x * pcl_ptr->points[i].x +
                                                pcl_ptr->points[i].y * pcl_ptr->points[i].y +
                                                pcl_ptr->points[i].z * pcl_ptr->points[i].z);
            tmp_ptr->points[row * Horizon_SCAN + col].x = pcl_ptr->points[i].x;
            tmp_ptr->points[row * Horizon_SCAN + col].y = pcl_ptr->points[i].y;
            tmp_ptr->points[row * Horizon_SCAN + col].z = pcl_ptr->points[i].z;
            tmp_ptr->points[row * Horizon_SCAN + col].intensity = pcl_ptr->points[i].intensity;
        }
        int count = 0;
        for (int i = 0; i < N_SCAN; i++) {
            start_ring[i] = count + 4;
            for (int j = 0; j < Horizon_SCAN; j++) {
                if (rangeMat.at<float>(i,j) != FLT_MAX) {
                    col_index[count] = j;
                    distance[count] = rangeMat.at<float>(i,j);
                    extracted_pcl_ptr->push_back(tmp_ptr->points[i * Horizon_SCAN + j]);
                    count++;
                }
            }
            end_ring[i] = count - 5;
        }
    }
    void feature(const pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_ptr){
        int cloud_size = pcl_ptr->points.size();
        for (int i = 5; i < cloud_size - 5; i++) {
            float t = distance[i-5] + distance[i-4] + distance[i-3] + distance[i-2] + distance[i-1] + 
                distance[i+5] + distance[i+4] + distance[i+3] + distance[i+2] + distance[i+1] - 
                10 * distance[i];
            curve_point[i].value = t * t;
            curve_point[i].ind = i;
            neigh_picked[i] = 0;
            label[i] = 0;
        }
        for (int i = 5; i < cloud_size - 5; i++) {
            if (abs(col_index[i+1]-col_index[i]) < 10) {
                if (distance[i] - distance[i+1] > 0.3) {
                    neigh_picked[i-5] = 1;
                    neigh_picked[i-4] = 1;
                    neigh_picked[i-3] = 1;
                    neigh_picked[i-2] = 1;
                    neigh_picked[i-1] = 1;
                } else if (distance[i+1] - distance[i] > 0.3) {
                    neigh_picked[i+5] = 1;
                    neigh_picked[i+4] = 1;
                    neigh_picked[i+3] = 1;
                    neigh_picked[i+2] = 1;
                    neigh_picked[i+1] = 1;
                }
            }
            if (abs(distance[i]-distance[i-1]) > 0.02*distance[i] && abs(distance[i]-distance[i+1]) > 0.02*distance[i])
                neigh_picked[i] = 1;
        }
        for (int i = 0; i < N_SCAN; i++) {
            for (int j = 0; j < 6; j++) {
                int sp = (start_ring[i] * (6 - j) + end_ring[i] * j) / 6;//线性插值
                int ep = (start_ring[i] * (5 - j) + end_ring[i] * (j + 1)) / 6 - 1;//下一个开始是j+1进行差值，再－1表示当前段结束
                sort(curve_point.begin()+sp, curve_point.begin()+ep, by_value());
                for (int k = ep; k >= sp; k--) {
                    int ind = curve_point[k].ind;
                    if (neigh_picked[ind] == 0 && curve_point[k].value > 1) {
                        neigh_picked[ind] = 1;
                        label[ind] = 1;
                        corner_pcl_ptr->push_back(pcl_ptr->points[k]);
                        for (int l = 1; l <= 5; l++) {
                            if(abs(col_index[ind+l] - col_index[ind+l-1]) > 10)
                                break;
                            neigh_picked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            if(abs(col_index[ind+l] - col_index[ind+l+1]) > 10)
                                break;
                            neigh_picked[ind + l] = 1;
                        }
                    }
                }
                for (int k = sp; k <= ep; k++) {
                    int ind = curve_point[k].ind;
                    if (neigh_picked[ind] == 0 && curve_point[k].value < 0.1) {
                        neigh_picked[ind] = 1;
                        label[ind] = -1;
                        for (int l = 1; l <= 5; l++) {
                            if(abs(col_index[ind+l] - col_index[ind+l-1]) > 10)
                                break;
                            neigh_picked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            if(abs(col_index[ind+l] - col_index[ind+l+1]) > 10)
                                break;
                            neigh_picked[ind + l] = 1;
                        }
                    }
                }
                for (int k = sp; k <= ep; k++) {
                    if (label[k] <= 0) {
                        surf_pcl_ptr->push_back(pcl_ptr->points[k]);
                    }
                }
            }
        }
        down_filter.setInputCloud(surf_pcl_ptr);
        down_filter.filter(*surf_pcl_ptr);

        sensor_msgs::PointCloud2 corner_msg;
        pcl::toROSMsg(*corner_pcl_ptr, corner_msg);
        corner_msg.header = cur_header;
        pub_corner_cloud.publish(corner_msg);

        sensor_msgs::PointCloud2 surf_msg;
        pcl::toROSMsg(*surf_pcl_ptr, surf_msg);
        surf_msg.header = cur_header;
        pub_surf_cloud.publish(surf_msg);
    }
    void match(const pcl::PointCloud<pcl::PointXYZI>::Ptr corner_ptr, const pcl::PointCloud<pcl::PointXYZI>::Ptr surf_ptr) {
        if (first_flag == 0) return;
        static int inited = 1;
        static vector<float> origin(3, 0);
        static vector<float> edge_left(3, 0);
        static vector<float> edge_right(3, 0);
        if (inited) {
            inited = 0;
            pre_pose = first_pose;
            origin = {first_pose(0, 3), first_pose(1, 3), first_pose(2, 3)};
            edge_left = {origin[0] - 150, origin[1] - 150, origin[2] - 150};
            edge_right = {origin[0] + 150, origin[1] + 150, origin[2] + 150};
            box_filter.setMin(Eigen::Vector4f(edge_left[0], edge_left[1], edge_left[2], 1.0e-6));
            box_filter.setMax(Eigen::Vector4f(edge_right[0], edge_right[1], edge_right[2], 1.0e6));
            
            box_filter.setInputCloud(global_corner_pcl_ptr);
            box_filter.filter(*local_corner_pcl_ptr);
            down_filter.setInputCloud(local_corner_pcl_ptr);
            down_filter.filter(*local_corner_pcl_ptr);

            box_filter.setInputCloud(global_surf_pcl_ptr);
            box_filter.filter(*local_surf_pcl_ptr);
            down_filter.setInputCloud(local_surf_pcl_ptr);
            down_filter.filter(*local_surf_pcl_ptr);
        }
            
        sensor_msgs::PointCloud2 local_corner_msg;
        pcl::toROSMsg(*local_corner_pcl_ptr, local_corner_msg);
        local_corner_msg.header.frame_id = "map";
        local_corner_msg.header.stamp = ros::Time::now();
        pub_local_corner_cloud.publish(local_corner_msg);

        sensor_msgs::PointCloud2 local_surf_msg;
        pcl::toROSMsg(*local_surf_pcl_ptr, local_surf_msg);
        local_surf_msg.header.frame_id = "map";
        local_surf_msg.header.stamp = ros::Time::now();
        pub_local_surf_cloud.publish(local_surf_msg);

        static Eigen::Matrix4f last_pose = pre_pose;
        kdtree_corner_map_->setInputCloud(local_corner_pcl_ptr);
        kdtree_surf_map_->setInputCloud(local_surf_pcl_ptr);
        auto t1 = chrono::steady_clock::now();
        result_pose = scanMatch(corner_ptr, surf_ptr, local_corner_pcl_ptr, local_surf_pcl_ptr, pre_pose);
        auto t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        
        Eigen::Matrix4f step_pose = last_pose.inverse() * result_pose;
        last_pose = result_pose;
        pre_pose = result_pose * step_pose;
        nav_msgs::Odometry odom_msg;
        odom_msg.header.frame_id = "map";
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.pose.pose.position.x = result_pose(0, 3);
        odom_msg.pose.pose.position.y = result_pose(1, 3);
        odom_msg.pose.pose.position.z = result_pose(2, 3);
        Eigen::Quaternionf q(result_pose.block<3, 3>(0, 0));
        odom_msg.pose.pose.orientation.w = q.w();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        pub_lidar_odom.publish(odom_msg);

        tf_msg.stamp_ = cur_header.stamp;
        tf_msg.frame_id_ = "map";
        tf_msg.child_frame_id_ = "velodyne";
        tf_msg.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        tf_msg.setOrigin(tf::Vector3(result_pose(0, 3), result_pose(1, 3), result_pose(2, 3)));
        tf_pub.sendTransform(tf_msg);

        if (fabs(result_pose(0,3) - edge_left[0]) < 30 || fabs(result_pose(0,3) - edge_right[0]) < 30 || 
            fabs(result_pose(1,3) - edge_left[1]) < 30 || fabs(result_pose(1,3) - edge_right[1]) < 30 || 
            fabs(result_pose(2,3) - edge_left[2]) < 30 || fabs(result_pose(2,3) - edge_right[2]) < 30) {
            origin = {result_pose(0, 3), result_pose(1, 3), result_pose(2, 3)};
            edge_left = {origin[0] - 150, origin[1] - 150, origin[2] - 150};
            edge_right = {origin[0] + 150, origin[1] + 150, origin[2] + 150};
            box_filter.setMin(Eigen::Vector4f(edge_left[0], edge_left[1], edge_left[2], 1.0e-6));
            box_filter.setMax(Eigen::Vector4f(edge_right[0], edge_right[1], edge_right[2], 1.0e6));

            box_filter.setInputCloud(global_corner_pcl_ptr);
            box_filter.filter(*local_corner_pcl_ptr);
            down_filter.setInputCloud(local_corner_pcl_ptr);
            down_filter.filter(*local_corner_pcl_ptr);

            box_filter.setInputCloud(global_surf_pcl_ptr);
            box_filter.filter(*local_surf_pcl_ptr);
            down_filter.setInputCloud(local_surf_pcl_ptr);
            down_filter.filter(*local_surf_pcl_ptr);
        }
    }
    Eigen::Matrix4f scanMatch(const pcl::PointCloud<pcl::PointXYZI>::Ptr& corner_ptr, const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_ptr, 
                              const pcl::PointCloud<pcl::PointXYZI>::Ptr& local_corner_pcl_ptr, const pcl::PointCloud<pcl::PointXYZI>::Ptr& local_surf_pcl_ptr, 
                              const Eigen::Matrix4f& pre_pose) {
        static double parameters_[7] = {0, 0, 0, 1, 0, 0, 0};
        Eigen::Map<Eigen::Quaterniond> q_w_curr_map(parameters_);
        Eigen::Map<Eigen::Vector3d> t_w_curr_map(parameters_ + 4);
        Eigen::Matrix3d predict_rot_mat = pre_pose.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d predict_trans(pre_pose(0,3), pre_pose(1,3), pre_pose(2,3));
        q_w_curr_map = Eigen::Quaterniond(predict_rot_mat);
        t_w_curr_map = predict_trans;
        static Eigen::Quaterniond q_w_curr_;
        static Eigen::Vector3d t_w_curr_;
        q_w_curr_ = q_w_curr_map;
        t_w_curr_ = t_w_curr_map;
        for(int iter_count = 0; iter_count < 3; iter_count++)
        {
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::LocalParameterization *q_paramterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);
            problem.AddParameterBlock(parameters_, 4, q_paramterization);
            problem.AddParameterBlock(parameters_ + 4, 3);
            pcl::PointXYZI point_ori, point_sel;
            std::vector<int> point_search_ind;
            std::vector<float> point_search_dist; 
            int corner_count = 0;
            int laser_corner_num = corner_ptr->size();
            for(int i = 0; i < laser_corner_num; i++)
            {
                point_ori = corner_ptr->points[i];
                Eigen::Vector3d point_curr(point_ori.x, point_ori.y, point_ori.z);
                Eigen::Vector3d point_w = q_w_curr_ * point_curr + t_w_curr_;
                point_sel.x = point_w.x();
                point_sel.y = point_w.y();
                point_sel.z = point_w.z();
                point_sel.intensity = point_ori.intensity;
                kdtree_corner_map_->nearestKSearch(point_sel, 5, point_search_ind, point_search_dist);
                //point_search_dist[4]表示最大值，升序数组
                if(point_search_dist[4] < 1.0)
                {
                    std::vector<Eigen::Vector3d> near_corners;
                    Eigen::Vector3d center(0, 0, 0);
                    for(int j = 0; j < 5; j++)
                    {
                        Eigen::Vector3d temp(local_corner_pcl_ptr->points[point_search_ind[j]].x,
                                                                    local_corner_pcl_ptr->points[point_search_ind[j]].y,
                                                                    local_corner_pcl_ptr->points[point_search_ind[j]].z);
                        center += temp;
                        near_corners.push_back(temp);
                    }
                    center /= 5.0;
                    Eigen::Matrix3d cov_mat = Eigen::Matrix3d::Zero();
                    for(int j = 0; j < 5; j++)
                    {
                        Eigen::Matrix<double, 3, 1> temp_zero_mean = near_corners[j] - center;
                        cov_mat = cov_mat + temp_zero_mean * temp_zero_mean.transpose();
                    }
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
                    Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                    Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                    if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                    { 
                        Eigen::Vector3d point_on_line = center;
                        Eigen::Vector3d point_a, point_b;
                        point_a = 0.1 * unit_direction + point_on_line;
                        point_b = -0.1 * unit_direction + point_on_line;
                        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                        // ceres::CostFunction *cost_function = new LidarEdgeAnalyticCostFunction(curr_point, point_a, point_b, 1.0);
                        problem.AddResidualBlock(cost_function, loss_function, parameters_, parameters_ + 4);
                        corner_count++;	
                    }		
                }
            }
            
            int surf_count = 0;
            int laser_surf_num = surf_ptr->size();
            for(int i = 0; i < laser_surf_num; i++)
            {
                point_ori = surf_ptr->points[i];
                Eigen::Vector3d point_curr(point_ori.x, point_ori.y, point_ori.z);
                Eigen::Vector3d point_w = q_w_curr_ * point_curr + t_w_curr_;
                point_sel.x = point_w.x();
                point_sel.y = point_w.y();
                point_sel.z = point_w.z();
                point_sel.intensity = point_ori.intensity;
                kdtree_surf_map_->nearestKSearch(point_sel, 5, point_search_ind, point_search_dist);
                Eigen::Matrix<double, 5, 3> matA0;
                Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                if(point_search_dist[4] < 1.0)
                {
                    for(int j = 0; j < 5; j ++)
                    {
                        matA0(j, 0) = local_surf_pcl_ptr->points[point_search_ind[j]].x;
                        matA0(j, 1) = local_surf_pcl_ptr->points[point_search_ind[j]].y;
                        matA0(j, 2) = local_surf_pcl_ptr->points[point_search_ind[j]].z;
                    }
                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();
                    bool planeValid = true;
                    for (int j = 0; j < 5; j++)
                    {
                        if (fabs(norm(0) * local_surf_pcl_ptr->points[point_search_ind[j]].x +
                                    norm(1) * local_surf_pcl_ptr->points[point_search_ind[j]].y +
                                    norm(2) * local_surf_pcl_ptr->points[point_search_ind[j]].z + negative_OA_dot_norm) > 0.2)
                        {
                            planeValid = false;
                            break;
                        }
                    }
                    Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                    if (planeValid)
                    {
                        ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
                        problem.AddResidualBlock(cost_function, loss_function, parameters_, parameters_ + 4);
                        surf_count++;
                    }
                }
            }
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            q_w_curr_ = q_w_curr_map;
            t_w_curr_ = t_w_curr_map;
        }

        q_w_curr_.normalize();
        Eigen::Matrix3d rot_mat(q_w_curr_);
        Eigen::Matrix4f tmp_pose = Eigen::Matrix4f::Identity();
        tmp_pose.block<3,3>(0,0) = rot_mat.cast<float>();
        tmp_pose(0,3) = t_w_curr_(0);
        tmp_pose(1,3) = t_w_curr_(1);
        tmp_pose(2,3) = t_w_curr_(2);
        return tmp_pose;
    }
    void freeMemory() {
        extracted_pcl_ptr->clear();
        corner_pcl_ptr->clear();
        surf_pcl_ptr->clear();
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    }
    typedef shared_ptr<Relocalization> Ptr;
};

int main(int argc, char** argv){
    ros::init(argc, argv, "relocalization");
    Relocalization::Ptr relocalization = std::make_shared<Relocalization>();
    ROS_INFO("\033[1;32m---> relocaliaztion START.\033[0m");//"\033[1;32m"表示文本颜色设置为亮绿色，"\033[0m" 则将颜色重置为终端默认颜色
    ros::Rate rate(100);
    while(ros::ok()){
        relocalization->run();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}