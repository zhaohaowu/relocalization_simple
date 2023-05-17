// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk
#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_LIDAR_FACTOR_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_LIDAR_FACTOR_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

inline Eigen::Matrix<double,3,3> skew(Eigen::Matrix<double,3,1>& mat_in){
    Eigen::Matrix<double,3,3> skew_mat;
    skew_mat.setZero();
    skew_mat(0,1) = -mat_in(2);
    skew_mat(0,2) =  mat_in(1);
    skew_mat(1,2) = -mat_in(0);
    skew_mat(1,0) =  mat_in(2);
    skew_mat(2,0) = -mat_in(1);
    skew_mat(2,1) =  mat_in(0);
    return skew_mat;
}

// EdgeAnalyticCostFunction
class LidarEdgeAnalyticCostFunction : public ceres::SizedCostFunction<1, 4, 3> {
public:
    LidarEdgeAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
								Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

  	virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const 
	{
		Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
		Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1]);
		Eigen::Vector3d lp;
		Eigen::Vector3d lp_r;
		lp_r = q_last_curr * curr_point; // for computing Jacobian of Rotation: dp_by_dr
		lp = q_last_curr * curr_point + t_last_curr; //new point
		Eigen::Vector3d nu = (lp - last_point_b).                                                                                                       cross(lp - last_point_a);
		Eigen::Vector3d de = last_point_b - last_point_a;

		residuals[0] = nu.norm() / de.norm();

		if(jacobians != NULL)
		{
			if(jacobians[0] != NULL)
			{
				Eigen::Vector3d X = nu;
				Eigen::Vector3d re = last_point_a - last_point_b;
				Eigen::Matrix3d skew_re = skew(re);
				//  Rotation
				Eigen::Matrix3d skew_lp_r = skew(lp_r);
				Eigen::Matrix<double, 3, 3> dp_by_dr;
				dp_by_dr.block<3,3>(0,0) = -skew_lp_r;
				Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J_so3_r(jacobians[0]);
				J_so3_r.setZero();
				J_so3_r.block<1,3>(0,0) = X.transpose() * skew_re * dp_by_dr / X.norm() / re.norm();

				// Translation
				Eigen::Matrix<double, 3, 3> dp_by_dt;
				(dp_by_dt.block<3,3>(0,0)).setIdentity();
				Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J_so3_t(jacobians[1]);
				J_so3_t.setZero();
				J_so3_t.block<1,3>(0,0) = X.transpose() * skew_re * dp_by_dt / X.norm() / re.norm();
			}
		}

		return true;
 
	}

protected:
	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};


// PlaneAnalyticCostFunction
class LidarPlaneAnalyticCostFunction : public ceres::SizedCostFunction<1, 4, 3> {
public:
    LidarPlaneAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
								Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_), last_point_m(last_point_m_), s(s_) {}

  	virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const 
	{
		Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
		Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1]);
		Eigen::Vector3d lp;
		Eigen::Vector3d lp_r;
		lp_r = q_last_curr * curr_point; // for computing Jacobian of Rotation: dp_dr
		lp = q_last_curr * curr_point + t_last_curr; //new point
		Eigen::Vector3d de = (last_point_l-last_point_j).cross(last_point_m-last_point_j);
		double nu = (lp-last_point_j).dot(de);
		
		residuals[0] = nu / de.norm();

		if(jacobians != NULL)
		{
			if(jacobians[0] != NULL)
			{
				Eigen::Vector3d dX_dp = de / de.norm();
				double X = nu / de.norm();
				Eigen::Vector3d ddh_dp = X * dX_dp / std::abs(X);
				//  Rotation
				Eigen::Matrix3d skew_lp_r = skew(lp_r);
				Eigen::Matrix<double, 3, 3> dp_dr;
				dp_dr.block<3,3>(0,0) = -skew_lp_r;
				Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J_so3_r(jacobians[0]);
				J_so3_r.setZero();
				J_so3_r.block<1,3>(0,0) = ddh_dp.transpose() * dp_dr;

				// Translation
				Eigen::Matrix<double, 3, 3> dp_dt;
				(dp_dt.block<3,3>(0,0)).setIdentity();
				Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J_so3_t(jacobians[1]);
				J_so3_t.setZero();
				J_so3_t.block<1,3>(0,0) = ddh_dp.transpose() * dp_dt;
			}
		}

		return true;
 
	}

protected:
	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	double s;
};


// Edgepoints 计算点到直线的距离作为误差函数
struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const // 构建误差函数，点到直线距离，跟论文中一样
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr); // 根据s进行插值
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr; // 得到了相对于这一帧起始坐标的点云位置（去畸变的点云坐标）

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb); // 两个向量叉乘的长度为这两个向量组成的平行四边形面积（两个三角形面积和）
		Eigen::Matrix<T, 3, 1> de = lpa - lpb; // 三角形的底边

		residual[0] = nu.x() / de.norm(); // 面积除以底边长度等于高的长度，即lp点到直线距离
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		// 使用ceres的自动求导
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

// Planar Points 计算点到平面的距离作为误差函数
struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m); // 得到平面的单位法向量
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr; // 得到了相对于这一帧起始坐标的点云位置（去畸变的点云坐标）

		residual[0] = (lp - lpj).dot(ljm); // 得到点到平面上一点lpj的连线（lp - lpj），这个连线到这个平面法向量的投影，即点到平面距离

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

// 根据拟合的平面法向量，计算点到平面的距离作为误差函数
struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};

// 两点之间的距离作为误差函数
struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};


#endif