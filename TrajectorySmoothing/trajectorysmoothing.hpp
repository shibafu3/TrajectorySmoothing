#ifndef TRAJECTORYSMOOTHING_HPP
#define TRAJECTORYSMOOTHING_HPP

#include "voronoifield.hpp"

#include "opencv2/opencv.hpp"
#include <vector>

#include "cvplot.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

#include <iomanip>


class TrajectrorySmoothing{
	/////////////////////////////////////////////////
	int debugi = 0;								   //
	CVPlot plt = CVPlot(cv::Point(512, 256));		   //
	CVPlot plt1 = CVPlot(cv::Point(512, 256), "lo");   //
	CVPlot plt2 = CVPlot(cv::Point(512, 256), "lk");   //
	CVPlot plt3 = CVPlot(cv::Point(512, 256), "ls");   //
	/////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	void DebugPlot(cv::Mat im, cv::Point2d p, int c){													 //
		for (int i = 1; i < 4; ++i){																	 //
			im.at<cv::Vec3b>(p)[c] = 255;																 //
			im.at<cv::Vec3b>(p + cv::Point2d(0, i))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(i, 0))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(i, i))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(0, -i))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(-i, 0))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(-i, -i))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(i, -i))[c] = 255;											 //
			im.at<cv::Vec3b>(p + cv::Point2d(-i, i))[c] = 255;											 //
		}																								 //
	}																									 //
	void Debug(std::vector<cv::Point2d> X, std::vector<cv::Point> obstacle, int c){						 //
		//cv::Mat image(cv::Size(512, 512), CV_8UC3, cv::Scalar::all(0));								 //
		//for (int i = 0; i < X.size(); ++i)       { DebugPlot(image, X[i], c); }						 //
		//for (int i = 0; i < obstacle.size(); ++i){ DebugPlot(image, obstacle[i], 2); }				 //
		//imshow("DebugImage", image);																	 //
		//cv::waitKey(1);																				 //
		//
		CVPlot plt(cv::Point(1024, 1024), "debug");														 //
		//plt.XLim(0, 128);																				 //
		//plt.YLim(0, 128);																				 //
		for (int i = 0; i < obstacle.size(); ++i){ plt.Plot(obstacle[i], cv::Scalar(0, 0, 255), 1); }	 //
		for (int i = 0; i < X.size(); ++i)       { plt.Plot(X[i], cv::Scalar(255, 0, 0), 1); }			 //
		plt.Show(1, 5000);																					 //
		//
	}																									 //
	///////////////////////////////////////////////////////////////////////////////////////////////////////


	double wo;
	double wk;
	double ws;
	double dmax;
	double kmax;
	int FIRST_POINT;
	int LAST_POINT;

	int interpolation_num;

	Voronoi vf;

	double NormalizationRad(double rad){
		while (rad < 0){
			rad += (2 * M_PI);
		}
		while (rad >= (2 * M_PI)){
			rad -= (2 * M_PI);
		}
		return rad;
	}
	double RadDifference(double rad){
		while (rad < -M_PI / 2){
			rad += (M_PI);
		}
		while (rad >= (M_PI / 2)){
			rad -= (M_PI);
		}
		return rad;
	}

	int Differential(
		std::vector<cv::Point2d> X,
		std::vector<cv::Point2d> &dodx,
		std::vector<cv::Point2d> &dkdxi,
		std::vector<cv::Point2d> &dkdxim1,
		std::vector<cv::Point2d> &dkdxip1,
		std::vector<cv::Point2d> &dsdx){

		double lo = 0;
		double lk = 0;
		double ls = 0;

		for (int i = FIRST_POINT; i < X.size() - LAST_POINT; ++i){
			cv::Point2d Xi = X[i];																			                          // Xi
			cv::Point2d Xim1 = X[i - 1];																	                          // Xi-1
			cv::Point2d Xip1 = X[i + 1];																	                          // Xi+1
			double lXil = norm(Xi);																		                          // |Xi|
			double lXim1l = norm(Xim1);																	                          // |Xi-1|
			double lXip1l = norm(Xip1);																	                          // |Xi+1|
			cv::Point2d Oi = cv::Point2d(vf.cells[int(Xi.x)][int(Xi.y)].point_mother.x, vf.cells[int(Xi.x)][int(Xi.y)].point_mother.y);   // Oi
			double lXimOil = vf.cells[int(Xi.x)][int(Xi.y)].distance;									                          // |Xi - Oi|
			cv::Point2d DXi = Xi - Xim1;																	                          // ƒ¢Xi = Xi - Xi-1
			double lDXil = norm(DXi);																	                          // |ƒ¢Xi|
			cv::Point2d DXp1 = Xip1 - Xi;																	                          // ƒ¢Xi+1 = Xi+1 - Xi
			cv::Point2d DXp1_DX = DXp1 - DXi;																                          // ƒ¢Xi+1 - ƒ¢Xi
			double Dx = Xi.x - Xim1.x;																	                          // ƒ¢x
			double Dy = Xi.y - Xim1.y;																	                          // ƒ¢y
			double Dxp1 = Xip1.x - Xi.x;																                          // ƒ¢x+1
			double Dyp1 = Xip1.y - Xi.y;																                          // ƒ¢y+1
			double Dphi = abs(RadDifference(atan2(Dyp1, Dxp1) - atan2(Dy, Dx)));										                          // ƒ¢ƒÓ
			double ki = Dphi / lDXil;																	                          // ki = ƒ¢ƒÓ / |ƒ¢Xi|
			double a = (-1 / lDXil) * (-1 / sqrt(1 - cos(Dphi)*cos(Dphi)));								                          // (-1 / |ƒ¢Xi|) * (-1 / (1 - cos^2(ƒ¢ƒÓ))^1/2)
			cv::Point2d b = (Dphi / DXi.ddot(DXi)) * cv::Point2d(1, 1);											                          // ƒ¢ƒÓ / (ƒ¢Xi)^2 * 1 or -1 or 0
			cv::Point2d p1 = (Xi - ((Xi.ddot(-Xip1) * -Xip1) / (lXip1l * lXip1l))) / (lXil * lXip1l);		                              // (Xi - (Xi.*-Xi+1 * -Xi+1)) / |Xi+1||Xi+1| / |Xi||Xi+1|
			cv::Point2d p2 = (-Xip1 - (((-Xip1.ddot(Xi)  *  Xi)) / (lXil   * lXil))) / (lXil * lXip1l);		                              // (-Xi+1 - (-Xi+1.*Xi * Xi)) / |Xi||Xi| / |Xi||Xi+1|


			//‹——£‚ªdmaxˆÈã—£‚ê‚Ä‚¢‚éê‡‚Í”÷•ª’l‚ª0
			dodx[i] = 2 * (std::min(0.0, lXimOil - dmax)) * ((Xi - Oi) / lXimOil);

			if (ki > kmax){
				dkdxi[i] = 2 * (ki - kmax) * (a * (-p1 - p2) - b);
				dkdxim1[i] = 2 * (ki - kmax) * (a * p2 + b);
				dkdxip1[i] = 2 * (ki - kmax) * (a * p1);
			}
			else{
				dkdxi[i] = cv::Point2d(0, 0);
				dkdxim1[i] = cv::Point2d(0, 0);
				dkdxip1[i] = cv::Point2d(0, 0);
			}

			dsdx[i] = -4 * (Xip1 - 2 * Xi + Xim1);

			////////////////////////////////////////////////////////////////
			//lo += (lXimOil - dmax) * (lXimOil - dmax);					//
			//lk += ((Dphi / lDXil) - kmax) * ((Dphi / lDXil) - kmax);	//
			//ls += DXp1_DX.ddot(DXp1_DX);								//
			//															//
			//plt1.Plot(cv::Point2d(debugi++, wo * lo), cv::Scalar(255, 0, 0));	//
			//plt2.Plot(cv::Point2d(debugi++, wk * lk), cv::Scalar(0, 255, 0));	//
			//plt3.Plot(cv::Point2d(debugi++, ws * ls), cv::Scalar(0, 0, 255));	//
			//plt.Plot(cv::Point2d(debugi++, wo * lo + wk * lk + ws * ls));	//
			//plt1.Show(1, 5000);											//
			//plt2.Show(1, 5000);											//
			//plt3.Show(1, 5000);											//
			//plt.Show(1, 5000);											//
			////////////////////////////////////////////////////////////////

			//std::cout << "Xi            " << setprecision(16) << Xi << std::endl;
			//std::cout << "Xi-1          " << setprecision(16) << Xim1 << std::endl;
			//std::cout << "Xi+1          " << setprecision(16) << Xip1 << std::endl;
			//std::cout << "lXil          " << setprecision(16) << lXil << std::endl;
			//std::cout << "lXim1l        " << setprecision(16) << lXim1l << std::endl;
			//std::cout << "lXip1l        " << setprecision(16) << lXip1l << std::endl;
			//std::cout << "-Xip1.ddot(Xi)" << setprecision(16) << -Xip1.ddot(Xi) << std::endl;
			//std::cout << "p1            " << setprecision(16) << p1 << std::endl;
			//std::cout << "p2            " << setprecision(16) << p2 << std::endl;
			//std::cout << "a             " << setprecision(16) << a << std::endl;
			//std::cout << "b             " << setprecision(16) << b << std::endl;
			//std::cout << "try num       " << n             << std::endl;
			//std::cout << "point num     " << i             << std::endl;
			//std::cout << "loss o        " << lo            << std::endl;
			//std::cout << "loss k        " << lk            << std::endl;
			//std::cout << "loss s        " << ls            << std::endl;
			//std::cout << "Xi-1          " << Xim1          << std::endl;
			//std::cout << "Xi            " << Xi            << std::endl;
			//std::cout << "Xi+1          " << Xip1          << std::endl;
			//std::cout << "ki            " << ki            << std::endl;
			//std::cout << "dkdxi         " << dkdxi[i]      << std::endl;
			//std::cout << "dkdxi-1       " << dkdxim1[i]    << std::endl;
			//std::cout << "dkdxi+1       " << dkdxip1[i]    << std::endl;
			//std::cout << "dodx          " << dodx[i]       << std::endl;
			//std::cout << "dsdx          " << dsdx[i]       << std::endl;
			//std::cout << "Xi            " << Xi            << std::endl;
			//std::cout << "Oi            " << Oi            << std::endl;
			//std::cout << "lXi - Oil     " << lXimOil       << std::endl;
			//std::cout << "norm(Xi - Oi) " << norm(Xi - Oi) << std::endl;
			//std::cout << "wo" << (wo*dodx[i].x) << ", " << (wo*dodx[i].y) << std::endl;
			//std::cout << "wk" << (wk*dkdxi[i].x) << ", " << (wk*dkdxi[i].y) << std::endl;
			//std::cout << "ws" << (ws*dsdx[i].x) << ", " << (ws*dsdx[i].y) << std::endl;
			//std::cout << std::endl;
		}
		//std::cout << "loss o        " << lo            << std::endl;
		//std::cout << "loss k        " << lk            << std::endl;
		//std::cout << "loss s        " << ls            << std::endl;
		//std::cout << "loss all      " << 0.03 * lo + 10000 * lk + 0.0025 * ls << std::endl;
		//std::cout << std::endl;
		return 0;
	}
public:
	TrajectrorySmoothing(cv::Size size){
		vf.SetMap(size);
		FIRST_POINT = 1;
		LAST_POINT = 1;
	}


	int Smooth(
		std::vector<cv::Point2d> X,
		std::vector<cv::Point2d> &Xout,
		std::vector<cv::Point> &obstacle,
		double wo_in, double wk_in, double ws_in,
		double dmax_in, double kmax_in,
		int num){

		vf.SetPoints(obstacle);
		vf.CreateVoronoi();

		wo = wo_in;
		wk = wk_in;
		ws = ws_in;
		dmax = dmax_in;
		kmax = kmax_in;

		for (int n = 0; n < num; ++n){
			std::vector<cv::Point2d> dodx(X.size());
			std::vector<cv::Point2d> dkdxi(X.size());
			std::vector<cv::Point2d> dkdxim1(X.size());
			std::vector<cv::Point2d> dkdxip1(X.size());
			std::vector<cv::Point2d> dsdx(X.size());

			TrajectrorySmoothing::Differential(X, dodx, dkdxi, dkdxim1, dkdxip1, dsdx);

			for (int i = FIRST_POINT; i < X.size() - LAST_POINT; ++i){
				X[i] -= cv::Point2d(
					(wo*dodx[i].x) + (wk*dkdxi[i].x) + (ws*dsdx[i].x),
					(wo*dodx[i].y) + (wk*dkdxi[i].y) + (ws*dsdx[i].y));
			}
			for (int i = FIRST_POINT + 1; i < X.size() - LAST_POINT - 1; ++i){
				X[i - 1] -= cv::Point2d(
					(wk*dkdxim1[i].x),
					(wk*dkdxim1[i].y));
				X[i + 1] -= cv::Point2d(
					(wk*dkdxip1[i].x),
					(wk*dkdxip1[i].y));
			}
			Debug(X, obstacle, 1);
		}
		Xout = X;
		return 0;
	}

	void Interpolation(std::vector<cv::Point2d> trajectory, std::vector<cv::Point2d> &trajectory_out, int num){
		trajectory_out.clear();
		interpolation_num = num;

		trajectory_out.push_back(trajectory[0]);
		for (int i = 0; i < trajectory.size() - 1; ++i){
			for (int n = 1; n < interpolation_num; ++n){
				trajectory_out.push_back(trajectory[i] + (trajectory[i + 1] - trajectory[i]) / interpolation_num * n);
			}
			trajectory_out.push_back(trajectory[i + 1]);
		}
	}
	void InterpolationSmooth(
		std::vector<cv::Point2d> X,
		std::vector<cv::Point2d> &Xout,
		std::vector<cv::Point> &obstacle,
		double wo_in,
		double wk_in,
		double ws_in,
		double dmax_in,
		double kmax_in,
		int num){

		wo = wo_in;
		wk = wk_in;
		ws = ws_in;
		dmax = dmax_in;
		kmax = kmax_in;

		for (int n = 0; n < num; ++n){
			std::vector<cv::Point2d> dodx(X.size());
			std::vector<cv::Point2d> dkdxi(X.size());
			std::vector<cv::Point2d> dkdxim1(X.size());
			std::vector<cv::Point2d> dkdxip1(X.size());
			std::vector<cv::Point2d> dsdx(X.size());

			TrajectrorySmoothing::Differential(X, dodx, dkdxi, dkdxim1, dkdxip1, dsdx);

			for (int i = FIRST_POINT; i < X.size() - LAST_POINT; ++i){
				if ((i % (interpolation_num))){
					X[i] -= cv::Point2d(
						(wo*dodx[i].x) + (wk*dkdxi[i].x) + (ws*dsdx[i].x),
						(wo*dodx[i].y) + (wk*dkdxi[i].y) + (ws*dsdx[i].y));
				}
			}
			for (int i = FIRST_POINT + 1; i < X.size() - LAST_POINT - 1; ++i){
				if ((i - 1) % (interpolation_num)){
					X[i - 1] -= cv::Point2d(
						(wk*dkdxim1[i].x),
						(wk*dkdxim1[i].y));
				}
				if ((i + 1) % (interpolation_num)){
					X[i + 1] -= cv::Point2d(
						(wk*dkdxip1[i].x),
						(wk*dkdxip1[i].y));
				}
			}
			//////////////////////////
			Debug(X, obstacle, 1);  //
			//////////////////////////
		}
		Xout = X;
		return;
	}
};

#endif