#ifdef _DEBUG
//Debugモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Releaseモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300.lib") 
#endif

#include "trajectorysmoothing.hpp"
#include "voronoifield.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>


using namespace std;
using namespace cv;

void PlotDot(Mat im, Point2d p, int c = 0){
	for (int i = 1; i < 4; ++i){
		im.at<Vec3b>(p)[c] = 255;
		im.at<Vec3b>(p + Point2d(0, i))[c] = 255;
		im.at<Vec3b>(p + Point2d(i, 0))[c] = 255;
		im.at<Vec3b>(p + Point2d(i, i))[c] = 255;
		im.at<Vec3b>(p + Point2d(0, -i))[c] = 255;
		im.at<Vec3b>(p + Point2d(-i, 0))[c] = 255;
		im.at<Vec3b>(p + Point2d(-i, -i))[c] = 255;
		im.at<Vec3b>(p + Point2d(i, -i))[c] = 255;
		im.at<Vec3b>(p + Point2d(-i, i))[c] = 255;
	}
}


int main(){
	double s = 50;
	double b = 10;

//	s = 25;
	vector<Point2d> X;


	//X.push_back(Point2d(10, 80));
	//X.push_back(Point2d(18.3093, 79.3709));
	//X.push_back(Point2d(26.3362, 77.1328));
	//X.push_back(Point2d(33.7723, 73.3717));
	//X.push_back(Point2d(40.3317, 68.2322));
	//X.push_back(Point2d(46.6521, 62.8015));
	//X.push_back(Point2d(53.1788, 57.6888));
	//X.push_back(Point2d(58.1158, 51.7736));
	//X.push_back(Point2d(62.7548, 44.8308));
	//X.push_back(Point2d(66.8481, 37.5723));
	//X.push_back(Point2d(70.6092, 30.1363));
	//X.push_back(Point2d(75.5704, 23.8437));
	//X.push_back(Point2d(81.1206, 18.9036));
	//X.push_back(Point2d(88.3791, 14.8103));
	//X.push_back(Point2d(96.2967, 12.2117));
	//X.push_back(Point2d(104.569, 11.2077));
	//X.push_back(Point2d(112.502, 10.6826));
	//X.push_back(Point2d(120.774, 9.67862));

	X.push_back(Point2d(10, 100));
	X.push_back(Point2d(18.2462, 101.001));
	X.push_back(Point2d(26.3982, 102.597));
	X.push_back(Point2d(34.6444, 103.599));
	X.push_back(Point2d(42.6895, 105.178));
	X.push_back(Point2d(50.9357, 104.176));
	X.push_back(Point2d(58.4689, 100.675));
	X.push_back(Point2d(64.5515, 95.0183));
	X.push_back(Point2d(68.5883, 87.7583));
	X.push_back(Point2d(70.1841, 79.6063));
	X.push_back(Point2d(71.1855, 71.3601));
	X.push_back(Point2d(74.6862, 63.827));
	X.push_back(Point2d(78.723, 56.5671));
	X.push_back(Point2d(82.1745, 49.1304));
	X.push_back(Point2d(87.8316, 43.0477));
	X.push_back(Point2d(95.0916, 39.0109));
	X.push_back(Point2d(103.244, 37.4151));
	X.push_back(Point2d(111.49, 38.4166));
	X.push_back(Point2d(119.642, 40.0124));


	//X.push_back(Point2d(0 * s + b, 0.00001 * s + b));
	//X.push_back(Point2d(1 * s + b, 0 * s + b));
	//X.push_back(Point2d(2 * s + b, 0 * s + b));
	//X.push_back(Point2d(2.000001 * s + b, 1 * s + b));
	//X.push_back(Point2d(2.000002 * s + b, 2. * s + b));
	//X.push_back(Point2d(2 * s + b, 3 * s + b));
	//X.push_back(Point2d(3 * s + b, 3 * s + b));
	//X.push_back(Point2d(4 * s + b, 3.00001 * s + b));
	//zji
	//X.push_back(Point2d(0 * s + b, 0 * s + b));
	//X.push_back(Point2d(1 * s + b, 0 * s + b));
	//X.push_back(Point2d(2 * s + b, 0 * s + b));
	//X.push_back(Point2d(2 * s + b, 1 * s + b));
	//X.push_back(Point2d(2 * s + b, 2 * s + b));
	//X.push_back(Point2d(2 * s + b, 3 * s + b));
	//X.push_back(Point2d(3 * s + b, 3 * s + b));
	//X.push_back(Point2d(4 * s + b, 3 * s + b));
	//konoji
	//X.push_back(Point2d(0 * s + b, 0 * s + b));
	//X.push_back(Point2d(1 * s + b, 0 * s + b));
	//X.push_back(Point2d(2 * s + b, 0 * s + b));
	//X.push_back(Point2d(3 * s + b, 0 * s + b));
	//X.push_back(Point2d(4 * s + b, 0 * s + b));
	//X.push_back(Point2d(4 * s + b, 1 * s + b));
	//X.push_back(Point2d(4 * s + b, 2 * s + b));
	//X.push_back(Point2d(4 * s + b, 3 * s + b));
	//X.push_back(Point2d(3 * s + b, 3 * s + b));
	//X.push_back(Point2d(2 * s + b, 3 * s + b));
	//X.push_back(Point2d(1 * s + b, 3 * s + b));
	//X.push_back(Point2d(0 * s + b, 3 * s + b));
	vector<Point> obstacle;
	//obstacle.push_back(Point(3 * s   + b, 2 * s   + b));
	//obstacle.push_back(Point(2.5 * s + b, 1.5 * s + b));
	//obstacle.push_back(Point(1 * s   + b, 1 * s   + b));
	//obstacle.push_back(Point(1 * s   + b, 1 * s   + b));
	Mat map;
	map = imread("C:\\Users\\0133752\\Desktop\\map.png", 0);
	for (int y = 0; y < map.size().height; ++y){
		for (int x = 0; x < map.size().width; ++x){
			if (map.at<unsigned char>(Point(x, y)) == 0){
				obstacle.push_back(Point(x, y));
			}
		}
	}


	TrajectrorySmoothing ts(Size(512, 512));

	double wo = 0.05 ;
	double wk = 3000 ;
	double ws = 0.002;
	double dmax = 1 * s;
	double kmax = 0.001;
	//wo = 0.001;
	//wk = 50;
	//ws = 0.02;
	//dmax = 1 * s;
	//kmax = 0.001;
	wo = 0.01;
	wk = 5;
	ws = 0.001;
	dmax = 10;
	kmax = 0.001;
	ts.Smooth(X, X, obstacle, wo, wk, ws, dmax, kmax, 100);

	int interpolation_num = 10;
	ts.Interpolation(X, X, interpolation_num);

	wo = 0.00025;
	wk = 0;
	ws = 0.01;
	dmax = 10;
	kmax = 0.001;
	ts.Smooth(X, X, obstacle, wo, wk, ws, dmax, kmax, 500);


	CVPlot plt(cv::Point(512, 512));
	Mat image(Size(512, 512), CV_8UC3, Scalar::all(0));
	for (auto &x : X)       { plt.Plot(x); }
	for (auto &o : obstacle){ plt.Plot(o); }
	//plt.XLim(0, 128);
	//plt.YLim(0, 128);
	plt.Show(0);

	return 0;
}




//#ifdef _DEBUG
////Debugモードの場合
//#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
//#else
////Releaseモードの場合
//#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300.lib") 
//#endif
//
//#include "opencv2/opencv.hpp"
//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
//
//#include <iostream>
//#define _USE_MATH_DEFINES
//#include <math.h>
//#include <vector>
//
//
//using namespace std;
//using namespace cv;
//
//
//class TrajectrorySmoothing{
//	double wo;
//	double wk;
//	double ws;
//	double dmax;
//	double kmax;
//public:
//	Point2d o;
//	TrajectrorySmoothing(double wo_in, double wk_in, double ws_in, double dmax_in, double kmax_in){
//		wo = wo_in;
//		wk = wk_in;
//		ws = ws_in;
//		dmax = dmax_in;
//		kmax = kmax_in;
//	}
//	int Smooth(vector<Point2d> X, vector<Point2d> &Xout, int n){
//		for (n = 0; n < 10; ++n){
//			double lo = 0;
//			double lk = 0;
//			double ls = 0;
//
//			vector<Point2d> dodx(X.size());
//			vector<Point2d> dkdxi(X.size());
//			vector<Point2d> dkdxim1(X.size());
//			vector<Point2d> dkdxip1(X.size());
//			vector<Point2d> dsdx(X.size());
//
//			for (int i = 1; i < X.size() - 1; ++i){
//				Point2d Xi = X[i];																			// Xi
//				Point2d Xim1 = X[i - 1];																	// Xi-1
//				Point2d Xip1 = X[i + 1];																	// Xi+1
//				double lXil = norm(Xi);																		// |Xi|
//				double lXim1l = norm(Xim1);																	// |Xi-1|
//				double lXip1l = norm(Xip1);																	// |Xi+1|
//				Point2d DXi = Xi - Xim1;																	// ΔXi = Xi - Xi-1
//				double lDXil = norm(DXi);																	// |ΔXi|
//				Point2d DXp1 = Xip1 - Xi;																	// ΔXi+1 = Xi+1 - Xi
//				Point2d DXp1_DX = DXp1 - DXi;																// ΔXi+1 - ΔXi
//				double Dx = Xi.x - Xim1.x;																	// Δx
//				double Dy = Xi.y - Xim1.y;																	// Δy
//				double Dxp1 = Xip1.x - Xi.x;																// Δx+1
//				double Dyp1 = Xip1.y - Xi.y;																// Δy+1
//				double Dphi = abs(atan2(Dyp1, Dxp1) - atan2(Dy, Dx));										// Δφ
//				double ki = Dphi / lDXil;																	// ki = Δφ / |ΔXi|
//				double a = (-1 / lDXil) * (-1 / sqrt(1 - cos(Dphi)*cos(Dphi)));								// (-1 / |ΔXi|) * (-1 / (1 - cos^2(Δφ))^1/2)
//				Point2d b = (Dphi / DXi.ddot(DXi)) * Point2d(1, 1);											// Δφ / (ΔXi)^2 * 1 or -1 or 0
//				Point2d p1 = (Xi - (Xi.ddot(-Xip1) * -Xip1)) / (lXip1l * lXip1l) / (lXil * lXip1l);		// (Xi - (Xi.*-Xi+1 * -Xi+1)) / |Xi+1||Xi+1| / |Xi||Xi+1|
//				Point2d p2 = (-Xip1 - (-Xip1.ddot(Xi)  *  Xi)) / (lXil   * lXil) / (lXil * lXip1l);		// (-Xi+1 - (-Xi+1.*Xi * Xi)) / |Xi||Xi| / |Xi||Xi+1|
//
//				lo += (norm(Xi - o) - dmax) *  (norm(Xi - o) - dmax);
//				lk += (Dphi / lDXil - kmax) * (Dphi / lDXil - kmax);
//				ls += DXp1_DX.ddot(DXp1_DX);
//
//				dodx[i] = 2 * (norm(Xi - o) - dmax) * (Xi - o) / norm(Xi - o);
//				//dkdxi[i]  = 2 * (ki - kmax) * (a * (-p1-p2) - b);
//				//dkdxim1[i] = 2 * (ki - kmax) * (a * p2       + b);
//				//dkdxip1[i] = 2 * (ki - kmax) * (a * p1          );
//				dsdx[i] = -4 * (Xip1 - 2 * Xi + Xim1);
//
//				cout << a << endl;
//				cout << b << endl;
//				cout << p1 << endl;
//				cout << p2 << endl;
//				cout << sqrt(1 - cos(Dphi)*cos(Dphi)) << endl;
//
//				//cout << dkdxi << endl;
//				//cout << dkdxim1 << endl;
//				//cout << dkdxip1 << endl;
//				cout << endl;
//			}
//			for (int i = 1; i < X.size() - 1; ++i){
//				X[i] -= Point2d(
//					(wo*dodx[i].x) + (wk*dkdxi[i].x) + (ws*dsdx[i].x),
//					(wo*dodx[i].y) + (wk*dkdxi[i].y) + (ws*dsdx[i].y));
//			}
//			for (int i = 2; i < X.size() - 2; ++i){
//				X[i - 1] -= Point2d(
//					(wk*dkdxim1[i].x),
//					(wk*dkdxim1[i].y));
//				X[i + 1] -= Point2d(
//					(wk*dkdxip1[i].x),
//					(wk*dkdxip1[i].y));
//			}
//		}
//		Xout = X;
//		return 0;
//	}
//};
//
//int main(){
//
//	double s = 50;
//
//	double wo = 0.03;
//	double wk = 0.03;
//	double ws = 0.04;
//	double dmax = 2 * s;
//	double kmax = 0.001;
//
//	vector<Point2d> X;
//	X.push_back(Point2d(0 * s, 0 * s));
//	X.push_back(Point2d(1 * s, 0 * s));
//	X.push_back(Point2d(2 * s, 0 * s));
//	X.push_back(Point2d(2 * s, 1 * s));
//	X.push_back(Point2d(2 * s, 2 * s));
//	X.push_back(Point2d(2 * s, 3 * s));
//	X.push_back(Point2d(3 * s, 3 * s));
//	X.push_back(Point2d(4 * s, 3 * s));
//
//	TrajectrorySmoothing ts(0.03, 0.03, 0.04, 2 * s, 0.001);
//
//	ts.o = Point2d(3 * s, 2 * s);
//	ts.Smooth(X, X, 10);
//
//	Mat image(Size(512, 512), CV_8UC1, Scalar::all(0));
//	image.at<unsigned char>(ts.o + Point2d(50, 50)) = 255;
//	for (int i = 0; i < X.size(); ++i){
//		image.at<unsigned char>(X[i] + Point2d(50, 50)) = 255;
//	}
//	imshow("image", image);
//	waitKey(0);
//
//	return 0;
//}