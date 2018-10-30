//
// Created by alex on 29.10.18.
//

#include <iostream>
#include <fstream>
#include <string>


#include "Net.h"


#include <opencv2/opencv.hpp>

void showDistribution(sn::Net * net, sn::Dataset * dataset = nullptr,
		float xMin = -1, float xMax = 1, 
		float yMin = -1, float yMax = 1, 
		int nX = 50, int nY = 50,
		int sizeX = 500, int sizeY = 500 
		)
{
	if (net->getNIn() != 2 || net->getNOut() != 1)
		return;

	cv::Mat M0(nY, nX, CV_8UC3);
	for (int i = 0; i < nY; i++)
		for (int j = 0; j < nX; j++)
		{
			float x = xMin + j * (xMax - xMin) / nX;
			float y = yMin + (nY - i) * (yMax - yMin) / nY;
			float xx[2] = {x, y};
			float p = *(net->process(xx));
			//p = 0.5 * (1 + p);
			M0.at<cv::Vec3b>(i, j) = cv::Vec3b::all(p * 255);
		}

	cv::Mat M(sizeY, sizeX, CV_8UC3);
	cv::resize(M0, M, cv::Size2i(sizeX, sizeY), 0, 0, cv::INTER_NEAREST);

	if (dataset == nullptr)
		goto draw;

	for (int point = 0; point < dataset->getNPoints(); point++)
	{
		float x = dataset->getPointIn(point)[0];
		float y = dataset->getPointIn(point)[1];
		int j =    (x - xMin) / (xMax - xMin)  * sizeX;
		int i = (1-(y - yMin) / (yMax - yMin)) * sizeY;
		const cv::Vec3b color = dataset->getPointOut(point)[0] > 0.5f ? cv::Vec3b(0,0,255) : cv::Vec3b(255,0,0);
		cv::circle(M, cv::Point(j, i), 5, color, 2);
	}

	draw:

	cv::imshow("Distribution", M);

	cv::waitKey();
}

using namespace std;

int main(int argc, char * argv[])
{	
	if (argc != 3)
	{
		cerr << "Usage:	./evaluate <net> <dataset>" << endl;
		exit(1);
	}

    ifstream file;

	
	file.open(argv[2]);

	if (!(file.good()))
	{
		cerr << "No such dataset file: \"" << argv[2] << "\"" << endl;
		exit(1);
	}

	sn::Dataset * dataset;
	try {
		dataset = new sn::Dataset(file);
	} catch (exception e) {
		cerr << "Failed to read dataset from " << argv[2] << endl;
		exit(1);
	}

	file.close();



	file.open(argv[1]);

	if (!(file.good()))
	{
		cerr << "No such net file: \"" << argv[1] << "\"" << endl;
		exit(1);
	}

	sn::Net * net;
	try {
		net = new sn::Net(file, sn::sigmoid, sn::sqrDiffNorm, sn::optimizeWeightNewton);
	} catch (exception e) {
		cerr << "Failed to read net from " << argv[1] << endl;
		exit(1);
	}


	file.close();


	cout << net->evaluate(dataset) << endl;
	
	showDistribution(net, dataset);

    return 0;
}
