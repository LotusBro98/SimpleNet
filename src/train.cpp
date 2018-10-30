//
// Created by alex on 29.10.18.
//

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include <unistd.h>

#include "Net.h"

/*
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

	//cv::waitKey(1);
}
*/

using namespace std;

int main(int argc, char * argv[])
{	
	if (argc < 2)
	{
		cerr << "Usage:	./train <dataset> [options]" << endl
			 << "Options:" << endl
			 << "-L <trainLoss>" << endl
		//	 << "-t threads" << endl
			 << "-n nNeurons -- add hidden layer with nNeurons" << endl;
		exit(1);
	}

    ifstream file(argv[1]);

	if (!(file.good()))
	{
		cerr << "No such dataset file: \"" << argv[1] << "\"" << endl;
		exit(1);
	}

	sn::Dataset * dataset;
	try {
		dataset = new sn::Dataset(file);
	} catch (exception e) {
		cerr << "Failed to read dataset from " << argv[1] << endl;
		exit(1);
	}

	float trainLoss = 0.05;
	int nHidden = 0;
	int nThreads = 1;
	std::list<int> nNeuronsList;


	int opt;
	while ((opt = getopt(argc, argv, "L:t:n:")) != -1)
	{
		try {
			switch (opt)
			{
				case 'L':
					trainLoss = stof(optarg);
					break;
				case 'n':
					nNeuronsList.push_back(stoi(optarg));
					break;
				case 't':
					nThreads = stoi(optarg);
					break;
				default:
					cerr << "Unrecognized option: -" << (char)opt << endl;
					exit(1);
					break;
			}
		} catch (exception e) {
			cerr << "Failed to parse option -" << (char)opt << " from \"" << optarg << "\""<< endl;
			exit(1);
		}
	}

	if (optind != argc - 1)
	{
		cerr << "Too many arguments." << endl;
		exit(-1);
	}

	nHidden = nNeuronsList.size();
	int * nNeurons = (nHidden == 0 ? nullptr : new int[nHidden]);
	for (int i = 0; i < nHidden; i++)
	{
		nNeurons[i] = nNeuronsList.front();
		nNeuronsList.pop_front();
	}


	//--------------------------

	srand(time(NULL));

    sn::Net * net = new sn::Net(dataset, nHidden, nNeurons, sn::sigmoid, sn::sqrDiffNorm, sn::optimizeWeightNewton, nThreads);

    float L = net->train(dataset, trainLoss);
    
	cout << net;

	delete nNeurons;

    return 0;
}
