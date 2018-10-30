//
// Created by alex on 29.10.18.
//

#include <cmath>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

namespace sn
{
	
	





    float sqrDiffNorm(Dataset * dataset1, Dataset * dataset2)
    {
        if (dataset1->getNPoints() != dataset2->getNPoints())
            throw;

        int nPoints = dataset1->getNPoints();
        int nOut = dataset1->getNOut();
        float L = 0;
        for (int point = 0; point < nPoints; ++point) {
            float Li = 0;
            for (int i = 0; i < nOut; i++) {
                float d = dataset1->getPointOut(point)[i] - dataset2->getPointOut(point)[i];
                Li += d * d;
            }
            L += Li / nOut;
        }

        return L / nPoints;
    }

    void optimizeWeightNewton(float & weight, float L, float dw, float dL)
    {
        const float maxDw = 0.07;
        const float speed = 1;

        float Dw = - speed * L * dw / dL;

        if (std::abs(Dw) > maxDw)
            Dw = Dw > 0 ? maxDw : -maxDw;

        weight += Dw;
    }

    void optimizeWeightGradient(float & weight, float L, float dw, float dL)
    {
        const float maxDw = 0.1;
        const float speed = 1;

        float Dw = - speed * dL / dw;

        if (std::abs(Dw) > maxDw)
            Dw = (Dw > 0) ? maxDw : -maxDw;

        weight += Dw;
    }

    class Net
    {
    public:
        Net(int nIn, int nOut, int nHiddenLayers, int nNeurons[],
                float (*activation)(float x) = sn::sigmoid,
                float (*norm)(Dataset * prediction, Dataset * experiment) = sqrDiffNorm,
                void (*optimizeWeight)(float & weight, float L, float dw, float dL) = optimizeWeightNewton)
        {
            this->nLayers = nHiddenLayers + 1;
            layers = new Layer*[nLayers];

            if (nHiddenLayers == 0)
                layers[0] = new Layer(nIn, nOut, activation);
            else {
                layers[0] = new Layer(nIn, nNeurons[0], activation);

                for (int i = 1; i < nHiddenLayers; i++)
                    layers[i] = new Layer(layers[i - 1], nNeurons[i]);

                layers[nHiddenLayers] = new Layer(layers[nHiddenLayers - 1], nOut);
            }

            this->norm = norm;
            this->optimizeWeight = optimizeWeight;
        }

		Net(std::istream& is, 
				float (*activation)(float x) = sn::sigmoid,
                float (*norm)(Dataset * prediction, Dataset * experiment) = sqrDiffNorm,
                void (*optimizeWeight)(float & weight, float L, float dw, float dL) = optimizeWeightGradient)
		{
			is >> this->nLayers;
			layers = new Layer*[nLayers];

			layers[0] = new Layer(is, activation);
			for (int i = 1; i < nLayers; i++)
			{
				layers[i] = new Layer(is, activation);
				layers[i]->setInput(layers[i - 1]->getOut());
			}

            this->norm = norm;
            this->optimizeWeight = optimizeWeight;
		}

        ~Net()
        {
            for (int i = 0; i < nLayers; i++)
                delete layers[i];
            delete [] layers;
		}

		friend std::ostream& operator << (std::ostream& os, Net* net)
		{
			os << net->nLayers <<std::endl;

			for (int i = 0; i < net->nLayers; i++)
				os << net->layers[i];

			return os;
		}

        int getNIn()
        {
            return layers[0]->getNIn();
        }

        int getNOut()
        {
            return layers[nLayers - 1]->getNOut();
        }

        float * process(float * input, float * output = nullptr)
        {
            layers[0]->setInput(input);
            for (int i = 0; i < (nLayers - 1); i++)
                layers[i]->process();
            layers[nLayers - 1]->process(output);

            //std::cout << layers[0]->weight(0, 0) << " " << layers[0]->weight(0, 1) << " " << layers[0]->weight(0, 2) << std::endl;

            if (output != nullptr)
                return output;
            else
                return layers[nLayers - 1]->getOut();
        }

        float evaluate(Dataset * dataset, Dataset * calcBuffer)
        {
            int nPoints = dataset->getNPoints();

            for (int point = 0; point < nPoints; ++point)
                process(dataset->getPointIn(point), calcBuffer->getPointOut(point));

            return norm(dataset, calcBuffer);
        }

        float evaluate(Dataset * dataset)
        {
            Dataset * calcBuffer = new Dataset(dataset);

            float L = evaluate(dataset, calcBuffer);

            delete calcBuffer;

            return L;
        }

        float trainEpoch(Dataset * dataset)
        {
            Dataset * calcBuffer = new Dataset(dataset);

            float dw = 0.001;

            for (int layer = 0; layer < nLayers; ++layer)
            {
                int nWeights = layers[layer]->getNOut() * (layers[layer]->getNIn() + 1);
                for (int i = 0; i < nWeights; ++i)
                {
                    float & w = layers[layer]->getWeights()[i];
                    float L0 = evaluate(dataset, calcBuffer);
                    w += dw;
                    float L = evaluate(dataset, calcBuffer);
                    w -= dw;
                    float dL = L - L0;
                    optimizeWeight(w, L0, dw, dL);
                }
            }

            float L = evaluate(dataset, calcBuffer);

            delete calcBuffer;

			//std::cout << this;

            return L;
        }

		void showDistribution(float xMin, float xMax, int nX, float yMin, float yMax, int nY, int sizeX, int sizeY, Dataset * dataset)
        {
            if (this->getNIn() != 2 || this->getNOut() != 1)
                return;

            cv::Mat M0(nY, nX, CV_8UC3);
            for (int i = 0; i < nY; i++)
                for (int j = 0; j < nX; j++)
                {
                    float x = xMin + j * (xMax - xMin) / nX;
                    float y = yMin + (nY - i) * (yMax - yMin) / nY;
                    float xx[2] = {x, y};
                    float p = *(this->process(xx));
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

    private:
        int nLayers;
        Layer ** layers;

        float (*norm)(Dataset * prediction, Dataset * experiment);
        void (*optimizeWeight)(float & weight, float L, float dw, float dL);
    };
}

int main()
{
    std::ifstream file("../circles.txt");

    sn::Dataset * dataset = new sn::Dataset(file);

	std::srand(std::time(NULL));

	/*
    sn::Net * net = new sn::Net(2, 1, 1, new int[1] {40}, sn::sigmoid, sn::sqrDiffNorm, sn::optimizeWeightNewton);

	float L = 1;
	int i = 0;

    while (L > 0.01) {
        L = net->trainEpoch(dataset);
		std::cerr << L << std::endl;
		//net->showDistribution(-1, 1, 50, -1, 1, 50, 500, 500, dataset);
		//cv::waitKey(1);
		i++;
    }
    
	std::cerr << "--------------- " << i << " ---------------" << std::endl;
	*/

	file.close();
	file.open("net.txt");

	auto net = new sn::Net(file);

	net->showDistribution(-1, 1, 50, -1, 1, 50, 500, 500, dataset);
	cv::waitKey();

	std::cout << net;

    return 0;
}
