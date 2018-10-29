//
// Created by alex on 29.10.18.
//

#include <cmath>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

namespace sn
{
    class Layer
    {
    public:

        Layer(int nIn, int nOut, float (*activation)(float x))
        {
            this->nIn = nIn;
            this->nOut = nOut;
            this->out = new float[nOut];
            this->in = nullptr;

            this->activation = activation;

            this->weights = new float[(nIn + 1) * nOut];
            for (int i = 0; i < nOut * (nIn + 1); i++)
                weights[i] = (float) std::rand() / RAND_MAX;
        }

        ~Layer()
        {
            delete [] weights;
            delete [] out;
        }

        Layer(Layer * prev, int nOut) : Layer(prev->nIn, nOut, prev->activation)
        {
            this->in = prev->out;
        }

        void setInput(float * in)
        {
            this->in = in;
        }

        float & weight(int iOut, int iIn)
        {
            return weights[iOut * (nIn + 1) + iIn];
        }

        float *getWeights() const {
            return weights;
        }

        void process(float * output = nullptr)
        {
            if (output == nullptr)
                output = out;

            for (int iOut = 0; iOut < nOut; iOut++) {
                float sum = weights[nIn];
                for (int iIn = 0; iIn < nIn; ++iIn)
                    sum += weight(iOut, iIn) * in[iIn];
                output[iOut] = activation(sum);
            }
        }

        float * getOut()
        {
            return out;
        }

        int getNIn() const {
            return nIn;
        }

        int getNOut() const {
            return nOut;
        }

    private:
        float (*activation)(float x);

        int nIn;
        int nOut;
        float * weights;
        float * in;
        float * out;
    };


    class Dataset
    {
    public:
        Dataset(int nPoints, int nIn, int nOut)
        {
            this->nPoints = nPoints;
            this->nIn = nIn;
            this->nOut = nOut;

            this->in = new float[nIn * nPoints];
            this->out = new float[nOut * nPoints];
        }

        ~Dataset()
        {
            delete [] in;
            delete [] out;
        }

        float * getPointIn(int point)
        {
            return in + point * nIn;
        }

        float * getPointOut(int point)
        {
            return out + point * nOut;
        }

        int getNPoints() const {
            return nPoints;
        }

        int getNIn() const {
            return nIn;
        }

        int getNOut() const {
            return nOut;
        }

        explicit Dataset(std::istream& is)
        {
            is >> this->nPoints;
            is >> this->nIn;
            is >> this->nOut;

            this->in = new float[nIn * nPoints];
            this->out = new float[nOut * nPoints];

            for (int point = 0; point < nPoints; ++point) {
                for (int iIn = 0; iIn < nIn; ++iIn) {
                    is >> getPointIn(point)[iIn];
                }

                for (int iOut = 0; iOut < nOut; ++iOut) {
                    is >> getPointOut(point)[iOut];
                }
            }
        }

        explicit Dataset(Dataset * dataset) : Dataset(dataset->nPoints, dataset->nIn, dataset->nOut)
        {
            std::copy(dataset->in, dataset->in + dataset->nIn * dataset->nPoints, this->in);
            std::copy(dataset->out, dataset->out + dataset->nOut * dataset->nPoints, this->out);
        }

        friend std::ostream& operator << (std::ostream& os, Dataset * ds)
        {
            os << ds->nPoints << " " << ds->nIn << " " << ds->nOut << std::endl;

            for (int point = 0; point < ds->nPoints; ++point) {
                for (int iIn = 0; iIn < ds->nIn; ++iIn) {
                    os << ds->getPointIn(point)[iIn] << " ";
                }

                for (int iOut = 0; iOut < ds->nOut; ++iOut) {
                    os << ds->getPointOut(point)[iOut] << " ";
                }
                os << std::endl;
            }

            return os;
        }

    private:
        int nPoints;
        int nIn;
        int nOut;
        float * in;
        float * out;

        int parseInt(std::istream& is)
        {
            int a;
            is >> a;
            return a;
        }
    };






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
        const float maxDw = 0.1;
        const float speed = 0.5;

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
                float (*activation)(float x) = std::tanh,
                float (*norm)(Dataset * prediction, Dataset * experiment) = sqrDiffNorm,
                void (*optimizeWeight)(float & weight, float L, float dw, float dL) = optimizeWeightGradient)
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

        ~Net()
        {
            for (int i = 0; i < nLayers; i++)
                delete layers[i];
            delete [] layers;
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
                    p = 0.5 * (1 + p);
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

            cv::waitKey(1);
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
    std::ifstream file("../dataset.txt");

    sn::Dataset * dataset = new sn::Dataset(file);

    sn::Net * net = new sn::Net(2, 1, 1, new int[1] {2}, std::tanh, sn::sqrDiffNorm, sn::optimizeWeightGradient);

    for (int i = 0; i < 400; i++) {
        std::cout << net->trainEpoch(dataset) << std::endl;
        net->showDistribution(-1, 1, 50, -1, 1, 50, 500, 500, dataset);
    }

    return 0;
}