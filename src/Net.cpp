#include "Net.h"

namespace sn
{
	Net::Net(int nIn, int nOut, int nHiddenLayers, int nNeurons[],
			float (*activation)(float x),
			float (*norm)(Dataset * prediction, Dataset * experiment),
			void (*optimizeWeight)(float & weight, float L, float dw, float dL),
			int nPoints, int nThreads)
	{
		this->nLayers = nHiddenLayers + 1;
		layers = new Layer*[nLayers];

		if (nHiddenLayers == 0)
			layers[0] = new Layer(nIn, nOut, activation, nPoints, nThreads);
		else {
			layers[0] = new Layer(nIn, nNeurons[0], activation, nPoints, nThreads);

			for (int i = 1; i < nHiddenLayers; i++)
				layers[i] = new Layer(layers[i - 1], nNeurons[i]);

			layers[nHiddenLayers] = new Layer(layers[nHiddenLayers - 1], nOut);
		}

		this->norm = norm;
		this->optimizeWeight = optimizeWeight;
	}

	Net::Net(Dataset * trainset, int nHiddenLayers, int nNeurons[],
			float (*activation)(float x),
			float (*norm)(Dataset * prediction, Dataset * experiment),
			void (*optimizeWeight)(float & weight, float L, float dw, float dL),
			int nThreads)
		: Net(trainset->getNIn(), trainset->getNOut(), nHiddenLayers, nNeurons, activation, norm, optimizeWeight, trainset->getNPoints(), nThreads)
	{
	}


	Net::Net(std::istream& is, 
			float (*activation)(float x),
			float (*norm)(Dataset * prediction, Dataset * experiment),
			void (*optimizeWeight)(float & weight, float L, float dw, float dL),
			int nPoints, int nThreads)
	{
		is >> this->nLayers;
		layers = new Layer*[nLayers];

		layers[0] = new Layer(is, activation, nPoints, nThreads);
		for (int i = 1; i < nLayers; i++)
		{
			layers[i] = new Layer(is, activation, nPoints, nThreads);
			layers[i]->setInput(layers[i - 1]->getOut());
		}

		this->norm = norm;
		this->optimizeWeight = optimizeWeight;
	}

	Net::~Net()
	{
		for (int i = 0; i < nLayers; i++)
			delete layers[i];
		delete [] layers;
	}

	std::ostream& operator << (std::ostream& os, Net* net)
	{
		os << net->nLayers <<std::endl;

		for (int i = 0; i < net->nLayers; i++)
			os << net->layers[i];

		return os;
	}

	int Net::getNIn()
	{
		return layers[0]->getNIn();
	}

	int Net::getNOut()
	{
		return layers[nLayers - 1]->getNOut();
	}

	int Net::getNLayers()
	{
		return nLayers;
	}

	float * Net::process(float * input, float * output)
	{
		if (output == nullptr)
			output = layers[nLayers - 1]->getOut();

		float * outPrev = layers[nLayers - 1]->getOut();

		layers[0]->setInput(input);
		layers[nLayers - 1]->setOut(output);

		for (int i = 0; i < nLayers; i++)
			layers[i]->process();

		layers[nLayers - 1]->setOut(outPrev);

		return output;
	}

	void Net::processAll()
	{
		for (int i = 0; i < nLayers; i++)
			layers[i]->processAll();
	}

	float Net::evaluateTrain(Dataset * dataset, Dataset * calcBuffer)
	{
		int nPoints = dataset->getNPoints();

		processAll();
		
		float L = norm(dataset, calcBuffer);
		
		return L;
	}

	float Net::evaluate(Dataset * dataset)
	{
		Dataset * calcBuffer = new Dataset(dataset);

		//TODO: improve this
		for (int point = 0; point < dataset->getNPoints(); point++)
			process(dataset->getPointIn(point), calcBuffer->getPointOut(point));

		float L = norm(dataset, calcBuffer);
		
		delete calcBuffer;

		return L;
	}

	float Net::train(Dataset * dataset, float toLoss)
	{
		Dataset * calcBuffer = new Dataset(dataset);

		float dw = 0.001;
		float Lfin = 1;

		int i = 0;

		layers[0]->setInput(dataset->getIn());
		float * outPrev = layers[nLayers - 1]->getOut();
		layers[nLayers - 1]->setOut(calcBuffer->getOut());

		do
		{
			for (int layer = 0; layer < nLayers; ++layer)
			{
				int nWeights = layers[layer]->getNOut() * (layers[layer]->getNIn() + 1);
				for (int i = 0; i < nWeights; ++i)
				{
					float & w = layers[layer]->getWeights()[i];
					float L0 = evaluateTrain(dataset, calcBuffer);
					w += dw;
					float L = evaluateTrain(dataset, calcBuffer);
					w -= dw;
					float dL = L - L0;
					optimizeWeight(w, L0, dw, dL);
				}
			}

			Lfin = evaluateTrain(dataset, calcBuffer);
			std::cerr << Lfin << std::endl;
			i++;
		}
		while (Lfin > toLoss);

		layers[nLayers - 1]->setOut(outPrev);

		delete calcBuffer;

		std::cerr << "--------------- " << i << " ---------------" << std::endl;
	
		return Lfin;
	}
}
