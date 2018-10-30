#include "Layer.h"

namespace sn
{
	Layer::Layer(int nIn, int nOut, float (*activation)(float x))
	{
		this->nIn = nIn;
		this->nOut = nOut;
		this->out = new float[nOut];
		this->in = nullptr;

		this->activation = activation;

		this->weights = new float[(nIn + 1) * nOut];
		for (int i = 0; i < nOut * (nIn + 1); i++)
			weights[i] = ((float) std::rand() / RAND_MAX - 0.5) / 0.5;

		//std::cout << this;
	}

	Layer::~Layer()
	{
		delete [] weights;
		delete [] out;
	}

	Layer::Layer(Layer * prev, int nOut) : Layer(prev->nOut, nOut, prev->activation)
	{
		this->in = prev->out;
	}

	Layer::Layer(std::istream& is, float (*activation)(float x))
	{
		is >> nIn;
		is >> nOut;
		out = new float[nOut];
		in = nullptr;

		this->activation = activation;

		weights = new float[(nIn + 1) * nOut];
		for (int i = 0; i < nOut * (nIn + 1); i++)
			is >> weights[i];

	}

	void Layer::setInput(float * in)
	{
		this->in = in;
	}

	float & Layer::weight(int iOut, int iIn)
	{
		return weights[iOut * (nIn + 1) + iIn];
	}

	float *Layer::getWeights() {
		return weights;
	}

	void Layer::process(float * output)
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

	float * Layer::getOut()
	{
		return out;
	}

	int Layer::getNIn(){
		return nIn;
	}

	int Layer::getNOut(){
		return nOut;
	}

	std::ostream& operator << (std::ostream& os, Layer * layer)
	{
		os << layer->nIn << " " << layer->nOut << std::endl;
		for (int iOut = 0; iOut < layer->nOut; iOut++)
		{
			for (int iIn = 0; iIn < layer->nIn + 1; iIn++)
				os << layer->weight(iOut, iIn) << " ";
			os << std::endl;
		}
		return os;
	}

}
