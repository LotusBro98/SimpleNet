#include "Layer.h"

#include <thread>
#include <pthread.h>

namespace sn
{
	Layer::Layer(int nIn, int nOut, float (*activation)(float x), int nPoints, int nProcessors)
	{
		this->nIn = nIn;
		this->nOut = nOut;
		this->nPoints = nPoints;
		this->out = new float[nOut * nPoints];
		this->in = nullptr;

		this->activation = activation;

		this->weights = new float[(nIn + 1) * nOut];
		for (int i = 0; i < nOut * (nIn + 1); i++)
			weights[i] = ((float) std::rand() / RAND_MAX - 0.5) / 0.5;

		this->nProcessors = nProcessors;

		this->processors = new Processor*[nProcessors];
		for (int i = 0; i < nProcessors; i++)
			processors[i] = new Processor(this, i);

		//std::cout << this;
	}

	Layer::~Layer()
	{
		delete [] weights;
		delete [] out;

		for (int i = 0; i < nProcessors; i++)
			delete processors[i];
		delete[] processors;
	}

	Layer::Layer(Layer * prev, int nOut) : Layer(prev->nOut, nOut, prev->activation, prev->nPoints, prev->nProcessors)
	{
		this->in = prev->out;
	}

	Layer::Layer(std::istream& is, float (*activation)(float x), int nPoints, int nProcessors)
	{
		is >> nIn;
		is >> nOut;
		this->nPoints = nPoints;
		out = new float[nOut * nPoints];
		in = nullptr;

		this->activation = activation;

		weights = new float[(nIn + 1) * nOut];
		for (int i = 0; i < nOut * (nIn + 1); i++)
			is >> weights[i];

		this->nProcessors = nProcessors;

		this->processors = new Processor*[nProcessors];
		for (int i = 0; i < nProcessors; i++)
			processors[i] = new Processor(this, i);
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

	void Layer::process(int point)
	{
		float * output = out + point * nOut;
		float * input = in + point * nIn;

		for (int iOut = 0; iOut < nOut; iOut++) {
			float sum = weights[nIn];
			for (int iIn = 0; iIn < nIn; ++iIn)
				sum += weight(iOut, iIn) * input[iIn];
			output[iOut] = activation(sum);
		}
	}

	void threadFunc(Layer::Processor * proc)
	{
		while (true)
		{
			proc->work.lock();
			if (proc->layer == nullptr)
				break;
			for (int i = proc->iThread; i < proc->layer->nPoints; i += proc->layer->nProcessors)
				proc->layer->process(i);
			proc->done.unlock();
		}
	}

	Layer::Processor::Processor(Layer * layer, int iThread)
	{
		this->layer = layer;
		this->iThread = iThread;

		this->done.unlock();
		this->work.lock();
		this->thread = new std::thread(threadFunc, this);

		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(iThread % 4, &set);
		pthread_setaffinity_np(pthread_self(), sizeof(set), &set);	
	}

	Layer::Processor::~Processor()
	{
		this->layer = nullptr;
		this->work.unlock();
		this->thread->join();
	}

	void Layer::Processor::process()
	{
		this->done.lock();
		this->work.unlock();
	}

	void Layer::Processor::wait()
	{
		this->done.lock();
		this->done.unlock();
	}

	void Layer::processAll()
	{
		if (nProcessors == 1)
			for (int i = 0; i < nPoints; i++)
				process(i);
		else
		{
			for (int i = 0; i < nProcessors; i++)
				processors[i]->process();
	
			for (int i = 0; i < nProcessors; i++)
				processors[i]->wait();
		}
	}

	float * Layer::getOut()
	{
		return out;
	}

	void Layer::setOut(float * out)
	{
		this->out = out;
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
