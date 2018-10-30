#include "Dataset.h"

namespace sn
{

	Dataset::Dataset(int nPoints, int nIn, int nOut)
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
}
