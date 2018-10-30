#include <iostream>

namespace sn 
{
	#ifndef DATASET_H
	#define DATASET_H

    class Dataset
    {
    public:
        Dataset(int nPoints, int nIn, int nOut);
        
		~Dataset();
       
		float * getPointIn(int point);
        
        float * getPointOut(int point);

		float * getIn();

		float * getOut();

        int getNPoints() const;

        int getNIn() const;

        int getNOut() const;

        explicit Dataset(std::istream& is);

        explicit Dataset(Dataset * dataset);

        friend std::ostream& operator << (std::ostream& os, Dataset * ds);

    private:
        int nPoints;
        int nIn;
        int nOut;
        float * in;
        float * out;
    };

	#endif

}
