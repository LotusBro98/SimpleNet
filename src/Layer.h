#ifndef SN_LAYER_H
#define SN_LAYER_H

#include <cmath>
#include <iostream>

namespace sn
{
	float sigmoid(float x)
	{
		return 1 / (1 + std::exp(-x));
	}


    class Layer
    {
    public:

        Layer(int nIn, int nOut, float (*activation)(float x)); 
        
		~Layer(); 
        
		Layer(Layer * prev, int nOut);
		
		Layer(std::istream& is, float (*activation)(float x));	
        
		void setInput(float * in);
        
		float & weight(int iOut, int iIn);
        
		float *getWeights();
        
		void process(float * output = nullptr);
            
        float * getOut();
        
        int getNIn();
		
		int getNOut();

		friend std::ostream& operator << (std::ostream& os, Layer * layer);

    private:
        float (*activation)(float x);

        int nIn;
        int nOut;
        float * weights;
        float * in;
        float * out;
    };
}

#endif
