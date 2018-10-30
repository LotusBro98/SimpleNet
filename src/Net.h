#include "Layer.h"
#include "Dataset.h"
#include "funcs.h"

namespace sn
{
	#ifndef NET_H
	#define NET_H

    class Net
    {
    public:
        Net(int nIn, int nOut = 1, int nHiddenLayers = 0, int nNeurons[] = nullptr,
                float (*activation)(float x) = sn::sigmoid,
                float (*norm)(Dataset * prediction, Dataset * experiment) = sqrDiffNorm,
                void (*optimizeWeight)(float & weight, float L, float dw, float dL) = optimizeWeightNewton,
				int nPoints = 1, int nThreads = 1);

		Net(Dataset * trainset, int nHiddenLayers = 0, int nNeurons[] = nullptr,
				float (*activation)(float x) = sn::sigmoid,
                float (*norm)(Dataset * prediction, Dataset * experiment) = sqrDiffNorm,
                void (*optimizeWeight)(float & weight, float L, float dw, float dL) = optimizeWeightNewton,
				int nThreads = 1);
      
		Net(std::istream& is, 
				float (*activation)(float x) = sn::sigmoid,
                float (*norm)(Dataset * prediction, Dataset * experiment) = sqrDiffNorm,
                void (*optimizeWeight)(float & weight, float L, float dw, float dL) = optimizeWeightGradient,
				int nPoints = 1, int nThreads = 1);
	
        ~Net();
   
		friend std::ostream& operator << (std::ostream& os, Net* net);

        int getNIn();

		int getNOut();

		int getNLayers();
       
        void processAll();
        
		float * process(float * input, float * output = nullptr);
      
     
        float evaluate(Dataset * dataset);
    
        float train(Dataset * dataset, float toLoss = 1);
   
    private:
        int nLayers;
        Layer ** layers;

        float evaluateTrain(Dataset * dataset, Dataset * calcBuffer);
		
		int nPoints;

        float (*norm)(Dataset * prediction, Dataset * experiment);
        void (*optimizeWeight)(float & weight, float L, float dw, float dL);
    };

	#endif
}
