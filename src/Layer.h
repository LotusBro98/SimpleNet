#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>

namespace sn
{
	#ifndef LAYER_H
	#define LAYER_H

	class Layer
    {
    public:

		class Processor
		{
		public:
			Processor(Layer * layer, int iThread);
			~Processor();

			void process();
			void wait();

		private:
			friend void threadFunc(Processor * proc);

			Layer * layer;
			int iThread;
			std::thread * thread;
			std::mutex work;
			std::mutex done;
		};

        Layer(int nIn, int nOut, float (*activation)(float x), int nPoints, int nProcessors); 
        
		~Layer(); 
        
		Layer(Layer * prev, int nOut);
		
		Layer(std::istream& is, float (*activation)(float x), int nPoints, int nProcessors);	
        
		void setInput(float * in);
        
		float & weight(int iOut, int iIn);
        
		float *getWeights();
        
		void process(int point = 0);

		void processAll();
            
        float * getOut();

		void setOut(float * out);
        
        int getNIn();
		
		int getNOut();

		friend void threadFunc(Processor * proc);

		friend std::ostream& operator << (std::ostream& os, Layer * layer);

    private:
        float (*activation)(float x);

		Processor ** processors;

        int nIn;
        int nOut;
		int nPoints;
		int nProcessors;
        float * weights;
        float * in;
        float * out;
    };

	#endif
}

