#include <cmath>

#include "funcs.h"

namespace sn
{
	float sigmoid(float x)
	{
		return 1 / (1 + std::exp(-x));
	}

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
}
