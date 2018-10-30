#include <cmath>

namespace sn
{
	float sigmoid(float x)
	{
		return 1 / (1 + std::exp(-x));
	}
}
