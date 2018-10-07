package helper

import "math"

func Sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Exp(-value))
}

func SigmoidDerivate(value float64) float64 {
	return value * (1.0 - value)
}
