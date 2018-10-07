package models

type Neuron struct {
	Weights  []float64
	Value    float64
	Error    float64
	Biais    float64
	Expected float64

	TotalErrorByWithOutput float64
	NewWeights             []float64
}
