package models

import "math"

type Neuron struct {
	Weights []float64
	Value float64
	Error float64
	Biais float64
}

func (n *Neuron) Sigmoid(value float64) {
	n.Value = 1.0/(1.0+math.Exp(-value))
}