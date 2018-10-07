package main

import (
	"NN/models"
)

func main() {

	trainSet := [][][]float64{{{1.0},{2.0}}, {{3.0}, {6.0}}}
	var nn models.NeuralNetwork

	nn.Init(1,1,1,1)
	nn.SetInput([]float64{1.0})
	nn.Propagate()
	nn.Print()

	nn.Train(trainSet)

}