package main

import "NN/models"

func main() {
	var nn models.NeuralNetwork

	nn.Init(1,1,1,1)
	nn.SetInput([]float64{1.0})
	nn.Propagate()
	nn.Print()

}