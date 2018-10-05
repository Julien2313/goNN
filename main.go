package main

import "NN/models"

func main() {
	var nn models.NeuralNetwork

	nn.Init(1,1,1,1)
	nn.Print()
	nn.SetInput([]float64{1.2})
	nn.Print()

}