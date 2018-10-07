/*package main

import (
	"fmt"

	"github.com/goNN/models"
)

func main() {

	trainSet := [][][]float64{{{0.05, 0.1}, {0.01, 0.99}}}
	var nn models.NeuralNetwork

	nn.Init(2, 2, 1, 2)
	nn.SetInput(trainSet[0][0])
	nn.Neurons[1][0].Weights[0] = 0.15
	nn.Neurons[1][0].Weights[1] = 0.2
	nn.Neurons[1][1].Weights[0] = 0.25
	nn.Neurons[1][1].Weights[1] = 0.30

	nn.Neurons[2][0].Weights[0] = 0.40
	nn.Neurons[2][0].Weights[1] = 0.45
	nn.Neurons[2][1].Weights[0] = 0.50
	nn.Neurons[2][1].Weights[1] = 0.55

	nn.Neurons[1][0].Biais = 0.35
	nn.Neurons[1][1].Biais = 0.35
	nn.Neurons[2][0].Biais = 0.60
	nn.Neurons[2][1].Biais = 0.60
	nn.Propagate()
	nn.Print()
	fmt.Println("!!!!!!!!!!!!!!!!!!!")

	for x := 0; x < 500; x++ {
		nn.Train(trainSet)
	}
	nn.SetInput(trainSet[0][0])
	nn.Propagate()
	nn.Print()

}

*/package main

import (
	"fmt"

	"github.com/goNN/models"
)

func main() {

	trainSet := [][][]float64{{{1}, {2}}, {{0}, {0}}, {{4}, {8}}, {{5}, {10}}}
	var nn models.NeuralNetwork

	nn.Init(1, 1, 0, 0)
	nn.SetInput([]float64{5})
	nn.Propagate()
	fmt.Println(nn.Neurons[2][0].Value)

	for x := 0; x < 500; x++ {
		nn.Train(trainSet)
	}
	nn.SetInput([]float64{5})
	nn.Propagate()
	fmt.Println(nn.Neurons[2][0].Value)

}
