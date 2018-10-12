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
	"github.com/goNN/models"
	"math/rand"
)

func main() {

	var trainSet [][][]float64

	trainSet = make([][][]float64, 500)

	for train := 0; train < len(trainSet); train++ {
		trainSet[train] = make([][]float64, 2)
		trainSet[train][0] = make([]float64, 3)
		trainSet[train][1] = make([]float64, 1)
		x := rand.Float64()*40.0 - 20.0
		y := rand.Float64()*40.0 - 20.0
		z := rand.Float64()*40.0 - 20.0
		a := x + y + z
		trainSet[train][0][0] = x
		trainSet[train][0][1] = y
		trainSet[train][0][2] = z
		trainSet[train][1][0] = a

	}

	var nn models.NeuralNetwork

	nn.Init(3, 1, 1, 32)
	//nn.SetInput([]float64{2})
	//nn.Propagate()
	//fmt.Println(nn.Neurons[1][0].Value)
	////
	for x := 0; x < 500; x++ {
		nn.Train(trainSet)
		nn.Draw(x)
	}
	//nn.SetInput([]float64{5})
	//nn.Propagate()
	//fmt.Println(nn.Neurons[1][0].Value)

}
