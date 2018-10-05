package models

import (
	"math/rand"
	"errors"
	"fmt"
)

type NeuralNetwork struct {
	NbrInput, NbrOutput, NbrHiddenLayers, NbrNeuronsPerLayer int
	LearningRate float64
	Neurons [][]Neuron
	Weights [][]float64
}

func (nn *NeuralNetwork) Init(nbrInput, NbrOutput, nbrHiddenLayers, nbrNeuronsPerLayer int) {
	nn.NbrInput = nbrInput
	nn.NbrOutput = NbrOutput
	nn.NbrHiddenLayers = nbrHiddenLayers
	nn.NbrNeuronsPerLayer = nbrNeuronsPerLayer

	nn.LearningRate = 0.1

	//Init Neurons
	nn.Neurons = make([][]Neuron, 2 + nbrHiddenLayers)
	nn.Neurons[0] = make([]Neuron, nbrInput)

	for numNeuron := 0; numNeuron < nbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
	}

	for numLayer := 1; numLayer < nbrHiddenLayers+1; numLayer++ {
		nn.Neurons[numLayer] = make([]Neuron, nbrNeuronsPerLayer)

		for numNeuron := 0; numNeuron < nbrNeuronsPerLayer; numNeuron++ {
			nn.Neurons[numLayer][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
		}
	}

	nn.Neurons[nbrHiddenLayers+1] = make([]Neuron, NbrOutput)

	for numNeuron := 0; numNeuron < NbrOutput; numNeuron++ {
		nn.Neurons[nbrHiddenLayers+1][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
	}

	//init weights
	nn.Weights = make([][]float64, 1 + nbrHiddenLayers)
	nn.Weights[0] = make([]float64, nbrInput)

	for numNeuron := 0; numNeuron < nbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
	}

	for numLayer := 1; numLayer < nbrHiddenLayers+1; numLayer++ {
		nn.Neurons[numLayer] = make([]Neuron, nbrNeuronsPerLayer)

		for numNeuron := 0; numNeuron < nbrNeuronsPerLayer; numNeuron++ {
			nn.Neurons[numLayer][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
		}
	}

	nn.Neurons[nbrHiddenLayers+1] = make([]Neuron, NbrOutput)

	for numNeuron := 0; numNeuron < NbrOutput; numNeuron++ {
		nn.Neurons[nbrHiddenLayers+1][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
	}



}

func (nn *NeuralNetwork) Print () {
	fmt.Println(nn.Neurons)
}

func (nn *NeuralNetwork) SetInput(inputs []float64) error{
	if len(inputs) != nn.NbrInput {
		return errors.New("Number of inputs doesn't match the model")
	}

	for numNeuron := 0; numNeuron < nn.NbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Value = inputs[numNeuron]
	}

	return nil
}

func (nn *NeuralNetwork) Propagate() {

}