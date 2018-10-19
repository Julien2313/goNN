package models

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/fogleman/gg"
	"github.com/goNN/helper"
)

type NeuralNetwork struct {
	NbrInput, NbrOutput, NbrHiddenLayers, NbrNeuronsPerLayer int
	LearningRate                                             float64
	Neurons                                                  [][]Neuron
	OutputWaited                                             []float64
}

func (nn *NeuralNetwork) Init(nbrInput, nbrOutput, nbrHiddenLayers, nbrNeuronsPerLayer int) {
	nn.NbrInput = nbrInput
	nn.NbrOutput = nbrOutput
	nn.NbrHiddenLayers = nbrHiddenLayers
	nn.NbrNeuronsPerLayer = nbrNeuronsPerLayer

	nn.LearningRate = 0.05

	//Init Neurons
	nn.Neurons = make([][]Neuron, 2+nbrHiddenLayers)
	nn.Neurons[0] = make([]Neuron, nbrInput)

	for numNeuron := 0; numNeuron < nbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Biais = rand.Float64()*MaxBiais*2 - MaxBiais
	}

	for numLayer := 1; numLayer < nbrHiddenLayers+1; numLayer++ {
		nn.Neurons[numLayer] = make([]Neuron, nbrNeuronsPerLayer)

		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			nn.Neurons[numLayer][numNeuron].Biais = rand.Float64()*MaxBiais*2 - MaxBiais
			nn.Neurons[numLayer][numNeuron].Weights = make([]float64, len(nn.Neurons[numLayer-1]))
			nn.Neurons[numLayer][numNeuron].NewWeights = make([]float64, len(nn.Neurons[numLayer-1]))
			for numWeightNeuron := 0; numWeightNeuron < len(nn.Neurons[numLayer-1]); numWeightNeuron++ {
				nn.Neurons[numLayer][numNeuron].Weights[numWeightNeuron] = rand.Float64()*MaxWeight*2 - MaxWeight
			}

		}
	}

	nn.Neurons[nbrHiddenLayers+1] = make([]Neuron, nbrOutput)

	for numNeuron := 0; numNeuron < nbrOutput; numNeuron++ {
		nn.Neurons[nbrHiddenLayers+1][numNeuron].Biais = rand.Float64()*MaxBiais*2 - MaxBiais

		nn.Neurons[nbrHiddenLayers+1][numNeuron].Weights = make([]float64, len(nn.Neurons[nbrHiddenLayers]))
		nn.Neurons[nbrHiddenLayers+1][numNeuron].NewWeights = make([]float64, len(nn.Neurons[nbrHiddenLayers]))
		for numWeightNeuron := 0; numWeightNeuron < len(nn.Neurons[nbrHiddenLayers]); numWeightNeuron++ {
			nn.Neurons[nbrHiddenLayers+1][numNeuron].Weights[numWeightNeuron] = rand.Float64()*MaxWeight*2 - MaxWeight
		}

	}
}

func (nn *NeuralNetwork) Print() {
	for _, layer := range nn.Neurons {
		for _, neuron := range layer {
			fmt.Print(neuron.Value, neuron.Weights, ", ")
		}
		fmt.Println()
		fmt.Println("----------------")
	}
}

func (nn *NeuralNetwork) Draw(epoch int) {

	const W = 1024 * 2
	const H = 1024 * 2
	dc := gg.NewContext(W, H)
	dc.SetRGBA(0, 0, 0, 0)
	dc.Clear()

	offSetX := W / (len(nn.Neurons) + 1)
	maxNumOfneurons := 0
	for numLayer := 0; numLayer < len(nn.Neurons); numLayer++ {
		if len(nn.Neurons[numLayer]) > maxNumOfneurons {
			maxNumOfneurons = len(nn.Neurons[numLayer])
		}
	}

	for numLayer := 1; numLayer < len(nn.Neurons); numLayer++ {
		x1 := float64(offSetX * (numLayer + 1))
		x2 := float64(offSetX * (numLayer))
		offSetY := H / (len(nn.Neurons[numLayer]) + 1)
		offSetYM1 := H / (len(nn.Neurons[numLayer-1]) + 1)
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			y1 := float64(offSetY * (numNeuron + 1))
			for numWeight := 0; numWeight < len(nn.Neurons[numLayer][numNeuron].Weights); numWeight++ {
				y2 := float64(offSetYM1 * (numWeight + 1))
				//x2 := rand.Float64() * W
				//y2 := rand.Float64() * H
				var r float64
				var b float64
				var w float64
				if nn.Neurons[numLayer][numNeuron].Weights[numWeight] > 0.0 {
					r = 1.0 // nn.Neurons[numLayer][numNeuron].Weights[numWeight]
					b = 0.0
				} else {
					r = 0.0 // nn.Neurons[numLayer][numNeuron].Weights[numWeight]
					b = 1.0
				}
				w = math.Abs(nn.Neurons[numLayer][numNeuron].Weights[numWeight]) / 5.0
				g := 0.0
				a := 1.0
				dc.SetRGBA(r, g, b, a)
				dc.SetLineWidth(w)
				dc.DrawLine(x1, y1, x2, y2)
				dc.Stroke()
			}
		}
	}

	for numLayer := 0; numLayer < len(nn.Neurons); numLayer++ {
		x := offSetX * (numLayer + 1)
		offSetY := H / (len(nn.Neurons[numLayer]) + 1)
		size := float64(len(nn.Neurons[numLayer])) / float64(maxNumOfneurons) / 3.0
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			y := offSetY * (numNeuron + 1)
			dc.SetRGBA(0, 0, 0, 1)
			dc.DrawCircle(float64(x), float64(y), size*float64(H)/float64(len(nn.Neurons[numLayer])+3))
			dc.Fill()
			dc.SetRGBA(1, 1, 1, 1)
			dc.DrawCircle(float64(x), float64(y), size*float64(H)/float64(len(nn.Neurons[numLayer])+3)*0.9)
			dc.Fill()
		}
	}

	dc.SavePNG(fmt.Sprintf("%04d", epoch) + "out.png")
}

func (nn *NeuralNetwork) SetInput(inputs []float64) error {
	if len(inputs) != nn.NbrInput {
		return errors.New("Number of inputs doesn't match the model")
	}

	for numNeuron := 0; numNeuron < nn.NbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Value = inputs[numNeuron]
	}

	return nil
}

func (nn *NeuralNetwork) Propagate() {
	for numLayer := 1; numLayer < len(nn.Neurons); numLayer++ {
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			sum := nn.Neurons[numLayer][numNeuron].Biais
			for numWeightNeuron := 0; numWeightNeuron < len(nn.Neurons[numLayer-1]); numWeightNeuron++ {
				sum += nn.Neurons[numLayer][numNeuron].Weights[numWeightNeuron] * nn.Neurons[numLayer-1][numWeightNeuron].Value
			}
			nn.Neurons[numLayer][numNeuron].Value = helper.Sigmoid(sum)
		}
	}
}

func (nn *NeuralNetwork) BackProp(output []float64) {
	for numNeuronOutput := 0; numNeuronOutput < nn.NbrOutput; numNeuronOutput++ {
		neuron := &nn.Neurons[nn.NbrHiddenLayers+1][numNeuronOutput]
		neuron.Error = (neuron.Value - output[numNeuronOutput])
		neuron.Biais -= (nn.LearningRate * neuron.Error)
		for numWeight := 0; numWeight < len(nn.Neurons[nn.NbrHiddenLayers]); numWeight++ {
			neuron.NewWeights[numWeight] = neuron.Weights[numWeight] - nn.LearningRate*neuron.Error*nn.Neurons[nn.NbrHiddenLayers][numWeight].Value*helper.SigmoidDerivate(neuron.Value)
		}
	}

	for numLayer := nn.NbrHiddenLayers; numLayer >= 1; numLayer-- {
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			errorAfter := 0.0
			for numWeightAfter := 0; numWeightAfter < len(nn.Neurons[numLayer+1]); numWeightAfter++ {
				errorAfter += nn.Neurons[numLayer+1][numWeightAfter].Error * helper.SigmoidDerivate(nn.Neurons[numLayer+1][numWeightAfter].Value)
			}

			nn.Neurons[numLayer][numNeuron].Error = errorAfter * helper.Sigmoid(nn.Neurons[numLayer][numNeuron].Value)

			for numWeight := 0; numWeight < len(nn.Neurons[numLayer-1]); numWeight++ {
				nn.Neurons[numLayer][numNeuron].NewWeights[numWeight] = nn.Neurons[numLayer][numNeuron].Weights[numWeight] - nn.LearningRate*nn.Neurons[numLayer][numNeuron].Error*nn.Neurons[numLayer-1][numWeight].Value
			}
		}
	}

	for numLayer := 1; numLayer < nn.NbrHiddenLayers+2; numLayer++ {
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			for numWeights := 0; numWeights < len(nn.Neurons[numLayer-1]); numWeights++ {
				if nn.Neurons[numLayer][numNeuron].NewWeights[numWeights] > MaxWeight {
					nn.Neurons[numLayer][numNeuron].NewWeights[numWeights] = MaxWeight
				}
				if nn.Neurons[numLayer][numNeuron].NewWeights[numWeights] < -MaxWeight {
					nn.Neurons[numLayer][numNeuron].NewWeights[numWeights] = -MaxWeight
				}
				nn.Neurons[numLayer][numNeuron].Weights[numWeights] = nn.Neurons[numLayer][numNeuron].NewWeights[numWeights]
			}
			if nn.Neurons[numLayer][numNeuron].Biais > MaxBiais {
				nn.Neurons[numLayer][numNeuron].Biais = MaxBiais
			}
			if nn.Neurons[numLayer][numNeuron].Biais < -MaxBiais {
				nn.Neurons[numLayer][numNeuron].Biais = -MaxBiais
			}
		}
	}
}

func (nn *NeuralNetwork) Train(dataSet [][][]float64) {
	for _, data := range dataSet {
		nn.SetInput(data[0])
		nn.Propagate()
		nn.BackProp(data[1])
	}
}

func (nn *NeuralNetwork) CheckTraining(dataSet [][][]float64) float64 {
	var error float64
	error = 0.0
	for _, data := range dataSet {
		nn.SetInput(data[0])
		nn.Propagate()
		for cpt, output := range data[1] {
			error += math.Pow(nn.Neurons[nn.NbrHiddenLayers+1][cpt].Value-output, 2)
		}

	}
	return error
}
