using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HelloNeuralNetwork.CostFunctions;
using HelloNeuralNetwork.Dropout;
using HelloNeuralNetwork.Helpers;
using HelloNeuralNetwork.Models;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork
{
    public class Network
    {

        private readonly int[] _sizes;

        private readonly int _numLayers;

        private readonly Random _random;

        private List<Matrix<double>> _biases;

        private List<Matrix<double>> _weights;

        private List<Matrix<double>> _previousWeights;

        private readonly IActivationFunction _activationFunction;

        private readonly MatrixBuilder<double> _matrixBuilder;

        private int _bestEfficiency = int.MinValue;

        private string _folderForResult = @"C:\Users\Gajowa Lion\Documents\Piotrek\Polibuda\Sieci Neuronowe\Results\";

        private string _folderForNetwork = @"C:\Users\Gajowa Lion\Documents\Piotrek\Polibuda\Sieci Neuronowe\Networks\";

        private readonly TemporaryNetworkBuilder _builder;

        private readonly NetworkUpdater _updater;

        private readonly bool _dropoutOn;

        private readonly ICostFunction _costFunction = new QuadraticCostFunction();

        private readonly MatrixHelpers _helpers = new MatrixHelpers();

        public Network(int[] sizes, IActivationFunction activationFunction, bool dropoutOn)
        {
            _random = new Random();
            _matrixBuilder = Matrix<double>.Build;
            _sizes = sizes;
            _numLayers = sizes.Length;
            InitializeBiases();
            InitializeWeights();
            _activationFunction = activationFunction;
            _previousWeights = _weights.ConvertAll(w => w.Clone());

            _builder = new TemporaryNetworkBuilder(new DropoutAdapter(), 
                _random, new DropoutListBuilderFactory().Create(dropoutOn, 
                _random));

            _updater = new NetworkUpdater();
            _dropoutOn = dropoutOn;
        }

        private void InitializeWeights()
        {
            _weights = new List<Matrix<double>>();

            for (var i = 1; i < _numLayers; i++)
            {
                var numNerons = _sizes[i];
                var numWeights = _sizes[i - 1];

                var wiegthsInIthLayer = Matrix<double>.Build.Dense(numNerons, numWeights,
                    (r, c) => _random.NextDouble()*2 - 1);

                _weights.Add(wiegthsInIthLayer);
            }
        }


        private void InitializeBiases()
        {
            _biases = new List<Matrix<double>>();

            for (var i = 1; i < _numLayers; i++)
            {
                var bias = Matrix<double>.Build.Dense(_sizes[i], 1, (r,c) => _random.NextDouble() * 2 - 1);

                _biases.Add(bias);
            }
        }

        public Matrix<double> FeedForward(Matrix<double> a)
        {
            for (var i = 0; i < _biases.Count; i++)
            {
               a =  _weights[i].Multiply(a).Add(_biases[i]).Map(_activationFunction.Function, Zeros.Include);
            }

            return a; 
        }

        public void Sgd(List<DigitImageForStuding>  trainingData, Parametrs param, List<DigitImageForEvaluating> testData)
        {
            var fileDiscryption = FileDescryption(param);
            var fileNameResult = "Result with "+fileDiscryption+".txt";
            var fileNameNetwork = "Network with " + fileDiscryption + ".bin"; 

            var nTest = 0;
            if (testData != null)
            {
                nTest = testData.Count;
            }

            using (var file = new StreamWriter(_folderForResult+fileNameResult))
            {


                for (var i = 0; i < param.Epochs; i++)
                {
                    trainingData.Shuffle();
                    List<List<DigitImageForStuding>> miniBatches = MakeMiniBatches(trainingData, param.MiniBatchSize);

                   

                   // miniBatches.ForEach(x => UpdateMiniBatch(x, param.Eta, param.Beta));

                    foreach (List<DigitImageForStuding> miniBatch in miniBatches)
                    {
                        var tempPrevoiusWeights = _weights.ConvertAll(m => m.Clone()); //For momentum

                        var tempNet = _builder.Build(this);

                        UpdateMiniBatch(tempNet, miniBatch, param);

                        _updater.Update(this, tempNet);
                        
                        _previousWeights = tempPrevoiusWeights; //For momentum
                    }

                    if (testData != null)
                    {
                        var currentEfficiency = Evaluate(testData);

                        if (_bestEfficiency < currentEfficiency)
                        {
                            SaveToBinary(_folderForNetwork+fileNameNetwork);
                            _bestEfficiency = currentEfficiency;
                        }

                        if (_dropoutOn)
                        {
                            _weights = _weights.Select(w => w * 0.5).ToList();
                            _biases = _biases.Select(b => b*0.5).ToList();
                        }

                        var agregateCostValidationData = AgregateCost(testData.Select(x => x.Pixels).ToList(),
                            testData.Select(x => _helpers.VectorizedResult(x.Label)).ToList());
                        var agregateCostStudyData = AgregateCost(trainingData.Select(x => x.Pixels).ToList(), 
                        trainingData.Select(x => x.Label).ToList());

                        Console.WriteLine("Epoch {0}: {1} / {2}", i, currentEfficiency, nTest);
                        Console.WriteLine("Cost on training set: {0}", agregateCostStudyData);
                        Console.WriteLine("Cost on validation set {0}", agregateCostValidationData);

                        file.WriteLine("Epoch {0}: {1} / {2}", i, currentEfficiency, nTest);
                        file.WriteLine("Cost on training set: {0}", agregateCostStudyData);
                        file.WriteLine("Cost on validation set {0}", agregateCostValidationData);

                        if (_dropoutOn)
                        {
                            _weights = _weights.Select(w => w * 2).ToList();
                            _biases = _biases.Select(b => b * 2).ToList();
                        }
                    }
                    else
                    {
                        Console.WriteLine("Epoch {0} complete", i);
                        file.WriteLine("Epoch {0} complete", i);
                    }
                }
                 
            }
        }

        public int Evaluate(List<DigitImageForEvaluating> data)
        {
           return  data.Select(sample => FeedForward(sample.Pixels).Column(0).MaximumIndex() == sample.Label).Sum(b => b ? 1: 0);
        }

        public void SaveToFile()
        {
            Console.WriteLine(typeof(Matrix<double>).IsSerializable);

            DelimitedWriter.Write("data.csv", _weights.First(), ",");
        }

        public void SaveToBinary(string fileName)
        {
            using (Stream stream = File.Open(fileName, FileMode.Create))
            {
                var bformatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();

                var weightAndBias = new List<List<Matrix<double>>> { _weights, _biases };

                bformatter.Serialize(stream, weightAndBias);
            }
        }


    

        public Matrix<double> GetWeightsBeetwenInputAndHiddenLayer()
        {
            return _weights[0];
        }

        public Matrix<double> GetWeightsBeetwenHiddenAndOutputLayer()
        {
            return _weights[1];
        }

        public Matrix<double> GetBiasOnHiddenLayer()
        {
            return _biases[0];
        }

        public Matrix<double> GetBiasOnOutputLayer()
        {
            return _biases[1];
        }

        public int HiddenLayerSize()
        {
            return _sizes[1];
        }

        public void SetWeightsForNeuronInHiddenLayer(int neuronLabel, Vector<double> weights)
        {
            _weights[0].SetRow(neuronLabel, weights);
        }

        public void SetWeightsForInputsFromHiddenLayer(int neronLabel, Vector<double> weights)
        {
            _weights[1].SetColumn(neronLabel, weights);
        }

        public void SetBiasOnHiddenLayer(int neuronLabel, Vector<double> bias)
        {
            _biases[0].Row(neuronLabel, bias);
        }

        public void SetBiasesOnOutputLayer(Matrix<double> biases)
        {
            _biases[1] = biases;
        }

        public void SetWeights(List<Matrix<double>> weights)
        {
            _weights = weights;
        }

        public void SetBiases(List<Matrix<double>> biases)
        {
            _biases = biases;
        }

        private double AgregateCost(List<Matrix<double>> inputs, List<Matrix<double>> expctedOutputs)
        {
            var actualOutput = inputs.Select(FeedForward).ToList();

            var outputsCount = actualOutput.Count();

            var agregateCost = 0.0;

            for (var i = 0; i < outputsCount; i++)
            {
                agregateCost += _costFunction.CountCost(expctedOutputs[i], actualOutput[i]);
            }

            return agregateCost;
        }

        private void UpdateMiniBatch(TemporaryNetwork tempNet,List<DigitImageForStuding> batch, Parametrs param)
        {
            var nablaB = CreateEmptyContenerForBiases(tempNet);

            var nablaW = CreateEmptyContenerForWeights(tempNet);

            foreach (var singleBatch in batch)
            {
                
                var delta = Backprop(tempNet, singleBatch);
                for (var i = 0; i < nablaB.Count; i++)
                {
                    nablaB[i] = nablaB[i].Add(delta.DeltaBiases[i]);
                }

                for (var i = 0; i < nablaW.Count; i++)
                {
                    nablaW[i] = nablaW[i].Add(delta.DeltaWeights[i]);
                }
            }

            for (var i = 0; i < _weights.Count; i++)
            {
                var layerNumber = i + 2;
                //var first = _weights[i].Subtract((eta/batch.Count)*nablaW[i]); dodano L2
                var first = tempNet.GetWeightsForNLayer(layerNumber).Multiply(1-param.Eta*(param.Lambda/param.NumberOfNeuronsInHiddenLayer)).Subtract((param.Eta/batch.Count)*nablaW[i]);


                //var momentum = param.Beta*(_weights[i].Subtract(_previousWeights[i])); Momentum na razie idzie do kosza

                tempNet.SetWeightsForNLayer(layerNumber, first);
            }

            for (var i = 0; i < _biases.Count; i++)
            {
                var layerNumber = i + 2;

                var newBias = tempNet.GetBiasOnNLayer(layerNumber).Subtract((param.Eta/batch.Count)*nablaB[i]);
                tempNet.SetBiasOnNLayer(layerNumber, newBias);
               }
        }

        private Delta Backprop(TemporaryNetwork tempNet, DigitImageForStuding singleBatch)
        {
            var nablaB = CreateEmptyContenerForBiases(tempNet);
            var nablaW = CreateEmptyContenerForWeights(tempNet);

            var activision = singleBatch.Pixels;
            var activisions = new List<Matrix<double>> {activision};

            //list to store all z vectors, layer by layer 
            var zs = new List<Matrix<double>>();

            for (var i = 0; i < 2; i++)
            {
                var layerNumber = i + 2;
                var z = tempNet.GetWeightsForNLayer(layerNumber).Multiply(activision) + tempNet.GetBiasOnNLayer(layerNumber);
                zs.Add(z);
                activision = z.Map(_activationFunction.Function);
                activisions.Add(activision);
            }

            var part1 = CostDerivate(activisions[activisions.Count - 1], singleBatch.Label);
            var part2 = zs[zs.Count - 1].Map(_activationFunction.Calcucus);

            var delta =  part1.PointwiseMultiply(part2);

            nablaB[nablaB.Count - 1] = delta;
            nablaW[nablaW.Count - 1] = delta.Multiply(activisions[activisions.Count - 2].Transpose());

            for (var i = 2; i < _numLayers; i++)
            {
                var z = zs[zs.Count - i];
                var sp = z.Map(_activationFunction.Calcucus);
                var layerLabel = _weights.Count - i + 1 + 2;
                delta =tempNet.GetWeightsForNLayer(layerLabel).Transpose().Multiply(delta).PointwiseMultiply(sp);
                nablaB[nablaB.Count - i] = delta;

                nablaW[nablaW.Count - i] = delta.Multiply(activisions[activisions.Count - i - 1].Transpose());
            }

            return new Delta(nablaW, nablaB);
        }

        private Matrix<double> CostDerivate(Matrix<double> outputActivisions, Matrix<double> y)
        {
            return outputActivisions.Subtract(y);
        }

        private string FileDescryption(Parametrs param)
        {
            return
                $"epochs={param.Epochs} miniBatch={param.MiniBatchSize} eta={param.Eta} hiddenLayer={param.NumberOfNeuronsInHiddenLayer} beta={param.Beta} lambda={param.Lambda} n={param.DataSize} dropoutOn={param.DropoutOn}";
        }


        private List<Matrix<double>> CreateEmptyContenerForWeights(TemporaryNetwork tempNet)
        {
            return new List<Matrix<double>>
            {
                _matrixBuilder.Dense(tempNet.ActiveNeuronInHiddenLayerCount(), tempNet.InputNeuronsCount()),
                _matrixBuilder.Dense(tempNet.OutputLayerCount(), tempNet.ActiveNeuronInHiddenLayerCount())
            };
        }

        private List<Matrix<double>> CreateEmptyContenerForBiases(TemporaryNetwork tempNet)
        {
            return new List<Matrix<double>>
            {
                _matrixBuilder.Dense(tempNet.ActiveNeuronInHiddenLayerCount(), 1),
                _matrixBuilder.Dense(tempNet.OutputLayerCount(), 1)
            };
        }

        private List<List<DigitImageForStuding>> MakeMiniBatches(List<DigitImageForStuding> trainingData, int miniBatchSize)
        {
            var batches = new List<List<DigitImageForStuding>>();

            int tailSize = trainingData.Count % miniBatchSize;
            int maxLenghtForFullMiniBatch = trainingData.Count - (tailSize);

            for (int i = 0; i < maxLenghtForFullMiniBatch; i += miniBatchSize)
            {
                batches.Add(trainingData.GetRange(i, miniBatchSize));
            }
            if (tailSize != 0)
            {
                batches.Add(trainingData.GetRange(maxLenghtForFullMiniBatch, tailSize));
            }

            return batches;
        }
    }
}
