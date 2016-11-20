using System;
using System.Collections.Generic;
using System.Linq;
using HelloNeuralNetwork.Models;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork
{
    class Network
    {

        private readonly int[] _sizes;

        private readonly int _numLayers;

        private readonly Random _random;

        private List<Matrix<double>> _biases;

        private List<Matrix<double>> _weights;

        private readonly IActivationFunction _activationFunction;

        private readonly MatrixBuilder<double> _matrixBuilder;

        public Network(int[] sizes, IActivationFunction activationFunction)
        {
            _random = new Random();
            _matrixBuilder = Matrix<double>.Build;
            _sizes = sizes;
            _numLayers = sizes.Length;
            InitializeBiases();
            InitializeWeights();
            _activationFunction = activationFunction;
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

            //_weights.ForEach(Console.WriteLine);

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

        public void Sgd(List<DigitImageForStuding>  trainingData, int epochs, int miniBatchSize, double eta, List<DigitImageForEvaluating> testData)
        {
            var nTest = 0;
            if (testData != null)
            {
                nTest = testData.Count;
            }

            for (var i = 0; i < epochs; i++)
            {
                trainingData.Shuffle();
                List<List<DigitImageForStuding>> miniBatches = MakeMiniBatches(trainingData, miniBatchSize);
                miniBatches.ForEach(x => UpdateMiniBatch(x, eta));

                if (testData != null)
                {
                    Console.WriteLine("Epoch {0}: {1} / {2}", i, Evaluate(testData), nTest);
                }
                else
                {
                    Console.WriteLine("Epoch {0} complete", i);
                }
            }
        }

        private int Evaluate(List<DigitImageForEvaluating> data)
        {
           return  data.Select(sample => FeedForward(sample.Pixels).Column(0).MaximumIndex() == sample.Label).Sum(b => b ? 1: 0);
        }

        private void UpdateMiniBatch(List<DigitImageForStuding> batch, double eta)
        {
            var nablaB = _biases.Select(bias => Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount)).ToList();

            var nablaW =
                _weights.Select(weight => Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount)).ToList();

            foreach (var singleBatch in batch)
            {
                var delta = Backprop(singleBatch);
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
                _weights[i] = _weights[i].Subtract((eta/batch.Count)*nablaW[i]);
            }

            for (var i = 0; i < _biases.Count; i++)
            {
                _biases[i] = _biases[i].Subtract((eta/batch.Count)*nablaB[i]);
            }

        }

        private Delta Backprop(DigitImageForStuding singleBatch)
        {
            var nablaB = _biases.Select(b => _matrixBuilder.Dense(b.RowCount, b.ColumnCount)).ToList();
            var nablaW = _weights.Select(w => _matrixBuilder.Dense(w.RowCount, w.ColumnCount)).ToList();

            var activision = singleBatch.Pixels;
            var activisions = new List<Matrix<double>> {activision};

            //list to store all z vectors, layer by layer 
            var zs = new List<Matrix<Double>>();

            for (var i = 0; i < _weights.Count; i++)
            {
                var z = _weights[i].Multiply(activision) + _biases[i];
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
                delta = _weights[_weights.Count - i + 1].Transpose().Multiply(delta).PointwiseMultiply(sp);
                nablaB[nablaB.Count - i] = delta;
                nablaW[nablaW.Count - i] = delta.Multiply(activisions[activisions.Count - i - 1].Transpose());
            }

            return new Delta(nablaW, nablaB);
        }

        private Matrix<double> CostDerivate(Matrix<double> outputActivisions, Matrix<double> y)
        {
            return outputActivisions.Subtract(y);
        }

        private List<List<DigitImageForStuding>> MakeMiniBatches(List<DigitImageForStuding> trainingData, int miniBatchSize)
        {
            var batches = new List<List<DigitImageForStuding>>();

            int tailSize = trainingData.Count % miniBatchSize;
            int maxLenghtForFullMiniBatch = trainingData.Count - (tailSize);

            for (int i = 0; i < maxLenghtForFullMiniBatch; i+=miniBatchSize)
            {
                batches.Add(trainingData.GetRange(i, miniBatchSize));
            }
            if (tailSize !=0)
            {
                batches.Add(trainingData.GetRange(maxLenghtForFullMiniBatch, tailSize));
            }

            return batches;
        }

    }
}
