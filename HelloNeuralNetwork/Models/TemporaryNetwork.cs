using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.Models
{
    public class TemporaryNetwork
    {

        private readonly List<int> _activeNeurons;

        private readonly List<Matrix<double>> _weights;

        private readonly List<Matrix<double>> _biases;

        public TemporaryNetwork(List<int> activeNeurons)
        {
            _activeNeurons = activeNeurons;
            _weights = Enumerable.Repeat((Matrix<double>)null, 2).ToList();

            _biases = Enumerable.Repeat((Matrix<double>) null, 2).ToList();
        }

        public void SetWeightsBeetwenInputAndHiddenLayer(Matrix<double> weights)
        {
            _weights[0] = weights;
        }

        public void SetWeightsBeetwenHiddenAndOutputLayer(Matrix<double> weights)
        {
            _weights[1] = weights;
        }

        public void SetBiasOnHiddenLayer(Matrix<double> bias)
        {
            _biases[0] = bias;
        }

        public void SetBiasOnOutputLayer(Matrix<double> bias)
        {
            _biases[1] = bias;
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

        public int ActiveNeuronInHiddenLayerCount()
        {
            return _activeNeurons.Count;
        }

        public int GetActiveNeuronLabel(int index)
        {
            return _activeNeurons[index];
        }

        public Vector<double> GetWeightsForNeuronFromHiddenLayer(int index)
        {
            return _weights[0].Row(index);
        }

        public Vector<double> GetBiasForNeuronFromHiddenLayer(int index)
        {
            return _biases[0].Row(index);
        }

        public int InputNeuronsCount()
        {
            return _weights[0].ColumnCount;
        }

        public int OutputLayerCount()
        {
            return _weights[1].RowCount;
        }

        public Matrix<double> GetBiasOnNLayer(int layerNumber)
        {
            return _biases[layerNumber - 2];
        }

        public void SetBiasOnNLayer(int layerNumber, Matrix<double> bias)
        {
            _biases[layerNumber - 2] = bias;
        }

        public Matrix<double> GetWeightsForNLayer(int layerNumber)
        {
            return _weights[layerNumber - 2];
        }

        public void SetWeightsForNLayer(int layerNumber, Matrix<double> weight)
        {
            _weights[layerNumber - 2] = weight;
        }

        public Vector<double> GetWeightsForInputFromHiddenLayer(int index)
        {
            return _weights[1].Column(index);
        }
    }
}
