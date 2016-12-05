using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.Dropout
{
    public class DropoutAdapter
    {
        private readonly MatrixBuilder<double> _builder = Matrix<double>.Build;

        public Matrix<double> CreateNewWeightsMatrixBeetwenInputAndHiddenLayers(Matrix<double> beforeDropout, List<int> activeNeurons)
        {
            

            var afterDropout = Matrix<double>.Build.Dense(activeNeurons.Count, beforeDropout.ColumnCount);

            for (var i = 0; i < activeNeurons.Count; i++)
            {
                afterDropout.SetRow(i, beforeDropout.Row(activeNeurons[i]));
            }

            return afterDropout;
        }

        public Matrix<double> CreateNewBiasOnHiddenLayer(Matrix<double> beforeDropout, List<int> activeNeurons)
        {
            var afterDropout = _builder.Dense(activeNeurons.Count, 1);
            for (var i = 0; i < activeNeurons.Count; i++)
            {
                afterDropout.SetRow(i, beforeDropout.Row(i));
            }

            return afterDropout;
        }

        public Matrix<double> CreateNewWeightsMatrixBeetwenHiddenAndOutputLayer(Matrix<double> beforeDropout, List<int> activeNeurons)
        {
            var afterDropout = Matrix<double>.Build.Dense(beforeDropout.RowCount, activeNeurons.Count);

            for (var i = 0; i < activeNeurons.Count; i++)
            {
                afterDropout.SetColumn(i,beforeDropout.Column(activeNeurons[i]));
            }

            return afterDropout;
        }

        public void UpdateMainWeightsBaseOnDropoutWeightsBeetwenInputAndHiddenLayer(Matrix<double> mainWeights, Matrix<double>  dropoutWeights, List<int> activeNeurons)
        {
            for (var i = 0; i < activeNeurons.Count; i++)
            {
                mainWeights.SetRow(activeNeurons[i], dropoutWeights.Row(i));
            }
        }

        public void UpdateMainWeightsBaseOnDropoutWeightsBeetwenHiddenAndOutputLayer(Matrix<double> mainWeights,
            Matrix<double> droputWeights, List<int> activeNeurons)
        {
            for (var i = 0; i < activeNeurons.Count; i++)
            {
                mainWeights.SetColumn(activeNeurons[i], droputWeights.Column(i));
            }
        }
            
    }
}
