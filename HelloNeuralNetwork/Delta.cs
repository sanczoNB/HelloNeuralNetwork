using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork
{
    public class Delta
    {
        public List<Matrix<double>> DeltaWeights { get; }

        public List<Matrix<double>> DeltaBiases { get; }

        public Delta(List<Matrix<double>> deltaWeights, List<Matrix<double>> deltaBiases)
        {
            DeltaWeights = deltaWeights;
            DeltaBiases = deltaBiases;
        }
    }
}
