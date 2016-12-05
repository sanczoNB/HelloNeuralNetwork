using System;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.CostFunctions
{
    class QuadraticCostFunction : ICostFunction
    {
        public double CountCost(Matrix<double> desire, Matrix<double> actual)
        {
            var distinction = desire.Subtract(actual);
            var quadratic = distinction.Map(x => Math.Pow(x, 2));
            var ones = Matrix<double>.Build.DenseIdentity(1, 3);

            return (ones.Multiply(0.5 * quadratic)).At(0, 0);
        }
    }
}
