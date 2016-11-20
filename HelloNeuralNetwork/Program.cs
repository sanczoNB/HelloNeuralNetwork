using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Matrix<Double> m1 = Matrix<Double>.Build.DiagonalIdentity(4,4);

            double[,] x = {{ 1.0 },{ 3.0},{2.0} };

            double[,] y = { { 3.0} , { 1.5} , { 7.0 } };

            m1 = Matrix<double>.Build.DenseOfArray(x);

            var m2 = Matrix<double>.Build.DenseOfArray(y);

            var hardman = m1.PointwiseMultiply(m2);

            var m = Matrix<double>.Build.DenseOfArray(x);

            var a = new List<int> {2,3,4,5};

            var sub = a.GetRange(2,2);

            m = 2 * m;

            var simulator = new Simulator();

            simulator.SimpleLearnig();
        }
    }
}
