using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.Helpers
{
    public class MatrixHelpers
    {
        public Matrix<double> VectorizedResult(byte label)
        {
            var vectorized = Matrix<double>.Build.Dense(10, 1);
            vectorized[label, 0] = 1;

            return vectorized;
        }
    }
}
