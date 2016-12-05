using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.CostFunctions
{
    public interface ICostFunction
    {
        double CountCost(Matrix<double> desire, Matrix<double> actual);
    }
}
