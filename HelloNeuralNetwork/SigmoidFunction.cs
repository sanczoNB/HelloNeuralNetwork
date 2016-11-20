using System;

namespace HelloNeuralNetwork
{
    public class SigmoidFunction : IActivationFunction
    {
        public double Function(double x)
        {
            return 1/(1 + Math.Exp(-x));
        }

        public double Calcucus(double x)
        {
           return Function(x)*(1 - Function(x));
        }
    }
}
