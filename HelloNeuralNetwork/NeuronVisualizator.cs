using System;
using System.Drawing;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork
{
    public class NeuronVisualizator
    {
        public Color[,] Visualize(Vector<double> weights)
        {
            var minWeights = weights.Min();
            var maxWeights = weights.Max();


           var normalizedWeights = weights.ToArray().Select(x => Normalized(x, minWeights, maxWeights)).ToList();

            var weightsLenghth = Math.Sqrt(normalizedWeights.Sum(x => Math.Pow(x, 2)));

            var scaleWeights = normalizedWeights.Select(x => x/weightsLenghth).ToList();

            var result = new Color[28,28];

            for (var i = 0; i < scaleWeights.Count(); i++)
            {
                var row = i%28;
                var column = i/28;
                var shadeOfGray = (int)((1 - normalizedWeights[i]) * 255);
                result[row,column] = Color.FromArgb(shadeOfGray, shadeOfGray, shadeOfGray) ;
            }
            return result;
        }

        private double Normalized(double d, double minValue, double maxValue)
        {
            return (d-minValue)/(maxValue-minValue);
        }
    }
}
