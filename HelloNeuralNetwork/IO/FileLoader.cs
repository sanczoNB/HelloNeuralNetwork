using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork
{
    class FileLoader
    {

        public List<List<Matrix<double>>> LoadFromBinary(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var bformatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();

                var weightAndBias = (List<List<Matrix<double>>>)bformatter.Deserialize(stream);

                return weightAndBias;
            }
        }

    }
}
