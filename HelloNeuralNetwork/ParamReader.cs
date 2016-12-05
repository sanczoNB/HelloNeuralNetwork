using System;
using System.Collections.Generic;

namespace HelloNeuralNetwork
{
    public class ParamReader
    {
        private string _filePath = @"C:\Users\Gajowa Lion\Documents\Piotrek\Polibuda\Sieci Neuronowe\Param\params.txt";

        private readonly ParamBuilder _builder = new ParamBuilder();
        public List<Parametrs> Read()
        {
            var result = new List<Parametrs>();

            string[] lines = System.IO.File.ReadAllLines(_filePath);

            foreach (var line in lines)
            {
                var dictionary = IntoDictionary(line);
                result.Add(_builder.Build(dictionary));
            }
            return result;
        }

        private Dictionary<string, string> IntoDictionary(string line)
        {
            var dictionary = new Dictionary<string, string>();

            var elms = line.Split(' ');
            foreach (var elm in elms)
            {
                var split = elm.Split('=');
                var key = split[0];
                if (split.Length > 1)
                {
                    var value = split[1];
                    dictionary.Add(key, value);
                }
                else
                {
                    dictionary.Add(key, "true");
                }

               
            }

            return dictionary;
        }
    }
}
