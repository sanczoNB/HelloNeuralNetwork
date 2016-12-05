using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.IO;
using HelloNeuralNetwork.Helpers;
using HelloNeuralNetwork.IO;
using ManyConsole;
using NDesk.Options;

namespace HelloNeuralNetwork.Commands
{
    public class VisualizeCommad : ConsoleCommand
    {
        private string _fileName;

        private readonly FileLoader _loader = new FileLoader();

        private readonly StringHelper _stringHelper = new StringHelper();
        
        private readonly NeuronVisualizator _visualizator = new NeuronVisualizator();

        private readonly FileSaver _fileSaver = new FileSaver();

        public VisualizeCommad()
        {

            Options = new OptionSet
            {
                {"f|fileName=", "Specify the file name", s => _fileName = s },
            };

            IsCommand("visualize", "Visualize weights in hidden layer");
        }

        public override int Run(string[] remainingArguments)
        {
            var directoryPath = ConfigurationManager.AppSettings["DirectoryPathForBinaryResults"];
            var weightsAndBiases = _loader.LoadFromBinary(directoryPath+_fileName);
            var weights = weightsAndBiases[0];
            var weightsForHiddenLayer = weights[0];

            var directoryBase = ConfigurationManager.AppSettings["DirectoryOutputForVisualization"];

            var cleanFileName = _stringHelper.CleanFileNameFromExtension(_fileName);

            Console.WriteLine("Wykonuje wizualizacje dla sieci pochodzącej z pliku {0}", cleanFileName);

            var filePath = directoryBase + cleanFileName + ".jpg";

            var images = new List<Color[,]>();

            for (var i = 0; i < weightsForHiddenLayer.RowCount; i++)
            {
                images.Add(_visualizator.Visualize(weightsForHiddenLayer.Row(i)));
            }

            var im = _fileSaver.Merge(images);

            im.Save(filePath);
            im.Dispose();

            return 0;
        }
    }
}
