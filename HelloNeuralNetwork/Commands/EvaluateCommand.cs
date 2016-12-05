using System;
using System.Configuration;
using System.Linq;
using HelloNeuralNetwork.Helpers;
using ManyConsole;
using NDesk.Options;

namespace HelloNeuralNetwork.Commands
{
    public class EvaluateCommand : ConsoleCommand
    {

        private string _fileName;
        
        private readonly StringHelper _stringHelper = new StringHelper();

        public EvaluateCommand()
        {
           Options = new OptionSet
            {
                {"n|name=", "Specify the file name", s => _fileName = s },
            };

            IsCommand("evaluate", "Evaluate neural netowrk save in binary file");
        }

        public override int Run(string[] remainingArguments)
        {
            var firstCut = _stringHelper.CleanFileNameFromExtension(_fileName);

            var split = firstCut.Split(' ');

            var stringContainHiddenLayer = split.First(x => x.Contains("hiddenLayer="));

            var startIndex = stringContainHiddenLayer.IndexOf('=') + 1;

            var hiddenLayer = int.Parse(stringContainHiddenLayer.Substring(startIndex));

            var directoryPath = ConfigurationManager.AppSettings["DirectoryPathForBinaryResults"];

            var accuracy =  new Simulator().Evaluate(directoryPath+_fileName, hiddenLayer);

            Console.WriteLine("Skutecznosc sieci pochodzacej z pliku \"{0}\" wynosi {1}/{2}", _fileName, accuracy, 10000);

            Console.ReadKey();

            return 0;
        }
    }
}
