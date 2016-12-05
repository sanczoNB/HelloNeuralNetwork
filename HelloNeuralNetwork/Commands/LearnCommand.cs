using System;
using HelloNeuralNetwork.Dropout;
using ManyConsole;
using NDesk.Options;

namespace HelloNeuralNetwork.Commands
{
    public class LearnCommand : ConsoleCommand
    {

        public LearnCommand()
        {
            IsCommand("learn", "Learn network and save results" );
        }

        public override int Run(string[] remainingArguments)
        {

            var simulator = new Simulator();

            simulator.LearningForDiffrentSetsOfParam();

            Console.ReadKey();

            return 0;
        }
    }
}
