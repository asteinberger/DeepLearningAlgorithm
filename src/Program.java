import java.util.Arrays;
import java.util.Random;

public class Program {
    public static void main(String[] args) throws Exception {

        final int numberOfInputNeurons = 25;
        final int[] numberOfHiddenNeurons = { 20, 15, 10 };
        final int numberOfOutputNeurons = 5;
        final String outputWeightsFileName = "weights.txt";
        final int maximumSteps = 10;
        final float minimumError = 0.9f;
        final double epsilon = 0.00000000001;
        final double learningRate = 0.9f;
        final double momentum = 0.7f;
        final String inputTrainingDataFileName = "inputTrainingData.txt";
        final String outputTrainingDataFileName = "outputTrainingData.txt";

        double[] testInputs = new double[numberOfInputNeurons];
        for (int index = 0; index < numberOfInputNeurons; index++) {
            testInputs[index] = new Random(System.currentTimeMillis()).nextDouble();
        }

        System.out.println("Input:");
        System.out.println(Arrays.toString(testInputs));

        NeuralNetwork neuralNetwork = new NeuralNetwork(
                numberOfInputNeurons,
                numberOfHiddenNeurons,
                numberOfOutputNeurons,
                outputWeightsFileName
        );

        neuralNetwork.setupTheNetwork(
                epsilon,
                learningRate,
                momentum,
                inputTrainingDataFileName,
                outputTrainingDataFileName
        );

        neuralNetwork.trainTheNetwork(maximumSteps, minimumError, outputWeightsFileName);

        neuralNetwork.runTheNetwork(testInputs);

        System.out.println("Output:");
        System.out.println(Arrays.toString(neuralNetwork.getOutputs()));
    }
}
