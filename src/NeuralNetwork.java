import java.io.*;
import java.util.*;

public class NeuralNetwork {

	private ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	private ArrayList<ArrayList<Neuron>> hiddenLayers = new ArrayList<>();
	private ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	private Neuron biasNeuron = new Neuron();
	private int[] layers;
	private double epsilon = 0.00000000001;
	private double learningRate = 0.9f;
	private double momentum = 0.7f;
	private double trainingInputs[][];
	private double expectedOutputs[][];
	private double resultingOutputs[][];
	private boolean isTheNetworkTrained = false;

	public NeuralNetwork(int numberOfInputNeurons,
						 int[] numberOfHiddenNeurons,
						 int numberOfOutputNeurons,
						 String providedWeightsFileName)
			throws Exception {

		File providedWeightsFile = new File(providedWeightsFileName);
		if (!providedWeightsFile.exists()) {
			FileWriter fileWriter = new FileWriter(providedWeightsFileName, true);
			BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
			for (int outerIndex = 0; outerIndex < numberOfHiddenNeurons.length; outerIndex++) {
				for (int index = 0 ; index < numberOfHiddenNeurons[outerIndex]; index++) {
					double weight = new Random(System.currentTimeMillis()).nextDouble();
					bufferedWriter.write(weight + " ");
				}
				bufferedWriter.write("\n");
			}
			for (int index = 0; index < numberOfOutputNeurons; index++) {
				double weight = new Random(System.currentTimeMillis()).nextDouble();
				bufferedWriter.write(weight + " ");
			}
			bufferedWriter.close();
		}

		FileInputStream inputFileStream = new FileInputStream(providedWeightsFileName);
		DataInputStream inputDataStream = new DataInputStream(inputFileStream);
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputDataStream));

		ArrayList<String> inputData = new ArrayList<String>();
		String inputLine;

		while ((inputLine = bufferedReader.readLine()) != null) {
			inputData.add(inputLine);
		}

		inputDataStream.close();

		// Build the Neural Network
		buildNeuralNetwork(
				numberOfInputNeurons,
				numberOfHiddenNeurons,
				numberOfOutputNeurons
		);

		ArrayList<String[]> hiddenAndOutputLayerWeights = new ArrayList<>();
		for (int lineNumber = 0; lineNumber < inputData.size(); lineNumber++) {
			String[] data = inputData.get(lineNumber).split(" ");
			hiddenAndOutputLayerWeights.add(data);
		}

		// set neural connection weights
		setWeights(false, hiddenAndOutputLayerWeights);

		// reset id counters
		resetIdCounters();

	} // end constructor

	private void buildNeuralNetwork(int numberOfInputNeurons,
									int[] numberOfHiddenNeurons,
									int numberOfOutputNeurons) {

		int numberOfLayers = numberOfHiddenNeurons.length + 2;
		this.layers = new int[numberOfLayers];
		this.layers[0] = numberOfInputNeurons;
		this.layers[numberOfLayers - 1] = numberOfOutputNeurons;
		for (int layerNumber = 0; layerNumber < numberOfHiddenNeurons.length; layerNumber++) {
			this.layers[layerNumber + 1] = numberOfHiddenNeurons[layerNumber];
		}

		for (int layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {

			if (layerIndex == 0) { // input layer

				for (int neuronIndex = 0; neuronIndex < this.layers[layerIndex]; neuronIndex++) {
					Neuron neuron = new Neuron();
					this.inputLayer.add(neuron);
				} // end if

			} else if (layerIndex > 0 && layerIndex < numberOfLayers - 1) { // hidden layer

				ArrayList<Neuron> layer = new ArrayList<>();
				for (int neuronIndex = 0; neuronIndex < this.layers[layerIndex]; neuronIndex++) {
					Neuron neuron = new Neuron();
					if (neuronIndex == 0) {
						neuron.addNeurons(this.inputLayer);
					} else {
						neuron.addNeurons(this.hiddenLayers.get(neuronIndex - 1));
					}
					neuron.addBias(this.biasNeuron);
					layer.add(neuron);
				} // end for
				this.hiddenLayers.add(layer);

			} else if (layerIndex == numberOfLayers - 1) { // output layer

				for (int neuronIndex = 0; neuronIndex < this.layers[layerIndex]; neuronIndex++) {
					Neuron neuron = new Neuron();
					neuron.addNeurons(this.hiddenLayers.get(this.hiddenLayers.size() - 1));
					neuron.addBias(this.biasNeuron);
					this.outputLayer.add(neuron);
				} // end for

			} else {
				System.err
						.println("Error: Neural Network could not be initialized!");
			} // end if

		} // end for

	} // end buildNeuralNetwork()

	private void setWeights(boolean weightsAreRandom,
							ArrayList<String[]> hiddenAndOutputLayerWeights) {

		int weightsIndex = 0;
		for (int hiddenLayerIndex = 0; hiddenLayerIndex < this.hiddenLayers.size(); hiddenLayerIndex++) {
			for (int hiddenLayerNeuronNumber = 0; hiddenLayerNeuronNumber < this.hiddenLayers.get(hiddenLayerIndex).size(); hiddenLayerNeuronNumber++) {
				Neuron neuron = this.hiddenLayers.get(hiddenLayerIndex).get(hiddenLayerNeuronNumber);
				ArrayList<Dendrite> dendrites = neuron.getDendrites();
				setDendriteWeight(
						weightsAreRandom,
						weightsIndex,
						dendrites,
						hiddenAndOutputLayerWeights.get(hiddenLayerNeuronNumber)
				);
			}
		} // end for

		weightsIndex = 0;
		for (int outputLayerIndex = 0; outputLayerIndex < this.outputLayer.size(); outputLayerIndex++) {
			Neuron neuron = this.outputLayer.get(outputLayerIndex);
			ArrayList<Dendrite> dendrites = neuron.getDendrites();
			setDendriteWeight(
					weightsAreRandom,
					weightsIndex,
					dendrites,
					hiddenAndOutputLayerWeights.get(hiddenAndOutputLayerWeights.size() - 1)
			);
		} // end for

	} // end setWeights()

	private void setDendriteWeight(boolean weightsAreRandom,
								  int weightsIndex,
								  ArrayList<Dendrite> dendrites,
								  String[] layerWeights) {

		for (int connectionIndex = 0; connectionIndex < dendrites.size(); connectionIndex++) {
			Dendrite dendrite = dendrites.get(connectionIndex);
			double connectionWeight;
			if (weightsAreRandom) {
				connectionWeight = new Random(System.currentTimeMillis()).nextDouble();
			} else {
				connectionWeight = Double.parseDouble(layerWeights[weightsIndex]);
			}
			dendrite.setWeight(connectionWeight);
			weightsIndex++;
		} // end for

	} // end setDendriteWeight()

	private void resetIdCounters() {
		Neuron.setCounter(0);
		Dendrite.setCounter(0);
	} // end resetIdCounters()

	public void setupTheNetwork(double epsilon,
								double learningRate,
								double momentum,
								String inputTrainingDataFileName,
								String outputTrainingDataFileName) {
		this.epsilon = epsilon;
		this.learningRate = learningRate;
		this.momentum = momentum;
		readInputTrainingData(inputTrainingDataFileName);
		readOutputTrainingData(outputTrainingDataFileName);
		this.resultingOutputs = new double[expectedOutputs.length][expectedOutputs[0].length];
		for (int index = 0; index < expectedOutputs.length; index++) {
			Arrays.fill(this.resultingOutputs[index], 0.0);
		}
	} // end setupNetwork()

	private void readInputTrainingData(String trainingDataFileName) {
		File trainingDataFile = new File(trainingDataFileName);
		try {
			if (!trainingDataFile.exists()) {
				FileWriter fileWriter = new FileWriter(trainingDataFileName, true);
				BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
				bufferedWriter.close();
			}

			FileInputStream inputFileStream = new FileInputStream(trainingDataFileName);
			DataInputStream inputDataStream = new DataInputStream(inputFileStream);
			BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputDataStream));

			String line;
			int index = 0;
			while ((line = bufferedReader.readLine()) != null) {
				String[] data = line.split(" ");
				for (int dataIndex = 0; dataIndex < data.length; dataIndex++) {
					this.trainingInputs[index][dataIndex] = Double.parseDouble(data[dataIndex]);
				}
			}

			inputDataStream.close();
		} catch (Exception exception) {
			System.out.println(exception);
		}
	}

	private void readOutputTrainingData(String trainingDataFileName) {
		File trainingDataFile = new File(trainingDataFileName);
		try {
			if (!trainingDataFile.exists()) {
				FileWriter fileWriter = new FileWriter(trainingDataFileName, true);
				BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
				bufferedWriter.close();
			}

			FileInputStream inputFileStream = new FileInputStream(trainingDataFileName);
			DataInputStream inputDataStream = new DataInputStream(inputFileStream);
			BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputDataStream));

			String line;
			int index = 0;
			while ((line = bufferedReader.readLine()) != null) {
				String[] data = line.split(" ");
				for (int dataIndex = 0; dataIndex < data.length; dataIndex++) {
					this.expectedOutputs[index][dataIndex] = Double.parseDouble(data[dataIndex]);
				}
			}

			inputDataStream.close();
		} catch (Exception exception) {
			System.out.println(exception);
		}
	}

	/**
	 * Input percept data for thinking
	 * 
	 * @param trainingInputs
	 */
	private void setInputLayerNeurons(double trainingInputs[]) {
		for (int index = 0; index < this.inputLayer.size(); index++) {
			this.inputLayer.get(index).setOutput(trainingInputs[index]);
		}
	} // end setInput()

	/**
	 * Get output learned from neural network thought
	 * 
	 * @return
	 */
	public double[] getOutputs() {
		double[] outputs = new double[this.outputLayer.size()];
		for (int index = 0; index < this.outputLayer.size(); index++) {
			outputs[index] = this.outputLayer.get(index).getOutput();
		}
		return outputs;
	} // end getOutputs()

	/**
	 * Run the Neural Network
	 *
	 * @return the output from the neural network
	 */
	public double[] runTheNetwork(double[] inputs) {
		this.setInputLayerNeurons(inputs);
		this.feedForward();
		double[] outputs = new double[outputLayer.size()];
		int outputIndex = 0;
		Iterator<Neuron> iterator = outputLayer.iterator();
		while (iterator.hasNext()) {
			Neuron neuron = iterator.next();
			outputs[outputIndex] = neuron.getOutput();
			outputIndex++;
		}
		return outputs;
	}

	/**
	 * Feed Forward
	 */
	public void feedForward() {

		for (int outerIndex = 0; outerIndex < this.hiddenLayers.size(); outerIndex++) {
			for (int index = 0; index < this.hiddenLayers.get(outerIndex).size(); index++) {
				Neuron neuron = this.hiddenLayers.get(outerIndex).get(index);
				neuron.computeOutput();
			} // end for
		}

		for (int index = 0; index < this.outputLayer.size(); index++) {
			Neuron neuron = this.outputLayer.get(index);
			neuron.computeOutput();
		} // end for

	} // end feedForward()

	/**
	 * Back Propagation
	 * 
	 * @param expectedOutput
	 *            first calculate the partial derivative of the error with
	 *            respect to each of the weight leading into the output neurons
	 *            bias is also updated here
	 */
	public void backPropagate(double expectedOutput[]) {

		// error check, normalize value [0,1]
		for (int expectedOutputIndex = 0; expectedOutputIndex < expectedOutput.length; expectedOutputIndex++) {
			double expected = expectedOutput[expectedOutputIndex];
			if ((expected < 0) || (expected > 1)) {
				if (expected < 0) {
					expectedOutput[expectedOutputIndex] = this.epsilon;
				} else {
					expectedOutput[expectedOutputIndex] = 1.0 - this.epsilon;
				}
			} // end if
		} // end for

		int index = 0;

		// update weights for the output layer
		for (int outputIndex = 0; outputIndex < this.outputLayer.size(); outputIndex++) {

			Neuron outputNeuron = this.outputLayer.get(outputIndex);
			ArrayList<Dendrite> dendrites = outputNeuron.getDendrites();

			for (int dendriteIndex = 0; dendriteIndex < dendrites.size(); dendriteIndex++) {
				Dendrite dendrite = dendrites.get(dendriteIndex);
				double output = outputNeuron.getOutput();
				double hiddenNeuronActivation = dendrite.getFromNeuron().getOutput();
				double desiredOutput = expectedOutput[index];
				double partialDerivative = -output * (1.0 - output) * hiddenNeuronActivation
						* (desiredOutput - output);
				double deltaWeight = -this.learningRate * partialDerivative;
				double adjustedWeight = dendrite.getWeight() + deltaWeight;
				dendrite.setDeltaWeight(deltaWeight);
				dendrite.setWeight(adjustedWeight + this.momentum
						* dendrite.getPreviousDeltaWeight());
			} // end for

			index++;

		} // end for

		// update weights for the hidden layer
		for (int outerHiddenIndex = this.hiddenLayers.size() - 1; outerHiddenIndex >= 0; outerHiddenIndex--) {

			for (int hiddenIndex = 0; hiddenIndex < this.hiddenLayers.get(outerHiddenIndex).size(); hiddenIndex++) {

				Neuron hiddenNeuron = this.hiddenLayers.get(outerHiddenIndex).get(hiddenIndex);
				ArrayList<Dendrite> dendrites = hiddenNeuron.getDendrites();

				for (int dendriteIndex = 0; dendriteIndex < dendrites.size(); dendriteIndex++) {

					Dendrite dendrite = dendrites.get(dendriteIndex);
					double hiddenNeuronOutput = hiddenNeuron.getOutput();
					double inputNeuronOutput = dendrite.getFromNeuron().getOutput();
					double computedSumOfOutputs = 0;

					for (int outputIndex = 0; outputIndex < this.outputLayer.size(); outputIndex++) {
						Neuron outputNeuron = this.outputLayer.get(outputIndex);
						double hiddenNeuronWeight = outputNeuron.getDendrite(hiddenNeuron.id).getWeight();
						double desiredOutput = expectedOutput[outputIndex];
						double actualOutput = outputNeuron.getOutput();
						computedSumOfOutputs += -(desiredOutput - actualOutput) * actualOutput * (1 - actualOutput) * hiddenNeuronWeight;
					} // end for

					double partialDerivative = hiddenNeuronOutput * (1 - hiddenNeuronOutput) * inputNeuronOutput * computedSumOfOutputs;
					double deltaWeight = -this.learningRate * partialDerivative;
					double adjustedWeight = dendrite.getWeight() + deltaWeight;
					dendrite.setDeltaWeight(deltaWeight);
					dendrite.setWeight(adjustedWeight + this.momentum
							* dendrite.getPreviousDeltaWeight());

				} // end for

			} // end for

		} // end for

	} // end backPropagate()

	public void trainTheNetwork(int maximumSteps, double minimumError, String trainedWeightsOutputFileName)
			throws Exception {

		int steps;
		double error = 1;

		for (steps = 0; (steps < maximumSteps) && (error > minimumError); steps++) {

			error = 0;

			for (int row = 0; row < this.trainingInputs.length; row++) {

				this.setInputLayerNeurons(this.trainingInputs[row]);
				this.feedForward();
				this.resultingOutputs[row] = this.getOutputs();

				for (int column = 0; column < this.expectedOutputs[row].length; column++) {
					double calculatedError = Math.pow(this.resultingOutputs[row][column]
							- this.expectedOutputs[row][column], 2.0);
					error += calculatedError;
				} // end for

				this.backPropagate(this.expectedOutputs[row]);

				this.saveWeights(trainedWeightsOutputFileName);

			} // end for

		} // end for

		this.isTheNetworkTrained = true;

		if (steps == maximumSteps) {
			System.err
					.println("Error: Neural Network training procedure has failed!");
		}

	} // end trainTheNetwork()

	@Override
	public String toString() {
		return "[NeuralNetwork \ninputLayer=" + this.inputLayer
				+ ", \nhiddenLayer=" + this.hiddenLayers + ", \noutputLayer="
				+ this.outputLayer + ", \nbiasNeuron=" + this.biasNeuron + ", layers="
				+ Arrays.toString(this.layers) + ", epsilon=" + this.epsilon
				+ ", learningRate=" + this.learningRate + ", momentum="
				+ this.momentum + ", \ninputs="
				+ Arrays.deepToString(this.trainingInputs) + ", \nexpectedOutputs="
				+ Arrays.deepToString(this.expectedOutputs)
				+ ", \nresultingOutputs="
				+ Arrays.deepToString(this.resultingOutputs) + ", \nisTheNetworkTrained="
				+ this.isTheNetworkTrained + "]";
	} // end toString()

	public void saveWeights(String outputFileName) throws Exception {

		ArrayList<ArrayList<Double>> hiddenWeights = new ArrayList<>();
		for (int neuronIndex = 0; neuronIndex < this.hiddenLayers.size(); neuronIndex++) {
			for (int index = 0; index < this.hiddenLayers.get(neuronIndex).size(); index++) {
				Neuron neuron = this.hiddenLayers.get(neuronIndex).get(index);
				ArrayList<Dendrite> dendrites = neuron.getDendrites();
				ArrayList<Double> weights = new ArrayList<>();
				for (int dendriteIndex = 0; dendriteIndex < dendrites.size(); dendriteIndex++) {
					Dendrite dendrite = dendrites.get(dendriteIndex);
					double weight = dendrite.getWeight();
					weights.add(weight);
				} // end for
				hiddenWeights.add(weights);
			}
		} // end for

		ArrayList<Double> outputWeights = new ArrayList<Double>();
		for (int neuronIndex = 0; neuronIndex < this.outputLayer.size(); neuronIndex++) {
			Neuron neuron = this.outputLayer.get(neuronIndex);
			ArrayList<Dendrite> dendrites = neuron.getDendrites();
			for (int dendriteIndex = 0; dendriteIndex < dendrites.size(); dendriteIndex++) {
				Dendrite dendrite = dendrites.get(dendriteIndex);
				double weight = dendrite.getWeight();
				outputWeights.add(weight);
			} // end for
		} // end for

		String data = "";

		for (int outerIndex = 0; outerIndex < hiddenWeights.size(); outerIndex++) {
			for (int index = 0; index < hiddenWeights.get(outerIndex).size(); index++) {
				double weight = hiddenWeights.get(outerIndex).get(index);
				data += weight + " ";
			}
			data += "\n";
		} // end for

		for (int index = 0; index < outputWeights.size(); index++) {
			double weight = outputWeights.get(index);
			data += weight + " ";
		} // end for

		FileWriter fileWriter = new FileWriter(outputFileName);
		BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
		bufferedWriter.write(data + "\n");
		bufferedWriter.close();

	} // end saveWeights()

} // end class