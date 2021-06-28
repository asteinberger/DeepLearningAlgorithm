import java.util.*;

public class Neuron {

	private Dendrite biasDendrite;
	private final double bias = -1;
	private double output;
	private ArrayList<Dendrite> dendrites = new ArrayList<Dendrite>();
	private HashMap<Integer, Dendrite> dendriteTable = new HashMap<Integer, Dendrite>();
	private static int counter = 0;
	public final int id; // auto increment, starts at 0

	public Neuron() {
		this.id = Neuron.counter;
		Neuron.counter++;
	} // end constructor

	public void computeOutput() {
		double sumOfWeightedInputActivations = 0;
		for (int index = 0; index < this.dendrites.size(); index++) {
			Dendrite neuralConnection = this.dendrites.get(index);
			Neuron connectingNeuron = neuralConnection.getFromNeuron();
			double weightFromConnectingNeuron = neuralConnection.getWeight();
			double activationFromConnectingNeuron = connectingNeuron.getOutput(); // output from previous layer
			sumOfWeightedInputActivations += weightFromConnectingNeuron * activationFromConnectingNeuron;
		} // end for
		sumOfWeightedInputActivations += this.biasDendrite.getWeight() * this.bias;
		this.output = this.sigmoid(sumOfWeightedInputActivations);
	} // end computeOutput()

	private double sigmoid(double input) {
		return 1.0 / (1.0 + (Math.exp(-input)));
	} // end sigmoid()

	public void addNeurons(ArrayList<Neuron> neurons) {
		for (int index = 0; index < neurons.size(); index++) {
			Neuron neuron = neurons.get(index);
			Dendrite dendrite = new Dendrite(neuron, this);
			this.dendrites.add(dendrite);
			this.dendriteTable.put(neuron.id, dendrite);
		} // end for
	} // end addNeurons()

	@Override
	public String toString() {
		return "[Neuron biasDendrite=" + this.biasDendrite + ", bias="
				+ this.bias + ", output=" + this.output + ", dendrites="
				+ this.dendrites + ", dendriteTable=" + this.dendriteTable + ", id="
				+ this.id + "]";
	} // end toString()

	public Dendrite getDendrite(int index) {
		return this.dendriteTable.get(index);
	} // end getDendrite()

	public void addBias(Neuron biasNeuron) {
		Dendrite dendrite = new Dendrite(biasNeuron, this);
		this.biasDendrite = dendrite;
		this.dendrites.add(dendrite);
	} // end addBias()

	public ArrayList<Dendrite> getDendrites() {
		return this.dendrites;
	} // end getDendrites()

	public double getOutput() {
		return this.output;
	} // end getOutput()

	public void setOutput(double output) {
		this.output = output;
	} // end setOutput()

	public static void setCounter(int counter) {
		Neuron.counter = counter;
	} // end setCounter()

} // end class
