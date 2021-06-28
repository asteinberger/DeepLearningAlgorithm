public class Dendrite {

	private double weight = 0;
	private double previousDeltaWeight = 0;
	private double deltaWeight = 0;
	private final Neuron fromNeuron;
	private final Neuron toNeuron;
	private static int counter = 0;
	public final int id; // auto increment, starts at 0

	public Dendrite(Neuron from, Neuron to) {
		this.fromNeuron = from;
		this.toNeuron = to;
		id = counter;
		counter++;
	} // end constructor

	@Override
	public String toString() {
		return "[Dendrite weight=" + this.weight + ", previousDeltaWeight="
				+ this.previousDeltaWeight + ", deltaWeight=" + this.deltaWeight
				+ ", fromNeuronID=" + this.fromNeuron.id + ", toNeuronID="
				+ this.toNeuron.id + ", id=" + this.id + "]";
	} // end toString()

	public double getWeight() {
		return this.weight;
	} // end getWeight()

	public void setWeight(double weight) {
		this.weight = weight;
	} // end setWeight()

	public void setDeltaWeight(double deltaWeight) {
		this.previousDeltaWeight = this.deltaWeight;
		this.deltaWeight = deltaWeight;
	} // end setDeltaWeight()

	public double getPreviousDeltaWeight() {
		return this.previousDeltaWeight;
	} // end getPrevDeltaWeight()

	public Neuron getFromNeuron() {
		return this.fromNeuron;
	} // end getFromNeuron()

	public static void setCounter(int c) {
		Dendrite.counter = c;
	} // end setCounter()

} // end class
