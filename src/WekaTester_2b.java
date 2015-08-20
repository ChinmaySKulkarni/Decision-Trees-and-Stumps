package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.*;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import cs446.weka.classifiers.trees.SGD;

//For SGD.
public class WekaTester_2b {

    public static void main(String[] args) throws Exception {

	if (args.length != 4) {
	    System.err.println("Usage: WekaTester_2b train_arff-file test_arff-file test_data prediction_output_on_test_data");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);

	//Used to find the average accuracy over training data for different folds.
	double sum = 0;
	double averageAccuracy;
	// Train on 80% of the data and test on 20%
	for (int dataPart = 0; dataPart < 5; dataPart ++)
	{
		Instances train = data.trainCV(5,dataPart);
		Instances test = data.testCV(5, dataPart);

		SGD classifier = new SGD();

		// Train
		classifier.buildClassifier(train);

		// Print the classfier
		//System.out.println(classifier);
		//System.out.println();

		// Evaluate on the test set
		Evaluation evaluation = new Evaluation(test);
		evaluation.evaluateModel(classifier, test);
		System.out.println(evaluation.toSummaryString());
		sum += evaluation.pctCorrect();
	}
	//Find the average accuracy over all folds.
	averageAccuracy = sum/5.0;
	System.out.println("The predictive accuracy over the five folds is: \t" + averageAccuracy + "%\n");
	SGD classifier = new SGD();
	classifier.buildClassifier(data);
	//System.out.println(classifier);
	//System.out.println();

	// Load the test data
	Instances test = new Instances(new FileReader(new File(args[1])));
	// The last attribute is the class label
	test.setClassIndex(test.numAttributes() - 1);
	String line;
	BufferedReader inFile = new BufferedReader(new FileReader(new File(args[2])));
	BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(args[3]), true));
	double predictedLabel;
	String label;
	
	int instanceNumber;
	//Find the predicted labels for each instance and write to file.
	for(instanceNumber = 0; instanceNumber < test.numInstances(); instanceNumber++)
	{	
		predictedLabel = classifier.classifyInstance(test.instance(instanceNumber));
		if (predictedLabel == 0.0)
			label ="-1";
		else
			label = "1";
		test.instance(instanceNumber).setClassValue(label);
		line = inFile.readLine();
		line = line.replace("?", label);
		outFile.write(line);
		outFile.newLine();
	}
 }
}
