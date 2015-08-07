package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.*;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
//import cs446.weka.classifiers.trees.SGD;

public class WekaTester_2c_d {

    public static void main(String[] args) throws Exception {

	if (args.length != 5) {
	    System.err.println("Usage: WekaTester_2c_d train_arff-file test_arff-file test_data prediction_output_on_test_data max_depth");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));
	//Take the depth as a command line arg.
	int depth = Integer.parseInt(args[4]);

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);

	//Used to find the average accuracy over training data for different folds.
	double sum = 0;
	double averageAccuracy;
	// Train on 80% of the data and test on 20%
	for (int dataPart = 0; dataPart < 5; dataPart++)
	{
		Instances train = data.trainCV(5,dataPart);
		Instances test = data.testCV(5, dataPart);

		// Create a new ID3 classifier. This is the modified one where you can
		// set the depth of the tree.
		Id3 classifier = new Id3();

		// An example depth. If this value is -1, then the tree is grown to full
		// depth.
		classifier.setMaxDepth(depth);

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
	averageAccuracy = sum/5.0;
	System.out.println("The predictive accuracy over the five folds is: \t" + averageAccuracy + "%\n");
	
	Id3 classifierBlind = new Id3();
	classifierBlind.setMaxDepth(depth);
	classifierBlind.buildClassifier(data);
	//System.out.println(classifierBlind);
	//System.out.println();

	// Load the test data
	Instances testId3 = new Instances(new FileReader(new File(args[1])));
	// The last attribute is the class label
	testId3.setClassIndex(testId3.numAttributes() - 1);
	String line;
	BufferedReader inFile = new BufferedReader(new FileReader(new File(args[2])));
	BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(args[3]), true));
	double predictedLabel;
	String label;
	
	int instanceNumber;
	for(instanceNumber = 0; instanceNumber < testId3.numInstances(); instanceNumber++)
	{
		predictedLabel = classifierBlind.classifyInstance(testId3.instance(instanceNumber));
		if (predictedLabel == 0.0)
			label ="-1";
		else
			label = "1";
		testId3.instance(instanceNumber).setClassValue(label);
		line = inFile.readLine();
		line = line.replace("?", label);
		outFile.write(line);
		outFile.newLine();
	}
 }
}
