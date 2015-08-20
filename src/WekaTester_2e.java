package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.*;
import java.util.Random;
import java.lang.*;
import weka.core.*;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.FastVector;
import cs446.weka.classifiers.trees.Id3;
import cs446.weka.classifiers.trees.SGD;

public class WekaTester_2e 
{

		private static FastVector zeroOne;
		private static FastVector labels;
		private static FastVector attributes;
		private static Instances data;
		private static Attribute oneAttr;
		private static Attribute classLabel;

		private static String getFeatureName(int instanceNum)
		{
			String name = "decisionStump_" + Integer.toString(instanceNum);
			return name;
		}
		
		private static void initializeAttributesFastVector()
		{
			attributes = new FastVector(101);

			zeroOne = new FastVector(2);
			zeroOne.addElement("1");
			zeroOne.addElement("0");

			labels = new FastVector(2);
			labels.addElement("-1");
			labels.addElement("1");

			for(int classifierNum = 0; classifierNum < 100; classifierNum++)
			{
				Attribute oneAttr = new Attribute(getFeatureName(classifierNum), zeroOne); // Creates a _textual_ attribute that only allows values "0" and "1".
				attributes.addElement(oneAttr);
			}
			Attribute classLabel = new Attribute("Class", labels);
			attributes.addElement(classLabel);
		}

  public static void main(String[] args) throws Exception 
	{
		if (args.length != 4) 
		{
	   	System.err.println("Usage: WekaTester_2e train_arff-file test_arff-file test_data prediction_output_on_test_data");
	    System.exit(-1);
		}

		// Load the data
		data = new Instances(new FileReader(new File(args[0])));
		// The last attribute is the class label
		data.setClassIndex(data.numAttributes() - 1);

		double predictedLabel;
		Id3 [] classifiers = new Id3[100];
		initializeAttributesFastVector();
		Instances decisionStumpLabels = new Instances("DecisionStumpLabels", attributes, data.numInstances());
		decisionStumpLabels.setClassIndex(decisionStumpLabels.numAttributes() - 1);	
		//Fill the dataSamples randomly and train the Id3 classifier on each data sample. 
		for(int sampleNumber = 0; sampleNumber < 100; sampleNumber++)
		{
			Random randomNum = new Random(System.currentTimeMillis());					//Providing the current system time as the seed to the random number generator.
			data.randomize(randomNum);
			Instances dataSample = new Instances(data,0,data.numInstances()/2);
			// The last attribute is the class label
			dataSample.setClassIndex(dataSample.numAttributes() - 1);

			// Depth = 4.	
			classifiers[sampleNumber] = new Id3();
			classifiers[sampleNumber].setMaxDepth(4);	
			// Train
			classifiers[sampleNumber].buildClassifier(dataSample);
		}

		//Find corresponding values of the decision trees for each instance.
		for(int instanceNumber = 0; instanceNumber < data.numInstances(); instanceNumber++)	
		{
			Instance instanceForSGD = new Instance(101);
			for(int classifierNum = 0; classifierNum < 100; classifierNum++)
			{
				predictedLabel = classifiers[classifierNum].classifyInstance(data.instance(instanceNumber));
				instanceForSGD.setValue((Attribute)attributes.elementAt(classifierNum), predictedLabel);
			}
			instanceForSGD.setValue((Attribute)attributes.elementAt(100), data.instance(instanceNumber).classValue());
			decisionStumpLabels.add(instanceForSGD);
		}
	
		//Used to find the average accuracy over training data for different folds.
		double sum = 0;
		double averageAccuracy;
		//Perform	five-fold cross-validation on the new instances data set that you have for SGD.
		for (int dataPart = 0; dataPart < 5; dataPart ++)
		{
			Instances train = decisionStumpLabels.trainCV(5,dataPart);
			Instances test = decisionStumpLabels.testCV(5, dataPart);

			SGD classifier = new SGD();

			// Train
			classifier.buildClassifier(train);

			// Evaluate on the test set
			Evaluation evaluation = new Evaluation(test);
			evaluation.evaluateModel(classifier, test);
			System.out.println(evaluation.toSummaryString());
			sum += evaluation.pctCorrect();
		}
		averageAccuracy = sum/5.0;
		System.out.println("The predictive accuracy over the five folds is: \t" + averageAccuracy + "%\n");
		
		//Test the classifier on badge.test.blind
		SGD classifier = new SGD();
		//Train on the entire training set.
		classifier.buildClassifier(data);
		// Load the test data
		Instances test = new Instances(new FileReader(new File(args[1])));
		// The last attribute is the class label
		test.setClassIndex(test.numAttributes() - 1);
		String line;
		BufferedReader inFile = new BufferedReader(new FileReader(new File(args[2])));
		BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(args[3]), true));
		double predictedLabelID3;
		double prediction;
		String label;
		Instance instanceForSGD;
		Instances decisionStumpLabelsTest = new Instances("DecisionStumpLabelsTest", attributes, test.numInstances());
		decisionStumpLabelsTest.setClassIndex(decisionStumpLabelsTest.numAttributes() - 1);	
		
		//Find corresponding values of the decision trees for each instance.
		for(int instanceNumber = 0; instanceNumber < test.numInstances(); instanceNumber++)	
		{
			instanceForSGD = new Instance(101);
			//This will give me the features corresponding to the decision tree predicted labels.
			for(int classifierNum = 0; classifierNum < 100; classifierNum++)
			{
				predictedLabelID3 = classifiers[classifierNum].classifyInstance(test.instance(instanceNumber));
				instanceForSGD.setValue((Attribute)attributes.elementAt(classifierNum), predictedLabelID3);
			}
			decisionStumpLabelsTest.add(instanceForSGD);
		}	
		
		for(int instanceNumber = 0; instanceNumber < test.numInstances(); instanceNumber++)	
		{
			//Find the predicted labels for each instance and write to file.
			prediction = classifier.classifyInstance(decisionStumpLabelsTest.instance(instanceNumber));
			if (prediction == 0.0)
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
