//package cs446.homework2;
package cs446.weka.classifiers.trees;

import weka.classifiers.*;
import weka.core.Instances;
import weka.core.Instance;
import java.util.Arrays;
import java.lang.*;

public class SGD extends Classifier{

	private boolean trained = false;
	private double learningRate = 0.25;
	private double [] currentWeight;
	private double currentWeight0 = 0.0;
	
	private double getLabel(double classValue)
	{
		if (classValue == 0.0)
			return -1;
		return 1;
	}

	//Calculate w^T * x.
	private double getPrediction(Instance currentInstance, double [] weight)
	{
		int attributeIndex;
		double prediction = 0.0;
		for (attributeIndex = 0; attributeIndex < (currentInstance.numAttributes() - 1); attributeIndex++)
		{
			double attributeVal = Double.parseDouble(currentInstance.stringValue(attributeIndex));
			prediction = prediction + (weight[attributeIndex] * attributeVal);
		}
		prediction = prediction + currentWeight0;
		if (prediction < 0)
			return -1.0;
		return 1.0;
	}	

	//Update the Weights.
	private double [] updateWeight(Instance currentInstance, double prediction, double [] currWeight)
	{
		int vectorIndex;
		double delta = prediction - getLabel(currentInstance.classValue());
		delta = delta * learningRate;
		for (vectorIndex = 0; vectorIndex < currentInstance.numAttributes() - 1; vectorIndex++)
			currWeight[vectorIndex] = currWeight[vectorIndex] - (delta * Double.parseDouble(currentInstance.stringValue(vectorIndex)));
		//Update theta:
		currentWeight0 = currentWeight0 - delta;
		return currWeight;
	}

/*	private double sign(double value)
	{
		if (value < 0)
			return 0;
		return 1;
	}
*/
/*	
	private double findPredictionAccuracy(double prediction, Instance currentInstance)
	{
		int accuratePrediction = 0;
	 	if (sign(prediction) == currentInstance.classValue())
			return 1;
		return 0;
	}

*/
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		/*
		 * STUDENTS : Implement your learning algorithm (e.g. SGD) inside this function, at the end of this function, your classifier
		 * should be prepared to classify data.
		 */
		//Initialize the weight vectors. Last element of currWeight corresponds to theta.
		Instance currentInstance;
		currentWeight = new double[arg0.numAttributes() - 1];
		Arrays.fill(currentWeight,0.00);
		double prediction;
		double accuracy = 0.0;
		int iterator;
		
		//Running this over the entire batch many number of times.
		//It is found that a better accuracy is achieved for iterations around 750.
		for(iterator = 0; iterator < 755; iterator++)
		{
			for (int instanceIndex = 0; instanceIndex < arg0.numInstances(); instanceIndex++)
			{
				currentInstance = arg0.instance(instanceIndex);
				prediction = getPrediction(currentInstance, currentWeight);
				currentWeight = updateWeight(currentInstance, prediction, currentWeight);
			}
		}
		trained = true;//keep this
	}
	
	@Override
	public double classifyInstance(Instance instance) throws java.lang.Exception 
	{
		if(!trained){
			throw new Exception("The classifier is not trained!");
		}
		double predictedLabel = getPrediction(instance, currentWeight);
		if (predictedLabel == -1)
			return 0;										//-1 corresponds to the zeroeth index of classValue.
		return 1;
		
		/*
		 * STUDENTS : Implement the decision function here.
		 *
		 * BEWARE: From the API, 
		 * 	Returns:
		 * 		index of the predicted class as a double if the class is nominal, otherwise the predicted value
		 *
		 * 		So for + -, if they are in that order in the ARFF file, you might return 0.0 and 1.0, respectively.
		 */
	}

}
