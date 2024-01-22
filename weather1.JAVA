import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class WeatherPrediction {

    public static void main(String[] args) throws Exception {
        // Load the weather dataset (replace this with your actual dataset)
        Instances data = loadData("path/to/your/dataset.arff");

        // Set the target attribute index (replace with the actual index)
        int targetIndex = 0;

        // Create and build the linear regression model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(data);

        // Predict temperatures for the next 7 days
        List<Date> futureDates = generateFutureDates(7);
        Instances futureData = createFutureData(futureDates, data);

        System.out.println("Predicted temperatures for the next 7 days:");
        for (int i = 0; i < futureData.numInstances(); i++) {
            double prediction = model.classifyInstance(futureData.instance(i));
            System.out.println(futureDates.get(i) + ": " + prediction + " °C");
        }
    }

    private static Instances loadData(String filePath) throws Exception {
        // Load dataset from ARFF file
        DataSource source = new DataSource(filePath);
        Instances data = source.getDataSet();

        // Set the class index (target attribute)
        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    private static List<Date> generateFutureDates(int numDays) {
        // Generate future dates for prediction
        List<Date> futureDates = new ArrayList<>();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");

        for (int i = 1; i <= numDays; i++) {
            Date currentDate = new Date();
            currentDate.setTime(currentDate.getTime() + i * 24 * 60 * 60 * 1000);
            futureDates.add(currentDate);
        }

        return futureDates;
    }

    private static Instances createFutureData(List<Date> futureDates, Instances originalData) {
        // Create Instances for future predictions
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < originalData.numAttributes() - 1; i++) {
            attributes.add(originalData.attribute(i));
        }

        // Add a new attribute for the target variable (temperature)
        attributes.add(new Attribute("temperature"));

        Instances futureData = new Instances("futureData", attributes, 0);

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");

        for (Date date : futureDates) {
            Instance instance = new DenseInstance(attributes.size());

            for (int i = 0; i < attributes.size() - 1; i++) {
                instance.setValue(attributes.get(i), 0);  // Set other attributes to 0 for simplicity
            }

            instance.setValue(attributes.size() - 1, 0);  // Set the target attribute to 0 for simplicity
            futureData.add(instance);
        }

        return futureData;
    }
}