package com.murlocks.maxent;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.net.URL;
import java.net.URISyntaxException;
import java.nio.file.Paths;

import opennlp.maxent.*;
import opennlp.maxent.io.*;
import opennlp.model.*;

public class Chunker {

    static final String dataFileName       = "training.data";
    static final String dataFeaturesName   = "training.features";
    static final String modelFileName      = "Chunker.model";
    static final String testFileName       = "test.data";
    static final String viterbiLogFile     = "viterbi.log";
    static final String bestOUtcomeLogFile = "bestoutcome.log";


    static final boolean readModelIn = false; // if true, read model in from modelFileName instead of building a new one


    /**
     *
     * Builds the following feature list:
     *   w   = Current word
     *   wa1 = Word that follows the current
     *   wb1 = Word that is before the current
     *   p1  = POS of the prior and current words in the format: POS-prior_POS-cur
     *   p2  = POS of the current and next words in the format: POS-current_POS-next
     *   p3  = POS of the prior, current, and next words in the format: POS-prior_POS-current_POS-next
     *
     * @param fileName The name of the file that contains data in the form of: WORD POS TAG per line
     * @param includeOutcome If true the tag will be included as the final value in the feature list
     *
     */
    static List<String> getFeatures(String fileName, boolean includeOutcome) throws IOException, URISyntaxException{

        // Separate the input file based on empty new lines

        URL path = ClassLoader.getSystemResource(fileName);
        Scanner s = new Scanner(new FileReader(Paths.get(path.toURI()).toFile())).useDelimiter(Pattern.compile("\\s^\\s*$\\s", Pattern.MULTILINE));
        List<String> features = new ArrayList<>();

        while (s.hasNext()) {

            // Break chunk into lines and extract words, POS, and outcomes (the assigned BIO tag) from each line

            String[] lines = s.next().split("\n");
            List<String> words = new ArrayList<>();
            List<String> pos = new ArrayList<>();
            List<String> outcomes = new ArrayList<>();

            for (String line : lines) {
                String[] t = line.split(" ");
                if (t.length == 3) {
                    words.add(t[0]);
                    pos.add(t[1]);
                    if (includeOutcome) {
                        outcomes.add(t[2]);
                    }
                }
            }

            // Build feature lists for each word

            for (int i = 0; i < words.size(); ++i) {

                String f, wb1, wb1p, w, wp, wa1, wa1p;

                if(i - 1 < 0) {
                    wb1  = "NONE";
                    wb1p = "NONE";
                } else {
                    wb1  = words.get(i-1);
                    wb1p = pos.get(i-1);
                }

                w  = words.get(i);
                wp = pos.get(i);

                if (i + 1 >= words.size()) {
                    wa1  = "NONE";
                    wa1p = "NONE";
                } else {
                    wa1  = words.get(i+1);
                    wa1p = pos.get(i+1);
                }

                // Replace any references to time or numbers with the generic: "TIME" or "NUMBER" respectively

                if (wp.contains("CD")) {
                    String[] temp = {"second", "minute", "hour", "day", "week", "month", "year"};;
                    for (String target : temp) {
                        if (wa1.contains(target)) {
                            w = "TIME";
                            break;
                        } else {
                            w = "NUMBER";
                        }
                    }
                }

                f = "wb1="  + wb1  + " "
                  + "w="    + w    + " "
                  + "wa1="  + wa1  + " "
                  + "p1="   + wb1p + "_" + wp   + " "
                  + "p2="   + wp   + "_" + wa1p + " "
                  + "p3="   + wb1p + "_" + wp   + "_" + wa1p;

                if (includeOutcome) {
                    f = f + " " + outcomes.get(i);
                }

                features.add(f);
            }
        }

        s.close();

        return features;
    }


    /**
     *
     * Turns a string of features: "feat=val feat=val ..." into an array of strings: ["feat=val", "feat=val", ... ]
     * @param features The string of features to be turned into an array
     *
     */
    static List<String[]> buildFeaturesList(List<String> features) {
        List<String[]> rtrn = new ArrayList<String[]>();
        for (String feature : features) {
            rtrn.add(feature.split(" "));
        }
        return rtrn;
    }


    /**
     *
     * Save a string of features into a file
     * @param features The string of features to be saved into a file
     * @param fileName The name of the file to save to
     *
     */
    static void writeListToFile(List<String> features, String fileName) throws IOException, URISyntaxException{
        PrintWriter pw = new PrintWriter(fileName, "UTF-8");
        for (String feature : features) {
            pw.println(feature);
        }
        pw.close();
    }


    /**
     *
     * Builds a feature list using a data file, saves the generated feature list, and finally saves and returns a model generated from the feature list
     * @param dataFileName File that contains a list of data in the form: WORD POS TAG per line
     * @param dataFeaturesName File to save features list to
     * @param modelFileName File to save model to
     *
     */
    static MaxentModel buildModel(String dataFileName, String dataFeaturesName, String modelFileName) throws IOException, URISyntaxException{
        writeListToFile(getFeatures(dataFileName, true), dataFeaturesName);
        EventStream es = new BasicEventStream(new PlainTextByLineDataStream(new FileReader(dataFeaturesName)));
        GISModel model = GIS.trainModel(es, 100, 4);
        File outputFile = new File(modelFileName);
        GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
        writer.persist();
        return model;
    }


    /**
     *
     * Outputs the number of incorrect BIO tags, precision, recall, and F-score
     * Also saves a log of the incorrect BIO tags with the expected tag, the output tag, and the feature list of that word
     * @param outcomes The produced outcome tags from the data file
     * @param featuresList the features list used to produce the outcomes
     * @param testFileName The name of the test file used with the correct tags in the final column of each line
     * @param enableLogging If true will log mismatches to logFileName
     * @param logFileName The name of the log file
     *
     */
    static void testOutcomes(List<String> outcomes, List<String[]> featuresList, String testFileName, boolean enableLogging, String logFileName) throws IOException, URISyntaxException{

        URL path = ClassLoader.getSystemResource(testFileName);
        Scanner s = new Scanner(new FileReader(Paths.get(path.toURI()).toFile())).useDelimiter(Pattern.compile("\\s^\\s*$\\s", Pattern.MULTILINE));
        List<String> expected = new ArrayList<>();

        // Extract the expected tags from the test file

        while (s.hasNext()) {
            String[] lines = s.next().split("\n");
            for (String line : lines) {
                expected.add(line.split(" ")[2]);
            }
        }

        Iterator<String> i = outcomes.iterator();
        Iterator<String> j = expected.iterator();
        Iterator<String[]> k = featuresList.iterator();
        List<String> log = new ArrayList<>();
        String out, exp;
        String[] context;

        int bWrong = 0, iWrong = 0, oWrong = 0;
        float expBlocks = 0, correctOutBlocks = 0, wrongOutBlocks = 0;
        boolean inExpBlock = false, inOutBlock = false;

        // While there are still tags to check

        while(i.hasNext() && j.hasNext()) {

            out = i.next();
            exp = j.next();
            context = k.next();

            // Determine chunk boundaries based on tags
            // If the current tag is I then we are inside a block
            // The block counters are incremented when we hit a tag that is not I while currently inside a block

            if (!out.equals(exp)) {
                if(enableLogging) {
                    log.add("[" + context[2] + "] Expected: " + exp + ", got: " + out + " in the context of: " + Arrays.toString(context));
                }
                switch (exp) {
                    case "B":
                        if (inExpBlock) expBlocks++;
                        bWrong++;
                        break;
                    case "I":
                        inExpBlock = true;
                        iWrong++;
                        break;
                    case "O":
                        if (inExpBlock) expBlocks++;
                        inExpBlock = false;
                        oWrong++;
                        break;
                }
                switch (out) {
                    case "B":
                        if (inOutBlock) wrongOutBlocks++;
                        break;
                    case "I":
                        inOutBlock = true;
                        break;
                    case "O":
                        if (inOutBlock) wrongOutBlocks++;
                        inOutBlock = false;
                        break;
                }
            } else {
                switch (exp) {
                    case "B":
                        if (inOutBlock) correctOutBlocks++;
                        if (inExpBlock) expBlocks++;
                        break;
                    case "I":
                        inExpBlock = true;
                        inOutBlock = true;
                        break;
                    case "O":
                        if (inExpBlock) expBlocks++;
                        if (inOutBlock) correctOutBlocks++;
                        inExpBlock = false;
                        inOutBlock = false;
                        break;
                }
            }
        }

        System.out.printf("Got %d B wrong\n", bWrong);
        System.out.printf("Got %d I wrong\n", iWrong);
        System.out.printf("Got %d O wrong\n", oWrong);

        float P = correctOutBlocks / (correctOutBlocks + wrongOutBlocks);
        float R = correctOutBlocks / expBlocks;
        System.out.printf("Precision = %.4f\n", P);
        System.out.printf("Recall = %.4f\n", R);
        System.out.printf("F-score = %.4f\n", 2*P*R/(P+R));

        writeListToFile(log, logFileName);

        s.close();
    }


    /**
     *
     * Uses the viterbi algorithm, outputs a list of tags based on the probability output of the maxent model
     * @param probabilities Probabilities output by the eval function of the maxent model
     * @param model The maxent model that was used to create the probabilities
     *
     */
    static List<String> viterbi(List<double[]> probabilities, MaxentModel model) {

        // Initilizing variables: s is the set of states: {I, O, B, start, finish}, t is time

        List<String> output = new ArrayList<String>();
        int s = model.getNumOutcomes(); // start state
        int f = s + 1; // finish state
        int t = probabilities.size();
        double[][] p = (double[][])probabilities.toArray(new double[t][s]);
        double[][] v = new double[t][s + 2];
        int[][] b = new int[t][s + 2];

        // Initialization: start -> first states

        for (int i = 0; i < s; i++) {
            v[0][i] = p[0][i];
            b[0][i] = s;
        }

        // Find best path from previous states to the current state

        for (int i = 1; i < t; i++) {
            for (int j = 0; j < s; j++) {
                for (int k = 0; k < s; k++) {
                    if (v[i][j] < v[i-1][k] * p[i][j]) {
                        v[i][j] = v[i-1][k] * p[i][j];
                        b[i][j] = k;
                    }
                }
            }
        }

        // Finalization: link up to the finish state

        for (int i = 0; i < s; i++) {
            if (v[t-1][f] < v[t-1][i]) {
                v[t-1][f] = v[t-1][i];
                b[t-1][f] = i;
            }
        }

        // Follow back pointer until it reaches start state and rebuild the best path

        int pointer = b[t-1][f];
        int time = t-1;

        while (pointer != s) {
            output.add(model.getOutcome(pointer));
            pointer = b[time--][pointer];
        }

        // Reverse the path because we followed it from end to start

        Collections.reverse(output);

        return output;
    }


    public static void main(String[] args) {
        try {

            // Get a GIS model

            MaxentModel model = null;

            if (readModelIn) {
                GISModelReader reader = new SuffixSensitiveGISModelReader(new File(modelFileName));
                model = reader.getModel();
            } else {
                model = buildModel(dataFileName, dataFeaturesName, modelFileName);
            }

            if (model != null) {
                // Generate features list for the test file

                List<String[]> testFeatures = buildFeaturesList(getFeatures(testFileName, false));
                List<double[]> events = new ArrayList<double[]>();
                List<String> getBestOut = new ArrayList<String>();

                // Iterate through the test files feature list and extract probabilities/BestOutcomes

                for (String[] features : testFeatures) {
                    events.add(model.eval(features));
                    getBestOut.add(model.getBestOutcome(model.eval(features)));
                }

                List<String> out = viterbi(events, model);

                System.out.println("\nNumber of outcomes: " + out.size());
                writeListToFile(out, "output.log");

                System.out.println("\nUsing viterbi:");
                testOutcomes(out, testFeatures, testFileName, false, viterbiLogFile);

                System.out.println("\nUsing getBestOutcome:");
                testOutcomes(getBestOut, testFeatures, testFileName, false, bestOUtcomeLogFile);
            } else {
                System.out.println("Unable to create model");
            }
        } catch (Exception e) {
            System.out.printf("IO exception: %s\n", e.getMessage());
        }
    }
}
