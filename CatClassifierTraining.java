import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class CatClassifierTraining {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Direktori dataset
        File catDir = new File("assets/CAT_00");
        File bananaDir = new File("banana");

        List<Mat> descriptors = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        // Proses gambar kucing (label 1)
        processCategory(catDir, descriptors, labels, 1);
        // Proses gambar non-kucing (label 0)
        processCategory(bananaDir, descriptors, labels, 0);

        // Gabungkan semua deskriptor menjadi satu Mat
        Mat trainingData = new Mat();
        for (Mat descriptor : descriptors) {
            trainingData.push_back(descriptor);
        }

        // Konversi labels ke Mat
        Mat labelMat = new Mat(labels.size(), 1, CvType.CV_32S);
        for (int i = 0; i < labels.size(); i++) {
            labelMat.put(i, 0, labels.get(i));
        }

        // Membuat dan melatih model SVM
        SVM svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.C_SVC);
        TrainData trainData = TrainData.create(trainingData, Ml.ROW_SAMPLE, labelMat);
        svm.train(trainData.getSamples(), Ml.ROW_SAMPLE, trainData.getResponses());

        // Simpan model
        svm.save("cat_classifier.xml");
        System.out.println("Model training complete and saved as 'cat_classifier.xml'.");
    }

    private static void processCategory(File directory, List<Mat> descriptors, List<Integer> labels, int label) {
        SIFT sift = SIFT.create();
        for (File file : directory.listFiles()) {
            Mat img = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            MatOfKeyPoint keypoints = new MatOfKeyPoint();
            Mat descriptor = new Mat();
            sift.detectAndCompute(img, new Mat(), keypoints, descriptor);
            descriptors.add(descriptor);
            for (int i = 0; i < descriptor.rows(); i++) {
                labels.add(label);
            }
        }
    }
}
