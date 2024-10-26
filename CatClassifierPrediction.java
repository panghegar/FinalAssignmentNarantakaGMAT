import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.SVM;

public class CatClassifierPrediction {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load model SVM yang sudah dilatih
        SVM svm = SVM.load("cat_classifier.xml");

        // Load gambar uji
        // assets/CAT_00/00000001_000.jpg
        // assets/download.jpeg
        // banana/0_100.jpg
        Mat testImage = Imgcodecs.imread("banana/0_100.jpg", Imgcodecs.IMREAD_GRAYSCALE);

        // Ekstraksi fitur menggunakan SIFT
        SIFT sift = SIFT.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptor = new Mat();
        sift.detectAndCompute(testImage, new Mat(), keypoints, descriptor);

        // Menghitung rata-rata deskriptor untuk prediksi
        Mat avgDescriptor = new Mat();
        Core.reduce(descriptor, avgDescriptor, 0, Core.REDUCE_AVG, -1);

        // Memastikan avgDescriptor berbentuk 1xN
        if (avgDescriptor.rows() > 1) {
            Core.transpose(avgDescriptor, avgDescriptor);
        }

        // Memprediksi menggunakan model SVM
        float result = svm.predict(avgDescriptor);

        // Menampilkan hasil prediksi
        if (result == 1) {
            System.out.println("Gambar adalah kucing");
        } else {
            System.out.println("Gambar bukan kucing");
        }
    }
}
