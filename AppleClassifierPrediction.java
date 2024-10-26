import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.SVM;

public class AppleClassifierPrediction {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load model SVM yang sudah dilatih
        SVM svm = SVM.load("Apple_classifier.xml");

        // Load gambar uji
        Mat testImage = Imgcodecs.imread("imagesCV/data/test/apple/XFXY6EOXUJ1H.jpg", Imgcodecs.IMREAD_GRAYSCALE);

        // Ekstraksi fitur menggunakan SIFT
        SIFT sift = SIFT.create();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptor = new Mat();
        sift.detectAndCompute(testImage, new Mat(), keypoints, descriptor);

        // Hitung rata-rata deskriptor untuk prediksi
        Mat avgDescriptor = new Mat();
        Core.reduce(descriptor, avgDescriptor, 0, Core.REDUCE_AVG, -1);

        // Pastikan avgDescriptor berbentuk 1xN
        if (avgDescriptor.rows() > 1) {
            Core.transpose(avgDescriptor, avgDescriptor);
        }

        // Prediksi menggunakan model SVM
        float result = svm.predict(avgDescriptor);

        // Tampilkan hasil prediksi
        if (result == 1) {
            System.out.println("Gambar adalah appel.");
        } else {
            System.out.println("Gambar bukan appel.");
        }
    }
}
