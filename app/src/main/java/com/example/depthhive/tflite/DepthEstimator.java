/* Copyright 2020 The TensorFlow Authors and Clarence Chen. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

        package com.example.depthhive.tflite;

        import android.app.Activity;
        import android.graphics.Bitmap;
        import android.graphics.RectF;
        import android.os.SystemClock;
        import android.os.Trace;
        import java.io.IOException;
        import java.nio.MappedByteBuffer;
        import org.tensorflow.lite.DataType;
        import org.tensorflow.lite.Interpreter;
        import com.example.depthhive.env.Logger;

        import org.tensorflow.lite.gpu.GpuDelegate;
        import org.tensorflow.lite.nnapi.NnApiDelegate;
        import org.tensorflow.lite.support.common.FileUtil;
        import org.tensorflow.lite.support.common.TensorOperator;
        import org.tensorflow.lite.support.common.TensorProcessor;
        import org.tensorflow.lite.support.image.ImageProcessor;
        import org.tensorflow.lite.support.image.TensorImage;
        import org.tensorflow.lite.support.image.ops.ResizeOp;
        import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
        import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
        import org.tensorflow.lite.support.image.ops.Rot90Op;
        import org.tensorflow.lite.support.image.ColorSpaceType;
        import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class DepthEstimator {
    private static final Logger LOGGER = new Logger();

    /** The model type used for classification. */
    public enum Model {
        FLOAT_MOBILENET,
        QUANTIZED_MOBILENET
    }

    /** The runtime device type used for executing classification. */
    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Image size along the x axis. */
    private final int imageSizeX;

    /** Image size along the y axis. */
    private final int imageSizeY;

    /** Optional GPU delegate for accleration. */
    private GpuDelegate gpuDelegate = null;

    /** Optional NNAPI delegate for accleration. */
    private NnApiDelegate nnApiDelegate = null;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    /** Output depth map TensorBuffer. */
    private final TensorBuffer outputTensorBuffer;

    /** Output TensorImage container used to create Bitmap */
    private final TensorImage outputDepthMapBuffer;

    /** Processer to apply post processing of the output probability. */
    private final TensorProcessor DepthMapProcessor;

    /**
     * Creates a classifier with the provided configuration.
     *
     * @param activity The current Activity.
     * @param model The model to use for classification.
     * @param device The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     */
    public static DepthEstimator create(Activity activity, Model model, Device device, int numThreads)
            throws IOException {
        if (model == Model.QUANTIZED_MOBILENET) {
            return new DepthEstimatorQuantizedMobileNet(activity, device, numThreads);
        } else if (model == Model.FLOAT_MOBILENET) {
            return new DepthEstimatorFloatMobileNet(activity, device, numThreads);
        } else {
            throw new UnsupportedOperationException();
        }
    }


    /** Initializes a {@code DepthEstimator}. */
    protected DepthEstimator(Activity activity, Device device, int numThreads) throws IOException {
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        switch (device) {
            case NNAPI:
                nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int depthMapTensorIndex = 0;
        int[] depthMapTensorShape =
                tflite.getOutputTensor(depthMapTensorIndex).shape(); // {1, height, width, 1}
        DataType depthMapDataType = tflite.getOutputTensor(depthMapTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputTensorBuffer = TensorBuffer.createFixedSize(depthMapTensorShape, depthMapDataType);
        outputDepthMapBuffer = new TensorImage(DataType.UINT8);

        // Creates the post processor for the output probability.
        DepthMapProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        LOGGER.d("Created a Tensorflow Lite Depth Map Estimator.");
    }

    /** Runs inference and returns the depth map. */
    public Bitmap recognizeImage(final Bitmap bitmap, int sensorOrientation) {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("loadImage");
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Trace.endSection();
        LOGGER.v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputTensorBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        // Gets the map of label and probability.
        outputDepthMapBuffer.load(
                DepthMapProcessor.process(outputTensorBuffer), ColorSpaceType.GRAYSCALE);
        Bitmap outputDepthMap = outputDepthMapBuffer.getBitmap();
        Trace.endSection();

        // Return Bitmap Depth Map.
        return outputDepthMap;
    }

    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
        tfliteModel = null;
    }

    /** Get the image size along the x axis. */
    public int getImageSizeX() {
        return imageSizeX;
    }

    /** Get the image size along the y axis. */
    public int getImageSizeY() {
        return imageSizeY;
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(numRotation))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    /** Gets the name of the model file stored in Assets. */
    protected abstract String getModelPath();

    /** Gets the TensorOperator to normalize the input image in preprocessing. */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();
}
