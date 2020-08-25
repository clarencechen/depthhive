/*
 * Copyright 2020 The TensorFlow Authors and Clarence Chen. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.depthhive;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import com.example.depthhive.env.BorderedText;
import com.example.depthhive.env.Logger;
import com.example.depthhive.tflite.DepthEstimator;
import com.example.depthhive.tflite.DepthEstimator.Device;
import com.example.depthhive.tflite.DepthEstimator.Model;

public class DepthEstActivity extends MainActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    private long lastProcessingTimeMs;
    private Integer sensorOrientation;
    private DepthEstimator depthEstimator;
    private BorderedText borderedText;
    /** Input image size of the model along x axis. */
    private int imageSizeX;
    /** Input image size of the model along y axis. */
    private int imageSizeY;

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        recreateDepthEstimator(getModel(), getDevice(), getNumThreads());
        if (depthEstimator == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final int cropSize = Math.min(previewWidth, previewHeight);

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (depthEstimator != null) {
                            final long startTime = SystemClock.uptimeMillis();
                            final Bitmap results =
                                    depthEstimator.recognizeImage(rgbFrameBitmap, sensorOrientation);
                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            //LOGGER.v("Detect: %s", results);

                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            showFrameInfo(previewWidth + "x" + previewHeight);
                                            showCropInfo(imageSizeX + "x" + imageSizeY);
                                            showCameraResolution(cropSize + "x" + cropSize);
                                            showRotationInfo(String.valueOf(sensorOrientation));
                                            showInference(lastProcessingTimeMs + "ms");
                                        }
                                    });

                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            displayBitmap(results);
                                        }
                                    });
                        }
                        readyForNextImage();
                    }
                });
    }

    @Override
    protected void onInferenceConfigurationChanged() {
        if (rgbFrameBitmap == null) {
            // Defer creation until we're getting camera frames.
            return;
        }
        final Device device = getDevice();
        final Model model = getModel();
        final int numThreads = getNumThreads();
        runInBackground(() -> recreateDepthEstimator(model, device, numThreads));
    }

    private void recreateDepthEstimator(Model model, Device device, int numThreads) {
        if (depthEstimator != null) {
            LOGGER.d("Closing depth estimator.");
            depthEstimator.close();
            depthEstimator = null;
        }
        if (device == Device.GPU
                && (model == Model.QUANTIZED_MOBILENET)) {
            LOGGER.d("Not creating depth estimator: GPU doesn't support quantized models.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show();
                    });
            return;
        }
        try {
            LOGGER.d(
                    "Creating depth estimator (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
            depthEstimator = DepthEstimator.create(this, model, device, numThreads);
        } catch (IOException e) {
            LOGGER.e(e, "Failed to create depth estimator.");
        }

        // Updates the input image size.
        imageSizeX = depthEstimator.getImageSizeX();
        imageSizeY = depthEstimator.getImageSizeY();
    }
}
