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
import java.io.IOException;

import com.example.depthhive.tflite.DepthEstimator.Device;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

/** This TensorFlow Lite classifier works with the quantized MobileNet model. */
public class DepthEstimatorQuantizedMobileNet extends DepthEstimator {

    /**
     * The quantized model does not require normalization, thus set mean as 0.0f, and std as 1.0f to
     * bypass the normalization.
     */
    private static final float IMAGE_MEAN = 0.0f;

    private static final float IMAGE_STD = 1.0f;

    /** Quantized MobileNet already outputs depth map in range [0, 255). */
    private static final float DEPTH_MEAN = 0.0f;

    private static final float DEPTH_STD = 1.0f;

    /**
     * Initializes a {@code ClassifierQuantizedMobileNet}.
     *
     * @param activity
     */
    public DepthEstimatorQuantizedMobileNet(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return "bts_nyu_mobilenet_quant.tflite";
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(DEPTH_MEAN, DEPTH_STD);
    }
}