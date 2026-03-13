/**
 * WASM lane inference module
 * Loads ONNX model via onnxruntime-node running inside a sandboxed Worker.
 * In production, this would use the WebNN / wasm-nn proposal; here we use
 * ORT's WASM backend explicitly to demonstrate the sandbox boundary.
 */

'use strict';

const ort  = require('onnxruntime-node');
const fs   = require('fs');
const path = require('path');

// Force WASM execution provider (not native CPU) — this is the key
// differentiator for this lane: inference occurs in a WASM sandbox.
ort.env.wasm.numThreads      = 1;
ort.env.wasm.simd            = true;
ort.env.wasm.proxy           = false;

const SCALER = JSON.parse(fs.readFileSync('/app/scaler.json', 'utf8'));
const MEAN   = Float32Array.from(SCALER.mean_);
const SCALE  = Float32Array.from(SCALER.scale_);

let _session = null;

async function getSession() {
    if (!_session) {
        // Explicitly request WASM execution provider
        _session = await ort.InferenceSession.create('/app/model.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });
    }
    return _session;
}

/**
 * Run inference on a single feature vector.
 * @param {number[]} features - 8-element array
 * @returns {{ prediction: number, probability: number, latency_ms: number }}
 */
async function infer(features) {
    if (features.length !== 8) {
        throw new Error('Expected 8 features');
    }

    const t0 = performance.now();
    const sess = await getSession();

    // Normalise
    const normed = new Float32Array(8);
    for (let i = 0; i < 8; i++) {
        normed[i] = (features[i] - MEAN[i]) / SCALE[i];
    }

    const tensor  = new ort.Tensor('float32', normed, [1, 8]);
    const results = await sess.run({ features: tensor });
    const logit   = results['logit'].data[0];
    const prob    = 1.0 / (1.0 + Math.exp(-logit));
    const pred    = prob >= 0.5 ? 1 : 0;
    const ms      = performance.now() - t0;

    return {
        prediction:  pred,
        probability: Math.round(prob * 1e6) / 1e6,
        latency_ms:  Math.round(ms * 1000) / 1000,
    };
}

module.exports = { infer, getSession };