/**
 * SPE Research — k6 Stress Test
 * Runs three load stages: ramp-up → sustained → spike
 * Env vars: PORT (required), LANE (label), LOAD_PROFILE (low|mid|high, default mid)
 *
 * Usage:
 *   k6 run -e PORT=8500 -e LANE=python -e LOAD_PROFILE=high \
 *          --summary-export=perf-python-high.json stress-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// ── Custom metrics ────────────────────────────────────────────────────────────
const errorRate        = new Rate('error_rate');
const inferenceLatency = new Trend('inference_latency_ms', true);
const timeouts         = new Counter('timeouts');

// ── Load profiles ─────────────────────────────────────────────────────────────
const PROFILES = {
  low:  { vus: 10,  peak: 20,  duration: '60s'  },
  mid:  { vus: 50,  peak: 100, duration: '90s'  },
  high: { vus: 100, peak: 200, duration: '120s' },
};

const profile = PROFILES[__ENV.LOAD_PROFILE || 'mid'];
const PORT    = __ENV.PORT || '8500';
const LANE    = __ENV.LANE || 'unknown';
const BASE    = `http://localhost:${PORT}`;

export const options = {
  stages: [
    { duration: '20s', target: profile.vus  },   // ramp-up
    { duration: profile.duration, target: profile.vus  },  // sustained
    { duration: '15s', target: profile.peak },   // spike
    { duration: '15s', target: profile.vus  },   // recover
    { duration: '10s', target: 0             },   // ramp-down
  ],
  thresholds: {
    'http_req_duration':    ['p(95)<2000', 'p(99)<5000'],
    'http_req_failed':      ['rate<0.05'],
    'error_rate':           ['rate<0.05'],
    'inference_latency_ms': ['p(95)<1500'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
};

// ── Realistic payloads (derived from Pima dataset test set) ───────────────────
// Each row: [Pregnancies, Glucose, BloodPressure, SkinThickness,
//            Insulin, BMI, DiabetesPedigreeFunction, Age]
const PAYLOADS = [
  { features: [6, 148.0, 72.0, 35.0, 169.5, 33.6, 0.627, 50] },
  { features: [1,  85.0, 66.0, 29.0, 102.5, 26.6, 0.351, 31] },
  { features: [8, 183.0, 64.0, 29.0, 102.5, 23.3, 0.672, 32] },
  { features: [1,  89.0, 66.0, 23.0,  94.0, 28.1, 0.167, 21] },
  { features: [0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33] },
  { features: [5, 116.0, 74.0, 29.0, 102.5, 25.6, 0.201, 30] },
  { features: [3,  78.0, 50.0, 32.0,  88.0, 31.0, 0.248, 26] },
  { features: [10, 115.0, 0.0, 29.0, 102.5, 35.3, 0.134, 29] },
  { features: [2, 197.0, 70.0, 45.0, 543.0, 30.5, 0.158, 53] },
  { features: [8, 125.0, 96.0, 29.0, 102.5, 0.0,  0.232, 54] },
];

const HEADERS = { 'Content-Type': 'application/json' };

export default function () {
  const payload = JSON.stringify(PAYLOADS[Math.floor(Math.random() * PAYLOADS.length)]);
  const start   = Date.now();

  const res = http.post(`${BASE}/predict`, payload, {
    headers: HEADERS,
    timeout: '5s',
  });

  const latency = Date.now() - start;
  inferenceLatency.add(latency);

  const ok = check(res, {
    'status 200':       r => r.status === 200,
    'has prediction':   r => {
      try { return JSON.parse(r.body).prediction !== undefined; } catch { return false; }
    },
    'prediction 0 or 1': r => {
      try {
        const p = JSON.parse(r.body).prediction;
        return p === 0 || p === 1;
      } catch { return false; }
    },
    'has probability':  r => {
      try { return typeof JSON.parse(r.body).probability === 'number'; } catch { return false; }
    },
    'latency < 2s':     () => latency < 2000,
  });

  errorRate.add(!ok);
  if (res.status === 0) timeouts.add(1);

  sleep(Math.random() * 0.5 + 0.1); // 100–600ms think time
}

export function handleSummary(data) {
  return {
    'stdout': JSON.stringify({
      lane:    LANE,
      profile: __ENV.LOAD_PROFILE || 'mid',
      metrics: {
        rps:              data.metrics.http_reqs?.values?.rate,
        p50_ms:           data.metrics.http_req_duration?.values?.['med'],
        p95_ms:           data.metrics.http_req_duration?.values?.['p(95)'],
        p99_ms:           data.metrics.http_req_duration?.values?.['p(99)'],
        error_rate:       data.metrics.http_req_failed?.values?.rate,
        inference_p95_ms: data.metrics.inference_latency_ms?.values?.['p(95)'],
        total_requests:   data.metrics.http_reqs?.values?.count,
        vus_peak:         data.metrics.vus_max?.values?.max,
      },
      thresholds_passed: Object.entries(data.metrics).every(
        ([, m]) => !m.thresholds || Object.values(m.thresholds).every(t => !t.ok === false)
      ),
    }, null, 2),
  };
}