import http from 'k6/http';
import { check, sleep } from 'k6';

export default function () {
  // Use the PORT passed from Ansible
  const url = `http://localhost:${__ENV.PORT}/predict`;
  
  // Ensure this array matches the N_FEATURES in your scaler.json
  const payload = JSON.stringify({
    features: [0.0, 126.0, 86.0, 27.0, 120.0, 27.399999618530273, 0.5149999856948853, 21.0, 3452.39990234375, 0.0, 3288.0]
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(0.5); // Pace the requests
}