import http from 'k6/http';
import { check, sleep } from 'k6';

export default function () {
  // Use the PORT passed from Ansible
  const url = `http://localhost:${__ENV.PORT}/predict`;
  
  // Ensure this array matches the N_FEATURES in your scaler.json
  const payload = JSON.stringify({
    features: [7, 159, 64, 29, 125, 27.4, 0.294, 40] 
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