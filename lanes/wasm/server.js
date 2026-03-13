'use strict';

const http  = require('http');
const { infer, getSession } = require('./infer');

const PORT  = 8502;
const ORT   = require('onnxruntime-node');

// Warm up session on start
getSession().then(() => {
    console.log('WASM session ready');
}).catch(e => {
    console.error('Session init failed:', e);
    process.exit(1);
});

const server = http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');

    if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200);
        res.end(JSON.stringify({ status: 'ok', lane: 'wasm' }));
        return;
    }

    if (req.method === 'POST' && req.url === '/predict') {
        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', async () => {
            try {
                const { features } = JSON.parse(body);
                if (!Array.isArray(features) || features.length !== 8) {
                    res.writeHead(422);
                    res.end(JSON.stringify({ error: 'Expected 8 features' }));
                    return;
                }
                const result = await infer(features);
                res.writeHead(200);
                res.end(JSON.stringify({
                    ...result,
                    lane:    'wasm',
                    runtime: `onnxruntime-wasm ${ORT.env.versions?.ortVersion || '1.17.3'}`,
                }));
            } catch (e) {
                res.writeHead(500);
                res.end(JSON.stringify({ error: e.message }));
            }
        });
        return;
    }

    res.writeHead(404);
    res.end(JSON.stringify({ error: 'Not found' }));
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`WASM lane listening on :${PORT}`);
});