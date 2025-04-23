import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [lang, setLang] = useState('en');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  // Downsample Float32Array to target rate
  const downsampleBuffer = (buffer, inputRate, outputRate) => {
    if (outputRate === inputRate) return buffer;
    const ratio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetRes = 0;
    let offsetBuf = 0;
    while (offsetRes < newLength) {
      result[offsetRes++] = buffer[Math.floor(offsetBuf)];
      offsetBuf += ratio;
    }
    return result;
  };

  const startWS = () => {
    setLoading(true);
    const wsUrl = `ws://localhost:8000/ws?lang=${lang}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen = () => {
      console.log('WebSocket opened');
      setConnected(true);
      setTranscripts([]);
      // Initialize microphone capture
      navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        streamRef.current = stream;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        audioContextRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = e => {
          const input = e.inputBuffer.getChannelData(0);
          const down = downsampleBuffer(input, audioCtx.sampleRate, 16000);
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(down.buffer);
          }
        };
        source.connect(processor);
        processor.connect(audioCtx.destination);
        processorRef.current = processor;
      });
    };
    ws.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'ready') {
        setLoading(false);
        return;
      }
      // transcript message
      setTranscripts(prev => [...prev, msg]);
    };
    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnected(false);
      setLoading(false);
    };
  };

  const stopWS = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    // Stop audio capture
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  return (
    <div className="App" style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      {loading && <div style={{ color: '#d00', marginBottom: '10px' }}>Loading models, please wait...</div>}
      <h1>Real-Time Transcription</h1>
      <div style={{ marginBottom: '10px' }}>
        <label>
          Language:
          <select value={lang} onChange={e => setLang(e.target.value)} style={{ marginLeft: '5px' }}>
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
            <option value="auto">Auto Detect</option>
          </select>
        </label>
        {!connected ? (
          <button onClick={startWS} style={{ marginLeft: '20px' }}>Start</button>
        ) : (
          <button onClick={stopWS} style={{ marginLeft: '20px' }}>Stop</button>
        )}
      </div>
      <div style={{ border: '1px solid #ccc', padding: '10px', height: '400px', overflowY: 'scroll' }}>
        {transcripts.map((t, i) => (
          <div key={i} style={{ marginBottom: '8px' }}>
            <strong>[{t.timestamp.toFixed(2)}s] {t.role.toUpperCase()}:</strong> {t.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
