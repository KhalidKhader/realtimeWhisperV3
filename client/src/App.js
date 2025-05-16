import React, { useState, useRef, useEffect } from 'react';
import {
  Container, Box, Typography, FormControl, InputLabel, Select, MenuItem,
  Button, Paper, Divider, Slider, IconButton, AppBar, Toolbar,
  CssBaseline, ThemeProvider, createTheme, FormHelperText
} from '@mui/material';
import { PlayArrow, Stop, Mic } from '@mui/icons-material';
import './App.css';

// Create a theme with a professional color palette
const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  const [lang, setLang] = useState('en');
  const [speakerCount, setSpeakerCount] = useState(2);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const transcriptEndRef = useRef(null);

  // Auto-scroll to bottom when new transcripts arrive
  useEffect(() => {
    if (transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcripts]);

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
    const wsUrl = `ws://localhost:8000/ws?lang=${lang}&speakers=${speakerCount}`;
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

  // Get color for speaker
  const getSpeakerColor = (speakerId) => {
    const colors = ['#3f51b5', '#f50057', '#00bcd4', '#ff9800', '#4caf50', '#9c27b0'];
    const id = speakerId.replace('Speaker_', '');
    return colors[(parseInt(id) - 1) % colors.length];
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Mic sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Real-Time Transcription
            </Typography>
          </Toolbar>
        </AppBar>
      </Box>
      <Container maxWidth="md" sx={{ mt: 4 }}>
        {loading && (
          <Paper elevation={2} sx={{ p: 2, mb: 2, bgcolor: '#ffebee' }}>
            <Typography color="error">
              Loading models, please wait... This may take a minute.
            </Typography>
          </Paper>
        )}

        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Configuration
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2, mb: 3 }}>
            <FormControl fullWidth>
              <InputLabel id="language-select-label">Language</InputLabel>
              <Select
                labelId="language-select-label"
                value={lang}
                label="Language"
                onChange={(e) => setLang(e.target.value)}
                disabled={connected}
              >
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="es">Spanish</MenuItem>
                <MenuItem value="fr">French</MenuItem>
                <MenuItem value="de">German</MenuItem>
                <MenuItem value="ar">Arabic</MenuItem>
                <MenuItem value="auto">Auto Detect</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel id="speakers-select-label">Number of Speakers</InputLabel>
              <Select
                labelId="speakers-select-label"
                value={speakerCount}
                label="Number of Speakers"
                onChange={(e) => setSpeakerCount(e.target.value)}
                disabled={connected}
              >
                {[1, 2, 3, 4, 5, 6].map(num => (
                  <MenuItem key={num} value={num}>{num}</MenuItem>
                ))}
              </Select>
              <FormHelperText>Expected number of speakers in conversation</FormHelperText>
            </FormControl>
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            {!connected ? (
              <Button 
                variant="contained" 
                color="primary" 
                size="large"
                startIcon={<PlayArrow />}
                onClick={startWS}
              >
                Start Transcription
              </Button>
            ) : (
              <Button 
                variant="contained" 
                color="secondary" 
                size="large"
                startIcon={<Stop />}
                onClick={stopWS}
              >
                Stop Transcription
              </Button>
            )}
          </Box>
        </Paper>

        <Paper 
          elevation={3} 
          sx={{ 
            p: 3, 
            height: '50vh', 
            mb: 3,
            overflowY: 'auto', 
            bgcolor: '#fafafa',
            border: '1px solid #e0e0e0'
          }}
        >
          <Typography variant="h5" gutterBottom>
            Transcription
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          {transcripts.length === 0 ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80%' }}>
              <Typography color="textSecondary">
                {connected ? 'Waiting for speech...' : 'Transcription will appear here'}
              </Typography>
            </Box>
          ) : (
            transcripts.map((t, i) => (
              <Box 
                key={i} 
                sx={{ 
                  mb: 2, 
                  display: 'flex', 
                  flexDirection: 'column'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                  <Box 
                    sx={{ 
                      width: 10, 
                      height: 10, 
                      borderRadius: '50%', 
                      bgcolor: getSpeakerColor(t.role),
                      mr: 1
                    }} 
                  />
                  <Typography 
                    variant="subtitle2" 
                    sx={{ 
                      color: getSpeakerColor(t.role),
                      fontWeight: 'bold'
                    }}
                  >
                    {t.role} [{t.timestamp.toFixed(1)}s]
                  </Typography>
                </Box>
                <Paper 
                  elevation={1} 
                  sx={{ 
                    p: 2, 
                    bgcolor: '#fff',
                    borderLeft: `4px solid ${getSpeakerColor(t.role)}`
                  }}
                >
                  <Typography>{t.text}</Typography>
                </Paper>
              </Box>
            ))
          )}
          <div ref={transcriptEndRef} />
        </Paper>
      </Container>
    </ThemeProvider>
  );
}

export default App;