# Real-Time Transcription React Client

This is the frontend client for the Real-Time Whisper application, built with React and Material UI.

## Features

- **Modern Material UI Interface**: Clean, professional design with intuitive controls
- **Real-time Transcription Display**: Visualize speech as it happens with color-coded speaker identification
- **Language Selection**: Choose from multiple languages for transcription
- **Speaker Count Configuration**: Optimize speaker diarization by specifying the number of speakers
- **Responsive Design**: Works well on desktop and mobile devices

## Installation

In the project directory, run:

```bash
npm install
```

This will install all dependencies, including React, Material UI, and other required packages.

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

## Usage

1. Make sure the backend server is running (see main README.md)
2. Start the React client with `npm start`
3. In the web interface:
   - Select your preferred language from the dropdown
   - Choose the number of speakers expected in the conversation
   - Click "Start Transcription" to begin
   - Speak into your microphone
   - View transcriptions in real-time with color-coded speaker identification
   - Click "Stop Transcription" when finished

## Configuration

The client connects to a WebSocket server at `ws://localhost:8000/ws` by default. If you need to change this URL (for example, if your backend is running on a different host or port), you can modify it in the `App.js` file.

## Dependencies

- React
- Material UI (@mui/material, @mui/icons-material)
- Emotion (@emotion/react, @emotion/styled)

## Troubleshooting

- **Connection issues**: Ensure the backend server is running and accessible
- **Microphone access**: Make sure you've granted microphone permissions in your browser
- **UI rendering problems**: Check that Material UI components are properly installed
