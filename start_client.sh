#!/bin/bash

# Change to client directory
cd client

# Set API URL environment variable to connect to the server
export REACT_APP_API_URL=http://localhost:8000

# Start React application in development mode
echo "Starting React client application..."
npm start 