import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPredictionResult(null); // Clear previous results
      setError(null); // Clear previous errors
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setSelectedFile(null);
      setPreview(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Replace with your actual backend API endpoint
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Try to get error message from backend if available
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      console.error("Error uploading file:", err);
      setError(err.message || "Failed to get prediction. Check console and backend logs.");
      setPredictionResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🖼️ Smart Image Classifier 🧠</h1>
        <p>Upload an image and let SqueezeNet tell you what it sees!</p>
      </header>
      <main className="App-main">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label htmlFor="fileUpload" className="file-label">
              {selectedFile ? selectedFile.name : 'Click to Select Image'}
            </label>
            <input 
              type="file" 
              id="fileUpload"
              onChange={handleFileChange} 
              accept="image/png, image/jpeg, image/gif" 
            />
          </div>
          <button type="submit" disabled={!selectedFile || isLoading} className="submit-button">
            {isLoading ? 'Analyzing...' : 'Classify Image'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <p>Error: {error}</p>
          </div>
        )}

        <div className="results-container">
          {preview && (
            <div className="image-preview-container">
              <h2>Image Preview:</h2>
              <img src={preview} alt="Selected preview" className="image-preview" />
            </div>
          )}

          {predictionResult && (
            <div className="prediction-result-container">
              <h2>Prediction Result:</h2>
              <div className="prediction-card">
                <p><strong>File:</strong> {predictionResult.filename}</p>
                <p><strong>Predicted Class:</strong> <span className="predicted-class">{predictionResult.predicted_class}</span></p>
                <p><strong>Confidence:</strong> <span className="confidence-score">{(predictionResult.confidence * 100).toFixed(2)}%</span></p>
                <p><small>Model: {predictionResult.model_used}</small></p>
                {predictionResult.saved_filename && 
                  <p><small>Saved as: {predictionResult.saved_filename}</small></p>
                }
              </div>
            </div>
          )}
        </div>
      </main>
      <footer className="App-footer">
        <p>Powered by FastAPI, React, and PyTorch (SqueezeNet)</p>
      </footer>
    </div>
  );
}

export default App;
