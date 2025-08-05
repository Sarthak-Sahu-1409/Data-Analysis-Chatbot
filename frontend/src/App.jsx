import React, { useState } from "react";
import axios from "axios";
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [chart, setChart] = useState(null);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);
    await axios.post("http://localhost:8000/upload", formData);
    alert("File uploaded!");
  };

  const handleQuery = async () => {
    const formData = new FormData();
    formData.append("query", query);
    const res = await axios.post("http://localhost:8000/query", formData);
    setAnswer(res.data.answer);
    setChart(res.data.chart);
  };

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
      <div style={{ padding: 32 }}>
      <h2>Data Analysis ChatBot</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      <br /><br />
      <input
        type="text"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Ask a question..."
        style={{ width: 400 }}
      />
      <button onClick={handleQuery}>Ask</button>
      <div style={{ marginTop: 32 }}>
        <h3>Answer:</h3>
        <div>{answer}</div>
        {chart && (
          <div>
            <h3>Chart:</h3>
            <img src={`data:image/png;base64,${chart}`} alt="Chart" />
          </div>
        )}
      </div>
    </div>
    </>
  )
}

export default App
