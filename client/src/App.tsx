import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)
  const [health, setHealth] = useState<string>('');

  useEffect(() => {
    const fetchHealth = async () => {
      let serverOrigin = import.meta.env.VITE_API_ORIGIN;
      if(import.meta.env.PROD){
        serverOrigin = window.location.origin
      }

      try {
        const healthEndpoint = `${serverOrigin}/health`
        const response = await fetch(healthEndpoint);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.text();
        setHealth(data);
      } catch (error) {
        console.error('Fetch error:', error);
      }
    };

    fetchHealth();
  }, []);

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
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      <p>{health || "Loading..."}
      </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
