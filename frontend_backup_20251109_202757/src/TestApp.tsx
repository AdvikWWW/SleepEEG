import React from 'react';

function TestApp() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1 style={{ color: '#333' }}>Sleep EEG Analysis App - Test Page</h1>
      <p>If you can see this, React is working!</p>
      <div style={{ 
        backgroundColor: '#f0f0f0', 
        padding: '20px', 
        borderRadius: '8px',
        margin: '20px 0'
      }}>
        <h2>Test Features:</h2>
        <ul>
          <li>✅ React is loading</li>
          <li>✅ TypeScript is working</li>
          <li>✅ Basic styling is applied</li>
        </ul>
      </div>
      <button 
        style={{
          backgroundColor: '#007bff',
          color: 'white',
          padding: '10px 20px',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
        onClick={() => alert('Button clicked! Frontend is working.')}
      >
        Test Button
      </button>
    </div>
  );
}

export default TestApp;
