import React, { useState } from 'react';

function SimpleApp() {
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('subject_id', 'TEST001');
    formData.append('age', '35');
    formData.append('sex', 'M');

    try {
      setError(null);
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze EEG data');
      }
      
      const data = await response.json();
      console.log('Analysis results:', data);
      setResults(data);
      setActiveTab('results');
    } catch (error: any) {
      console.error('Error:', error);
      setError(error.message || 'Failed to analyze EEG data. Please check the file format and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ color: '#2563eb', fontSize: '2.5rem', marginBottom: '10px' }}>
          🧠 Sleep EEG Analysis
        </h1>
        <p style={{ color: '#6b7280', fontSize: '1.1rem' }}>
          Upload your EEG data for comprehensive sleep analysis
        </p>
      </header>

      <div style={{ backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
        <nav style={{ borderBottom: '1px solid #e5e7eb', padding: '0' }}>
          <div style={{ display: 'flex', gap: '0' }}>
            {['upload', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: '16px 24px',
                  border: 'none',
                  backgroundColor: 'transparent',
                  borderBottom: activeTab === tab ? '2px solid #2563eb' : '2px solid transparent',
                  color: activeTab === tab ? '#2563eb' : '#6b7280',
                  fontWeight: '500',
                  cursor: 'pointer',
                  textTransform: 'capitalize'
                }}
              >
                {tab === 'upload' ? '📤 Upload' : '📊 Results'}
              </button>
            ))}
          </div>
        </nav>

        <div style={{ padding: '24px' }}>
          {activeTab === 'upload' && (
            <div style={{ maxWidth: '600px', margin: '0 auto' }}>
              <h2 style={{ fontSize: '1.5rem', marginBottom: '24px', textAlign: 'center' }}>
                Upload EEG Data
              </h2>
              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                    Select EEG File (CSV format)
                  </label>
                  <input
                    type="file"
                    accept=".csv,.edf"
                    onChange={handleFileUpload}
                    style={{
                      width: '100%',
                      padding: '12px',
                      border: '2px dashed #d1d5db',
                      borderRadius: '8px',
                      backgroundColor: '#f9fafb'
                    }}
                  />
                </div>
                
                {file && (
                  <div style={{ 
                    backgroundColor: '#f0f9ff', 
                    padding: '16px', 
                    borderRadius: '8px',
                    border: '1px solid #bae6fd'
                  }}>
                    <p style={{ margin: '0', color: '#0369a1' }}>
                      ✅ File selected: {file.name}
                    </p>
                  </div>
                )}

                {error && (
                  <div style={{ 
                    backgroundColor: '#fee2e2', 
                    padding: '16px', 
                    borderRadius: '8px',
                    border: '1px solid #fca5a5'
                  }}>
                    <p style={{ margin: '0', color: '#dc2626', fontWeight: '500' }}>
                      ❌ {error}
                    </p>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={!file || loading}
                  style={{
                    padding: '12px 24px',
                    backgroundColor: !file || loading ? '#d1d5db' : '#2563eb',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '1rem',
                    fontWeight: '500',
                    cursor: !file || loading ? 'not-allowed' : 'pointer',
                    transition: 'background-color 0.2s'
                  }}
                >
                  {loading ? '⏳ Analyzing...' : '🚀 Analyze EEG Data'}
                </button>
              </form>
            </div>
          )}

          {activeTab === 'results' && (
            <div>
              {results ? (
                <div>
                  <h2 style={{ fontSize: '1.5rem', marginBottom: '24px', textAlign: 'center' }}>
                    📊 Analysis Results
                  </h2>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                    <div style={{ backgroundColor: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                      <h3 style={{ color: '#1e40af', marginBottom: '16px', fontSize: '1.2rem' }}>💤 Sleep Statistics</h3>
                      {results.statistical_results && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                          <p style={{ margin: 0 }}><strong>Sleep Efficiency:</strong> {results.statistical_results.sleepEfficiency?.toFixed(1)}%</p>
                          <p style={{ margin: 0 }}><strong>Total Sleep Time:</strong> {(results.statistical_results.totalSleepTime / 60)?.toFixed(1)} hours</p>
                          <p style={{ margin: 0 }}><strong>Sleep Quality Score:</strong> {results.statistical_results.sleepQualityScore}/100</p>
                          <p style={{ margin: 0 }}><strong>Sleep Onset Latency:</strong> {results.statistical_results.sleepOnsetLatency?.toFixed(1)} min</p>
                          <p style={{ margin: 0 }}><strong>REM Latency:</strong> {results.statistical_results.remLatency?.toFixed(1)} min</p>
                          <p style={{ margin: 0 }}><strong>Awakenings:</strong> {results.statistical_results.awakenings}</p>
                        </div>
                      )}
                    </div>
                    
                    <div style={{ backgroundColor: '#f0fdf4', padding: '20px', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
                      <h3 style={{ color: '#166534', marginBottom: '16px', fontSize: '1.2rem' }}>🌙 Sleep Stages</h3>
                      {results.statistical_results?.stagePercentages && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                          {Object.entries(results.statistical_results.stagePercentages).map(([stage, percentage]) => (
                            <p key={stage} style={{ margin: 0 }}><strong>{stage}:</strong> {(percentage as number)?.toFixed(1)}%</p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Novel Neuroscience Metrics */}
                  {results.novel_metrics && (
                    <div style={{ marginBottom: '20px' }}>
                      <h3 style={{ fontSize: '1.3rem', marginBottom: '16px', color: '#7c3aed' }}>🧠 Advanced Neuroscience Metrics</h3>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
                        <div style={{ backgroundColor: '#faf5ff', padding: '16px', borderRadius: '8px', border: '1px solid #e9d5ff' }}>
                          <h4 style={{ color: '#7c3aed', margin: '0 0 8px 0', fontSize: '0.95rem' }}>Sleep Fragmentation Index</h4>
                          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0', color: '#6b21a8' }}>{results.novel_metrics.sleep_fragmentation_index?.toFixed(2)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Transitions per hour</p>
                        </div>
                        <div style={{ backgroundColor: '#fef3c7', padding: '16px', borderRadius: '8px', border: '1px solid #fde68a' }}>
                          <h4 style={{ color: '#d97706', margin: '0 0 8px 0', fontSize: '0.95rem' }}>REM-to-N3 Ratio</h4>
                          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0', color: '#92400e' }}>{results.novel_metrics.rem_to_n3_ratio?.toFixed(2)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Cognitive vs restorative balance</p>
                        </div>
                        <div style={{ backgroundColor: '#dbeafe', padding: '16px', borderRadius: '8px', border: '1px solid #bfdbfe' }}>
                          <h4 style={{ color: '#1d4ed8', margin: '0 0 8px 0', fontSize: '0.95rem' }}>Delta/Theta Ratio</h4>
                          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0', color: '#1e40af' }}>{results.novel_metrics.delta_theta_ratio?.toFixed(2)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Sleep depth indicator</p>
                        </div>
                        <div style={{ backgroundColor: '#dcfce7', padding: '16px', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
                          <h4 style={{ color: '#15803d', margin: '0 0 8px 0', fontSize: '0.95rem' }}>Sleep Consolidation Index</h4>
                          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0', color: '#166534' }}>{results.novel_metrics.sleep_consolidation_index?.toFixed(2)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Sleep continuity measure</p>
                        </div>
                        <div style={{ backgroundColor: '#fce7f3', padding: '16px', borderRadius: '8px', border: '1px solid #fbcfe8' }}>
                          <h4 style={{ color: '#be185d', margin: '0 0 8px 0', fontSize: '0.95rem' }}>Spindle Efficiency</h4>
                          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0', color: '#9f1239' }}>{results.novel_metrics.spindle_efficiency?.toFixed(2)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Memory consolidation marker</p>
                        </div>
                        <div style={{ backgroundColor: '#fff7ed', padding: '16px', borderRadius: '8px', border: '1px solid #fed7aa' }}>
                          <h4 style={{ color: '#c2410c', margin: '0 0 8px 0', fontSize: '0.95rem' }}>Architecture Stability</h4>
                          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '0', color: '#9a3412' }}>{results.novel_metrics.architecture_stability?.toFixed(2)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Sleep pattern consistency</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* EEG Power Spectral Density */}
                  {results.statistical_results?.powerSpectralDensity && (
                    <div style={{ marginBottom: '20px', backgroundColor: '#f0f9ff', padding: '20px', borderRadius: '8px', border: '1px solid #bae6fd' }}>
                      <h3 style={{ fontSize: '1.3rem', marginBottom: '16px', color: '#0369a1' }}>⚡ EEG Power Spectral Density</h3>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                        <div style={{ textAlign: 'center' }}>
                          <h4 style={{ color: '#0c4a6e', margin: '0 0 8px 0' }}>Delta (0.5-4 Hz)</h4>
                          <p style={{ fontSize: '1.8rem', fontWeight: 'bold', margin: '0', color: '#075985' }}>{results.statistical_results.powerSpectralDensity.delta?.toFixed(1)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Deep sleep waves</p>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <h4 style={{ color: '#0c4a6e', margin: '0 0 8px 0' }}>Theta (4-8 Hz)</h4>
                          <p style={{ fontSize: '1.8rem', fontWeight: 'bold', margin: '0', color: '#075985' }}>{results.statistical_results.powerSpectralDensity.theta?.toFixed(1)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Light sleep & REM</p>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <h4 style={{ color: '#0c4a6e', margin: '0 0 8px 0' }}>Alpha (8-13 Hz)</h4>
                          <p style={{ fontSize: '1.8rem', fontWeight: 'bold', margin: '0', color: '#075985' }}>{results.statistical_results.powerSpectralDensity.alpha?.toFixed(1)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Relaxed wakefulness</p>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <h4 style={{ color: '#0c4a6e', margin: '0 0 8px 0' }}>Beta (13-30 Hz)</h4>
                          <p style={{ fontSize: '1.8rem', fontWeight: 'bold', margin: '0', color: '#075985' }}>{results.statistical_results.powerSpectralDensity.beta?.toFixed(1)}</p>
                          <p style={{ fontSize: '0.85rem', color: '#6b7280', margin: '4px 0 0 0' }}>Active thinking</p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {results.summary_report && (
                    <div style={{ marginTop: '24px', backgroundColor: '#fefce8', padding: '20px', borderRadius: '8px' }}>
                      <h3 style={{ color: '#a16207', marginBottom: '16px' }}>📋 Summary Report</h3>
                      <pre style={{ whiteSpace: 'pre-wrap', fontSize: '14px', lineHeight: '1.5' }}>
                        {results.summary_report}
                      </pre>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#6b7280' }}>
                  <p>No analysis results yet. Please upload an EEG file first.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default SimpleApp;
