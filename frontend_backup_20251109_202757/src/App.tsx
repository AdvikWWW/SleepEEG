import React, { useState } from 'react';
import { Upload, Brain, BarChart3, Download, FileText, Zap, Activity, Users, Clock, Star, Lightbulb } from 'lucide-react';
import axios from 'axios';

// Define types for our application
type SleepStage = 'N1' | 'N2' | 'N3' | 'REM' | 'Wake';

interface AnalysisResult {
  stage: SleepStage;
  start: string;
  end: string;
  duration: number;
}

interface StatisticalResults {
  totalSleepTime: number;
  sleepEfficiency: number;
  sleepLatency: number;
  remLatency: number;
  wakeAfterSleepOnset: number;
  stagePercentages: Record<SleepStage, number>;
  stageCounts: Record<SleepStage, number>;
  powerSpectralDensity: {
    delta: number;
    theta: number;
    alpha: number;
    beta: number;
  };
  sleepQualityScore: number;
}

interface PopulationComparison {
  userVsAverage: Record<string, { user: number; average: number; percentile: number; is_normal: boolean; normal_range: number[] }>;
  normative_flags: string[];
}

interface NovelMetrics {
  sleep_fragmentation_index: number;
  rem_to_n3_ratio: number;
  delta_theta_ratio: number;
  sleep_consolidation_index: number;
  spindle_efficiency: number;
  architecture_stability: number;
}

interface ResearchSummary {
  title: string;
  abstract: string;
  key_findings: string[];
}

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [statisticalResults, setStatisticalResults] = useState<StatisticalResults | null>(null);
  const [populationComparison, setPopulationComparison] = useState<PopulationComparison | null>(null);
  const [novelMetrics, setNovelMetrics] = useState<NovelMetrics | null>(null);
  const [verificationFlags, setVerificationFlags] = useState<string[]>([]);
  const [researchSummary, setResearchSummary] = useState<ResearchSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [summaryReport, setSummaryReport] = useState<string>('');

  // Form state for EEG analysis
  const [subjectId, setSubjectId] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [memoryScore, setMemoryScore] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadedFile) return;

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('subject_id', subjectId);
    formData.append('age', age);
    formData.append('sex', sex);
    formData.append('memory_score', memoryScore);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(progress);
          }
        },
      });

      setAnalysisResults(response.data.analysis_results || []);
      setStatisticalResults(response.data.statistical_results || null);
      setPopulationComparison(response.data.population_comparison || null);
      setNovelMetrics(response.data.novel_metrics || null);
      setVerificationFlags(response.data.verification_flags || []);
      setResearchSummary(response.data.research_summary || null);
      setSummaryReport(response.data.summary_report || '');
      setActiveTab('results');
    } catch (err) {
      console.error('Error uploading file:', err);
      setError('Failed to analyze the EEG file. Please try again.');
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  // Helper function to get color for each sleep stage
  const getStageColor = (stage: SleepStage): string => {
    const colors: Record<SleepStage, string> = {
      'N1': '#3b82f6', // blue-500
      'N2': '#6366f1', // indigo-500
      'N3': '#8b5cf6', // violet-500
      'REM': '#ec4899', // pink-500
      'Wake': '#9ca3af', // gray-400
    };
    return colors[stage];
  };

  // Render the upload form
  const renderUploadForm = () => (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-6 text-center">Upload EEG Data</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Subject ID
            </label>
            <input
              type="text"
              value={subjectId}
              onChange={(e) => setSubjectId(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Age
              </label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Sex
              </label>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select...</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Memory Score (optional)
            </label>
            <input
              type="number"
              value={memoryScore}
              onChange={(e) => setMemoryScore(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              step="0.01"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              EEG Data File (EDF/CSV)
            </label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
              <div className="space-y-1 text-center">
                <div className="flex text-sm text-gray-600">
                  <label
                    htmlFor="file-upload"
                    className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none"
                  >
                    <span>Upload a file</span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      accept=".edf,.csv"
                      onChange={handleFileChange}
                      required
                    />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">EDF or CSV files up to 100MB</p>
                {uploadedFile && (
                  <p className="text-sm text-green-600">
                    Selected: {uploadedFile.name} ({(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB)
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {loading ? (
          <div className="pt-2">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600 mt-2 text-center">
              Analyzing EEG data... {uploadProgress}%
            </p>
          </div>
        ) : (
          <button
            type="submit"
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            disabled={!uploadedFile}
          >
            Analyze EEG Data
          </button>
        )}

        {error && (
          <div className="rounded-md bg-red-50 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg
                  className="h-5 w-5 text-red-400"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">{error}</h3>
              </div>
            </div>
          </div>
        )}
      </form>
    </div>
  );

  // Render the analysis results
  const renderResults = () => {
    if (!statisticalResults) return null;

    return (
      <div className="p-6 max-w-6xl mx-auto">
        <h2 className="text-2xl font-bold mb-6">Sleep Analysis Results</h2>
        
        {/* Research Summary Banner */}
        {researchSummary && (
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg shadow-lg mb-8">
            <h3 className="text-xl font-bold mb-2">{researchSummary.title}</h3>
            <p className="text-blue-100 mb-4">{researchSummary.abstract}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {researchSummary.key_findings.map((finding, index) => (
                <div key={index} className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-2"></div>
                  <span className="text-sm">{finding}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Verification Flags */}
        {verificationFlags.length > 0 && (
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-8">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800">Data Quality Flags</h3>
                <div className="mt-2 text-sm text-yellow-700">
                  <ul className="list-disc list-inside space-y-1">
                    {verificationFlags.map((flag, index) => (
                      <li key={index}>{flag}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600 mr-4">
                <Clock className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Sleep Time</p>
                <p className="text-2xl font-semibold">
                  {(statisticalResults.totalSleepTime / 60).toFixed(1)} hrs
                </p>
                <p className="text-xs text-gray-400">
                  {statisticalResults.totalSleepTime.toFixed(1)} minutes
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 text-green-600 mr-4">
                <Activity className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Sleep Efficiency</p>
                <p className="text-2xl font-semibold">
                  {statisticalResults.sleepEfficiency.toFixed(1)}%
                </p>
                {populationComparison?.userVsAverage?.sleep_efficiency && (
                  <p className="text-xs text-gray-400">
                    {populationComparison.userVsAverage.sleep_efficiency.is_normal ? '✅ Normal' : '⚠️ Abnormal'}
                  </p>
                )}
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100 text-purple-600 mr-4">
                <Zap className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm text-gray-500">REM Latency</p>
                <p className="text-2xl font-semibold">
                  {statisticalResults.remLatency.toFixed(1)} min
                </p>
                {populationComparison?.userVsAverage?.rem_latency && (
                  <p className="text-xs text-gray-400">
                    {populationComparison.userVsAverage.rem_latency.is_normal ? '✅ Normal' : '⚠️ Abnormal'}
                  </p>
                )}
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-yellow-100 text-yellow-600 mr-4">
                <Star className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Sleep Quality Score</p>
                <p className="text-2xl font-semibold">
                  {statisticalResults.sleepQualityScore.toFixed(1)}/100
                </p>
                <p className="text-xs text-gray-400">
                  {statisticalResults.sleepQualityScore >= 75 ? '✅ Good' : 
                   statisticalResults.sleepQualityScore >= 50 ? '⚠️ Fair' : '❌ Poor'}
                </p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Sleep Stage Distribution */}
        <div className="bg-white p-6 rounded-lg shadow mb-8">
          <h3 className="text-lg font-medium mb-4">Sleep Stage Distribution</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-3">Time Spent (%)</h4>
              <div className="space-y-3">
                {Object.entries(statisticalResults.stagePercentages).map(([stage, percentage]) => (
                  <div key={stage} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{stage}</span>
                      <span className="text-gray-600">{percentage.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div
                        className="h-2.5 rounded-full transition-all duration-500"
                        style={{
                          width: `${percentage}%`,
                          backgroundColor: getStageColor(stage as SleepStage)
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-500 mb-3">Stage Counts</h4>
              <div className="space-y-3">
                {Object.entries(statisticalResults.stageCounts).map(([stage, count]) => (
                  <div key={stage} className="flex items-center">
                    <div
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: getStageColor(stage as SleepStage) }}
                    ></div>
                    <span className="text-sm font-medium mr-2">{stage}:</span>
                    <span className="text-sm text-gray-600">{count} epochs</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Novel Neuroscience Metrics */}
        {novelMetrics && (
          <div className="bg-white p-6 rounded-lg shadow mb-8">
            <h3 className="text-lg font-medium mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-600" />
              Advanced Neuroscience Metrics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="p-4 bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg border border-purple-200">
                <div className="text-sm text-purple-600 font-medium mb-1">Sleep Fragmentation Index</div>
                <div className="text-2xl font-bold text-purple-700 mb-1">
                  {novelMetrics.sleep_fragmentation_index.toFixed(1)}
                </div>
                <div className="text-xs text-gray-600">transitions/hour</div>
                <div className="text-xs text-purple-600 mt-2">
                  {novelMetrics.sleep_fragmentation_index > 15 ? '⚠️ High fragmentation' : 
                   novelMetrics.sleep_fragmentation_index > 10 ? '⚠️ Moderate' : '✅ Low fragmentation'}
                </div>
              </div>

              <div className="p-4 bg-gradient-to-br from-green-50 to-teal-50 rounded-lg border border-green-200">
                <div className="text-sm text-green-600 font-medium mb-1">REM/Deep Sleep Ratio</div>
                <div className="text-2xl font-bold text-green-700 mb-1">
                  {novelMetrics.rem_to_n3_ratio.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600">cognitive vs restorative</div>
                <div className="text-xs text-green-600 mt-2">
                  {novelMetrics.rem_to_n3_ratio > 2 ? '⚠️ REM dominant' : 
                   novelMetrics.rem_to_n3_ratio < 0.5 ? '⚠️ Deep sleep dominant' : '✅ Balanced'}
                </div>
              </div>

              <div className="p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-lg border border-orange-200">
                <div className="text-sm text-orange-600 font-medium mb-1">Delta/Theta Ratio</div>
                <div className="text-2xl font-bold text-orange-700 mb-1">
                  {novelMetrics.delta_theta_ratio.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600">sleep depth indicator</div>
                <div className="text-xs text-orange-600 mt-2">
                  {novelMetrics.delta_theta_ratio > 3 ? '✅ Deep sleep' : '⚠️ Light sleep'}
                </div>
              </div>

              <div className="p-4 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg border border-indigo-200">
                <div className="text-sm text-indigo-600 font-medium mb-1">Sleep Consolidation</div>
                <div className="text-2xl font-bold text-indigo-700 mb-1">
                  {novelMetrics.sleep_consolidation_index.toFixed(1)}
                </div>
                <div className="text-xs text-gray-600">continuity index</div>
                <div className="text-xs text-indigo-600 mt-2">
                  {novelMetrics.sleep_consolidation_index > 1.5 ? '✅ Well consolidated' : '⚠️ Fragmented'}
                </div>
              </div>

              <div className="p-4 bg-gradient-to-br from-pink-50 to-rose-50 rounded-lg border border-pink-200">
                <div className="text-sm text-pink-600 font-medium mb-1">Spindle Efficiency</div>
                <div className="text-2xl font-bold text-pink-700 mb-1">
                  {novelMetrics.spindle_efficiency.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600">memory consolidation</div>
                <div className="text-xs text-pink-600 mt-2">
                  {novelMetrics.spindle_efficiency > 2 ? '✅ High efficiency' : '⚠️ Low efficiency'}
                </div>
              </div>

              <div className="p-4 bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg border border-cyan-200">
                <div className="text-sm text-cyan-600 font-medium mb-1">Architecture Stability</div>
                <div className="text-2xl font-bold text-cyan-700 mb-1">
                  {(novelMetrics.architecture_stability * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-600">pattern consistency</div>
                <div className="text-xs text-cyan-600 mt-2">
                  {novelMetrics.architecture_stability > 0.8 ? '✅ Stable patterns' : '⚠️ Unstable patterns'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Power Spectral Density */}
        <div className="bg-white p-6 rounded-lg shadow mb-8">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-blue-600" />
            EEG Power Spectral Density
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
              <div className="text-2xl font-bold text-red-600">
                {statisticalResults.powerSpectralDensity.delta.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 font-medium">Delta (0.5-4 Hz)</div>
              <div className="text-xs text-red-600 mt-1">Deep Sleep Waves</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
              <div className="text-2xl font-bold text-orange-600">
                {statisticalResults.powerSpectralDensity.theta.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 font-medium">Theta (4-8 Hz)</div>
              <div className="text-xs text-orange-600 mt-1">REM & Drowsiness</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-600">
                {statisticalResults.powerSpectralDensity.alpha.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 font-medium">Alpha (8-13 Hz)</div>
              <div className="text-xs text-green-600 mt-1">Relaxed Wakefulness</div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">
                {statisticalResults.powerSpectralDensity.beta.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600 font-medium">Beta (13-30 Hz)</div>
              <div className="text-xs text-blue-600 mt-1">Active Wakefulness</div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-600">
              <strong>Neuroscience Context:</strong> Delta waves dominate deep sleep and are crucial for memory consolidation. 
              Theta activity increases during REM sleep. Alpha waves indicate relaxed states, while beta waves suggest alertness.
            </p>
          </div>
        </div>
        
        {/* Hypnogram Visualization */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">Sleep Architecture (Hypnogram)</h3>
          <div className="h-64 bg-gray-100 rounded-md p-4 flex items-center justify-center">
            <div className="text-center">
              <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Interactive hypnogram visualization</p>
              <p className="text-sm text-gray-400 mt-2">
                Shows sleep stage transitions over time
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render population comparison
  const renderPopulationComparison = () => {
    if (!populationComparison || !statisticalResults) return null;

    return (
      <div className="p-6 max-w-6xl mx-auto">
        <h2 className="text-2xl font-bold mb-6">Population Comparison</h2>
        
        {/* Normative Flags */}
        {populationComparison?.normative_flags && populationComparison.normative_flags.length > 0 && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-8">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Abnormal Sleep Metrics</h3>
                <div className="mt-2 text-sm text-red-700">
                  <ul className="list-disc list-inside space-y-1">
                    {populationComparison.normative_flags.map((flag, index) => (
                      <li key={index}>{flag}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Population Comparison */}
        <div className="bg-white p-6 rounded-lg shadow mb-8">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-green-600" />
            Normative Data Comparison
          </h3>
          <div className="space-y-6">
            {Object.entries(populationComparison.userVsAverage).map(([metric, data]) => (
              <div key={metric} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-medium capitalize">{metric.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">
                      {data.percentile.toFixed(1)}th percentile
                    </span>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      data.is_normal ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {data.is_normal ? '✅ Normal' : '⚠️ Abnormal'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Your Value: <strong>{data.user.toFixed(1)}</strong></span>
                      <span>Normal Range: {data.normal_range[0]}-{data.normal_range[1]}</span>
                      <span>Population Average: {data.average.toFixed(1)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3 relative">
                      {/* Normal range indicator */}
                      <div 
                        className="absolute bg-green-200 h-3 rounded-full"
                        style={{ 
                          left: `${(data.normal_range[0] / (data.normal_range[1] * 1.2)) * 100}%`,
                          width: `${((data.normal_range[1] - data.normal_range[0]) / (data.normal_range[1] * 1.2)) * 100}%`
                        }}
                      ></div>
                      {/* User value indicator */}
                      <div
                        className={`h-3 rounded-full ${data.is_normal ? 'bg-green-600' : 'bg-red-600'}`}
                        style={{ width: `${Math.min(100, Math.max(5, (data.user / (data.normal_range[1] * 1.2)) * 100))}%` }}
                      ></div>
                      {/* Population average marker */}
                      <div 
                        className="absolute w-1 h-3 bg-blue-600 rounded"
                        style={{ left: `${(data.average / (data.normal_range[1] * 1.2)) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // Render sleep tips
  const renderSleepTips = () => (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Sleep Improvement Tips</h2>
      
      <div className="grid gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-start">
            <div className="p-3 rounded-full bg-blue-100 text-blue-600 mr-4">
              <Lightbulb className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-medium mb-2">Sleep Hygiene</h3>
              <p className="text-gray-600 mb-4">
                Maintain a consistent sleep schedule and create a relaxing bedtime routine.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                <li>Go to bed and wake up at the same time every day</li>
                <li>Keep your bedroom cool, dark, and quiet</li>
                <li>Avoid screens 1 hour before bedtime</li>
                <li>Limit caffeine intake after 2 PM</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-start">
            <div className="p-3 rounded-full bg-green-100 text-green-600 mr-4">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-medium mb-2">Exercise & Diet</h3>
              <p className="text-gray-600 mb-4">
                Regular exercise and proper nutrition can significantly improve sleep quality.
              </p>
              <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                <li>Exercise regularly, but not close to bedtime</li>
                <li>Avoid large meals 2-3 hours before sleep</li>
                <li>Consider magnesium or melatonin supplements</li>
                <li>Stay hydrated, but limit fluids before bed</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Render summary report
  const renderSummaryReport = () => (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Summary Report</h2>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="prose max-w-none">
          {summaryReport ? (
            <div className="whitespace-pre-wrap text-gray-700">
              {summaryReport}
            </div>
          ) : (
            <div className="text-center py-8">
              <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">
                Upload and analyze EEG data to generate a comprehensive summary report.
              </p>
            </div>
          )}
        </div>
        
        {summaryReport && (
          <div className="mt-6 pt-6 border-t border-gray-200">
            <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
              <Download className="w-4 h-4 mr-2" />
              Download PDF Report
            </button>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900 flex items-center">
            <Brain className="w-8 h-8 mr-3 text-blue-600" />
            Sleep EEG Analysis
          </h1>
        </div>
      </header>
      
      <main>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="bg-white shadow rounded-lg overflow-hidden mb-8">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'upload'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-center">
                    <Upload className="w-5 h-5 mr-2" />
                    Upload
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('results')}
                  disabled={!analysisResults.length && !statisticalResults}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'results'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed'
                  }`}
                >
                  <div className="flex items-center justify-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Results
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('population')}
                  disabled={!populationComparison}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'population'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 disabled:opacity-50 disabled:cursor-not-allowed'
                  }`}
                >
                  <div className="flex items-center justify-center">
                    <Users className="w-5 h-5 mr-2" />
                    Population Comparison
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('tips')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'tips'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-center">
                    <Lightbulb className="w-5 h-5 mr-2" />
                    Sleep Tips
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('report')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeTab === 'report'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-center">
                    <FileText className="w-5 h-5 mr-2" />
                    Summary Report
                  </div>
                </button>
              </nav>
            </div>
            
            <div className="p-0">
              {activeTab === 'upload' && renderUploadForm()}
              {activeTab === 'results' && renderResults()}
              {activeTab === 'population' && renderPopulationComparison()}
              {activeTab === 'tips' && renderSleepTips()}
              {activeTab === 'report' && renderSummaryReport()}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
