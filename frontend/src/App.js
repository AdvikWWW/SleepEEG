import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { Upload, Brain, BarChart3, Download, FileText, Database, Zap, Activity, TrendingUp, Users, Clock, Microscope } from 'lucide-react';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Separator } from './components/ui/separator';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [statisticalResults, setStatisticalResults] = useState(null);
  const [visualizations, setVisualizations] = useState({});
  const [loading, setLoading] = useState(false);
  const [datasets, setDatasets] = useState([]);
  
  // Form data for EEG analysis
  const [subjectId, setSubjectId] = useState('');
  const [memoryScore, setMemoryScore] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');

  useEffect(() => {
    fetchDatasets();
    fetchAnalysisResults();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const fetchAnalysisResults = async () => {
    try {
      const response = await axios.get(`${API}/analysis-results`);
      setAnalysisResults(response.data);
    } catch (error) {
      console.error('Error fetching analysis results:', error);
    }
  };

  const fetchStatisticalAnalysis = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/statistical-analysis`);
      setStatisticalResults(response.data);
    } catch (error) {
      console.error('Error fetching statistical analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchVisualizations = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/visualizations`);
      setVisualizations(response.data);
    } catch (error) {
      console.error('Error fetching visualizations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && (file.name.endsWith('.edf') || file.name.endsWith('.bdf'))) {
      setUploadedFile(file);
    } else {
      alert('Please upload a valid EDF or BDF file');
    }
  };

  const analyzeEEG = async () => {
    if (!uploadedFile || !subjectId) {
      alert('Please select a file and enter subject ID');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('subject_id', subjectId);
    if (memoryScore) formData.append('memory_score', memoryScore);
    if (age) formData.append('age', age);
    if (sex) formData.append('sex', sex);

    try {
      const response = await axios.post(`${API}/analyze-eeg`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setAnalysisResults([...analysisResults, response.data]);
      setUploadedFile(null);
      setSubjectId('');
      setMemoryScore('');
      setAge('');
      setSex('');
      
      alert('Analysis completed successfully!');
    } catch (error) {
      console.error('Error analyzing EEG:', error);
      alert('Analysis failed. Please check your file format.');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    try {
      const response = await axios.get(`${API}/download-report`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sleep_spindle_analysis_report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  const downloadJupyter = async () => {
    try {
      const response = await axios.get(`${API}/download-jupyter`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sleep_spindle_analysis.ipynb');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading notebook:', error);
    }
  };

  const renderPlotlyChart = (plotData) => {
    if (!plotData) return null;
    
    // This is a simplified version - in a real app you'd use react-plotly.js
    return (
      <div className="w-full h-96 bg-gray-50 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-300">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 mx-auto text-gray-400 mb-2" />
          <p className="text-gray-600">Interactive Chart Available</p>
          <p className="text-sm text-gray-500">Chart data loaded successfully</p>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-r from-slate-900 via-purple-900 to-slate-800">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative container mx-auto px-6 py-24">
          <div className="flex flex-col lg:flex-row items-center gap-12">
            <div className="flex-1 text-center lg:text-left">
              <div className="flex items-center justify-center lg:justify-start mb-6">
                <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl mr-4">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <span className="text-2xl font-bold text-white tracking-wide">NeuroSleep Research</span>
              </div>
              
              <h1 className="text-4xl lg:text-6xl font-bold text-white mb-6 leading-tight">
                Advanced EEG Sleep 
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400"> Analysis</span>
              </h1>
              
              <p className="text-xl text-gray-300 mb-8 leading-relaxed max-w-2xl">
                Comprehensive platform for analyzing sleep spindles, REM theta power, and memory consolidation. 
                Built for researchers and students with publication-ready results.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Button 
                  size="lg" 
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3 rounded-full font-semibold shadow-lg transition-all duration-300 hover:shadow-xl hover:scale-105"
                  onClick={() => setActiveTab('upload')}
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Start Analysis
                </Button>
                <Button 
                  variant="outline" 
                  size="lg"
                  className="border-2 border-white/20 text-white hover:bg-white/10 backdrop-blur-sm px-8 py-3 rounded-full font-semibold transition-all duration-300"
                  onClick={() => setActiveTab('datasets')}
                >
                  <Database className="w-5 h-5 mr-2" />
                  Browse Datasets
                </Button>
              </div>
            </div>
            
            <div className="flex-1 relative">
              <div className="relative w-full max-w-lg mx-auto">
                <img 
                  src="https://images.unsplash.com/photo-1549925245-f20a1bac6454" 
                  alt="Neuroscience Research"
                  className="w-full h-96 object-cover rounded-2xl shadow-2xl border border-white/10"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-purple-900/30 to-transparent rounded-2xl"></div>
                
                {/* Floating stats */}
                <div className="absolute -bottom-6 -left-6 bg-white/90 backdrop-blur-md rounded-xl p-4 shadow-lg border border-white/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Activity className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-gray-800">Sleep Spindles</p>
                      <p className="text-xs text-gray-600">11-16 Hz Analysis</p>
                    </div>
                  </div>
                </div>
                
                <div className="absolute -top-6 -right-6 bg-white/90 backdrop-blur-md rounded-xl p-4 shadow-lg border border-white/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <Zap className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-gray-800">REM Theta</p>
                      <p className="text-xs text-gray-600">4-8 Hz Power</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          <Card className="group hover:shadow-xl transition-all duration-300 border-0 shadow-lg bg-white/80 backdrop-blur-sm">
            <CardHeader className="text-center pb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Microscope className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-xl font-bold text-gray-800">Research Grade</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-gray-600">Publication-ready analysis using MNE-Python and YASA toolboxes with statistical validation.</p>
            </CardContent>
          </Card>

          <Card className="group hover:shadow-xl transition-all duration-300 border-0 shadow-lg bg-white/80 backdrop-blur-sm">
            <CardHeader className="text-center pb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Users className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-xl font-bold text-gray-800">Educational</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-gray-600">Perfect learning tool for students with interactive visualizations and downloadable notebooks.</p>
            </CardContent>
          </Card>

          <Card className="group hover:shadow-xl transition-all duration-300 border-0 shadow-lg bg-white/80 backdrop-blur-sm">
            <CardHeader className="text-center pb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-green-500 to-teal-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <TrendingUp className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-xl font-bold text-gray-800">Comprehensive</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-gray-600">Complete pipeline from EEG upload to statistical analysis with correlation and regression modeling.</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Interface */}
        <Card className="shadow-2xl border-0 bg-white/90 backdrop-blur-sm">
          <CardHeader className="border-b bg-gradient-to-r from-gray-50 to-blue-50">
            <CardTitle className="text-2xl font-bold text-gray-800 flex items-center">
              <Brain className="w-6 h-6 mr-3 text-blue-600" />
              Analysis Dashboard
            </CardTitle>
            <CardDescription className="text-lg text-gray-600">
              Upload EEG data, analyze sleep architecture, and generate research reports
            </CardDescription>
          </CardHeader>
          
          <CardContent className="p-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-5 mb-8 bg-gray-100 rounded-xl p-1">
                <TabsTrigger value="upload" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload
                </TabsTrigger>
                <TabsTrigger value="results" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Activity className="w-4 h-4 mr-2" />
                  Results
                </TabsTrigger>
                <TabsTrigger value="statistics" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Statistics
                </TabsTrigger>
                <TabsTrigger value="visualizations" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Charts
                </TabsTrigger>
                <TabsTrigger value="datasets" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Database className="w-4 h-4 mr-2" />
                  Datasets
                </TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-6">
                    <div className="border-2 border-dashed border-blue-300 bg-blue-50/30 rounded-xl p-8 text-center hover:border-blue-400 transition-colors">
                      <Upload className="w-12 h-12 mx-auto text-blue-500 mb-4" />
                      <h3 className="text-lg font-semibold text-gray-800 mb-2">Upload EEG File</h3>
                      <p className="text-gray-600 mb-4">Select EDF or BDF format files</p>
                      <Input
                        type="file"
                        accept=".edf,.bdf"
                        onChange={handleFileUpload}
                        className="max-w-xs mx-auto"
                      />
                      {uploadedFile && (
                        <Badge variant="outline" className="mt-4 bg-green-50 text-green-700 border-green-200">
                          {uploadedFile.name}
                        </Badge>
                      )}
                    </div>
                  </div>

                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-800">Subject Information</h3>
                    
                    <div className="grid grid-cols-1 gap-4">
                      <div>
                        <Label htmlFor="subject-id" className="text-sm font-medium text-gray-700">Subject ID *</Label>
                        <Input
                          id="subject-id"
                          placeholder="e.g., S001"
                          value={subjectId}
                          onChange={(e) => setSubjectId(e.target.value)}
                          className="mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="memory-score" className="text-sm font-medium text-gray-700">Memory Score (optional)</Label>
                        <Input
                          id="memory-score"
                          type="number"
                          placeholder="0-100"
                          value={memoryScore}
                          onChange={(e) => setMemoryScore(e.target.value)}
                          className="mt-1"
                        />
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="age" className="text-sm font-medium text-gray-700">Age</Label>
                          <Input
                            id="age"
                            type="number"
                            placeholder="Years"
                            value={age}
                            onChange={(e) => setAge(e.target.value)}
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <Label htmlFor="sex" className="text-sm font-medium text-gray-700">Sex</Label>
                          <Input
                            id="sex"
                            placeholder="M/F"
                            value={sex}
                            onChange={(e) => setSex(e.target.value)}
                            className="mt-1"
                          />
                        </div>
                      </div>
                    </div>

                    <Button 
                      onClick={analyzeEEG} 
                      disabled={loading || !uploadedFile || !subjectId}
                      className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 rounded-lg font-semibold shadow-lg transition-all duration-300"
                    >
                      {loading ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Brain className="w-5 h-5 mr-2" />
                          Analyze EEG
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="results" className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold text-gray-800">Analysis Results</h3>
                  <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                    {analysisResults.length} subjects analyzed
                  </Badge>
                </div>

                {analysisResults.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No analysis results yet. Upload and analyze EEG data to see results here.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-4 max-h-96 overflow-y-auto">
                    {analysisResults.map((result, index) => (
                      <Card key={index} className="border-l-4 border-l-blue-500 shadow-sm">
                        <CardContent className="p-4">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                              <p className="text-sm font-medium text-gray-600">Subject</p>
                              <p className="text-lg font-semibold text-gray-800">{result.subject_id}</p>
                            </div>
                            <div>
                              <p className="text-sm font-medium text-gray-600">Sleep Efficiency</p>
                              <p className="text-lg font-semibold text-blue-600">{result.sleep_efficiency?.toFixed(1)}%</p>
                            </div>
                            <div>
                              <p className="text-sm font-medium text-gray-600">Spindle Density</p>
                              <p className="text-lg font-semibold text-purple-600">{result.spindle_density?.toFixed(2)}/min</p>
                            </div>
                            <div>
                              <p className="text-sm font-medium text-gray-600">REM Theta Power</p>
                              <p className="text-lg font-semibold text-green-600">{result.rem_theta_power?.toFixed(1)} μV²</p>
                            </div>
                          </div>
                          
                          {result.memory_score && (
                            <div className="mt-4 pt-4 border-t">
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-600">Memory Score</span>
                                <span className="text-lg font-semibold text-indigo-600">{result.memory_score.toFixed(1)}</span>
                              </div>
                              <Progress value={result.memory_score} className="mt-2" />
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="statistics" className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold text-gray-800">Statistical Analysis</h3>
                  <Button 
                    onClick={fetchStatisticalAnalysis}
                    disabled={loading || analysisResults.length < 2}
                    className="bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 text-white"
                  >
                    {loading ? 'Analyzing...' : 'Run Analysis'}
                  </Button>
                </div>

                {statisticalResults ? (
                  <div className="space-y-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center">
                          <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                          Correlation Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div>
                            <h4 className="font-semibold text-gray-800 mb-3">Key Correlations</h4>
                            {statisticalResults.correlation_analysis?.correlations && 
                              Object.entries(statisticalResults.correlation_analysis.correlations).map(([key, value]) => (
                                <div key={key} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                                  <span className="text-sm text-gray-600">{key.replace(/_/g, ' ')}</span>
                                  <span className={`font-semibold ${Math.abs(value) > 0.5 ? 'text-red-600' : Math.abs(value) > 0.3 ? 'text-orange-600' : 'text-gray-600'}`}>
                                    {value.toFixed(3)}
                                  </span>
                                </div>
                              ))
                            }
                          </div>
                          
                          <div>
                            <h4 className="font-semibold text-gray-800 mb-3">Sample Statistics</h4>
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Sample Size</span>
                                <span className="font-semibold">{statisticalResults.sample_size}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {statisticalResults.regression_analysis && !statisticalResults.regression_analysis.error && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center">
                            <TrendingUp className="w-5 h-5 mr-2 text-purple-600" />
                            Multiple Regression Results
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-semibold text-gray-800 mb-3">Model Fit</h4>
                              <div className="space-y-2">
                                <div className="flex justify-between">
                                  <span className="text-sm text-gray-600">R²</span>
                                  <span className="font-semibold">{statisticalResults.regression_analysis.r_squared?.toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-sm text-gray-600">Adjusted R²</span>
                                  <span className="font-semibold">{statisticalResults.regression_analysis.adjusted_r_squared?.toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-sm text-gray-600">F-statistic</span>
                                  <span className="font-semibold">{statisticalResults.regression_analysis.f_statistic?.toFixed(2)}</span>
                                </div>
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-semibold text-gray-800 mb-3">Coefficients</h4>
                              {statisticalResults.regression_analysis.coefficients && 
                                Object.entries(statisticalResults.regression_analysis.coefficients).map(([key, value]) => (
                                  <div key={key} className="flex justify-between items-center py-1">
                                    <span className="text-sm text-gray-600">{key}</span>
                                    <span className="font-semibold">{value.toFixed(3)}</span>
                                  </div>
                                ))
                              }
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Statistical analysis will appear here after running the analysis.</p>
                    <p className="text-sm mt-2">Requires at least 2 analyzed subjects.</p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="visualizations" className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold text-gray-800">Data Visualizations</h3>
                  <Button 
                    onClick={fetchVisualizations}
                    disabled={loading || analysisResults.length < 2}
                    className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
                  >
                    {loading ? 'Generating...' : 'Generate Charts'}
                  </Button>
                </div>

                {Object.keys(visualizations).length > 0 ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {visualizations.correlation_heatmap && (
                      <Card>
                        <CardHeader>
                          <CardTitle>Correlation Heatmap</CardTitle>
                        </CardHeader>
                        <CardContent>
                          {renderPlotlyChart(visualizations.correlation_heatmap)}
                        </CardContent>
                      </Card>
                    )}

                    {visualizations.spindle_memory_scatter && (
                      <Card>
                        <CardHeader>
                          <CardTitle>Spindle Density vs Memory</CardTitle>
                        </CardHeader>
                        <CardContent>
                          {renderPlotlyChart(visualizations.spindle_memory_scatter)}
                        </CardContent>
                      </Card>
                    )}

                    {visualizations.theta_memory_scatter && (
                      <Card>
                        <CardHeader>
                          <CardTitle>REM Theta vs Memory</CardTitle>
                        </CardHeader>
                        <CardContent>
                          {renderPlotlyChart(visualizations.theta_memory_scatter)}
                        </CardContent>
                      </Card>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Interactive visualizations will appear here after generation.</p>
                    <p className="text-sm mt-2">Requires at least 2 analyzed subjects.</p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="datasets" className="space-y-6">
                <h3 className="text-xl font-semibold text-gray-800">Available Sleep Datasets</h3>
                
                <div className="grid grid-cols-1 gap-4">
                  {datasets.map((dataset, index) => (
                    <Card key={index} className="hover:shadow-md transition-shadow">
                      <CardHeader>
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-lg">{dataset.name}</CardTitle>
                            <CardDescription className="mt-2">{dataset.description}</CardDescription>
                          </div>
                          <Badge variant="secondary">{dataset.subjects} subjects</Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                          <div>
                            <p className="text-sm font-medium text-gray-600">Duration</p>
                            <p className="text-sm text-gray-800">{dataset.duration}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-600">Sampling Rate</p>
                            <p className="text-sm text-gray-800">{dataset.sampling_rate} Hz</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-600">Subjects</p>
                            <p className="text-sm text-gray-800">{dataset.subjects}</p>
                          </div>
                          <div className="flex items-end">
                            <a 
                              href={dataset.download_url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                            >
                              Download Dataset →
                            </a>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>
            </Tabs>

            {/* Download Section */}
            {analysisResults.length > 0 && (
              <div className="mt-12 pt-8 border-t">
                <h3 className="text-xl font-semibold text-gray-800 mb-6 flex items-center">
                  <Download className="w-5 h-5 mr-2 text-green-600" />
                  Download Results
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button 
                    onClick={downloadReport}
                    variant="outline"
                    className="flex items-center justify-center py-6 border-2 border-red-200 hover:border-red-300 hover:bg-red-50 transition-colors"
                  >
                    <FileText className="w-5 h-5 mr-2 text-red-600" />
                    Download PDF Report
                  </Button>
                  
                  <Button 
                    onClick={downloadJupyter}
                    variant="outline"
                    className="flex items-center justify-center py-6 border-2 border-orange-200 hover:border-orange-300 hover:bg-orange-50 transition-colors"
                  >
                    <Download className="w-5 h-5 mr-2 text-orange-600" />
                    Download Jupyter Notebook
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <div className="flex items-center mb-4">
                <Brain className="w-6 h-6 mr-2 text-blue-400" />
                <span className="text-xl font-bold">NeuroSleep Research</span>
              </div>
              <p className="text-gray-400">Advanced EEG sleep analysis platform for neuroscience research and education.</p>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Features</h4>
              <ul className="space-y-2 text-gray-400">
                <li>• Sleep spindle detection (11-16 Hz)</li>
                <li>• REM theta power analysis (4-8 Hz)</li>
                <li>• Memory consolidation correlations</li>
                <li>• Statistical regression modeling</li>
                <li>• Publication-ready reports</li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Research Focus</h4>
              <ul className="space-y-2 text-gray-400">
                <li>• Sleep architecture analysis</li>
                <li>• Neural oscillation patterns</li>
                <li>• Memory consolidation mechanisms</li>
                <li>• EEG signal processing</li>
                <li>• Educational neuroscience tools</li>
              </ul>
            </div>
          </div>
          
          <Separator className="my-8 bg-gray-700" />
          
          <div className="text-center text-gray-400">
            <p>&copy; 2025 NeuroSleep Research Platform. Built for advancing sleep neuroscience research.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;