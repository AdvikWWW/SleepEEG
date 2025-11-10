import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { Upload, Brain, BarChart3, Download, FileText, Database, Zap, Activity, TrendingUp, Users, Clock, Microscope, Star, Target, Lightbulb, FileSpreadsheet, Timeline } from 'lucide-react';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Separator } from './components/ui/separator';
import { Alert, AlertDescription } from './components/ui/alert';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [statisticalResults, setStatisticalResults] = useState(null);
  const [visualizations, setVisualizations] = useState({});
  const [populationComparison, setPopulationComparison] = useState(null);
  const [sleepTips, setSleepTips] = useState(null);
  const [longitudinalData, setLongitudinalData] = useState(null);
  const [selectedSubject, setSelectedSubject] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
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

  const fetchPopulationComparison = async (subjectId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/population-comparison/${subjectId}`);
      setPopulationComparison(response.data);
    } catch (error) {
      console.error('Error fetching population comparison:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSleepTips = async (subjectId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/sleep-tips/${subjectId}`);
      setSleepTips(response.data);
    } catch (error) {
      console.error('Error fetching sleep tips:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchLongitudinalData = async (subjectId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/longitudinal-tracking/${subjectId}`);
      setLongitudinalData(response.data);
    } catch (error) {
      console.error('Error fetching longitudinal data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && (file.name.endsWith('.edf') || file.name.endsWith('.bdf') || file.name.endsWith('.csv'))) {
      setUploadedFile(file);
      setUploadProgress(0);
    } else {
      alert('Please upload a valid EDF, BDF, or CSV file');
    }
  };

  const analyzeEEG = async () => {
    if (!uploadedFile || !subjectId) {
      alert('Please select a file and enter subject ID');
      return;
    }

    setLoading(true);
    setUploadProgress(10);
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('subject_id', subjectId);
    if (memoryScore) formData.append('memory_score', memoryScore);
    if (age) formData.append('age', age);
    if (sex) formData.append('sex', sex);

    try {
      setUploadProgress(30);
      const response = await axios.post(`${API}/analyze-eeg`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(Math.min(progress, 90));
        }
      });
      
      setUploadProgress(100);
      setAnalysisResults([...analysisResults, response.data]);
      setUploadedFile(null);
      setSubjectId('');
      setMemoryScore('');
      setAge('');
      setSex('');
      
      alert('Analysis completed successfully!');
      setActiveTab('results');
    } catch (error) {
      console.error('Error analyzing EEG:', error);
      alert('Analysis failed. Please check your file format.');
    } finally {
      setLoading(false);
      setUploadProgress(0);
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
      link.setAttribute('download', 'comprehensive_sleep_analysis_report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading report:', error);
    }
  };

  const downloadCSV = async () => {
    try {
      const response = await axios.get(`${API}/export-csv`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sleep_eeg_analysis_results.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading CSV:', error);
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

  const renderHypnogram = (hypnogramData) => {
    if (!hypnogramData || hypnogramData.length === 0) return null;
    
    const stageColors = {
      'Wake': '#ff6b6b',
      'REM': '#4ecdc4',
      'N1': '#45b7d1',
      'N2': '#96ceb4',
      'N3': '#2d3436'
    };
    
    return (
      <div className="w-full h-32 bg-gray-50 rounded-lg p-4 border">
        <h4 className="text-sm font-semibold mb-2">Sleep Hypnogram</h4>
        <div className="flex h-20 items-end space-x-0.5">
          {hypnogramData.slice(0, 100).map((epoch, index) => (
            <div
              key={index}
              className="flex-1 rounded-t-sm"
              style={{
                backgroundColor: stageColors[epoch.stage] || '#gray',
                height: epoch.stage === 'Wake' ? '100%' : 
                       epoch.stage === 'REM' ? '80%' :
                       epoch.stage === 'N1' ? '60%' :
                       epoch.stage === 'N2' ? '40%' : '20%'
              }}
              title={`${epoch.stage} at ${epoch.time_minutes.toFixed(1)} min`}
            />
          ))}
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0 min</span>
          <span>{(hypnogramData[hypnogramData.length - 1]?.time_minutes || 0).toFixed(0)} min</span>
        </div>
      </div>
    );
  };

  const renderSleepStageChart = (result) => {
    const stages = [
      { name: 'Wake', value: result.wake_percent || 0, color: '#ff6b6b' },
      { name: 'REM', value: result.rem_percent || 0, color: '#4ecdc4' },
      { name: 'N1', value: result.n1_percent || 0, color: '#45b7d1' },
      { name: 'N2', value: result.n2_percent || 0, color: '#96ceb4' },
      { name: 'N3', value: result.n3_percent || 0, color: '#2d3436' }
    ];

    return (
      <div className="space-y-3">
        <h4 className="text-sm font-semibold">Sleep Stage Distribution</h4>
        {stages.map((stage) => (
          <div key={stage.name} className="flex items-center space-x-3">
            <div 
              className="w-4 h-4 rounded"
              style={{ backgroundColor: stage.color }}
            />
            <span className="text-sm w-12">{stage.name}</span>
            <div className="flex-1">
              <Progress value={stage.value} className="h-2" />
            </div>
            <span className="text-sm font-medium w-12 text-right">
              {stage.value.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  const renderPopulationComparison = () => {
    if (!populationComparison) return null;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Population Comparison</h3>
          <Badge variant="outline">
            Age: {populationComparison.age} | {populationComparison.sex || 'Not specified'}
          </Badge>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(populationComparison.comparison).map(([metric, data]) => (
            <Card key={metric} className="border-l-4" style={{ borderLeftColor: data.color }}>
              <CardContent className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-sm font-medium capitalize">
                    {metric.replace(/_/g, ' ')}
                  </h4>
                  <Badge 
                    variant="outline" 
                    className={`text-xs ${
                      data.category === 'Above Average' ? 'bg-green-50 text-green-700' :
                      data.category === 'Average' ? 'bg-blue-50 text-blue-700' :
                      'bg-orange-50 text-orange-700'
                    }`}
                  >
                    {data.category}
                  </Badge>
                </div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Your value:</span>
                    <span className="font-semibold">{data.user_value.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Population avg:</span>
                    <span>{data.population_mean.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Percentile:</span>
                    <span className="font-semibold">{data.percentile.toFixed(0)}th</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  };

  const renderSleepTips = () => {
    if (!sleepTips || !sleepTips.tips) return null;

    const priorityColors = {
      high: 'border-red-200 bg-red-50',
      medium: 'border-yellow-200 bg-yellow-50',
      low: 'border-green-200 bg-green-50'
    };

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Personalized Sleep Tips</h3>
          <div className="flex items-center space-x-2">
            <Star className="w-5 h-5 text-yellow-500" />
            <span className="font-semibold">{sleepTips.sleep_quality_score.toFixed(1)}/100</span>
          </div>
        </div>

        <div className="space-y-3">
          {sleepTips.tips.map((tip, index) => (
            <Alert key={index} className={priorityColors[tip.priority]}>
              <div className="flex items-start space-x-3">
                <span className="text-2xl">{tip.icon}</span>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-semibold text-sm">{tip.category}</h4>
                    <Badge 
                      variant="outline" 
                      className={`text-xs ${
                        tip.priority === 'high' ? 'border-red-300 text-red-700' :
                        tip.priority === 'medium' ? 'border-yellow-300 text-yellow-700' :
                        'border-green-300 text-green-700'
                      }`}
                    >
                      {tip.priority} priority
                    </Badge>
                  </div>
                  <AlertDescription className="text-sm">
                    {tip.tip}
                  </AlertDescription>
                </div>
              </div>
            </Alert>
          ))}
        </div>
      </div>
    );
  };

  const renderLongitudinalTracking = () => {
    if (!longitudinalData) return null;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Longitudinal Tracking</h3>
          <Badge variant="outline">
            {longitudinalData.total_analyses} analyses
          </Badge>
        </div>

        {longitudinalData.trends && Object.keys(longitudinalData.trends).length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {Object.entries(longitudinalData.trends).map(([metric, trend]) => (
              <Card key={metric}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium capitalize">
                      {metric.replace(/_/g, ' ')}
                    </h4>
                    <div className={`flex items-center space-x-1 ${
                      trend.direction === 'improving' ? 'text-green-600' :
                      trend.direction === 'declining' ? 'text-red-600' :
                      'text-gray-600'
                    }`}>
                      <TrendingUp className="w-4 h-4" />
                      <span className="text-xs font-semibold">
                        {trend.direction}
                      </span>
                    </div>
                  </div>
                  <div className="text-lg font-bold">
                    {trend.change > 0 ? '+' : ''}{trend.change.toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {trend.percentage_change.toFixed(1)}% change
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold mb-3">Analysis History</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {longitudinalData.tracking_data.map((analysis, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                <div>
                  <div className="font-medium text-sm">{analysis.filename}</div>
                  <div className="text-xs text-gray-500">
                    {new Date(analysis.date).toLocaleDateString()}
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-sm">
                    {analysis.sleep_quality_score?.toFixed(1) || 'N/A'}/100
                  </div>
                  <div className="text-xs text-gray-500">
                    {analysis.sleep_efficiency?.toFixed(1) || 'N/A'}% efficiency
                  </div>
                </div>
              </div>
            ))}
          </div>
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
              <TabsList className="grid w-full grid-cols-7 mb-8 bg-gray-100 rounded-xl p-1">
                <TabsTrigger value="upload" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload
                </TabsTrigger>
                <TabsTrigger value="results" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Activity className="w-4 h-4 mr-2" />
                  Results
                </TabsTrigger>
                <TabsTrigger value="comparison" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Target className="w-4 h-4 mr-2" />
                  Compare
                </TabsTrigger>
                <TabsTrigger value="tips" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Lightbulb className="w-4 h-4 mr-2" />
                  Tips
                </TabsTrigger>
                <TabsTrigger value="tracking" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <Timeline className="w-4 h-4 mr-2" />
                  Tracking
                </TabsTrigger>
                <TabsTrigger value="statistics" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Statistics
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
                      <p className="text-gray-600 mb-4">Select EDF, BDF, or CSV format files</p>
                      <Input
                        type="file"
                        accept=".edf,.bdf,.csv"
                        onChange={handleFileUpload}
                        className="max-w-xs mx-auto"
                      />
                      {uploadedFile && (
                        <Badge variant="outline" className="mt-4 bg-green-50 text-green-700 border-green-200">
                          {uploadedFile.name}
                        </Badge>
                      )}
                      {uploadProgress > 0 && (
                        <div className="mt-4">
                          <Progress value={uploadProgress} className="w-full" />
                          <p className="text-sm text-gray-600 mt-1">Processing... {uploadProgress}%</p>
                        </div>
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
                  <div className="space-y-6">
                    {analysisResults.map((result, index) => (
                      <Card key={index} className="border-l-4 border-l-blue-500 shadow-sm">
                        <CardContent className="p-6">
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h4 className="text-lg font-semibold text-gray-800">{result.subject_id}</h4>
                              <p className="text-sm text-gray-600">{result.filename}</p>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Star className="w-5 h-5 text-yellow-500" />
                              <span className="font-semibold text-lg">{result.sleep_quality_score?.toFixed(1) || 'N/A'}/100</span>
                            </div>
                          </div>

                          {result.hypnogram && result.hypnogram.length > 0 && (
                            <div className="mb-6">
                              {renderHypnogram(result.hypnogram)}
                            </div>
                          )}

                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="space-y-4">
                              <h5 className="font-semibold text-gray-800">Sleep Architecture</h5>
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <p className="text-sm font-medium text-gray-600">Sleep Efficiency</p>
                                  <p className="text-lg font-semibold text-blue-600">{result.sleep_efficiency?.toFixed(1)}%</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-gray-600">Total Sleep Time</p>
                                  <p className="text-lg font-semibold text-green-600">{(result.total_sleep_time/60)?.toFixed(1)}h</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-gray-600">REM Sleep</p>
                                  <p className="text-lg font-semibold text-purple-600">{result.rem_percent?.toFixed(1)}%</p>
                                </div>
                                <div>
                                  <p className="text-sm font-medium text-gray-600">Deep Sleep (N3)</p>
                                  <p className="text-lg font-semibold text-indigo-600">{result.n3_percent?.toFixed(1)}%</p>
                                </div>
                              </div>
                            </div>

                            <div>
                              {renderSleepStageChart(result)}
                            </div>
                          </div>

                          <div className="mt-6 pt-4 border-t">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                              <div>
                                <p className="text-sm font-medium text-gray-600">Spindle Density</p>
                                <p className="text-lg font-semibold text-orange-600">{result.spindle_density?.toFixed(2)}/min</p>
                              </div>
                              <div>
                                <p className="text-sm font-medium text-gray-600">REM Theta Power</p>
                                <p className="text-lg font-semibold text-cyan-600">{result.rem_theta_power?.toFixed(1)} μV²</p>
                              </div>
                              <div>
                                <p className="text-sm font-medium text-gray-600">Awakenings</p>
                                <p className="text-lg font-semibold text-red-600">{result.num_awakenings || 'N/A'}</p>
                              </div>
                              <div>
                                <p className="text-sm font-medium text-gray-600">Sleep Onset</p>
                                <p className="text-lg font-semibold text-teal-600">{result.sleep_onset_latency?.toFixed(1)}min</p>
                              </div>
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

              <TabsContent value="comparison" className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold text-gray-800">Population Comparison</h3>
                  <div className="flex items-center space-x-2">
                    <Label htmlFor="subject-select" className="text-sm font-medium">Select Subject:</Label>
                    <select
                      id="subject-select"
                      value={selectedSubject}
                      onChange={(e) => {
                        setSelectedSubject(e.target.value);
                        if (e.target.value) {
                          fetchPopulationComparison(e.target.value);
                        }
                      }}
                      className="px-3 py-1 border rounded-md text-sm"
                    >
                      <option value="">Choose subject...</option>
                      {analysisResults.map((result) => (
                        <option key={result.subject_id} value={result.subject_id}>
                          {result.subject_id}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {!selectedSubject ? (
                  <div className="text-center py-12 text-gray-500">
                    <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Select a subject to view population comparison.</p>
                  </div>
                ) : loading ? (
                  <div className="text-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p>Loading comparison data...</p>
                  </div>
                ) : (
                  renderPopulationComparison()
                )}
              </TabsContent>

              <TabsContent value="tips" className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold text-gray-800">Sleep Improvement Tips</h3>
                  <div className="flex items-center space-x-2">
                    <Label htmlFor="tips-subject-select" className="text-sm font-medium">Select Subject:</Label>
                    <select
                      id="tips-subject-select"
                      value={selectedSubject}
                      onChange={(e) => {
                        setSelectedSubject(e.target.value);
                        if (e.target.value) {
                          fetchSleepTips(e.target.value);
                        }
                      }}
                      className="px-3 py-1 border rounded-md text-sm"
                    >
                      <option value="">Choose subject...</option>
                      {analysisResults.map((result) => (
                        <option key={result.subject_id} value={result.subject_id}>
                          {result.subject_id}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {!selectedSubject ? (
                  <div className="text-center py-12 text-gray-500">
                    <Lightbulb className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Select a subject to view personalized sleep tips.</p>
                  </div>
                ) : loading ? (
                  <div className="text-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p>Generating personalized tips...</p>
                  </div>
                ) : (
                  renderSleepTips()
                )}
              </TabsContent>

              <TabsContent value="tracking" className="space-y-6">
                <div className="flex justify-between items-center">
                  <h3 className="text-xl font-semibold text-gray-800">Longitudinal Tracking</h3>
                  <div className="flex items-center space-x-2">
                    <Label htmlFor="tracking-subject-select" className="text-sm font-medium">Select Subject:</Label>
                    <select
                      id="tracking-subject-select"
                      value={selectedSubject}
                      onChange={(e) => {
                        setSelectedSubject(e.target.value);
                        if (e.target.value) {
                          fetchLongitudinalData(e.target.value);
                        }
                      }}
                      className="px-3 py-1 border rounded-md text-sm"
                    >
                      <option value="">Choose subject...</option>
                      {analysisResults.map((result) => (
                        <option key={result.subject_id} value={result.subject_id}>
                          {result.subject_id}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {!selectedSubject ? (
                  <div className="text-center py-12 text-gray-500">
                    <Timeline className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Select a subject to view longitudinal tracking data.</p>
                  </div>
                ) : loading ? (
                  <div className="text-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p>Loading tracking data...</p>
                  </div>
                ) : (
                  renderLongitudinalTracking()
                )}
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
                    onClick={downloadCSV}
                    variant="outline"
                    className="flex items-center justify-center py-6 border-2 border-orange-200 hover:border-orange-300 hover:bg-orange-50 transition-colors"
                  >
                    <FileSpreadsheet className="w-5 h-5 mr-2 text-orange-600" />
                    Download CSV Data
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