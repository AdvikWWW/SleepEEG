#!/usr/bin/env python3

import requests
import sys
import json
import tempfile
import os
from datetime import datetime
import numpy as np
import mne
from io import BytesIO

class NeuroSleepAPITester:
    def __init__(self, base_url="https://rem-theta-research.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.analysis_results = []

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        if details:
            print(f"   Details: {details}")

    def test_api_root(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                expected_keys = ["message", "version"]
                has_expected_keys = all(key in data for key in expected_keys)
                success = has_expected_keys
                details = f"Response: {data}" if has_expected_keys else f"Missing keys in response: {data}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("API Root Endpoint", success, details)
            return success
            
        except Exception as e:
            self.log_test("API Root Endpoint", False, str(e))
            return False

    def test_datasets_endpoint(self):
        """Test datasets endpoint"""
        try:
            response = requests.get(f"{self.api_url}/datasets", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                success = isinstance(data, list) and len(data) > 0
                
                if success:
                    # Check if datasets have required fields
                    required_fields = ["name", "description", "subjects", "duration", "sampling_rate", "download_url"]
                    first_dataset = data[0]
                    has_required_fields = all(field in first_dataset for field in required_fields)
                    success = has_required_fields
                    details = f"Found {len(data)} datasets" if has_required_fields else f"Missing fields in dataset: {first_dataset}"
                else:
                    details = f"Expected list with datasets, got: {type(data)}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("Datasets Endpoint", success, details)
            return success
            
        except Exception as e:
            self.log_test("Datasets Endpoint", False, str(e))
            return False

    def create_mock_edf_file(self):
        """Create a mock EDF file for testing"""
        try:
            # Create synthetic EEG data
            sfreq = 100  # 100 Hz sampling rate
            duration = 30  # 30 seconds
            n_channels = 2
            
            # Generate synthetic data
            times = np.arange(0, duration, 1/sfreq)
            data = np.random.randn(n_channels, len(times)) * 1e-6  # Convert to volts
            
            # Add some realistic EEG patterns
            # Add alpha waves (8-12 Hz)
            alpha_freq = 10
            data[0] += 0.5e-6 * np.sin(2 * np.pi * alpha_freq * times)
            
            # Add sleep spindles (11-16 Hz) 
            spindle_freq = 13
            spindle_times = (times > 10) & (times < 15)  # 5 second spindle
            data[1, spindle_times] += 1e-6 * np.sin(2 * np.pi * spindle_freq * times[spindle_times])
            
            # Create channel info
            ch_names = ['EEG1', 'EEG2']
            ch_types = ['eeg', 'eeg']
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            
            # Create Raw object
            raw = mne.io.RawArray(data, info)
            
            # Save to temporary EDF file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.edf')
            temp_file.close()
            
            # Export to EDF
            raw.export(temp_file.name, fmt='edf', overwrite=True, verbose=False)
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error creating mock EDF file: {e}")
            return None

    def test_eeg_analysis(self):
        """Test EEG analysis endpoint with file upload"""
        try:
            # Create mock EDF file
            edf_file_path = self.create_mock_edf_file()
            if not edf_file_path:
                self.log_test("EEG Analysis - File Creation", False, "Could not create mock EDF file")
                return False
            
            # Prepare form data
            files = {'file': ('test_eeg.edf', open(edf_file_path, 'rb'), 'application/octet-stream')}
            data = {
                'subject_id': 'TEST_001',
                'memory_score': 85.5,
                'age': 25,
                'sex': 'M'
            }
            
            response = requests.post(f"{self.api_url}/analyze-eeg", files=files, data=data, timeout=30)
            
            # Clean up
            files['file'][1].close()
            os.unlink(edf_file_path)
            
            success = response.status_code == 200
            
            if success:
                result = response.json()
                required_fields = ['id', 'filename', 'subject_id', 'sleep_efficiency', 'spindle_density', 'rem_theta_power']
                has_required_fields = all(field in result for field in required_fields)
                success = has_required_fields
                
                if success:
                    self.analysis_results.append(result)
                    details = f"Analysis completed for subject {result['subject_id']}"
                else:
                    details = f"Missing required fields in response: {result}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("EEG Analysis Upload", success, details)
            return success
            
        except Exception as e:
            self.log_test("EEG Analysis Upload", False, str(e))
            return False

    def test_analysis_results(self):
        """Test analysis results endpoint"""
        try:
            response = requests.get(f"{self.api_url}/analysis-results", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                success = isinstance(data, list)
                
                if success and len(data) > 0:
                    # Check if results have required structure
                    first_result = data[0]
                    required_fields = ['id', 'subject_id', 'sleep_efficiency', 'spindle_density']
                    has_required_fields = all(field in first_result for field in required_fields)
                    success = has_required_fields
                    details = f"Found {len(data)} analysis results" if has_required_fields else f"Missing fields: {first_result}"
                else:
                    details = f"Got {len(data) if isinstance(data, list) else 'non-list'} results"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("Analysis Results Endpoint", success, details)
            return success
            
        except Exception as e:
            self.log_test("Analysis Results Endpoint", False, str(e))
            return False

    def test_statistical_analysis(self):
        """Test statistical analysis endpoint"""
        try:
            # First, ensure we have at least 2 results by uploading another analysis
            if len(self.analysis_results) < 2:
                print("   Creating second analysis for statistical testing...")
                self.test_eeg_analysis()  # Upload second analysis
            
            response = requests.get(f"{self.api_url}/statistical-analysis", timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                required_keys = ['sample_size', 'correlation_analysis', 'regression_analysis']
                has_required_keys = all(key in data for key in required_keys)
                success = has_required_keys
                
                if success:
                    sample_size = data.get('sample_size', 0)
                    details = f"Statistical analysis completed for {sample_size} subjects"
                else:
                    details = f"Missing required keys in response: {data}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("Statistical Analysis", success, details)
            return success
            
        except Exception as e:
            self.log_test("Statistical Analysis", False, str(e))
            return False

    def test_visualizations(self):
        """Test visualizations endpoint"""
        try:
            response = requests.get(f"{self.api_url}/visualizations", timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                success = isinstance(data, dict) and len(data) > 0
                
                if success:
                    # Check for expected visualization types
                    expected_charts = ['correlation_heatmap']
                    has_charts = any(chart in data for chart in expected_charts)
                    success = has_charts
                    details = f"Generated {len(data)} visualizations: {list(data.keys())}" if has_charts else f"No expected charts found: {list(data.keys())}"
                else:
                    details = f"Expected dict with charts, got: {type(data)}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("Visualizations Endpoint", success, details)
            return success
            
        except Exception as e:
            self.log_test("Visualizations Endpoint", False, str(e))
            return False

    def test_download_report(self):
        """Test PDF report download"""
        try:
            response = requests.get(f"{self.api_url}/download-report", timeout=20)
            success = response.status_code == 200
            
            if success:
                # Check if response is PDF
                content_type = response.headers.get('content-type', '')
                is_pdf = 'pdf' in content_type.lower() or len(response.content) > 1000
                success = is_pdf
                details = f"PDF report downloaded, size: {len(response.content)} bytes" if is_pdf else f"Invalid content type: {content_type}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("PDF Report Download", success, details)
            return success
            
        except Exception as e:
            self.log_test("PDF Report Download", False, str(e))
            return False

    def test_download_jupyter(self):
        """Test Jupyter notebook download"""
        try:
            response = requests.get(f"{self.api_url}/download-jupyter", timeout=20)
            success = response.status_code == 200
            
            if success:
                # Check if response is a notebook file
                content_length = len(response.content)
                is_notebook = content_length > 1000  # Notebooks should be substantial
                success = is_notebook
                details = f"Jupyter notebook downloaded, size: {content_length} bytes" if is_notebook else f"File too small: {content_length} bytes"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
                
            self.log_test("Jupyter Notebook Download", success, details)
            return success
            
        except Exception as e:
            self.log_test("Jupyter Notebook Download", False, str(e))
            return False

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🧠 Starting NeuroSleep Research Platform Backend Tests")
        print("=" * 60)
        
        # Basic connectivity tests
        print("\n📡 Testing Basic API Connectivity...")
        api_available = self.test_api_root()
        
        if not api_available:
            print("❌ API not available, stopping tests")
            return False
        
        self.test_datasets_endpoint()
        
        # Core functionality tests
        print("\n🔬 Testing EEG Analysis Pipeline...")
        analysis_success = self.test_eeg_analysis()
        
        if analysis_success:
            self.test_analysis_results()
            
            # Advanced analysis tests (require data)
            print("\n📊 Testing Statistical Analysis...")
            self.test_statistical_analysis()
            self.test_visualizations()
            
            # Download tests
            print("\n📥 Testing Download Functionality...")
            self.test_download_report()
            self.test_download_jupyter()
        else:
            print("⚠️  Skipping advanced tests due to analysis failure")
        
        # Results summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed! Backend is working correctly.")
            return True
        else:
            failed_tests = self.tests_run - self.tests_passed
            print(f"⚠️  {failed_tests} test(s) failed. Check the details above.")
            return False

def main():
    """Main test execution"""
    tester = NeuroSleepAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())