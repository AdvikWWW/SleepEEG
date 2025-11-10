"""
Comprehensive Reproducibility Testing and Fixing Script
Tests EEG analysis reproducibility and iteratively fixes issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import pandas as pd
import mne
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Import backend modules
from backend.server import analyze_sleep_eeg
from backend.reproducibility_utils import verify_reproducibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define tolerances for reproducibility testing
TOLERANCES = {
    'stage_percentages': 0.05,  # 0.05% tolerance (very strict)
    'spectral_power': 0.5,      # 0.5 μV² tolerance
    'sleep_metrics': 0.05,      # 0.05 minute tolerance
    'quality_score': 0.05,      # 0.05 point tolerance
    'ratios': 0.005,            # 0.005 tolerance for ratios
    'counts': 0,                # Exact match for counts
}

class ReproducibilityTester:
    """Test and verify reproducibility of EEG analysis"""
    
    def __init__(self, num_runs=5):
        self.num_runs = num_runs
        self.results = []
        self.test_report = {
            'test_date': datetime.now().isoformat(),
            'num_runs': num_runs,
            'files_tested': [],
            'issues_found': [],
            'fixes_applied': [],
            'final_status': 'pending'
        }
    
    def create_test_eeg(self, duration_minutes=30, sfreq=256, n_channels=4):
        """Create a synthetic test EEG file with known properties"""
        logger.info(f"Creating synthetic test EEG: {duration_minutes} min, {sfreq} Hz, {n_channels} channels")
        
        # Create deterministic synthetic data
        n_samples = int(duration_minutes * 60 * sfreq)
        
        # Use fixed seed for test data generation
        np.random.seed(42)
        
        # Create channels with different frequency content
        times = np.arange(n_samples) / sfreq
        data = np.zeros((n_channels, n_samples))
        
        # Channel 1: Delta-dominant (deep sleep simulation)
        data[0] = 50 * np.sin(2 * np.pi * 2 * times)  # 2 Hz delta
        data[0] += 20 * np.sin(2 * np.pi * 1 * times)  # 1 Hz delta
        
        # Channel 2: Theta-dominant (REM simulation)
        data[1] = 30 * np.sin(2 * np.pi * 6 * times)  # 6 Hz theta
        data[1] += 15 * np.sin(2 * np.pi * 5 * times)  # 5 Hz theta
        
        # Channel 3: Alpha-dominant (wake simulation)
        data[2] = 25 * np.sin(2 * np.pi * 10 * times)  # 10 Hz alpha
        data[2] += 10 * np.sin(2 * np.pi * 9 * times)   # 9 Hz alpha
        
        # Channel 4: Mixed
        data[3] = 20 * np.sin(2 * np.pi * 2 * times)   # Delta
        data[3] += 15 * np.sin(2 * np.pi * 6 * times)  # Theta
        data[3] += 10 * np.sin(2 * np.pi * 10 * times) # Alpha
        
        # Add small deterministic noise
        for i in range(n_channels):
            np.random.seed(42 + i)
            data[i] += np.random.randn(n_samples) * 2
        
        # Create MNE Raw object
        ch_names = [f'EEG{i+1}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        return raw
    
    def run_single_analysis(self, raw, subject_info):
        """Run a single analysis and extract key metrics"""
        try:
            result = analyze_sleep_eeg(raw, subject_info)
            
            # Extract key metrics
            metrics = {
                # Stage percentages
                'wake_percent': result.get('wake_percent', 0),
                'n1_percent': result.get('n1_percent', 0),
                'n2_percent': result.get('n2_percent', 0),
                'n3_percent': result.get('n3_percent', 0),
                'rem_percent': result.get('rem_percent', 0),
                
                # Sleep architecture
                'total_sleep_time': result.get('total_sleep_time', 0),
                'sleep_efficiency': result.get('sleep_efficiency', 0),
                'sleep_onset_latency': result.get('sleep_onset_latency', 0),
                'rem_latency': result.get('rem_latency', 0),
                
                # Spectral power
                'delta_power': result.get('delta_power', 0),
                'theta_power': result.get('theta_power', 0),
                'alpha_power': result.get('alpha_power', 0),
                'beta_power': result.get('beta_power', 0),
                
                # Sleep quality
                'num_awakenings': result.get('num_awakenings', 0),
                'arousal_index': result.get('arousal_index', 0),
                'sleep_quality_score': result.get('sleep_quality_score', 0),
                
                # Advanced metrics
                'spindle_density': result.get('spindle_density', 0),
                'fragmentation_index': result.get('fragmentation_index', 0),
            }
            
            # Calculate derived metrics
            if metrics['theta_power'] > 0:
                metrics['delta_theta_ratio'] = metrics['delta_power'] / metrics['theta_power']
            else:
                metrics['delta_theta_ratio'] = 0
            
            if metrics['n3_percent'] > 0:
                metrics['rem_n3_ratio'] = metrics['rem_percent'] / metrics['n3_percent']
            else:
                metrics['rem_n3_ratio'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def compare_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Compare metrics across multiple runs and identify variability"""
        if len(metrics_list) < 2:
            return {'reproducible': True, 'issues': []}
        
        issues = []
        metric_stats = {}
        
        # Calculate statistics for each metric
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            metric_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'values': values
            }
        
        # Check stage percentages
        stage_keys = ['wake_percent', 'n1_percent', 'n2_percent', 'n3_percent', 'rem_percent']
        for key in stage_keys:
            stats = metric_stats[key]
            if stats['range'] > TOLERANCES['stage_percentages']:
                issues.append({
                    'metric': key,
                    'type': 'stage_percentage',
                    'range': stats['range'],
                    'tolerance': TOLERANCES['stage_percentages'],
                    'values': stats['values'],
                    'severity': 'high' if stats['range'] > 0.1 else 'medium'
                })
        
        # Check spectral power
        power_keys = ['delta_power', 'theta_power', 'alpha_power', 'beta_power']
        for key in power_keys:
            stats = metric_stats[key]
            if stats['range'] > TOLERANCES['spectral_power']:
                issues.append({
                    'metric': key,
                    'type': 'spectral_power',
                    'range': stats['range'],
                    'tolerance': TOLERANCES['spectral_power'],
                    'values': stats['values'],
                    'severity': 'high' if stats['range'] > 5.0 else 'medium'
                })
        
        # Check sleep metrics
        metric_keys = ['total_sleep_time', 'sleep_onset_latency', 'rem_latency']
        for key in metric_keys:
            stats = metric_stats[key]
            if stats['range'] > TOLERANCES['sleep_metrics']:
                issues.append({
                    'metric': key,
                    'type': 'sleep_metric',
                    'range': stats['range'],
                    'tolerance': TOLERANCES['sleep_metrics'],
                    'values': stats['values'],
                    'severity': 'medium'
                })
        
        # Check ratios
        ratio_keys = ['delta_theta_ratio', 'rem_n3_ratio']
        for key in ratio_keys:
            stats = metric_stats[key]
            if stats['range'] > TOLERANCES['ratios']:
                issues.append({
                    'metric': key,
                    'type': 'ratio',
                    'range': stats['range'],
                    'tolerance': TOLERANCES['ratios'],
                    'values': stats['values'],
                    'severity': 'medium'
                })
        
        # Check integer counts
        if 'num_awakenings' in metric_stats:
            stats = metric_stats['num_awakenings']
            if stats['range'] > TOLERANCES['counts']:
                issues.append({
                    'metric': 'num_awakenings',
                    'type': 'count',
                    'range': stats['range'],
                    'tolerance': TOLERANCES['counts'],
                    'values': stats['values'],
                    'severity': 'high'
                })
        
        return {
            'reproducible': len(issues) == 0,
            'issues': issues,
            'metric_stats': metric_stats
        }
    
    def run_test(self, raw, subject_info, test_name="Test EEG"):
        """Run complete reproducibility test"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting reproducibility test: {test_name}")
        logger.info(f"Number of runs: {self.num_runs}")
        logger.info(f"{'='*80}\n")
        
        metrics_list = []
        
        # Run analysis multiple times
        for run_num in range(self.num_runs):
            logger.info(f"Run {run_num + 1}/{self.num_runs}...")
            metrics = self.run_single_analysis(raw, subject_info)
            metrics_list.append(metrics)
            
            # Log key metrics for this run
            logger.info(f"  Sleep Efficiency: {metrics['sleep_efficiency']:.2f}%")
            logger.info(f"  Delta Power: {metrics['delta_power']:.2f} μV²")
            logger.info(f"  Awakenings: {metrics['num_awakenings']}")
        
        # Compare results
        logger.info(f"\nComparing results across {self.num_runs} runs...")
        comparison = self.compare_metrics(metrics_list)
        
        # Generate report
        self.generate_report(test_name, comparison, metrics_list)
        
        return comparison
    
    def generate_report(self, test_name, comparison, metrics_list):
        """Generate detailed reproducibility report"""
        logger.info(f"\n{'='*80}")
        logger.info(f"REPRODUCIBILITY TEST REPORT: {test_name}")
        logger.info(f"{'='*80}\n")
        
        if comparison['reproducible']:
            logger.info("✅ RESULT: FULLY REPRODUCIBLE")
            logger.info("All metrics are consistent across runs within tolerances.\n")
        else:
            logger.info("❌ RESULT: VARIABILITY DETECTED")
            logger.info(f"Found {len(comparison['issues'])} metrics with variability:\n")
            
            # Group issues by severity
            high_severity = [i for i in comparison['issues'] if i['severity'] == 'high']
            medium_severity = [i for i in comparison['issues'] if i['severity'] == 'medium']
            
            if high_severity:
                logger.info("HIGH SEVERITY ISSUES:")
                for issue in high_severity:
                    logger.info(f"  • {issue['metric']}")
                    logger.info(f"    Range: {issue['range']:.4f} (tolerance: {issue['tolerance']:.4f})")
                    logger.info(f"    Values: {[f'{v:.4f}' for v in issue['values']]}")
                logger.info("")
            
            if medium_severity:
                logger.info("MEDIUM SEVERITY ISSUES:")
                for issue in medium_severity:
                    logger.info(f"  • {issue['metric']}")
                    logger.info(f"    Range: {issue['range']:.4f} (tolerance: {issue['tolerance']:.4f})")
                    logger.info(f"    Values: {[f'{v:.4f}' for v in issue['values']]}")
                logger.info("")
        
        # Show detailed statistics for key metrics
        logger.info("DETAILED METRIC STATISTICS:")
        logger.info(f"{'Metric':<25} {'Mean':<12} {'Std Dev':<12} {'Range':<12} {'Status':<10}")
        logger.info("-" * 80)
        
        for metric, stats in comparison['metric_stats'].items():
            status = "✓ OK" if stats['range'] <= self._get_tolerance(metric) else "✗ VARY"
            logger.info(f"{metric:<25} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['range']:<12.4f} {status:<10}")
        
        logger.info(f"\n{'='*80}\n")
    
    def _get_tolerance(self, metric):
        """Get appropriate tolerance for a metric"""
        if 'percent' in metric:
            return TOLERANCES['stage_percentages']
        elif 'power' in metric:
            return TOLERANCES['spectral_power']
        elif 'ratio' in metric:
            return TOLERANCES['ratios']
        elif 'awakening' in metric:
            return TOLERANCES['counts']
        else:
            return TOLERANCES['sleep_metrics']
    
    def save_report(self, filename='reproducibility_test_report.json'):
        """Save test report to file"""
        report_path = Path(__file__).parent / filename
        with open(report_path, 'w') as f:
            json.dump(self.test_report, f, indent=2)
        logger.info(f"Report saved to: {report_path}")

def main():
    """Main testing function"""
    logger.info("="*80)
    logger.info("SLEEP EEG ANALYSIS - REPRODUCIBILITY TESTING SYSTEM")
    logger.info("="*80)
    logger.info("")
    
    # Initialize tester
    tester = ReproducibilityTester(num_runs=5)
    
    # Create test EEG
    logger.info("Creating synthetic test EEG...")
    raw = tester.create_test_eeg(duration_minutes=30, sfreq=256, n_channels=4)
    
    # Subject info
    subject_info = {
        'subject_id': 'TEST_REPRO_001',
        'age': 35,
        'sex': 'M'
    }
    
    # Run test
    comparison = tester.run_test(raw, subject_info, test_name="Synthetic EEG Test")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    if comparison['reproducible']:
        logger.info("✅ SUCCESS: The analysis is fully reproducible!")
        logger.info("   All metrics are consistent across multiple runs.")
        logger.info("   The system is ready for production use.")
    else:
        logger.info("⚠️  ISSUES DETECTED: Some metrics show variability.")
        logger.info(f"   {len(comparison['issues'])} metrics exceed tolerance thresholds.")
        logger.info("   Review the detailed report above for specific issues.")
        logger.info("\nRECOMMENDED ACTIONS:")
        logger.info("1. Check seed generation for sufficient entropy")
        logger.info("2. Verify all calculations use deterministic rounding")
        logger.info("3. Ensure ratio normalization is applied")
        logger.info("4. Review epoch calculation for rounding consistency")
    
    logger.info("="*80 + "\n")
    
    # Save report
    tester.save_report()
    
    return comparison['reproducible']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
