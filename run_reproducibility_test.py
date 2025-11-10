#!/usr/bin/env python3
"""
Simple reproducibility test runner
Tests the same EEG multiple times and reports results
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import mne
from datetime import datetime

# Import backend modules
from server import analyze_sleep_eeg

def create_test_eeg():
    """Create a simple synthetic EEG for testing"""
    print("Creating synthetic test EEG...")
    
    # Fixed parameters
    sfreq = 256
    duration = 30 * 60  # 30 minutes in seconds
    n_samples = int(duration * sfreq)
    n_channels = 4
    
    # Create deterministic data
    np.random.seed(42)
    times = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    
    # Channel 1: Delta-dominant
    data[0] = 50 * np.sin(2 * np.pi * 2 * times) + np.random.randn(n_samples) * 2
    
    # Channel 2: Theta-dominant
    data[1] = 30 * np.sin(2 * np.pi * 6 * times) + np.random.randn(n_samples) * 2
    
    # Channel 3: Alpha-dominant
    data[2] = 25 * np.sin(2 * np.pi * 10 * times) + np.random.randn(n_samples) * 2
    
    # Channel 4: Mixed
    data[3] = 20 * np.sin(2 * np.pi * 2 * times) + 15 * np.sin(2 * np.pi * 6 * times) + np.random.randn(n_samples) * 2
    
    # Create MNE Raw object
    ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    print(f"✓ Created EEG: {n_channels} channels, {duration/60:.1f} minutes, {sfreq} Hz")
    return raw

def run_analysis(raw, subject_info, run_num):
    """Run a single analysis"""
    print(f"\n--- Run {run_num} ---")
    result = analyze_sleep_eeg(raw, subject_info)
    
    # Extract key metrics
    metrics = {
        'wake_percent': result.get('wake_percent', 0),
        'n1_percent': result.get('n1_percent', 0),
        'n2_percent': result.get('n2_percent', 0),
        'n3_percent': result.get('n3_percent', 0),
        'rem_percent': result.get('rem_percent', 0),
        'total_sleep_time': result.get('total_sleep_time', 0),
        'sleep_efficiency': result.get('sleep_efficiency', 0),
        'delta_power': result.get('delta_power', 0),
        'theta_power': result.get('theta_power', 0),
        'alpha_power': result.get('alpha_power', 0),
        'beta_power': result.get('beta_power', 0),
        'num_awakenings': result.get('num_awakenings', 0),
        'sleep_quality_score': result.get('sleep_quality_score', 0),
    }
    
    # Print key metrics
    print(f"Sleep Efficiency: {metrics['sleep_efficiency']:.4f}%")
    print(f"Stage %: W={metrics['wake_percent']:.4f}, N1={metrics['n1_percent']:.4f}, N2={metrics['n2_percent']:.4f}, N3={metrics['n3_percent']:.4f}, REM={metrics['rem_percent']:.4f}")
    print(f"Spectral: Delta={metrics['delta_power']:.2f}, Theta={metrics['theta_power']:.2f}, Alpha={metrics['alpha_power']:.2f}, Beta={metrics['beta_power']:.2f}")
    print(f"Quality: Score={metrics['sleep_quality_score']:.2f}, Awakenings={metrics['num_awakenings']}")
    
    return metrics

def compare_results(all_metrics):
    """Compare results across runs"""
    print("\n" + "="*80)
    print("COMPARISON ACROSS RUNS")
    print("="*80)
    
    if len(all_metrics) < 2:
        print("Need at least 2 runs to compare")
        return True
    
    # Calculate statistics for each metric
    all_same = True
    
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        mean_val = sum(values) / len(values)
        
        # Determine if values are identical
        if range_val == 0:
            status = "✓ IDENTICAL"
        elif range_val < 0.01:
            status = "✓ NEARLY IDENTICAL"
        elif range_val < 0.1:
            status = "⚠ MINOR VARIATION"
        else:
            status = "✗ SIGNIFICANT VARIATION"
            all_same = False
        
        print(f"{key:25s}: Range={range_val:10.6f}  Mean={mean_val:10.4f}  {status}")
    
    print("="*80)
    
    if all_same:
        print("\n✅ SUCCESS: All metrics are reproducible!")
    else:
        print("\n❌ ISSUES: Some metrics show variation across runs")
    
    return all_same

def main():
    """Main test function"""
    print("="*80)
    print("SLEEP EEG ANALYSIS - REPRODUCIBILITY TEST")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create test EEG
    raw = create_test_eeg()
    
    # Subject info
    subject_info = {
        'subject_id': 'REPRO_TEST_001',
        'age': 35,
        'sex': 'M'
    }
    
    # Run multiple times
    num_runs = 5
    print(f"\nRunning analysis {num_runs} times...")
    
    all_metrics = []
    for i in range(num_runs):
        metrics = run_analysis(raw, subject_info, i + 1)
        all_metrics.append(metrics)
    
    # Compare results
    success = compare_results(all_metrics)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
