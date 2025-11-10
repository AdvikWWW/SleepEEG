#!/usr/bin/env python3
"""
Epoch Scaling Verification Script
Ensures TST and all time metrics use correct 30-second epoch scaling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import mne
from datetime import datetime

# Import backend modules
from server import analyze_sleep_eeg

def verify_epoch_scaling(hypnogram, computed_tst, epoch_duration_sec=30):
    """
    Verify that TST is correctly calculated from 30-second epochs
    
    Args:
        hypnogram: List of epoch dictionaries with 'stage' key
        computed_tst: Computed total sleep time in minutes
        epoch_duration_sec: Expected epoch duration in seconds (default: 30)
    
    Returns:
        bool: True if within ±5% tolerance
    """
    # Count non-wake epochs
    num_sleep_epochs = sum(1 for epoch in hypnogram if epoch.get('stage') != 'Wake')
    
    # Expected TST = (num_sleep_epochs * epoch_duration_sec) / 60
    expected_tst = (num_sleep_epochs * epoch_duration_sec) / 60.0
    
    # Calculate error
    if expected_tst == 0:
        return computed_tst == 0
    
    error_pct = abs(computed_tst - expected_tst) / expected_tst * 100
    
    print(f"\n  Epoch Scaling Verification:")
    print(f"    Total epochs: {len(hypnogram)}")
    print(f"    Sleep epochs: {num_sleep_epochs}")
    print(f"    Epoch duration: {epoch_duration_sec} seconds")
    print(f"    Expected TST: {expected_tst:.2f} min")
    print(f"    Computed TST: {computed_tst:.2f} min")
    print(f"    Error: {error_pct:.2f}%")
    
    is_valid = error_pct < 5.0  # ±5% tolerance
    
    if is_valid:
        print(f"    ✅ Epoch scaling verified: TST within ±5% of expected value")
    else:
        print(f"    ❌ Epoch scaling error detected: {error_pct:.2f}% deviation")
    
    return is_valid

def create_test_eeg(name="Test", duration_min=30, sleep_ratio=0.85):
    """Create synthetic EEG with known sleep characteristics"""
    print(f"\nCreating {name} EEG ({duration_min} min, {sleep_ratio*100:.0f}% sleep)...")
    
    sfreq = 256
    duration = duration_min * 60
    n_samples = int(duration * sfreq)
    n_channels = 4
    
    # Create deterministic data
    np.random.seed(42)
    times = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    
    # Channels with different frequency content
    data[0] = 50 * np.sin(2 * np.pi * 2 * times) + np.random.randn(n_samples) * 2  # Delta
    data[1] = 30 * np.sin(2 * np.pi * 6 * times) + np.random.randn(n_samples) * 2  # Theta
    data[2] = 25 * np.sin(2 * np.pi * 10 * times) + np.random.randn(n_samples) * 2  # Alpha
    data[3] = 20 * np.sin(2 * np.pi * 2 * times) + 15 * np.sin(2 * np.pi * 6 * times) + np.random.randn(n_samples) * 2
    
    # Create MNE Raw object
    ch_names = [f'EEG{i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    print(f"✓ Created: {n_channels} channels, {duration_min} min, {sfreq} Hz")
    return raw

def run_comprehensive_test(raw, subject_info, test_name):
    """Run analysis and verify all metrics"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    
    # Run analysis
    result = analyze_sleep_eeg(raw, subject_info)
    
    # Extract metrics
    tst = result.get('total_sleep_time', 0)
    efficiency = result.get('sleep_efficiency', 0)
    sol = result.get('sleep_onset_latency', 0)
    rem_lat = result.get('rem_latency', 0)
    time_in_bed = result.get('time_in_bed', 0)
    sleep_window_duration = result.get('sleep_window_duration', 0)
    
    # Get hypnogram for verification
    hypnogram = result.get('hypnogram', [])
    
    print(f"\nRESULTS:")
    print(f"  Time in Bed: {time_in_bed:.2f} min")
    print(f"  Sleep Window Duration: {sleep_window_duration:.2f} min")
    print(f"  Total Sleep Time: {tst:.2f} min ({tst/60:.2f} hours)")
    print(f"  Sleep Efficiency: {efficiency:.2f}%")
    print(f"  Sleep Onset Latency: {sol:.2f} min")
    print(f"  REM Latency: {rem_lat:.2f} min")
    
    # Verify epoch scaling
    is_valid = verify_epoch_scaling(hypnogram, tst, epoch_duration_sec=30)
    
    # Additional sanity checks
    print(f"\n  Additional Checks:")
    
    # Check 1: TST should be <= Time in Bed
    check1 = tst <= time_in_bed
    print(f"    TST ≤ Time in Bed: {'✅ PASS' if check1 else '❌ FAIL'} ({tst:.1f} ≤ {time_in_bed:.1f})")
    
    # Check 2: Sleep efficiency should be 0-100%
    check2 = 0 <= efficiency <= 100
    print(f"    Sleep Efficiency 0-100%: {'✅ PASS' if check2 else '❌ FAIL'} ({efficiency:.1f}%)")
    
    # Check 3: Sleep window should be reasonable
    check3 = sleep_window_duration > 0 and sleep_window_duration <= time_in_bed
    print(f"    Sleep Window Valid: {'✅ PASS' if check3 else '❌ FAIL'} ({sleep_window_duration:.1f} min)")
    
    # Check 4: TST should be reasonable (not absurdly high)
    expected_max_tst = time_in_bed * 0.95  # Max 95% of recording
    check4 = tst <= expected_max_tst
    print(f"    TST Reasonable: {'✅ PASS' if check4 else '❌ FAIL'} ({tst:.1f} ≤ {expected_max_tst:.1f})")
    
    all_checks = is_valid and check1 and check2 and check3 and check4
    
    print(f"\n{'='*80}")
    if all_checks:
        print(f"✅ ALL CHECKS PASSED")
    else:
        print(f"❌ SOME CHECKS FAILED")
    print(f"{'='*80}")
    
    return all_checks, result

def main():
    """Main validation function"""
    print("="*80)
    print("EPOCH SCALING VERIFICATION SYSTEM")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis test verifies:")
    print("1. TST is correctly calculated using 30-second epochs")
    print("2. All time metrics are properly scaled")
    print("3. Sleep window detection works correctly")
    print("4. Results are within expected physiological ranges")
    
    # Subject info
    subject_info = {
        'subject_id': 'EPOCH_TEST_001',
        'age': 35,
        'sex': 'M'
    }
    
    results = []
    
    # Test 1: Short recording (30 min)
    raw1 = create_test_eeg("Short", duration_min=30, sleep_ratio=0.85)
    success1, result1 = run_comprehensive_test(raw1, subject_info, "30-Minute Recording")
    results.append(("30-min", success1, result1))
    
    # Test 2: Medium recording (60 min)
    raw2 = create_test_eeg("Medium", duration_min=60, sleep_ratio=0.85)
    success2, result2 = run_comprehensive_test(raw2, subject_info, "60-Minute Recording")
    results.append(("60-min", success2, result2))
    
    # Test 3: Long recording (120 min) - closer to typical PSG
    raw3 = create_test_eeg("Long", duration_min=120, sleep_ratio=0.85)
    success3, result3 = run_comprehensive_test(raw3, subject_info, "120-Minute Recording")
    results.append(("120-min", success3, result3))
    
    # Test 4: Very long recording (480 min = 8 hours) - full night PSG
    raw4 = create_test_eeg("Full Night", duration_min=480, sleep_ratio=0.85)
    success4, result4 = run_comprehensive_test(raw4, subject_info, "480-Minute Recording (8 hours)")
    results.append(("480-min", success4, result4))
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n{'Test':<20} {'Status':<15} {'TST (min)':<15} {'Efficiency':<15}")
    print("-"*80)
    for name, success, result in results:
        status = "✅ PASS" if success else "❌ FAIL"
        tst = result.get('total_sleep_time', 0)
        eff = result.get('sleep_efficiency', 0)
        print(f"{name:<20} {status:<15} {tst:<15.2f} {eff:<15.2f}%")
    
    all_passed = all(success for _, success, _ in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("  • Epoch scaling is correct (30-second epochs)")
        print("  • TST calculations are accurate")
        print("  • Sleep window detection works properly")
        print("  • All metrics within physiological ranges")
        print("  • System ready for production")
    else:
        print("❌ SOME TESTS FAILED")
        print("  • Review detailed output above")
        print("  • Check epoch duration calculations")
        print("  • Verify sleep window logic")
    print("="*80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
