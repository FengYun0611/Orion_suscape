#!/usr/bin/env python3
"""Test script to verify SUScapeOrionDataset has CLASSES attribute."""

import sys
sys.path.insert(0, '/home/runner/work/Orion_suscape/Orion_suscape')

try:
    # Test 1: Import the dataset class
    print("=" * 80)
    print("TEST 1: Importing SUScapeOrionDataset")
    print("=" * 80)
    from mmcv.datasets.suscape_orion_dataset import SUScapeOrionDataset
    print("✓ Successfully imported SUScapeOrionDataset")
    
    # Test 2: Check if CLASSES attribute exists
    print("\n" + "=" * 80)
    print("TEST 2: Checking CLASSES attribute")
    print("=" * 80)
    if hasattr(SUScapeOrionDataset, 'CLASSES'):
        print("✓ SUScapeOrionDataset has CLASSES attribute")
        print(f"  CLASSES = {SUScapeOrionDataset.CLASSES}")
        print(f"  Number of classes: {len(SUScapeOrionDataset.CLASSES)}")
    else:
        print("✗ FAIL: SUScapeOrionDataset does NOT have CLASSES attribute")
        sys.exit(1)
    
    # Test 3: Verify get_classes method works
    print("\n" + "=" * 80)
    print("TEST 3: Testing get_classes() method")
    print("=" * 80)
    try:
        classes = SUScapeOrionDataset.get_classes(None)
        print(f"✓ get_classes(None) returned: {classes}")
        if classes == SUScapeOrionDataset.CLASSES:
            print("✓ get_classes(None) matches CLASSES attribute")
        else:
            print("✗ WARNING: get_classes(None) does not match CLASSES attribute")
    except Exception as e:
        print(f"✗ FAIL: get_classes() failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ FAIL: Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
