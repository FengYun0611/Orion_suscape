#!/usr/bin/env python3
"""Test that the data_infos structure handling works correctly."""

def test_data_infos_dict_format():
    """Test handling of dict format with 'infos' key."""
    # Simulate what might be in pkl file
    data_infos_dict = {
        'infos': [
            {'folder': 'scene-000000', 'timestamp': 123.0},
            {'folder': 'scene-000001', 'timestamp': 124.0},
        ],
        'metadata': {'version': '1.0'}
    }
    
    # Simulate the logic from load_annotations
    if isinstance(data_infos_dict, dict) and 'infos' in data_infos_dict:
        infos_list = data_infos_dict['infos']
    else:
        infos_list = data_infos_dict
    
    # Verify we got the list
    assert isinstance(infos_list, list), "Should extract list from dict"
    assert len(infos_list) == 2, "Should have 2 items"
    assert infos_list[0]['folder'] == 'scene-000000', "Should have correct scene name"
    
    # Simulate iterating (what happens in _load_csv_data_from_infos)
    scene_names = set()
    for info in infos_list:
        scene_name = info.get('folder', None)
        if scene_name:
            scene_names.add(scene_name)
    
    assert len(scene_names) == 2, "Should extract 2 unique scene names"
    assert 'scene-000000' in scene_names
    assert 'scene-000001' in scene_names
    print("✓ Dict format test passed")


def test_data_infos_list_format():
    """Test handling of list format (direct list of dicts)."""
    # Simulate what might be in pkl file (old format)
    data_infos_list = [
        {'folder': 'scene-000000', 'timestamp': 123.0},
        {'folder': 'scene-000001', 'timestamp': 124.0},
    ]
    
    # Simulate the logic from load_annotations
    if isinstance(data_infos_list, dict) and 'infos' in data_infos_list:
        infos_list = data_infos_list['infos']
    else:
        infos_list = data_infos_list
    
    # Verify we got the list
    assert isinstance(infos_list, list), "Should keep as list"
    assert len(infos_list) == 2, "Should have 2 items"
    
    # Simulate iterating
    scene_names = set()
    for info in infos_list:
        scene_name = info.get('folder', None)
        if scene_name:
            scene_names.add(scene_name)
    
    assert len(scene_names) == 2, "Should extract 2 unique scene names"
    print("✓ List format test passed")


if __name__ == '__main__':
    test_data_infos_dict_format()
    test_data_infos_list_format()
    print("\n✓ All tests passed!")
