"""Independent masks exercise spatial topology diagnostics."""
import numpy as np
import pytest

from app.pipeline.topology import compare_topology
from app.pipeline.assessment import assess_result


def ring():
    mask = np.zeros((80, 100), dtype=bool)
    mask[10:70, 10:90] = True
    mask[25:45, 25:45] = False
    return mask


def test_identical_regions_pass():
    result = compare_topology(ring(), ring())
    assert result['passed']
    assert result['holes']['reference_count'] == 1
    assert result['holes']['regions'][0]['iou'] == 1


def test_same_hole_count_at_different_position_fails():
    src = ring()
    dst = src.copy()
    dst[25:45, 25:45] = True
    dst[25:45, 55:75] = False
    result = compare_topology(src, dst)
    assert result['holes']['reference_count'] == result['holes']['rendered_count'] == 1
    assert result['holes']['regions'][0]['status'] == 'missing'
    assert len(result['holes']['extra_regions']) == 1
    assert not result['passed']


def test_hole_opened_to_exterior_fails():
    src = ring()
    dst = src.copy()
    dst[10:26, 30:35] = False
    result = compare_topology(src, dst)
    assert result['holes']['regions'][0]['status'] == 'missing'
    assert not result['passed']


def test_two_strokes_merged_fails():
    src = np.zeros((80, 100), dtype=bool)
    src[10:70, 10:30] = True
    src[10:70, 60:80] = True
    dst = src.copy()
    dst[35:40, 30:60] = True
    result = compare_topology(src, dst)
    assert {r['status'] for r in result['components']['regions']} == {'merged'}
    assert not result['passed']


def test_stroke_broken_fails():
    src = np.zeros((80, 100), dtype=bool)
    src[10:70, 10:30] = True
    dst = src.copy()
    dst[35:40] = False
    result = compare_topology(src, dst)
    assert result['components']['regions'][0]['status'] == 'split'
    assert not result['passed']


def test_hole_shrink_fails():
    src = ring()
    dst = src.copy()
    dst[25:35, 25:45] = True
    result = compare_topology(src, dst)
    assert result['holes']['regions'][0]['status'] == 'deformed'
    assert not result['passed']


def test_extra_foreground_fails():
    src = ring()
    dst = src.copy()
    dst[1:5, 1:5] = True
    result = compare_topology(src, dst)
    assert len(result['components']['extra_regions']) == 1
    assert not result['passed']


def test_small_hole_is_kept_and_noise_floor_is_explicit():
    src = np.zeros((30, 30), dtype=bool)
    src[5:25, 5:25] = True
    src[12:14, 12:14] = False
    dst = src.copy()
    dst[12:14, 12:14] = True
    assert not compare_topology(src, dst)['passed']
    assert compare_topology(src, dst, min_area=5)['passed']


def test_diagonal_background_is_not_a_hole_opening():
    src = np.ones((9, 9), dtype=bool)
    src[0, 0] = src[1, 1] = False
    result = compare_topology(src, src, min_area=1)
    assert result['holes']['reference_count'] == 1
    assert result['passed']


@pytest.mark.parametrize('reference,rendered', [
    (np.zeros((0, 0)), np.zeros((0, 0))),
    (np.zeros((5, 5)), np.zeros((4, 5))),
    (np.zeros(5), np.zeros(5)),
])
def test_invalid_mask_shape_rejected(reference, rendered):
    with pytest.raises(ValueError):
        compare_topology(reference, rendered)


@pytest.mark.parametrize('setting', [{'min_area':0}, {'min_iou':0}, {'min_iou':float('nan')}])
def test_invalid_settings_rejected(setting):
    with pytest.raises(ValueError):
        compare_topology(ring(), ring(), **setting)


def test_assessment_requires_truth_topology_evidence():
    result = assess_result({}, {}, None, (1000, 1000))
    assert result['items']['truth_topology']['status'] == 'untested'
    result = assess_result({'truth_topology':compare_topology(ring(),ring())}, {}, None, (1000,1000))
    assert result['items']['truth_topology']['status'] == 'pass'
    dst = ring()
    dst[25:45,25:45] = True
    result = assess_result({'truth_topology':compare_topology(ring(),dst)}, {}, None, (1000,1000))
    assert result['items']['truth_topology']['status'] == 'fail'
    assert result['status'] == 'fail'
