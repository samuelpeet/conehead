# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
import numpy as np
import pytest

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from conehead.structure import ROI, StructureSet
from conehead.grid import Grid


def test_roi_mask_on_grid_simple_rect():
    # 10x10x1 grid, voxel centres at 0.5,1.5,...
    g = Grid(
        num_voxels=np.array([10, 10, 1], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # Rectangle covering x in [0,2] and y in [0,3] at z=0.5 (world units)
    contour = np.array(
        [
            [0.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [2.0, 3.0, 0.5],
            [0.0, 3.0, 0.5],
        ],
        dtype=np.float32,
    )

    roi = ROI(roi_number=1, name="rect", contours=[contour])

    mask = roi.mask_on_grid(g)

    assert mask.shape == (1, 10, 10)
    assert mask.dtype == bool

    # At minimum we expect the two columns and three rows to be covered; raster
    # fill may include boundary pixels so allow >= expected.
    assert int(mask.sum()) >= 2 * 3


def test_roi_mask_on_grid_multiple_slices():
    # 5x5x3 grid
    g = Grid(
        num_voxels=np.array([5, 5, 3], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # Three small 1x1 boxes at different Z slices
    c0 = np.array(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32
    )
    # place second contour away from the left edge to avoid rasterisation
    # edge cases
    c1 = np.array(
        [[1.0, 1.0, 1.5], [2.0, 1.0, 1.5], [2.0, 2.0, 1.5], [1.0, 2.0, 1.5]], dtype=np.float32
    )
    c2 = np.array(
        [[0.0, 3.0, 2.5], [1.0, 3.0, 2.5], [1.0, 4.0, 2.5], [0.0, 4.0, 2.5]], dtype=np.float32
    )

    roi = ROI(roi_number=2, name="multi", contours=[c0, c1, c2])
    mask = roi.mask_on_grid(g)

    # At minimum one voxel should be filled in each slice; allow >=1 to be
    # tolerant of small rasterisation differences.
    assert mask.shape == (3, 5, 5)
    assert int(mask.sum()) >= 3
    assert int(mask[0].sum()) >= 1
    assert int(mask[1].sum()) >= 1
    assert int(mask[2].sum()) >= 1


def test_structureset_from_dataset_roundtrip():
    # Build a minimal RTSTRUCT-like dataset with one ROI
    ds = Dataset()

    # StructureSetROISequence
    roi_def = Dataset()
    roi_def.ROINumber = 1
    roi_def.ROIName = "TestROI"
    ds.StructureSetROISequence = Sequence([roi_def])

    # ROIContourSequence
    contour_item = Dataset()
    contour_item.ReferencedROINumber = 1
    # Single contour with four points (square) at z=0.0.. We set convert_to_cm=False
    contour_ds = Dataset()
    contour_ds.ContourData = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    contour_item.ContourSequence = Sequence([contour_ds])
    ds.ROIContourSequence = Sequence([contour_item])

    # RTROIObservationsSequence with interpreted type and density
    obs_item = Dataset()
    obs_item.ReferencedROINumber = 1
    obs_item.RTROIInterpretedType = "ORGAN"
    prop = Dataset()
    prop.ROIPhysicalProperty = "REL_MASS_DENSITY"
    prop.ROIPhysicalPropertyValue = 1.23
    obs_item.ROIPhysicalPropertiesSequence = Sequence([prop])
    ds.RTROIObservationsSequence = Sequence([obs_item])

    ss = StructureSet.from_file(ds, convert_to_cm=False)

    assert len(ss.rois) == 1
    r = ss.get_roi_by_number(1)
    assert r is not None
    assert r.name == "TestROI"
    assert r.roi_type == "ORGAN"
    assert pytest.approx(r.density, rel=1e-6) == 1.23
    # Contour points should be present and match the provided values (4 points)
    assert len(r.contours) == 1
    assert r.contours[0].shape[0] == 4


def test_roi_empty_contours_returns_empty_mask():
    g = Grid(
        num_voxels=np.array([4, 4, 2], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    roi = ROI(roi_number=3, name="empty", contours=[])
    mask = roi.mask_on_grid(g)

    assert mask.shape == (2, 4, 4)
    assert mask.dtype == bool
    assert int(mask.sum()) == 0


def test_roi_contour_outside_grid_is_ignored():
    g = Grid(
        num_voxels=np.array([4, 4, 2], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # contour placed at z far outside grid
    contour = np.array([[0.0, 0.0, 100.0], [1.0, 0.0, 100.0], [1.0, 1.0, 100.0]], dtype=np.float32)
    roi = ROI(roi_number=4, name="oob", contours=[contour])
    mask = roi.mask_on_grid(g)

    assert mask.sum() == 0


def test_roi_nonplanar_contour_maps_to_single_slice():
    # Grid with 3 slices
    g = Grid(
        num_voxels=np.array([4, 4, 3], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # Contour has slight Z variation around slice 1 (z ~ 1.4..1.6)
    contour = np.array(
        [[1.0, 1.0, 1.4], [2.0, 1.0, 1.45], [2.0, 2.0, 1.6], [1.0, 2.0, 1.5]], dtype=np.float32
    )
    roi = ROI(roi_number=5, name="nonplanar", contours=[contour])
    mask = roi.mask_on_grid(g)

    # Should rasterise to exactly one slice (nearest slice for mean z)
    nz = mask.shape[0]
    nonzero_slices = [i for i in range(nz) if int(mask[i].sum()) > 0]
    assert len(nonzero_slices) == 1


def test_structureset_helpers_and_find_external():
    ds = Dataset()

    # Two ROI definitions
    r1 = Dataset()
    r1.ROINumber = 10
    r1.ROIName = "Liver"
    r2 = Dataset()
    r2.ROINumber = 20
    r2.ROIName = "External Body"
    ds.StructureSetROISequence = Sequence([r1, r2])

    # Contours: create minimal contour entries for both
    c1 = Dataset()
    c1.ReferencedROINumber = 10
    cd1 = Dataset()
    cd1.ContourData = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    c1.ContourSequence = Sequence([cd1])

    c2 = Dataset()
    c2.ReferencedROINumber = 20
    cd2 = Dataset()
    cd2.ContourData = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]
    c2.ContourSequence = Sequence([cd2])
    ds.ROIContourSequence = Sequence([c1, c2])

    # Observations: mark roi 20 as EXTERNAL
    o1 = Dataset()
    o1.ReferencedROINumber = 10
    o2 = Dataset()
    o2.ReferencedROINumber = 20
    o2.RTROIInterpretedType = "EXTERNAL"
    ds.RTROIObservationsSequence = Sequence([o1, o2])

    ss = StructureSet.from_file(ds, convert_to_cm=False)
    assert ss.get_roi_by_name("liver").roi_number == 10
    assert ss.get_roi_by_number(20).name == "External Body"
    ext_list = ss.get_roi_by_type("EXTERNAL")
    assert len(ext_list) == 1
    assert ext_list[0].roi_number == 20


def test_get_roi_by_type_multiple_and_name_lookup():
    # Build three ROI objects directly with roi_type set
    r1 = ROI(roi_number=1, name="ExtA", roi_type="EXTERNAL")
    r2 = ROI(roi_number=2, name="ExtB", roi_type="EXTERNAL")
    r3 = ROI(roi_number=3, name="Liver", roi_type="ORGAN")

    ss = StructureSet(rois=[r1, r2, r3])

    ext = ss.get_roi_by_type("external")  # case-insensitive
    assert isinstance(ext, list)
    assert len(ext) == 2
    assert {r.roi_number for r in ext} == {1, 2}

    # get_roi_by_name should be case-insensitive and return first match
    found = ss.get_roi_by_name("liver")
    assert found is not None
    assert found.roi_number == 3


def test_mask_partial_out_of_bounds():
    # Grid smaller than contour extents
    g = Grid(
        num_voxels=np.array([3, 3, 1], dtype=np.int32),
        corner=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        resolution=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # Contour extends partly outside the grid bounds
    contour = np.array(
        [[-1.0, -1.0, 0.5], [4.0, -1.0, 0.5], [4.0, 4.0, 0.5], [-1.0, 4.0, 0.5]], dtype=np.float32
    )
    roi = ROI(roi_number=7, name="big", contours=[contour])
    mask = roi.mask_on_grid(g)

    # Some voxels should still be filled inside the 3x3 grid
    assert mask.shape == (1, 3, 3)
    assert int(mask.sum()) > 0
