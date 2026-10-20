"""Affine-index36 input summaries on actual, shifted3x3 native-index stencils.

Preparation only: import/application and qualification require later admission.
No weather loader, learner, target, scaler, writer, padding, geographic transform
or physical fine-grid closure is defined here. Input shape is (...,36,2), with
stored vector components last and row-major parent IDs. Both coordinates and
derivatives use native-index units: coordinate order is (column,row).
"""
from __future__ import annotations
import numpy as np

OPERATOR_VERSION = 'AFFINE_INDEX36_SHIFTED3_V1'
RESIDUAL_ROUNDOFF_FACTOR = 64.0
RESIDUAL_SCALE_REFERENCE_MPS2 = 1.0


def build_geometry() -> dict[str, np.ndarray | str]:
    """Fixed36 actual-node templates and their analytical uniform-weight OLS.

    Every template is a Cartesian product of three distinct rows and columns.
    Thus centered index columns are orthogonal, each with sum-of-squares6.
    The target intercept is the fit at offset(0,0), including boundary targets.
    This is an algebraic input operator; no data-dependent geometry is chosen.
    """
    ids=np.empty((36,9),dtype=np.int64)
    offsets=np.empty((36,9,2),dtype=np.float64)
    for parent in range(36):
        row,col=divmod(parent,6)
        start_row=min(max(row-1,0),3)
        start_col=min(max(col-1,0),3)
        points=[(r,c) for r in range(start_row,start_row+3)
                       for c in range(start_col,start_col+3)]
        ids[parent]=[6*r+c for r,c in points]
        offsets[parent]=[(c-col,r-row) for r,c in points]
    offset_mean=offsets.mean(axis=1)
    centered=offsets-offset_mean[:,None,:]
    # Fixed Cartesian3x3 identity, not a pseudoinverse fitted to weather.
    slope_weights=np.swapaxes(centered,1,2)/6.0
    intercept_weights=np.full((36,9),1/9)-np.einsum('pi,pij->pj',offset_mean,slope_weights)
    coefficient_operator=np.concatenate([intercept_weights[:,None,:],slope_weights],axis=1)
    design=np.concatenate([np.ones((36,9,1)),offsets],axis=2)
    coordinate_covariance=np.einsum('pji,pjk->pik',centered,centered)/9
    for parent in range(36):
        assert len(set(ids[parent].tolist()))==9 and parent in ids[parent]
        assert ids[parent].min()>=0 and ids[parent].max()<36
    assert np.all(np.sum(centered*centered,axis=1)==6)
    assert np.all(np.sum(centered[:,:,0]*centered[:,:,1],axis=1)==0)
    return dict(version=OPERATOR_VERSION,neighbor_ids=ids,
                offsets_col_row=offsets,offset_mean_col_row=offset_mean,
                weights=np.full((36,9),1/9),design=design,
                coefficient_operator=coefficient_operator,
                coordinate_covariance=coordinate_covariance)


def apply_operator(field: np.ndarray) -> dict[str, np.ndarray | str]:
    """Summarize each target using only its nine real stored native vectors.

    Returned G axes are (...,parent,vector_component,index_axis), where index
    axes are(column,row). ``target_affine_vector`` is the affine intercept at
    the target coordinate, not a prediction of the actual fine-cell mean.

    Residual shape is Cres/trace(Cres) only when its trace exceeds the fixed
    floating-point resolution floor. Otherwise a zero storage matrix plus
    ``residual_shape_valid=False`` denotes an undefined/unresolved shape, never
    an isotropic tensor. Raw covariance and raw trace are never thresholded.
    The resolution floor is numerical, not a calm gate or a learned cutoff.
    """
    f=np.asarray(field,dtype=np.float64)
    if f.ndim<2 or f.shape[-2:]!=(36,2):
        raise ValueError('Expected (...,36,2) stored-component vectors')
    if not np.isfinite(f).all():
        raise ValueError('Nonfinite input vector')
    geometry=build_geometry()
    neighbors=f[...,geometry['neighbor_ids'],:]
    coefficients=np.einsum('pij,...pjc->...pic',geometry['coefficient_operator'],neighbors)
    fitted=np.einsum('pji,...pic->...pjc',geometry['design'],coefficients)
    residual=neighbors-fitted
    G=np.swapaxes(coefficients[...,1:,:],-1,-2)
    neighbor_mean=neighbors.mean(axis=-2)
    centered=neighbors-neighbor_mean[...,None,:]
    covariance=np.einsum('...pja,...pjb->...pab',centered,centered)/9
    residual_covariance=np.einsum('...pja,...pjb->...pab',residual,residual)/9
    residual_trace=np.trace(residual_covariance,axis1=-2,axis2=-1)
    explained_covariance=np.einsum('...pai,pij,...pbj->...pab',G,
                                  geometry['coordinate_covariance'],G)
    decomposition_error=covariance-explained_covariance-residual_covariance
    energy_scale=np.maximum(RESIDUAL_SCALE_REFERENCE_MPS2,
                            np.mean(np.sum(neighbors*neighbors,axis=-1),axis=-1))
    numerical_floor=(RESIDUAL_ROUNDOFF_FACTOR*np.finfo(np.float64).eps)**2*energy_scale
    shape_valid=residual_trace>numerical_floor
    shape=np.divide(residual_covariance,residual_trace[...,None,None],
                    out=np.zeros_like(residual_covariance),where=shape_valid[...,None,None])
    answer=dict(version=OPERATOR_VERSION,target_affine_vector=coefficients[...,0,:],
        G_col_row=G,G_frobenius_squared=np.sum(G*G,axis=(-2,-1)),
        neighbor_mean_vector=neighbor_mean,neighbor_covariance=covariance,
        neighbor_covariance_trace=np.trace(covariance,axis1=-2,axis2=-1),
        affine_explained_covariance=explained_covariance,
        residual_covariance=residual_covariance,residual_covariance_trace=residual_trace,
        residual_normalized_shape=shape,residual_shape_valid=shape_valid,
        residual_trace_numerical_floor=numerical_floor,
        covariance_decomposition_error=decomposition_error,
        target_minus_affine_vector=f-coefficients[...,0,:],
        neighbor_affine_vectors=fitted,neighbor_residual_vectors=residual,
        residual_mean_vector=residual.mean(axis=-2),
        neighbor_ids=geometry['neighbor_ids'],offsets_col_row=geometry['offsets_col_row'],
        weights=geometry['weights'])
    for key,value in answer.items():
        if isinstance(value,np.ndarray) and not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite operator output: '+key)
    return answer
