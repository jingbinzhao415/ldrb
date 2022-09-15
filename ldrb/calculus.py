import numba
import numpy as np
import quaternion


def bislerp(
    Qa: np.ndarray,
    Qb: np.ndarray,
    t: float,
) -> np.ndarray:
    r"""
    Linear interpolation of two orthogonal matrices.
    Assiume that :math:`Q_a` and :math:`Q_b` refers to
    timepoint :math:`0` and :math:`1` respectively.
    Using spherical linear interpolation (slerp) find the
    orthogonal matrix at timepoint :math:`t`.
    """

    if ~Qa.any() and ~Qb.any():
        return np.zeros((3, 3))
    if ~Qa.any():
        return Qb
    if ~Qb.any():
        return Qa

    tol = 1e-12  #1e-12
    qa = quaternion.from_rotation_matrix(Qa)
    qb = quaternion.from_rotation_matrix(Qb)

    quat_i = quaternion.quaternion(0, 1, 0, 0)
    quat_j = quaternion.quaternion(0, 0, 1, 0)
    quat_k = quaternion.quaternion(0, 0, 0, 1)

    quat_array = [
        qa,
        -qa,
        qa * quat_i,
        -qa * quat_i,
        qa * quat_j,
        -qa * quat_j,
        qa * quat_k,
        -qa * quat_k,
    ]

    dot_arr = [abs((qi.components * qb.components).sum()) for qi in quat_array]
    max_idx = int(np.argmax(dot_arr))
    max_dot = dot_arr[max_idx]
    qm = quat_array[max_idx]

    if max_dot > 1 - tol:
        return Qb

    qm_slerp = quaternion.slerp(qm, qb, 0, 1, t)
    qm_norm = qm_slerp.normalized()
    Qab = quaternion.as_rotation_matrix(qm_norm)
    return Qab


def system_at_dof(
    lv: float,
    rv: float,
    epi: float,
    grad_lv: np.ndarray,
    grad_rv: np.ndarray,
    grad_epi: np.ndarray,
    grad_ab: np.ndarray,
    alpha_endo: float,
    alpha_epi: float,
    beta_endo: float,
    beta_epi: float,
    tol: float = 1e-7# 1e-7,
) -> np.ndarray:
    """
    Compte the fiber, sheet and sheet normal at a
    single degre of freedom

    Arguments
    ---------
    lv : float
        Value of the Laplace solution for the LV at the dof.
    rv : float
        Value of the Laplace solution for the RV at the dof.
    epi : float
        Value of the Laplace solution for the EPI at the dof.
    grad_lv : np.ndarray
        Gradient of the Laplace solution for the LV at the dof.
    grad_rv : np.ndarray
        Gradient of the Laplace solution for the RV at the dof.
    grad_epi : np.ndarray
        Gradient of the Laplace solution for the EPI at the dof.
    grad_epi : np.ndarray
        Gradient of the Laplace solution for the apex to base.
        at the dof
    alpha_endo : scalar
        Fiber angle at the endocardium.
    alpha_epi : scalar
        Fiber angle at the epicardium.
    beta_endo : scalar
        Sheet angle at the endocardium.
    beta_epi : scalar
        Sheet angle at the epicardium.
    tol : scalar
        Tolerance for whether to consider the scalar values.
        Default: 1e-7
    """

    if lv + rv < tol:
        depth = 0.5
    else:
        depth = rv / (lv + rv)

    alpha_s = alpha_endo * (1 - depth) - alpha_endo * depth
    alpha_w = alpha_endo * (1 - epi) + alpha_epi * epi
    beta_s = beta_endo * (1 - depth) - beta_endo * depth
    beta_w = beta_endo * (1 - epi) + beta_epi * epi

    Q_epi = np.zeros((3, 3))
    Q_epiaxis = np.zeros((3, 3))

    if epi >= tol:
        Q_epi = axis(grad_ab, grad_epi)
        Q_epiaxis = axis(grad_ab, grad_epi)
        Q_epi = orient(Q_epiaxis, alpha_w, beta_w)

    if lv >= tol:
        Q_lv = np.zeros((3, 3))
        Q_lvaxis = np.zeros((3, 3))

        Q_lv = axis(grad_ab, -1 * grad_lv)
        Q_lvaxis = axis(grad_ab, -1 * grad_lv)
        Q_lv = orient(Q_lvaxis, alpha_w, beta_w)
        Q_rvaxis = np.zeros((3, 3))
        Q_rv = np.zeros((3, 3))

        if rv >= tol:

            Q_rv = np.zeros((3, 3))
            Q_rvaxis = np.zeros((3, 3))
        
            Q_rv = axis(grad_ab, grad_rv)
            Q_rvaxis = axis(grad_ab, grad_rv)
            Q_rv = orient(Q_rvaxis, -alpha_w, -beta_w)

    Q_endo = bislerp(Q_lv, Q_rv, depth)
    Q_fiber = bislerp(Q_endo, Q_epi, epi)

    return Q_fiber, Q_lvaxis, Q_rvaxis, Q_epiaxis,Q_lv,Q_rv,Q_epi,Q_endo,alpha_w


@numba.njit
def normalize(u: np.ndarray) -> np.ndarray:
    """
    Normalize vector
    """
    return u / np.linalg.norm(u)


@numba.njit
def axis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Construct the fiber orientation coordinate system.

    Given two vectors :math:`u` and :math:`v` in the apicobasal
    and transmural direction respectively return a matrix that
    represents an orthonormal basis in the circumferential (first),
    apicobasal (second) and transmural (third) direction.
    """

    e1 = normalize(u)

    # Create an initial guess for e0
    e2 = normalize(v)
    e2 -= e1.dot(e2) * e1
    e2 = normalize(e2)

    e0 = np.cross(e1, e2)
    e0 = normalize(e0)

    Q = np.zeros((3, 3))
    Q[:, 0] = e0
    Q[:, 1] = e1
    Q[:, 2] = e2

    return Q


@numba.njit
def orient(Q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    r"""
    Define the orthotropic fiber orientations.

    Given a coordinate system :math:`Q`, in the canonical
    basis, rotate it in order to align with the fiber, sheet
    and sheet-normal axis determine by the angles :math:`\alpha`
    (fiber) and :math:`\beta` (sheets).
    """
    A = np.zeros((3, 3))
    A[0, :] = [np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0]
    A[1, :] = [np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0]
    A[2, :] = [0, 0, 1]

    B = np.zeros((3, 3))
    B[0, :] = [1, 0, 0]
    B[1, :] = [0, np.cos(np.radians(beta)), np.sin(np.radians(beta))]
    B[2, :] = [0, -np.sin(np.radians(beta)), np.cos(np.radians(beta))]

    C = np.dot(Q.real, A).dot(B)
    return C


def _compute_fiber_sheet_system(
    f0,
    s0,
    n0,
    lvaf0,
    lvas0,
    lvan0, 
    rvaf0,
    rvas0,
    rvan0,
    epiaf0,
    epias0,
    epian0, 
    Q_lvf0,
    Q_lvs0,
    Q_lvn0,
    Q_rvf0,
    Q_rvs0,
    Q_rvn0,
    Q_epif0,
    Q_epis0,
    Q_epin0,
    Q_endof0,
    Q_endos0,
    Q_endon0,
    alphaw,              
    xdofs,
    ydofs,
    zdofs,
    sdofs,
    lv_scalar,
    rv_scalar,
    epi_scalar,
    lv_rv_scalar,
    lv_gradient,
    rv_gradient,
    epi_gradient,
    apex_gradient,
    marker_scalar,
    alpha_endo_lv,
    alpha_epi_lv,
    alpha_endo_rv,
    alpha_epi_rv,
    alpha_endo_sept,
    alpha_epi_sept,
    beta_endo_lv,
    beta_epi_lv,
    beta_endo_rv,
    beta_epi_rv,
    beta_endo_sept,
    beta_epi_sept,
    tol,
):
    grad_lv = np.zeros(3)
    grad_rv = np.zeros(3)
    grad_epi = np.zeros(3)
    grad_ab = np.zeros(3)

    for i in range(len(xdofs)):

        lv = lv_scalar[sdofs[i]]
        if lv < tol:
            lv = tol
        rv = rv_scalar[sdofs[i]]
        if rv is None and lv >= tol:
            rv = 0
            if rv < tol:
                rv = tol        
        epi = epi_scalar[sdofs[i]]
        if epi < tol:
            epi = tol
        lv_rv = lv_rv_scalar[sdofs[i]]
        if lv_rv < tol:
            lv_rv = tol
        #print("lv is#######", lv)    
        #print("rv is#######", rv)
        #print("epi is#######", epi)
        #print("lv_rv is#######", lv_rv)


        grad_lv[0] = lv_gradient[xdofs[i]]
        grad_lv[1] = lv_gradient[ydofs[i]]
        grad_lv[2] = lv_gradient[zdofs[i]]

        grad_rv[0] = rv_gradient[xdofs[i]]
        grad_rv[1] = rv_gradient[ydofs[i]]
        grad_rv[2] = rv_gradient[zdofs[i]]

        grad_epi[0] = epi_gradient[xdofs[i]]
        grad_epi[1] = epi_gradient[ydofs[i]]
        grad_epi[2] = epi_gradient[zdofs[i]]

        grad_ab[0] = apex_gradient[xdofs[i]]
        grad_ab[1] = apex_gradient[ydofs[i]]
        grad_ab[2] = apex_gradient[zdofs[i]]

        if epi > 0.5:
            
            if lv_rv >= 0.5:
                # We are in the LV region
                marker_scalar[sdofs[i]] = 1
                alpha_endo = alpha_endo_lv
                beta_endo = beta_endo_lv
                alpha_epi = alpha_epi_lv
                beta_epi = beta_epi_lv
            else:
                # We are in the RV region
                marker_scalar[sdofs[i]] = 2
                alpha_endo = alpha_endo_rv
                beta_endo = beta_endo_rv
                alpha_epi = alpha_epi_rv
                beta_epi = beta_epi_rv
        else:
            if lv_rv >= 0.9:
                # We are in the LV region
                marker_scalar[sdofs[i]] = 1
                alpha_endo = alpha_endo_lv
                beta_endo = beta_endo_lv
                alpha_epi = alpha_epi_lv
                beta_epi = beta_epi_lv
            #elif lv_rv <= tol :
            elif  lv_rv <= 0.1:
                # We are in the RV region
                marker_scalar[sdofs[i]] = 2
                alpha_endo = alpha_endo_rv
                beta_endo = beta_endo_rv
                alpha_epi = alpha_epi_rv
                beta_epi = beta_epi_rv
            else:
                # We are in the septum
                marker_scalar[sdofs[i]] = 3
                alpha_endo = alpha_endo_sept
                beta_endo = beta_endo_sept
                alpha_epi = alpha_epi_sept
                beta_epi = beta_epi_sept

        Q_fiber, Q_lvaxis, Q_rvaxis, Q_epiaxis,Q_lv,Q_rv,Q_epi,Q_endo,alpha_w = system_at_dof(
            lv=lv,
            rv=rv,
            epi=epi,
            grad_lv=grad_lv,
            grad_rv=grad_rv,
            grad_epi=grad_epi,
            grad_ab=grad_ab,
            alpha_endo=alpha_endo,
            alpha_epi=alpha_epi,
            beta_endo=beta_endo,
            beta_epi=beta_epi,
            tol=tol,
        )
        if Q_fiber is None:
            continue


        f0[xdofs[i]] = Q_fiber[0, 0]
        f0[ydofs[i]] = Q_fiber[1, 0]
        f0[zdofs[i]] = Q_fiber[2, 0]

        s0[xdofs[i]] = Q_fiber[0, 1]
        s0[ydofs[i]] = Q_fiber[1, 1]
        s0[zdofs[i]] = Q_fiber[2, 1]

        n0[xdofs[i]] = Q_fiber[0, 2]
        n0[ydofs[i]] = Q_fiber[1, 2]
        n0[zdofs[i]] = Q_fiber[2, 2]
##
        lvaf0[xdofs[i]] = Q_lvaxis[0, 0]
        lvaf0[ydofs[i]] = Q_lvaxis[1, 0]
        lvaf0[zdofs[i]] = Q_lvaxis[2, 0]

        lvas0[xdofs[i]] = Q_lvaxis[0, 1]
        lvas0[ydofs[i]] = Q_lvaxis[1, 1]
        lvas0[zdofs[i]] = Q_lvaxis[2, 1]

        lvan0[xdofs[i]] = Q_lvaxis[0, 2]
        lvan0[ydofs[i]] = Q_lvaxis[1, 2]
        lvan0[zdofs[i]] = Q_lvaxis[2, 2]

##
        rvaf0[xdofs[i]] = Q_rvaxis[0, 0]
        rvaf0[ydofs[i]] = Q_rvaxis[1, 0]
        lvaf0[zdofs[i]] = Q_rvaxis[2, 0]

        rvas0[xdofs[i]] = Q_rvaxis[0, 1]
        rvas0[ydofs[i]] = Q_rvaxis[1, 1]
        rvas0[zdofs[i]] = Q_rvaxis[2, 1]

        rvan0[xdofs[i]] = Q_rvaxis[0, 2]
        rvan0[ydofs[i]] = Q_rvaxis[1, 2]
        rvan0[zdofs[i]] = Q_rvaxis[2, 2]


##
        epiaf0[xdofs[i]] = Q_epiaxis[0, 0]
        epiaf0[ydofs[i]] = Q_epiaxis[1, 0]
        epiaf0[zdofs[i]] = Q_epiaxis[2, 0]

        epias0[xdofs[i]] = Q_epiaxis[0, 1]
        epias0[ydofs[i]] = Q_epiaxis[1, 1]
        epias0[zdofs[i]] = Q_epiaxis[2, 1]

        epian0[xdofs[i]] = Q_epiaxis[0, 2]
        epian0[ydofs[i]] = Q_epiaxis[1, 2]
        epian0[zdofs[i]] = Q_epiaxis[2, 2] 
##
        Q_lvf0[xdofs[i]] = Q_lv[0, 0]
        Q_lvf0[ydofs[i]] = Q_lv[1, 0]
        Q_lvf0[zdofs[i]] = Q_lv[2, 0]

        Q_lvs0[xdofs[i]] = Q_lv[0, 1]
        Q_lvs0[ydofs[i]] = Q_lv[1, 1]
        Q_lvs0[zdofs[i]] = Q_lv[2, 1]

        Q_lvn0[xdofs[i]] = Q_lv[0, 2]
        Q_lvn0[ydofs[i]] = Q_lv[1, 2]
        Q_lvn0[zdofs[i]] = Q_lv[2, 2]
##
        Q_rvf0[xdofs[i]] = Q_rv[0, 0]
        Q_rvf0[ydofs[i]] = Q_rv[1, 0]
        Q_rvf0[zdofs[i]] = Q_rv[2, 0]

        Q_rvs0[xdofs[i]] = Q_rv[0, 1]
        Q_rvs0[ydofs[i]] = Q_rv[1, 1]
        Q_rvs0[zdofs[i]] = Q_rv[2, 1]

        Q_rvn0[xdofs[i]] = Q_rv[0, 2]
        Q_rvn0[ydofs[i]] = Q_rv[1, 2]
        Q_rvn0[zdofs[i]] = Q_rv[2, 2]
##
        Q_epif0[xdofs[i]] = Q_epi[0, 0]
        Q_epif0[ydofs[i]] = Q_epi[1, 0]
        Q_epif0[zdofs[i]] = Q_epi[2, 0]

        Q_epis0[xdofs[i]] = Q_epi[0, 1]
        Q_epis0[ydofs[i]] = Q_epi[1, 1]
        Q_epis0[zdofs[i]] = Q_epi[2, 1]

        Q_epin0[xdofs[i]] = Q_epi[0, 2]
        Q_epin0[ydofs[i]] = Q_epi[1, 2]
        Q_epin0[zdofs[i]] = Q_epi[2, 2]
##
        Q_endof0[xdofs[i]] = Q_endo[0, 0]
        Q_endof0[ydofs[i]] = Q_endo[1, 0]
        Q_endof0[zdofs[i]] = Q_endo[2, 0]

        Q_endos0[xdofs[i]] = Q_endo[0, 1]
        Q_endos0[ydofs[i]] = Q_endo[1, 1]
        Q_endos0[zdofs[i]] = Q_endo[2, 1]

        Q_endon0[xdofs[i]] = Q_endo[0, 2]
        Q_endon0[ydofs[i]] = Q_endo[1, 2]
        Q_endon0[zdofs[i]] = Q_endo[2, 2]

        #marker_scalar[sdofs[i]] = marker_scalar
        alphaw[sdofs[i]] = alpha_w
        #alpha_s[sdofs[i]] = alpha_s



