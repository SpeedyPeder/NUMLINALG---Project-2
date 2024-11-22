import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, LinearOperator
from time import perf_counter

def lhs_func(U, N, v=np.array([1, 1])):
    """Computes the left-hand side vector based on the input array U for diffusion-advection problem."""
    h = 1 / N
    U = np.reshape(U, (N+1, N+1))
    lhs = np.zeros((N + 1, N + 1))
    #Use numpy index to calculate
    index = np.arange(1,N) # indices corresponding to internal nodes
    ixy = np.ix_(index,index)
    ixm_y = np.ix_(index-1,index)
    ixp_y = np.ix_(index+1,index)
    ix_ym = np.ix_(index,index-1)
    ix_yp = np.ix_(index,index+1)
    #Lhs with numpy index
    lhs[ixy] = ((4*U[ixy] - U[ixp_y] - U[ixm_y] - U[ix_yp] - U[ix_ym])
                + h* v[0] * (U[ixy] - U[ixm_y])
                + h* v[1] * (U[ixy] - U[ix_ym]))
    return lhs

def rhs_func(N, v=np.array([1, 1])):
    """Constructs the right-hand side vector f based on the analytical solution u(x, y) = sin(pi x) * sin(2 * pi y)."""
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    
    # Analytical function for f based on the exact solution u(x, y) = sin(pi x) * sin(2 * pi y)
    f = (5 * np.pi**2 * np.sin(np.pi * X) * np.sin(2 * np.pi * Y) +
         v[0] * np.pi * np.cos(np.pi * X) * np.sin(2 * np.pi * Y) +
         v[1] * 2 * np.pi * np.sin(np.pi * X) * np.cos(2 * np.pi * Y))
    
    #Set the boundaries to zero, since u = g = 0 at the boundaries
    f[:, 0] = f[:, -1] = f[0, :] = f[-1, :] = 0

    # Scale by h^2 as per the finite difference scheme
    return f * h**2

def solve_scipy_gmres(U0, rhs, N, v=np.array([1, 1]), tol=1e-10):
    """Solve the diffusion-advection problem using GMRES for a grid of size (N+1) x (N+1)."""
    x0 = U0.flatten()  # Flatten the initial guess
    
    # Define the linear operator for the problem
    def matvec(u_flat):
        u = u_flat.reshape((N + 1, N + 1))
        return lhs_func(u, N, v).flatten()

    iteration_count = [0]
    def count_iterations(xk):
        """Callback to increment the iteration count."""
        iteration_count[0] += 1

    A = LinearOperator((x0.size, x0.size), matvec=matvec)
    b = rhs.flatten()  # Flatten the RHS

    # Solve using GMRES
    u_flat, exitCode = gmres(A, b, tol=tol, callback = count_iterations)

    if exitCode != 0:
        print(f"GMRES did not converge, exit code: {exitCode}")

    # Reshape the solution back into a grid
    return u_flat.reshape((N + 1, N + 1)), iteration_count[0]

def u_exact(N):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    # Compute the exact solution at each grid point
    u_exact = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    
    return u_exact

def arnoldi(m, N, omega, v=np.array([1, 1])):
    """
    Arnoldi's method to generate an orthonormal basis for the Krylov subspace
    
    Parameters:
        v: The starting velocity field
        m: The number of basis vectors to compute.
        N: Grid size
        omega: Initial residual vector/ starting vector (N+1)^2
    
    Returns:
        V: The orthonormal basis vectors
        H: The upper Hessenberg matrix 
    """
    n = (N+1)**2  # Total number of grid points including boundary
    V = np.zeros((n, m+1))  # Orthonormal basis
    H = np.zeros((m+1, m))  # Upper Hessenberg matrix with m+1 rows
    omega = omega.flatten() #Make sure omega is in vector format
    
    # Normalize the first vector
    V[:,0] = omega / np.linalg.norm(omega)
    for j in range(m):
        # Compute the next vector using lhs_func to apply the operator
        w = lhs_func(V[:, j], N, v).flatten()  # Apply the operator to the vector V[:, j]
        # Orthogonalization against previous vectors
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if np.abs(H[j+1,j]) <= 1.e-12:
            print("H[j+1,j] is close to 0")            
            break
        V[:,j+1] = w/H[j+1,j]
    return V, H 

def restarted_gmres(U0, rhs, m, N, tol=1e-4, v=np.array([1, 1])):
    """
    Restarted GMRES method using Arnoldi's method to solve Ax = b.

    Parameters:
        U0: The initial guess matrix 
        rhs: Righrt hand side matrix
        m: Number of iterations before restart.
        tol: Convergence tolerance.
        N: Grid size
        v: Advection speed
    
    Returns:
        x: The computed solution vector
        count: Number of iterations
        time: Converging time
    """
    start = perf_counter()
    x = U0.copy().flatten()      #Initial guess vector
    b = rhs.flatten()  # Right-hand side vector
    r_0 = b - lhs_func(x, N, v).flatten()  #Error matrix
    norm_r0 = np.linalg.norm(r_0)
    
    if norm_r0 < tol:  # Already converged
        return x, r_0, 0, 0
    
    count = 0
    res_arr = [norm_r0]
    while True:
        r_m = b - lhs_func(x, N, v).flatten()  # New residual
        omega_1 = r_m / np.linalg.norm(r_m)
        V, H = arnoldi(m, N, omega=omega_1, v=v)  # Create an Arnoldi basis
        e1 = np.zeros(m + 1)
        e1[0] = np.linalg.norm(r_m)
        y = np.linalg.lstsq(H, e1, rcond = None)[0]
        x += V[:, :y.size] @ y  # Only use the relevant part of V and y
        count += 1
        res_arr.append(np.linalg.norm(r_m))
        if np.linalg.norm(r_m) / norm_r0 < tol:  # Check convergence
            break
    stop = perf_counter()
    time = stop - start
    return x.reshape((N+1, N+1)), res_arr, count, time  # Reshape the solution into (N+1)x(N+1) grid
    
def jacobi_relax(u, rhs, omega, N, nu, v = np.array([1,1])):
    """
    Perform weighted Jacobi iterations with upwind differencing for advection.
    
    Parameters:
        u: Initial guess (N+1)x(N+1) matrix.
        rhs: Right-hand side (N+1)x(N+1) matrix.
        omega: Relaxation parameter.
        N: Grid size.
        nu: Number of Jacobi iterations.
        v: Velocity field [v_x, v_y].
        
    Returns:
        Updated solution u after nu iterations.
    """
    h = 1 / N
    index = np.arange(1, N)  # Indices corresponding to internal nodes
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)  # i-1, j
    ixp_y = np.ix_(index + 1, index)  # i+1, j
    ix_ym = np.ix_(index, index - 1)  # i, j-1
    ix_yp = np.ix_(index, index + 1)  # i, j+1
    
    div = (4 + v[0] * h + v[1] * h) # Denominator includes advection terms
    for _ in range(nu):
        u_new = u.copy() # Create a copy to avoid overwriting in the same iteration
        # Jacobi iteration with upwind differencing for advection
        u_new[ixy] = ((1 - omega) * u[ixy] + 
                     omega*( (u[ixp_y] + u[ixm_y] * (1 + v[0] * h) + 
                      u[ix_yp] + u[ix_ym] * (1 + v[1] * h))/div + (rhs[ixy]/div) ))
        u = u_new  #Update the solution
    return u

    
def residual(u, rhs, N, v = np.array([1,1])):
    """
    Compute the residual rf = rhs - A @ u, where A is the discretized operator.
    
    Parameters:
        u: Current solution (N+1)x(N+1) matrix.
        rhs: Right-hand side (N+1)x(N+1) matrix.
        N: Grid size.
        
    Returns:
        Residual matrix.
    """
    rhs = np.reshape(rhs,(N+1,N+1)) #In case rhs is not in matrix format
    rf = rhs - lhs_func(u, N, v=v)
    return rf

def restriction(e, N):
    """
    Restrict the error from the fine grid to the coarse grid using a 2D full-weighting restriction.

    Parameters:
        e: Error matrix on the fine grid (N+1)x(N+1).
        N: Grid size of the fine grid, allways choose N to be even.
        
    Returns:
        Restricted error matrix on the coarse grid ((N//2)+1)x((N//2)+1).
    """
    # Calculate the coarse grid size
    Nc = N // 2
    rc = np.zeros((Nc + 1, Nc + 1))
    
    
    # Full-weighting restriction for the interior points
    for i in range(1, Nc):
        for j in range(1, Nc):
            rc[i, j] = (1/16) * (
                4 * e[2*i, 2*j] +         # Center point
                2 * (e[2*i-1, 2*j] + e[2*i+1, 2*j] + e[2*i, 2*j-1] + e[2*i, 2*j+1]) +  # Adjacent points
                (e[2*i-1, 2*j-1] + e[2*i-1, 2*j+1] + e[2*i+1, 2*j-1] + e[2*i+1, 2*j+1])  # Diagonal points
            )
    
    #Keep the edges 
    for j in range(1, Nc):
        rc[0, j] = e[0,2*j] 
        rc[-1, j] = e[-1,2*j] 
        rc[j, 0] = e[2*j,0] 
        rc[j, -1] = e[2*j,-1]
        
    #Corners
    rc[0, 0] = e[0, 0]
    rc[0, -1] = e[0, -1]
    rc[-1, 0] = e[-1, 0]
    rc[-1, -1] = e[-1, -1] 

    return rc


def interpolation(e, N):
    """
    Perform interpolation (prolongation) from the coarse grid to the fine grid
    as 4 times the transpose of the restriction operator.

    Parameters:
        e: Error matrix on the coarse grid (N//2 +1)x(N//2 +1).
        N: Grid size of the course grid

    Returns:
        Interpolated error matrix on the fine grid (N+1)x(N+1).
    """
    Nf = N*2
    ef = np.zeros((Nf + 1, Nf + 1))
    if e.ndim == 1:
        e = e.reshape((N + 1, N + 1))  # Reshape to 2D if accidentally flattened
    
    # Interpolate by setting ef to be 4 times the transpose of restriction
    for i in range(0, N):
        for j in range(0, N):
            ef[2*i, 2*j] =  e[i, j]  # Center point from coarse grid
            ef[2*i+1, 2*j] = (1/2) * (e[i, j] + e[i+1, j])  # Horizontal interpolation
            ef[2*i, 2*j+1] = (1/2) * (e[i, j] + e[i, j+1])  # Vertical interpolation
            ef[2*i+1, 2*j+1] = (1/4) * (e[i, j] + e[i+1, j] + e[i, j+1] + e[i+1, j+1])  # Diagonal interpolation

    # Interpolate the boundary edges which is not captured by the previous loop, where I use linear interpolation along the edges
    for j in range(0, N):
        
        #Right boundary
        ef[-1, 2*j] = e[-1, j]
        ef[-1, 2*j +1] = (1/2) * (e[-1, j] + e[-1, j+1]) 
        
        
        #Bottom boundary (y indices goes from top to bottom)
        ef[2*j, -1] = e[j, -1] 
        ef[2*j+1, -1] =(1/2) * (e[j, -1] + e[j+1, -1])
        

    #Corners
    ef[0, 0] = e[0, 0]
    ef[0, -1] = e[0, -1]
    ef[-1, 0] = e[-1, 0]
    ef[-1, -1] = e[-1, -1]
        
    return ef

def mgv(u0, rhs, m,  N, nu1, nu2, level, max_level, v = np.array([1,1])):
    """
    Perform one multigrid V-cycle on the 2D advection-diffusion equation.

    Parameters:
        u0: Initial guess (N+1)x(N+1) matrix.
        rhs: Right-hand side (N+1)x(N+1) matrix.
        N: Grid size.
        m: Number of iterations in GMRES
        nu1: Number of pre-smoothings.
        nu2: Number of post-smoothings.
        level: Current level of the V-cycle.
        max_level: Total number of levels, remember to choose this such that N gives a square matrix and not zero
        v: Velocity field (not used in the given context).
        
    Returns:
        u: The computed solution vector
        time: Converging time
        
    """
    if level == max_level:
        # On the coarsest level, solve using GMRES
        u, _, _, _ = restarted_gmres(u0, rhs, m, N, tol=1e-10, v = v)
    else:
        u = jacobi_relax(u0, rhs, 2/3, N, nu1,v) # Pre-smoothing
        rf = residual(u, rhs, N, v)         # Compute the residual
        rc = restriction(rf, N)          # Restrict the residual to a coarser grid
        ec = mgv(np.zeros_like(rc), rc, m,  N//2, nu1, nu2, level + 1, max_level, v)   #Recursive call on the coarser grid
        ef = interpolation(ec, N//2)   # Interpolate the error back to the fine grid
        u += ef   # Update the solution with the interpolated error
        u = jacobi_relax(u,rhs,2/3,N,nu2,v)  # Post-smoothing
    return u

def precon_gmres(U0, m, N, nu1, nu2, level, max_level, tol=1e-4, max_iter=200, v=np.array([1, 1])):
    """
    Preconditioned GMRES with Restart and Multigrid Preconditioning.

    Parameters:
        U0: ndarray
            Initial guess for the solution.
        m: int
            Dimension of the Krylov subspace before restart.
        N: int
            Size of the problem.
        nu1, nu2: int
            Number of pre- and post-smoothing steps in multigrid.
        level: int
            Current multigrid level.
        max_level: int
            Maximum multigrid levels.
        tol: float
            Convergence tolerance.
        max_iter: int
            Maximum GMRES restart iterations.
        v: ndarray
            Advection speed.

    Returns:
        x: ndarray
            Approximate solution.
        res_arr: list
            List of residuals at each iteration.
        count: int
            Number of iterations performed.
        time: float
            Time taken to converge.
    """
    start = perf_counter()
    x = U0.copy().flatten()  # Initial guess vector
    b = rhs_func(N, v).flatten()  # Right-hand side vector
    r_0 = b - lhs_func(x, N, v).flatten()  # Initial residual
    norm_r0 = np.linalg.norm(r_0)
    if norm_r0 < tol:  # Already converged
        return x.reshape((N + 1, N + 1)), [r_0], 0, 0
    
    res_arr = [norm_r0]  # Track residuals
    omega_1 = r_0 / norm_r0  # Normalize the initial residual
    count = 0
    for outer_iter in range(max_iter):
        # Preconditioned residual using multigrid
        z = mgv(np.zeros((N + 1, N + 1)), r_0.reshape((N + 1, N + 1)), m, N, nu1, nu2, level, max_level, v).flatten()
        norm_z = np.linalg.norm(z)
        if norm_z < tol:  # Check convergence of the preconditioned residual
            break

        omega_1 = z / norm_z  # Normalize the preconditioned residual
        V, H = arnoldi(m, N, omega=omega_1, v=v)  # Generate Krylov subspace
        y = np.linalg.lstsq(H, np.dot(V.T, b - lhs_func(x, N, v).flatten()), rcond=None)[0]
        x += V[:, :y.size] @ y  # Update the solution
        
        # Recompute residual
        r_0 = b - lhs_func(x, N, v).flatten()
        res_arr.append(np.linalg.norm(r_0))
        count += 1
        
        norm_rm = np.linalg.norm(r_0)
        print(f"Outer Iteration {count}, Residual Norm: {norm_rm}")

        if norm_rm / norm_r0 < tol:  # Check convergence relative to preconditioned residual
            break
    
    stop = perf_counter()
    time = stop - start
    return x.reshape((N + 1, N + 1)), res_arr, count, time

def lhs_func_space_dependent(U, N, v_field):
    """Computes the left-hand side vector based on the input array U for diffusion-advection problem."""
    h = 1 / N
    U = np.reshape(U, (N+1, N+1))
    lhs = np.zeros((N + 1, N + 1))
    
    index = np.arange(1, N)  # Indices corresponding to internal nodes
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)
    ixp_y = np.ix_(index + 1, index)
    ix_ym = np.ix_(index, index - 1)
    ix_yp = np.ix_(index, index + 1)
    
    # Velocity field on grid points
    vx, vy = v_field

    lhs[ixy] = ((4 * U[ixy] - U[ixp_y] - U[ixm_y] - U[ix_yp] - U[ix_ym])
                + h * vx[ixy] * (U[ixy] - U[ixm_y])
                + h * vy[ixy] * (U[ixy] - U[ix_ym]))
    return lhs

def rhs_func_space_dependent(N, v_field):
    """Constructs the right-hand side vector f based on the analytical solution u(x, y) = sin(pi x) * sin(2 * pi y)."""
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    
    # Velocity field on grid points
    vx, vy = v_field
    
    f = (5 * np.pi**2 * np.sin(np.pi * X) * np.sin(2 * np.pi * Y) +
         vx * np.pi * np.cos(np.pi * X) * np.sin(2 * np.pi * Y) +
         vy * 2 * np.pi * np.sin(np.pi * X) * np.cos(2 * np.pi * Y))
    
    f[:, 0] = f[:, -1] = f[0, :] = f[-1, :] = 0  # Boundary conditions
    return f * h**2

def arnoldi_space_dependent(m, N, omega, v_field):
    """Arnoldi's method to generate an orthonormal basis for the Krylov subspace."""
    n = (N+1)**2  # Total number of grid points including boundary
    V = np.zeros((n, m+1))  # Orthonormal basis
    H = np.zeros((m+1, m))  # Upper Hessenberg matrix with m+1 rows
    omega = omega.flatten()  # Make sure omega is in vector format
    
    V[:, 0] = omega / np.linalg.norm(omega)
    for j in range(m):
        w = lhs_func_space_dependent(V[:, j], N, v_field).flatten()
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j+1, j] = np.linalg.norm(w)
        if np.abs(H[j+1, j]) <= 1e-12:
            break
        V[:, j+1] = w / H[j+1, j]
    return V, H

def restarted_gmres_space_dependent(U0, rhs, m, N, tol=1e-4, v_field=None):
    """Restarted GMRES method using Arnoldi's method to solve Ax = b."""
    start = perf_counter()
    x = U0.copy().flatten()
    b = rhs.flatten()
    r_0 = b - lhs_func_space_dependent(x, N, v_field).flatten()
    norm_r0 = np.linalg.norm(r_0)
    
    if norm_r0 < tol:
        return x, r_0, 0, 0
    
    omega_1 = r_0 / norm_r0
    count = 0
    while True:
        V, H = arnoldi_space_dependent(m, N, omega_1, v_field)
        y = np.linalg.lstsq(H, np.dot(V.T, b - lhs_func_space_dependent(x, N, v_field).flatten()), rcond=None)[0]
        x += V[:, :y.size] @ y
        r_m = b - lhs_func_space_dependent(x, N, v_field).flatten()
        count += 1
        if np.linalg.norm(r_m) / norm_r0 < tol:
            break
        omega_1 = r_m / np.linalg.norm(r_m)
    stop = perf_counter()
    time = stop - start
    return x.reshape((N+1, N+1)), r_m, count, time

def jacobi_relax_space_dependent(u, rhs, omega, N, nu, v_field):
    """Perform weighted Jacobi iterations with space-dependent velocity."""
    h = 1 / N
    index = np.arange(1, N)
    ixy = np.ix_(index, index)
    ixm_y = np.ix_(index - 1, index)
    ixp_y = np.ix_(index + 1, index)
    ix_ym = np.ix_(index, index - 1)
    ix_yp = np.ix_(index, index + 1)

    vx, vy = v_field
    div = 4 + vx[ixy] * h + vy[ixy] * h
    
    for _ in range(nu):
        u_new = np.zeros_like(u)
        u_new[ixy] = ((1 - omega) * u[ixy] +
                      omega * ((u[ixp_y] + u[ixm_y] * (1 + vx[ixy] * h) +
                                u[ix_yp] + u[ix_ym] * (1 + vy[ixy] * h)) / div +
                               (rhs[ixy] / div)))
        u = u_new
    return u

def residual_space_dependent(u, rhs, N, v_field):
    """Compute the residual for space-dependent velocity."""
    rf = rhs - lhs_func_space_dependent(u, N, v_field)
    return rf

def mgv_space_dependent(u0, rhs, m, N, nu1, nu2, level, max_level, v_field):
    """Perform one multigrid V-cycle with space-dependent velocity."""
    if level == max_level:
        u, _, _, _ = restarted_gmres_space_dependent(u0, rhs, m, N, tol=1e-10, v_field=v_field)
    else:
        u = jacobi_relax_space_dependent(u0, rhs, 2/3, N, nu1, v_field)
        rf = residual_space_dependent(u, rhs, N, v_field)
        rc = restriction(rf, N)
        ec = mgv_space_dependent(np.zeros_like(rc), rc, m, N//2, nu1, nu2, level+1, max_level, v_field)
        ef = interpolation(ec, N//2)
        u += ef
        u = jacobi_relax_space_dependent(u, rhs, 2/3, N, nu2, v_field)
    return u


def rhs_func_2(N, v = np.array([1,1])):
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    
    # Analytical function for f based on the exact solution u(x, y) = x(1-x) + y(1-y)
    f = (-2*Y**2 + 2*Y - 2*X**2 + 2*X 
        + v[0] *( -X*Y*(1-Y) + Y*(1-X)*(1-Y)) 
        + v[1] * ( -X*Y*(1-X) + X*(1-X)*(1-Y)))
    
    #Set the boundaries to zero, since u = g = 0 at the boundaries
    f[:, 0] = f[:, -1] = f[0, :] = f[-1, :] = 0

    # Scale by h^2 as per the finite difference scheme
    return f * h**2

def u_exact_2(N):
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    u = X*(1-X)*Y*(1-Y)
    return u
    