import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

def track_time(func):
    """Decorator to track the execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time for the entire function
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # End time for the entire function
        print(f"[{func.__name__}] Total time: {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


def track_part_time(part_name):
    """Helper function to track execution time of a specific part of a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            part_start = time.time()
            result = func(*args, **kwargs)
            part_end = time.time()
            print(f"[{part_name}] Time: {part_end - part_start:.4f} seconds.")
            return result

        return wrapper

    return decorator


def find_nearest_index(grid, value):
    """Find index of the nearest grid point to a given value"""
    idx = np.abs(grid - value).argmin()
    return idx


def find_nearest_lower_index(grid, value):
    """Find index of the nearest grid point less than or equal to a given value"""
    indices = np.where(grid <= value)[0]
    if len(indices) == 0:
        return 0
    return indices[-1]

def find_closest_indices(grid, value):

    """
    Find the two closest indices in the grid to the given value.
    
    Parameters:
    -----------
    value : float
        The value to find the closest indices for
    grid : numpy.ndarray
        The grid of values
    
    Returns:
    --------
    tuple
        (lower_index, upper_index, lower_weight)
    """
    # Check if value is outside the grid
    if value <= grid[0]:
        return 0, 0, 1.0
    elif value >= grid[-1]:
        return len(grid)-1, len(grid)-1, 1.0
    
    # Find the index of the grid point just below the value
    lower_idx = np.searchsorted(grid, value, side='right') - 1
    upper_idx = lower_idx + 1
    
    # Calculate weights for interpolation
    grid_range = grid[upper_idx] - grid[lower_idx]
    if grid_range == 0:  # Avoid division by zero
        lower_weight = 1.0
    else:
        lower_weight = (grid[upper_idx] - value) / grid_range
        # upper_weight = (value - grid[lower_idx]) / grid_range
    
    return lower_idx, upper_idx, lower_weight


class LagosWrightAiyagariSolver:
    def __init__(self, params):
        """Initialise solver with model parameters"""
        # Preference parameters
        self.beta = params['beta']      # Discount factor
        self.alpha = params['alpha']    # Probability of DM consumption opportunity
        self.alpha_1 = params['alpha_1']  # Probability of accepting both money and assets
        self.alpha_0 = 1.0 - self.alpha_1  # Probability of accepting only money
        self.gamma = params['gamma']    # Risk aversion parameter
        self.Psi = params['Psi']       # DM utility scaling parameter
        
        # Production and labor market parameters
        self.psi = params['psi']       # Matching function elasticity
        self.zeta = params['zeta']     # Worker bargaining power
        self.nu = params['nu']         # Labor supply elasticity
        self.mu = params['mu']         # Matching efficiency
        self.delta = params['delta']   # Job separation rate
        self.kappa = params['kappa']   # Vacancy posting cost
        self.replace_rate = params['repl_rate']  # Unemployment benefit replacement rate
        
        # Grid specifications
        self.n_a = params['n_a']       # Number of asset grid points
        self.n_m = params['n_m']       # Number of money grid points
        self.n_f = params['n_f']       # Number of illiquid asset grid points
        self.n_b = params['n_b']       # Number of bank grid points
        self.a_max = params['a_max']   # Maximum asset value
        self.a_min = params['a_min']   # Minimum asset value
        self.m_min = params['m_min']   # Minimum money holdings
        self.m_max = params['m_max']   # Maximum money holdings
        self.f_min = params['f_min']   # Minimum illiquid asset holdings
        self.f_max = params['f_max']   # Maximum illiquid asset holdings
        self.b_min = params['b_min']   # Minimum deposit value
        self.b_max = params['b_max']   # Maximum deposit value
        self.n_e = 2                   # Employment states: [0=unemployed, 1=employed]
        self.n_z = 3                   # Skill types: [0=low, 1=medium, 2=high]
        self.ny = params['ny']         # Number of grid points for DM goods

        # For computation
        self.c_min = params['c_min']   # Minimum consumption
        self.Rm = params['Rm']         # Gross return of real money balances (exogenous)
        
        # Price parameters
        self.prices = np.array([
            params['py'],              # Price of DM goods
            params['Rl'],              # Return on illiquid assets
            params['i']                # Nominal rate from banks
        ])
        
        # Convergence parameters
        self.max_iter = params['max_iter']
        self.tol = params['tol']
        
        # Initialise grids
        self.setup_grids(params)
        
        # Initialise value and policy functions
        self.initialise_functions()
        
        # For storing value function history
        self.value_history = {
            'W': [],
            'V': []
        }

    
    def setup_grids(self, params):
        """Set up state space grids with appropriate spacing"""
        # Asset grid
        self.a_grid = np.linspace(params['a_min'], params['a_max'], self.n_a)

        # Money holdings grid
        self.m_grid = np.linspace(params['m_min'], params['m_max'], self.n_m)

        # Illiquid asset grid
        self.f_grid = np.linspace(params['f_min'], params['f_max'], self.n_f)
        
        # Bank grid (real terms)
        self.b_grid = np.linspace(params['b_min'], params['b_max'], self.n_b)
        
        # Employment grid
        self.e_grid = np.array([0, 1])  # 0 = unemployed, 1 = employed
        
        # Productivity/skill grid
        self.z_grid = np.array([0.58, 0.98, 2.15])  # Low, medium, high skill
        
        # Skill type distribution (fixed - doesn't change over time)
        self.z_dist = np.array([0.62, 0.28, 0.1])  # Distribution of skill types

    
    def initialise_functions(self):
        """Initialise value [V & W] and policy functions with reasonable guesses"""
        # Initialise arrays using m and f as state variables
        self.W = np.zeros((self.n_a, self.n_b, self.n_z, self.n_e))
        self.V = np.zeros((self.n_m, self.n_f, self.n_z, self.n_e))
        firm_result = self.firm_problem(self.prices)
        wages = firm_result['wages']
        
        # Initialise W with reasonable guesses using a utility-based approach
        for e_idx, e in enumerate(self.e_grid):
            for z_idx, z in enumerate(self.z_grid):
                # Estimate income based on skill and employment
                income_est = wages[z_idx, e_idx]
                
                for a_idx, a in enumerate(self.a_grid):
                    for b_idx, b in enumerate(self.b_grid):
                        # Estimate resources available for consumption
                        resources = a + (1 + self.prices[2]) * b + income_est

                        # Ensure minimum consumption
                        consumption = max(resources * 0.9, self.c_min)  # Consume ~90% of resources

                        # Calculate utility and scale for perpetuity
                        self.W[a_idx, b_idx, z_idx, e_idx] = self.utility(consumption)
        
        # Initialise V based on W
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for m_idx, m in enumerate(self.m_grid):
                    for f_idx, f in enumerate(self.f_grid):
                        # Simple estimate: deposit all money, no loans
                        b_est = m
                        a = m + f
                        
                        # Find closest grid points
                        a_idx = min(find_nearest_index(self.a_grid, a), self.n_a - 1)
                        b_idx = min(find_nearest_index(self.b_grid, b_est), self.n_b - 1)
                        
                        # Map to V
                        self.V[m_idx, f_idx, z_idx, e_idx] = self.W[a_idx, b_idx, z_idx, e_idx]
         # Print size of W and V
        print("W shape:", self.W.shape)
        print("V shape:", self.V.shape)

        # Print some sample values for inspection
        print("\nFirst two rows from initial guess W:")
        print(self.W[:min(2, self.n_a), :min(2, self.n_b), :min(2, self.n_z), :min(2, self.n_e)])

        print("\nFirst two rows from initial guess V:")
        print(self.V[:min(2, self.n_m), :min(2, self.n_f), :min(2, self.n_z), :min(2, self.n_e)])

        
    def utility(self, c):
        """
        1. Utility function for centralised market
        2. c : endogenously determined by the skill type e, which impacts the budget constraint
        """
        c = np.asarray(c)
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        safe_c = np.maximum(c, eps)
        
        return np.where(c <= 0,
                    -1e3,  # penalty for non-positive consumption
                    (safe_c ** (1 - self.gamma) - 1) / (1 - self.gamma))
    
    def utility_dm(self, y):
        """
        Utility function for decentralised market -- parameters are different from the centralised market
        Generates 26% increase in consumption during preference shocks.
        """
        y = np.asarray(y)
        return np.where(y <= 0,
                    0,  # normalisation in the paper
                    self.Psi * (y ** (1 - self.psi) - 1) / (1 - self.psi))
    
    def κ_prime_inv(self, py):
        """
        Solve the firm's optimal production y (skill-z adjusted), given goods price.
        Assumed: q_bar = y_bar = 1
        :param py:
        :return: the optimal supply of early consumption goods
        """
        y_temp = 1 / ((1 + py ** (1 / (self.zeta - 1))) ** self.zeta)
        y_tilda = max(min(y_temp, 1), 0)
        return y_tilda

    def κ_fun(self, y):
        """
        κ(y) = q̄ - Q(y), where Q is a PPF and q̄ == 1.
        Opportunity cost of producing y (skill-z adjusted).
        :param y: a vector of early consumption.
        :return:
        """
        cost = np.zeros_like(y)
        positive_mask = (y >= 0)
        cost[positive_mask] = 1 - (1 - y[positive_mask] ** (1 / self.zeta)) ** self.zeta
        return cost

    
    def q_fun(self, py):
        """
        The productivity-adjusted revenue of a job (independent of productivity z).
        """
        y_tilda = self.κ_prime_inv(py)
        q = 1 + py * y_tilda - self.κ_fun(y_tilda)
        return q

    @track_time
    def firm_post_rev(self, prices):
        """
        Calculates the productivity-normalised value of a filled job.
        This represents the value per unit of worker productivity (z).
        Both revenue and costs scale with z, so we normalise by z to get 
        a single market tightness for all productivity types.
        
        Derived from Eq (13): ϕᶠ(z) = zq(pʸ) - zw₁ + (1-δ)ϕᶠ(z)/Rⁱ
        When normalised by z: ϕ̃ᶠ = ϕᶠ(z)/z = q(pʸ) - w₁ + (1-δ)ϕ̃ᶠ/Rⁱ
        
        Returns:
            normalised_rev: The value per unit of productivity (independent of z)
        """
        py = prices[0]
        Rl = prices[1]

        q = self.q_fun(py)
        normalised_rev = (1 - self.mu) * q / (1 - (1 - self.delta) / Rl)
        return normalised_rev
    
    def job_finding(self, θ):
        """
        Job finding probability λ for a worker.
        :param θ: tightness in the labor market:  job vacancy / seekers
        :return: vacancy filling rate between 0 and 1.
        """
        prob = 1 / (1 + (1 / θ) ** (self.zeta)) ** (1 / self.zeta)
        return min(max(prob, 1e-3), 1)

    def job_filling(self, θ):
        """
        Job filling rate for a firm.
        :param θ: tightness in the labor market job_opening / seekers
        :return: vacancy filling rate between 0 and 1.
        """
        prob = 1 / (1 + θ ** self.zeta) ** (1 / self.zeta)
        return min(max(prob, 1e-3), 1)
    
    def job_fill_inv(self, x):
        """
        Pin down the market tightness θ given the free-entry condition.
        :param x: the free-entry condition (right-hand side of the equation)
        :return: tightness
        """
        θ = ((1 / x) ** self.zeta - 1) ** (1 / self.zeta)
        return θ

    def solve_θ(self, prices):
        """
        Solves for equilibrium market tightness using the free-entry condition.
        
        The key insight is that both the vacancy creation cost (kappa*z) and the expected 
        benefit of hiring (rev*z) scale with productivity z. Therefore, when comparing 
        costs and benefits, z cancels out, leading to a single market tightness for all 
        productivity types.
        
        Derived from Eq (17): -z*kappa + λ(θ)zϕ̃ᶠ/θRⁱ ≤ 0
        Dividing by z:        -kappa + λ(θ)ϕ̃ᶠ/θRⁱ ≤ 0
        
        Parameters:
            prices: Vector of prices [py, Rl]
        
        Returns:
            θ: Market tightness
            λ: Job-finding probability
            filling: Vacancy-filling probability
        """
        kappa = self.kappa
        Rl = prices[1]
        # Get productivity-normalised value of a filled job
        normalised_rev = self.firm_post_rev(prices=prices)
        
        # Compare expected cost with expected benefit (both normalised by z)
        if kappa >= normalised_rev / Rl:
            θ = 1e-3  # Almost no vacancies if cost exceeds revenue
        else:
            θ = self.job_fill_inv(kappa * Rl / normalised_rev)
        
        # Compute job finding probability & vacancy filling probability
        λ = self.job_finding(θ)
        # filling = self.job_filling(θ)
        return λ

    @track_time
    def firm_problem(self, prices):
        """
        Compute firm-side decisions and total asset supply for the illiquid asset market.
        This is skill-dependent and mirrors Bethune and Rocheteau's logic.

        Parameters:
        - prices: [py, Rl, i]
        - gz: stationary skill distribution array (n_z, )

        Returns:
        - dict containing Ys, frev, wages, profits, Js, qz, J_total
        """

        # labor market stuff
        py, Rl = prices[0], prices[1]
        labor_share = self.mu  # assuming μ = labor share in production

        # Step 1: Optimal production per worker (independent of z)
        y_star = self.κ_prime_inv(py)  # scalar
        kappa_y = self.κ_fun(np.array([y_star]))[0]  # scalar

        # Step 2: Firm revenue per worker for each productivity z
        frev = self.z_grid * (1 + py * y_star - kappa_y)  # shape: (n_z, )

        # Step 3: Wage per skill type (employed workers)
        wages_em = labor_share * frev  # wage_bar[1,:] in BR

        # Step 4: Per-firm profit
        profits = frev - wages_em

        # Step 5: Firm value J(z), capitalised with Rl
        Js = profits / (1 - (1 - self.delta) / Rl)

        # Step 6: Solve for job finding probability
        job_finding_prob = self.solve_θ(self.prices)

        # Step 7: Employment rate
        emp = job_finding_prob / (self.delta + job_finding_prob)

        # Step 8: Total supply of illiquid asset (firm equity)
        J_total = emp * np.sum(self.z_dist * Js)
        
        # Calculate wages and unemployment benefits (based on labor share)
        wages_bar = np.zeros((self.n_z, self.n_e))
        wages_bar[:, 1] = wages_em
        wages_bar[:, 0] = self.replace_rate * wages_em  # Unemployed benefits
        
        # Taxes and transfers
        self.Ag0 = params['Ag0']  # Government bond supply
        taulumpsum = ((1.0 / Rl) - 1.0) * self.Ag0  # Revenue from money creation
        
        # Apply lump-sum transfer to all households using broadcasting
        wages = wages_bar + taulumpsum
        
        # Employment transition matrix: rows = current state, cols = next state
        # [0,0] = P(unemployed → unemployed), [0,1] = P(unemployed → employed)
        # [1,0] = P(employed → unemployed), [1,1] = P(employed → employed)
        P = np.array([
            [1 - job_finding_prob, job_finding_prob],
            [self.delta, 1 - self.delta]
        ])

        return {
            'Ys': y_star,
            'frev': frev,
            'wages': wages,
            'profits': profits,
            'Js': Js,
            'J_total': J_total,
            'emp': emp,
            'transition_matrix': P
        }

    
    def solve_dm_problem_vectorised(self, W_guess, prices):
        """
        BLOCK 1: Solve the DM problem.
        Solve the policy functions for the DM.
        Input:
            W_guess: Initial guess for value function W.
            prices: Current prices [py, Rl, i]
        Output:
            Updated value function V and policy functions
        """
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialise arrays - using float32 for memory efficiency if precision allows
        shape = (self.n_m, self.n_f, self.n_z, self.n_e)
        policy_y0 = np.zeros(shape, dtype=np.float32)
        policy_b0 = np.zeros(shape, dtype=np.float32)
        V0 = np.zeros(shape, dtype=np.float32)
        policy_y1 = np.zeros(shape, dtype=np.float32)
        policy_b1 = np.zeros(shape, dtype=np.float32)
        V1 = np.zeros(shape, dtype=np.float32)
        policy_b_noshock = np.zeros(shape, dtype=np.float32)
        V_noshock = np.zeros(shape, dtype=np.float32)
        
        # Precompute y-grids for all asset levels
        y0_nb_grids = {}  # No borrowing grids for ω=0
        y1_nb_grids = {}  # No borrowing grids for ω=1

        # precompute firm problem
        firm_result = self.firm_problem(prices)
        wages = firm_result['wages']
        
        start_time = time.time()
        for m_idx, a_m in enumerate(self.m_grid):
            for f_idx, a_f in enumerate(self.f_grid):
                a = min(a_m + a_f, self.a_max)  # Total assets
                y1_nb_grids[(m_idx, f_idx)] = np.linspace(0, a/py, self.ny)  # Can use all assets when ω=1
                y0_nb_grids[(m_idx, f_idx)] = np.linspace(0, a_m/py, self.ny)
        print(f"Part 1 loop in dm took {time.time() - start_time:.4f} seconds.")
        
        # Loop over employment states and skill types
        start_time = time.time()
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                income = wages[z_idx, e_idx]
                # Vectorise for different asset levels
                for f_idx, a_f in enumerate(self.f_grid):
                    # Case 3: No preference shock
                    # When there's no preference shock, optimal policy is to deposit all money
                    w_noshock_values = np.zeros(self.n_m)
                    for m_idx, a_m in enumerate(self.m_grid):
                        # Total assets
                        a = min(a_m + a_f, self.a_max)
                        # Calculate continuation value
                        w_noshock_values[m_idx] = self.interpolate_2d_vectorised(
                            np.array([a - a_m]), np.array([a_m]),
                            self.a_grid, self.b_grid,
                            W_guess[:, :, z_idx, e_idx]
                        )[0]   # since all values are the same in this vector (inputs are scalar)
                        # Deposit all money
                        policy_b_noshock[m_idx, f_idx, z_idx, e_idx] = a_m
                        
                            
                    # Store no-shock values for all money levels at once
                    V_noshock[:, f_idx, z_idx, e_idx] = w_noshock_values
                    
                    # Process each money level separately for shock cases
                    for m_idx, a_m in enumerate(self.m_grid):
                        a = min(a_m + a_f, self.a_max)
                        
                        # -------------------------------------------------------------------------
                        # Case 1: Preference shock + ω=0 (only money accepted)
                        # -------------------------------------------------------------------------
                        
                        # Case 1.1: No borrowing (can use precomputed grid)
                        y0_nb = y0_nb_grids[(m_idx, f_idx)]
                        d0 = a_m - py*y0_nb  # won't exceed the maximum
                        
                        w0_nb = self.interpolate_2d_vectorised(
                            a_f, d0, 
                            self.a_grid, self.b_grid,                # a_f is the total asset
                            W_guess[:, :, z_idx, e_idx]           # when money is either consumed or deposited
                        )
                        v0_nb = self.utility_dm(y0_nb) + w0_nb
                        
                        # Case 1.2: With borrowing
                        # Efficient borrowing constraint calculation
                        income_based_max = (a_f + income - self.c_min)/(1 + i)
                        collateral_based_max = a_f/(1 + i)
                        effective_max_borrow = min(income_based_max, collateral_based_max)
                        
                        # Skip borrowing calculations if no borrowing is possible
                        if effective_max_borrow > 0:
                            y0_b = np.linspace(0, (a_m + effective_max_borrow)/py, self.ny)
                            l0 = np.clip(py*y0_b - a_m, 0, effective_max_borrow)   # borrow exactly the amount needed to consume
                            
                            w0_b = self.interpolate_2d_vectorised(
                                a + l0 - py*y0_b, -l0,
                                self.a_grid, self.b_grid,
                                W_guess[:, :, z_idx, e_idx]
                            )
                            v0_b = self.utility_dm(y0_b) + w0_b
                            
                            # Find optimal ω=0 policy efficiently
                            max_nb_idx = np.argmax(v0_nb)
                            max_b_idx = np.argmax(v0_b)
                            max_nb_val = v0_nb[max_nb_idx]
                            max_b_val = v0_b[max_b_idx]
                            
                            # compare all options when ω=0: borrow or not borrow
                            if max_nb_val >= max_b_val:
                                policy_y0[m_idx, f_idx, z_idx, e_idx] = y0_nb[max_nb_idx]
                                policy_b0[m_idx, f_idx, z_idx, e_idx] = d0[max_nb_idx]
                                V0[m_idx, f_idx, z_idx, e_idx] = max_nb_val
                            else:
                                policy_y0[m_idx, f_idx, z_idx, e_idx] = y0_b[max_b_idx]
                                policy_b0[m_idx, f_idx, z_idx, e_idx] = -l0[max_b_idx]
                                V0[m_idx, f_idx, z_idx, e_idx] = max_b_val
                        else:
                            # No borrowing possible, only use no-borrowing case
                            max_idx = np.argmax(v0_nb)
                            policy_y0[m_idx, f_idx, z_idx, e_idx] = y0_nb[max_idx]
                            policy_b0[m_idx, f_idx, z_idx, e_idx] = d0[max_idx]
                            V0[m_idx, f_idx, z_idx, e_idx] = v0_nb[max_idx]
                        
                        # -------------------------------------------------------------------------
                        # Case 2: Preference shock + ω=1 (both assets accepted)
                        # -------------------------------------------------------------------------
                        
                        # Case 2.1: No borrowing (use precomputed grid)
                        y1_nb = y1_nb_grids[(m_idx, f_idx)]
                        d1 = a - py*y1_nb
                        
                        w1_nb = self.interpolate_2d_vectorised(
                            a_f, d1,       # money is either consumed or deposited; only a_f is the total asset
                            self.a_grid, self.b_grid,
                            W_guess[:, :, z_idx, e_idx]
                        )
                        v1_nb = self.utility_dm(y1_nb) + w1_nb
                        
                        # Case 2.2: With borrowing
                        max_borrow = (income - self.c_min)/(1 + i)
                        
                        # Skip borrowing calculations if no borrowing is possible
                        if max_borrow > 0:
                            y1_b = np.linspace(0, (a + max_borrow)/py, self.ny)
                            l1 = np.clip(py*y1_b - a, 0, max_borrow)
                            
                            w1_b = self.interpolate_2d_vectorised(
                                a + l1 - py*y1_b, -l1,
                                self.a_grid, self.b_grid,
                                W_guess[:, :, z_idx, e_idx]
                            )
                            v1_b = self.utility_dm(y1_b) + w1_b
                            
                            # Find optimal ω=1 policy efficiently
                            max_nb_idx = np.argmax(v1_nb)
                            max_b_idx = np.argmax(v1_b)
                            max_nb_val = v1_nb[max_nb_idx]
                            max_b_val = v1_b[max_b_idx]
                            
                            if max_nb_val >= max_b_val:
                                policy_y1[m_idx, f_idx, z_idx, e_idx] = y1_nb[max_nb_idx]
                                policy_b1[m_idx, f_idx, z_idx, e_idx] = d1[max_nb_idx]
                                V1[m_idx, f_idx, z_idx, e_idx] = max_nb_val
                            else:
                                policy_y1[m_idx, f_idx, z_idx, e_idx] = y1_b[max_b_idx]
                                policy_b1[m_idx, f_idx, z_idx, e_idx] = -l1[max_b_idx]
                                V1[m_idx, f_idx, z_idx, e_idx] = max_b_val
                        else:
                            # No borrowing possible, only use no-borrowing case
                            max_idx = np.argmax(v1_nb)
                            policy_y1[m_idx, f_idx, z_idx, e_idx] = y1_nb[max_idx]
                            policy_b1[m_idx, f_idx, z_idx, e_idx] = d1[max_idx]
                            V1[m_idx, f_idx, z_idx, e_idx] = v1_nb[max_idx]

        print(f"Part 2 loop in dm took {time.time() - start_time:.4f} seconds.")
        # Calculate expected value with correct probability weighting
        V_dm = self.alpha * (self.alpha_0 * V0 + self.alpha_1 * V1) + (1 - self.alpha) * V_noshock

         # Collect outputs
        result = {
            'V_dm': V_dm,
            'V0': V0,
            'V1': V1,
            'V_noshock': V_noshock,
            'policy_y0': policy_y0,
            'policy_b0': policy_b0,
            'policy_y1': policy_y1,
            'policy_b1': policy_b1,
            'policy_b_noshock': policy_b_noshock
        }

        # Print shape and sample values (first two m and f indices, for each grid)
        for key, array in result.items():
            print(f"\n[{key}] shape: {array.shape}")
            sample = array[
                :min(2, array.shape[0]),
                :min(2, array.shape[1]),
                :min(2, array.shape[2]),
                :min(2, array.shape[3])
            ]
            print(f"[{key}] sample (first 2 in each dimension):\n{sample}\n")

        return result
    
    @track_time
    def solve_cm_problem_vectorised(self, V_guess, prices):
        """
         BLOCK 1: Solve the CM problem.
        Solve the policy functions for the CM.
        Input: 
            V_guess: Initial guess for value function V.
            prices: Current prices [py, Rl, i]
        Output:
            Updated value function W and policy functions.
        """
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialise value and policy functions
        w_value = np.zeros((self.n_a, self.n_b, self.n_z, self.n_e))
        policy_m = np.zeros_like(w_value)
        policy_f = np.zeros_like(w_value)

        # Precompute firm problem
        firm_result = self.firm_problem(prices)
        wages = firm_result['wages']
        P = firm_result['transition_matrix']

        # Precompute portfolio costs
        m_grid_reshaped = self.m_grid.reshape(-1, 1)  # (n_m, 1)
        f_grid_reshaped = self.f_grid.reshape(1, -1)  # (1, n_f)
        portfolio_costs = m_grid_reshaped / self.Rm + f_grid_reshaped / Rl  # (n_m, n_f)

        # Loop over employment status and skill types
        for e_idx in range(self.n_e):
            # Reshape transition probabilities for correct broadcasting
            P_reshaped = P[e_idx, :].reshape(1, 1, -1)  # (1, 1, n_e)
            
            for z_idx in range(self.n_z):
                income = wages[z_idx, e_idx]
                
                # Get value function slice for current z_idx
                V_slice = V_guess[:, :, z_idx, :]  # (n_m, n_f, n_e)
                
                # Calculate expected continuation value
                cont_values = np.sum(V_slice * P_reshaped, axis=2)  # (n_m, n_f)
                
                # Vectorise over current state (a, b)
                for a_idx, a in enumerate(self.a_grid):
                    for b_idx, b in enumerate(self.b_grid):
                        # Calculate total resources available
                        total_resources = a + (1 + i) * b + income
                        
                        # Calculate consumption for all portfolio combinations
                        c_values = total_resources - portfolio_costs  # (n_m, n_f)
                        
                        # Create mask for valid consumption levels
                        valid_c = c_values >= self.c_min  # (n_m, n_f)
                        
                        # Calculate utility and continuation value where consumption is valid
                        w_choice = np.where(valid_c, 
                                        self.utility(c_values) + self.beta * cont_values,
                                        -1e3)  # (n_m, n_f)
                        
                        # Find the maximum value and corresponding indices
                        max_val = np.max(w_choice)
                        max_indices = np.unravel_index(np.argmax(w_choice), w_choice.shape)
                        
                        # Store the maximum value and corresponding policies
                        w_value[a_idx, b_idx, z_idx, e_idx] = max_val
                        policy_m[a_idx, b_idx, z_idx, e_idx] = self.m_grid[max_indices[0]]
                        policy_f[a_idx, b_idx, z_idx, e_idx] = self.f_grid[max_indices[1]]
        
         # Collect results
        result = {
            'W': w_value,
            'policy_m': policy_m,
            'policy_f': policy_f,
        }

        # Print shapes and sample values
        for key, array in result.items():
            print(f"\n[{key}] shape: {array.shape}")
            sample = array[
                :min(2, array.shape[0]),
                :min(2, array.shape[1]),
                :min(2, array.shape[2]),
                :min(2, array.shape[3])
            ]
            print(f"[{key}] sample (first 2 in each dimension):\n{sample}\n")

        return result
    
    def solver_iteration(self, prices, W_guess=None):
        """
        BLOCK 1: Solve the DM and CM problems via iteration.
        Performs one iteration of the solution algorithm:
        1. Solve DM problem given W_guess to get V
        2. Solve CM problem given V to get updated W
        3. Calculate convergence metrics (max_diff, mean_diff).
        4. Store current value functions for history.
        5. Return updated W, V, and convergence metrics.
        
        Input:
            prices: Current prices [py, Rl, i]
            W_guess: Initial guess for value function W.
        Output:
            W_updated: Updated value function W.
            V: Value function V.
            dm_results: Output from the DM problem.
            cm_results: Output from the CM problem.
            max_diff: Maximum difference between updated and previous W.
            mean_diff: Mean difference between updated and previous W.
            converged: Boolean indicating convergence status.
        """
        # Use current W if no guess provided
        if W_guess is None:
            W_guess = self.W
            
        # Step 1: Solve DM problem to get V
        dm_results = self.solve_dm_problem_vectorised(W_guess, prices)
        V = dm_results['V_dm']
        
        # Step 2: Solve CM problem to get updated W
        cm_results = self.solve_cm_problem_vectorised(V, prices)
        W_updated = cm_results['W']
        
        # Calculate convergence metrics
        diff = np.abs(W_updated - W_guess)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Store current value functions for history
        self.value_history['W'].append(W_updated.copy())
        self.value_history['V'].append(V.copy())
        
        return {
            'W': W_updated,
            'V': V,
            'dm_results': dm_results,
            'cm_results': cm_results,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'converged': max_diff < self.tol
        }


    @track_time
    def household_transition(self, G_guess, prices, cm_output, dm_output):
        """
        BLOCK 2: Household Transition
        Update the household mass distribution based on the current policies and prices.
        This function computes the new mass distribution of households in the economy  
        by iterating over the state space and applying the policy functions obtained from
        the centralised and decentralised market problems.
        Input:   
            G_guess: Initial guess for household mass distribution
            prices: Current prices [py, Rl, i]
            cm_output: output from the CM problem: W function, policy_m, policy_f
            dm_output: output from the DM problem: V function, policy_y0, policy_b0, 
                        policy_y1, policy_b1, policy_b_noshock
        Output:
            G_new: Updated household mass distribution
        """
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialise transition matrix
        G_new = np.zeros_like(G_guess)

        # Unpack transition matrix
        firm_result = self.firm_problem(prices)
        P = firm_result['transition_matrix']

        # unpack policy functions
        policy_m = cm_output['policy_m']
        policy_f = cm_output['policy_f']
        policy_y0 = dm_output['policy_y0']
        policy_b0 = dm_output['policy_b0']
        policy_y1 = dm_output['policy_y1']
        policy_b1 = dm_output['policy_b1']
        policy_b_noshock = dm_output['policy_b_noshock']

        # Loop over employment status and skill types
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):

                # Vectorise over current state (a, b)
                for a_idx, a in enumerate(self.a_grid):
                    for b_idx, b in enumerate(self.b_grid):
                        # Get the current mass of households in this state
                        g = G_guess[a_idx, b_idx, z_idx, e_idx]

                        # Calculate the next state based on the policies
                        # Formed from the pre-determined grids
                        m_prime = policy_m[a_idx, b_idx, z_idx, e_idx]
                        f_prime = policy_f[a_idx, b_idx, z_idx, e_idx]
                        total_asset = m_prime + f_prime
                        # No need to calculate the weights since it came from the grid exactly
                        m_idx = find_nearest_index(self.m_grid, m_prime)
                        f_idx = find_nearest_index(self.f_grid, f_prime)
                        
                        # Iterate over the next-period employment status
                        for e_next_idx in range(self.n_e):
                            # Transition probabilities
                            prob = P[e_idx, e_next_idx]

                            # unpack the consumption and borrowing policies
                            y_0 = policy_y0[m_idx, f_idx, z_idx, e_next_idx]
                            b_0 = policy_b0[m_idx, f_idx, z_idx, e_next_idx]
                            y_1 = policy_y1[m_idx, f_idx, z_idx, e_next_idx]
                            b_1 = policy_b1[m_idx, f_idx, z_idx, e_next_idx]
                            b_noshock = policy_b_noshock[m_idx, f_idx, z_idx, e_next_idx]

                            # Calculate the next state
                            # 1: No shock (ϵ = 0)
                            a_next_noshock = total_asset - b_noshock
                            a_noshock_lower, a_noshock_upper, a_noshock_wt = find_closest_indices(self.a_grid, a_next_noshock)
                            b_noshock_lower, b_noshock_upper, b_noshock_wt = find_closest_indices(self.b_grid, b_noshock)
                            G_new[a_noshock_lower, b_noshock_lower, z_idx, e_next_idx] += (g * (1 - self.alpha) * 
                                                                                            a_noshock_wt * b_noshock_wt * prob)
                            G_new[a_noshock_upper, b_noshock_lower, z_idx, e_next_idx] += (g * (1 - self.alpha) * 
                                                                                            (1 - a_noshock_wt) * b_noshock_wt * prob)
                            G_new[a_noshock_lower, b_noshock_upper, z_idx, e_next_idx] += (g * (1 - self.alpha) * 
                                                                                            a_noshock_wt * (1 - b_noshock_wt) * prob)
                            G_new[a_noshock_upper, b_noshock_upper, z_idx, e_next_idx] += (g * (1 - self.alpha) * 
                                                                                            (1 - a_noshock_wt) * (1 - b_noshock_wt) * prob)
                            # 2. Shock (ϵ = 1)
                            # 2.1: Only cash: ω=0
                            a0_next = total_asset - py * y_0 - b_0
                            a0_next_lower, a0_next_upper, a0_wt = find_closest_indices(self.a_grid, a0_next)
                            b0_lower, b0_upper, b0_wt = find_closest_indices(self.b_grid, b_0)
                            G_new[a0_next_lower, b0_lower, z_idx, e_next_idx] += (g * self.alpha * self.alpha_0 * a0_wt * b0_wt * prob)
                            G_new[a0_next_upper, b0_lower, z_idx, e_next_idx] += (g * self.alpha * self.alpha_0 *
                                                                                    (1 - a0_wt) * b0_wt * prob)
                            G_new[a0_next_lower, b0_upper, z_idx, e_next_idx] += (g * self.alpha * self.alpha_0 *
                                                                                    a0_wt * (1 - b0_wt) * prob)
                            G_new[a0_next_upper, b0_upper, z_idx, e_next_idx] += (g * self.alpha * self.alpha_0 *
                                                                                    (1 - a0_wt) * (1 - b0_wt) * prob)

                            # 2.2: Only illiquid asset: ω=1
                            a1_next = total_asset - py * y_1 - b_1
                            a1_next_lower, a1_next_upper, a1_wt = find_closest_indices(self.a_grid, a1_next)
                            b1_lower, b1_upper, b1_wt = find_closest_indices(self.b_grid, b_1)
                            G_new[a1_next_lower, b1_lower, z_idx, e_next_idx] += (g * self.alpha * self.alpha_1 *
                                                                                    a1_wt * b1_wt * prob)
                            G_new[a1_next_upper, b1_lower, z_idx, e_next_idx] += (g * self.alpha * self.alpha_1 *
                                                                                    (1 - a1_wt) * b1_wt * prob)
                            G_new[a1_next_lower, b1_upper, z_idx, e_next_idx] += (g * self.alpha * self.alpha_1 *
                                                                                    a1_wt * (1 - b1_wt) * prob)
                            G_new[a1_next_upper, b1_upper, z_idx, e_next_idx] += (g * self.alpha * self.alpha_1 *
                                                                                    (1 - a1_wt) * (1 - b1_wt) * prob)
        # Normalise the new distribution
        G_new /= np.sum(G_new)

        # Print shape and a sample slice
        print(f"\n[G_new] shape: {G_new.shape}")
        sample = G_new[
            :min(2, G_new.shape[0]),
            :min(2, G_new.shape[1]),
            :min(2, G_new.shape[2]),
            :min(2, G_new.shape[3])
        ]
        print(f"[G_new] sample (first 2 in each dimension):\n{sample}\n")      

        # Return the updated distribution
        return G_new


    def iterate_household_distribution(self, prices, cm_output, dm_output, tol=1e-5, max_iter=1000, verbose=True):
        """
        BLOCK 2: Iteration of the household distribution
        Iterate the household distribution until convergence.
        Input:
            prices: Current prices [py, Rl, i]
            cm_output: Output from the CM problem: W function, policy_m, policy_f
            dm_output: Output from the DM problem: V function, policy_y0, policy_b0,
                       policy_y1, policy_b1, policy_b_noshock
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
            verbose: Print convergence information
        Output:
            G_new: Updated household distribution
            error_list: List of errors at each iteration
        """
        # Initialise a uniform distribution
        shape = (len(self.a_grid), len(self.b_grid), self.n_z, self.n_e)
        G = np.ones(shape) / np.prod(shape)
        error_list = []

        for it in range(max_iter):
            G_new = self.household_transition(G, prices, cm_output, dm_output)
            error = np.max(np.abs(G_new - G))
            error_list.append(error)

            if verbose:
                print(f"Iteration {it+1}: max error = {error:.2e}")

            if error < tol:
                if verbose:
                    print(f"\nConverged after {it+1} iterations.")
                break

            G = G_new.copy()
        else:
            raise RuntimeError("Household distribution did not converge within the max number of iterations.")

        # Plot the convergence error
        plt.figure(figsize=(6, 4))
        plt.plot(error_list, marker='o')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Max Error')
        plt.title('Convergence of Household Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return G_new, error_list
                


    def interpolate_2d_vectorised(self, x_values, y_values, x_grid, y_grid, grid_values):
        """
        Vectorised bilinear interpolation for arrays of query points
        
        Parameters:
        -----------
        x_values, y_values : numpy.ndarray
            Arrays of query points (same shape)
        x_grid, y_grid : numpy.ndarray
            1D arrays of grid points
        grid_values : numpy.ndarray
            2D array of values at grid points
        """
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        
        # Find indices using broadcasting
        x_idx = np.searchsorted(x_grid, x_values) - 1
        y_idx = np.searchsorted(y_grid, y_values) - 1
        
        # Clip indices to valid range
        x_idx = np.clip(x_idx, 0, len(x_grid) - 2)
        y_idx = np.clip(y_idx, 0, len(y_grid) - 2)
        
        # Get surrounding points
        x0 = x_grid[x_idx]
        x1 = x_grid[x_idx + 1]
        y0 = y_grid[y_idx]
        y1 = y_grid[y_idx + 1]
        
        # Calculate weights using broadcasting
        wx = (x1 - x_values) / (x1 - x0)
        wy = (y1 - y_values) / (y1 - y0)
        
        # Handle edge cases
        wx = np.where((x1 - x0) == 0, 1.0, wx)
        wy = np.where((y1 - y0) == 0, 1.0, wy)
        
        # Get values at corners using advanced indexing
        v00 = grid_values[x_idx, y_idx]
        v10 = grid_values[x_idx + 1, y_idx]
        v01 = grid_values[x_idx, y_idx + 1]
        v11 = grid_values[x_idx + 1, y_idx + 1]
        
        # Bilinear interpolation using broadcasting
        return (v00 * wx * wy + 
                v10 * (1 - wx) * wy + 
                v01 * wx * (1 - wy) + 
                v11 * (1 - wx) * (1 - wy))

    


    @track_time
    def market_clearing(self, prices, G, dm_result, cm_result):
        """
        BLOCK 3: Market Clearing and Calculation of excess demand for 3 goods.
        Calculate the demand and supply for each goods: 
            Goods market: 
              - Demand: y_omega + y [am, af, z, e] in the DM;
              - Supply: employment * production, integreated across different z.
            Illiquid asset market:
              - Demand: af [a, b, z, e] in the CM;
              - Supply: exogenous government bond + profits from operating firms.
            Bank market:
              - Demand: borrowing from households;
              - Supply: deposits from households.
        """

        # precompute firm problem
        firm_result = self.firm_problem(prices)
        y_star = firm_result['Ys']
        emp = firm_result['emp']
        J_total = firm_result['J_total']
        P = firm_result['transition_matrix']


        # Unpack policy functions
        policy_y0 = dm_result['policy_y0']
        policy_b0 = dm_result['policy_b0']
        policy_y1 = dm_result['policy_y1']
        policy_b1 = dm_result['policy_b1']
        policy_b_noshock = dm_result['policy_b_noshock']
        policy_f = cm_result['policy_f']
        policy_m = cm_result['policy_m']

        # Initialise asset distribution F
        F = np.zeros((self.n_m, self.n_f, self.n_z, self.n_e))
        # Initialise market aggregates
        excess_demand = np.zeros(3)
        demand_vector = np.zeros(3)
        supply_vector = np.zeros(3)

        # Initialise market agregates
        # Goods market
        Yd = 0.0
        Ys = 0.0
        # Bank market
        Bd = 0.0
        Bs = 0.0
        # Illiquid asset market
        Fd = 0.0
        Fs = 0.0

        # Call out firm's profits + Supply of government bonds (However for the sign change we code it as demand)
        Fs += self.Ag0
        Fs += J_total  # given prices in the initialisation (self)

        # Calculate the supply of goods
        Ys += emp * (self.z_grid @ self.z_dist) * y_star

        # Loop over all states in household distribution G
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for a_idx in range(self.n_a):
                    for b_idx in range(self.n_b):
                        # Get the current mass of households in this state
                        g = G[a_idx, b_idx, z_idx, e_idx]

                        # Calculate the asset distribution F
                        af = policy_f[a_idx, b_idx, z_idx, e_idx]
                        am = policy_m[a_idx, b_idx, z_idx, e_idx]
                        af_lower, af_upper, af_wt = find_closest_indices(self.f_grid, af)
                        am_lower, am_upper, am_wt = find_closest_indices(self.m_grid, am)
                        F[am_lower, af_lower, z_idx, e_idx] += g * am_wt * af_wt
                        F[am_upper, af_lower, z_idx, e_idx] += g * (1 - am_wt) * af_wt
                        F[am_lower, af_upper, z_idx, e_idx] += g * am_wt * (1 - af_wt)
                        F[am_upper, af_upper, z_idx, e_idx] += g * (1 - am_wt) * (1 - af_wt)

                        # Calculate the demand of illiquid assets (however higher return demand more)
                        Fd += g * af
                        

                for m_idx in range(self.n_m):
                    for f_idx in range(self.n_f):
                        for e_next_idx in range(self.n_e):
                            # Get the current mass of households in this state
                            f_mass = F[m_idx, f_idx, z_idx, e_next_idx]

                            # Calculate the demand for goods (when preference shock occurs)
                            yd0 = policy_y0[m_idx, f_idx, z_idx, e_next_idx]
                            yd1 = policy_y1[m_idx, f_idx, z_idx, e_next_idx]
                            
                            Yd += f_mass * self.alpha *(self.alpha_0 * yd0 + 
                                    self.alpha_1 * yd1) * P[e_idx, e_next_idx]

                            # Extract policy functions
                            b0 = policy_b0[m_idx, f_idx, z_idx, e_next_idx]
                            b1 = policy_b1[m_idx, f_idx, z_idx, e_next_idx]
                            b_noshock = policy_b_noshock[m_idx, f_idx, z_idx, e_next_idx]

                            # Separate the positive and negative values
                            # Bank supply (deposit): only positive values
                            Bs_0 = max(0.0, b0)
                            Bs_1 = max(0.0, b1)
                            Bs_noshock = max(0.0, b_noshock)

                            # Bank demand (loans): only negative values (multiplied by -1 to be positive)
                            Bd_0 = -min(0.0, b0)
                            Bd_1 = -min(0.0, b1)
                            Bd_noshock = -min(0.0, b_noshock)

                            # Calculate the demand and supply for the state b
                            Bd += f_mass * (self.alpha * (self.alpha_0 * Bd_0 + self.alpha_1 * 
                                        Bd_1) + (1 - self.alpha) * Bd_noshock) * P[e_idx, e_next_idx]
                            Bs += f_mass * (self.alpha * (self.alpha_0 * Bs_0 + self.alpha_1 * 
                                        Bs_1) + (1 - self.alpha) * Bs_noshock) * P[e_idx, e_next_idx]

        # Store demand and supply vectors
        demand_vector[:] = [Yd, Fd, Bd]
        supply_vector[:] = [Ys, Fs, Bs]
        excess_demand[:] = [Yd - Ys, Fs - Fd, Bd - Bs]

        # Print results
        print(f"\n[F] shape: {F.shape}")
        F_sample = F[
            :min(2, F.shape[0]),
            :min(2, F.shape[1]),
            :min(2, F.shape[2]),
            :min(2, F.shape[3])
        ]
        print(f"[F] sample (first 2 in each dimension):\n{F_sample}\n")

        print("[demand_vector]:", demand_vector)
        print("[supply_vector]:", supply_vector)
        print("[excess_demand]:", excess_demand)

        return excess_demand, demand_vector, supply_vector

    
    @staticmethod
    def update_price_bisection(prices, excess_demand, price_bounds, lambda_=0.5):
        """
        Update prices using the bisection method based on the sign of excess demand.
        
        Parameters
        ----------
        prices : array-like
            Current price vector [py, Rl, i].
        excess_demand : array-like
            Excess demand vector [goods, illiquid asset, bank].
        price_bounds : list of tuples
            Bounds for each price [(py_min, py_max), (Rl_min, Rl_max), (i_min, i_max)].
        lambda_ : float
            Weighting factor between 0 and 1.
            
        Returns
        -------
        new_prices : np.ndarray
            Updated price vector.
        new_bounds : list of tuples
            Updated bounds for the next iteration.
        """
        new_prices = prices.copy()
        new_bounds = price_bounds.copy()
        
        for i in range(len(prices)):
            p_old = prices[i]
            lower, upper = price_bounds[i]
            
            if excess_demand[i] > 0:
                # Excess demand: increase price → new lower bound = current price
                lower = p_old
                p_new = lambda_ * upper + (1 - lambda_) * p_old
            else:
                # Excess supply: decrease price → new upper bound = current price
                upper = p_old
                p_new = lambda_ * lower + (1 - lambda_) * p_old

            new_prices[i] = p_new
            new_bounds[i] = (lower, upper)
            
        return new_prices, new_bounds



    def solve_model(self, prices=None, plot_frequency=50, report_frequency=100, tol_dist=1e-5, 
                        max_iter_dist=100000, tol_market=1e-3, max_iter_market=1000, lambda_=0.5):
        """
        Solve the model by iterating on the market clearing prices and the household distribution.
         BLOCK 3: Iterate prices using the bisection method.
                It contains BLOCK 1 & 2.
        Input:
            prices: Initial guess for prices [py, Rl, i]
            plot_frequency: Frequency of plotting the value functions
            report_frequency: Frequency of reporting convergence information
            tol_dist: Convergence tolerance for household distribution
            max_iter_dist: Maximum number of iterations for household distribution
            tol_market: Convergence tolerance for market clearing
            max_iter_market: Maximum number of iterations for market clearing
            lambda_: Weighting factor for price update
        Output:
            prices: Final prices [py, Rl, i]
            G: Final household distribution
            dm_result: Output from the DM problem
            cm_result: Output from the CM problem
        """
        if prices is None:
            prices = self.prices

        price_bounds = [
            (0.5, 5.0 * prices[0]),   # py bounds
            (0.5, 5.0 * prices[1]),   # Rl bounds
            (0.0, 10.0 * prices[2])    # i bounds
        ]

        market_iter = 0
        while market_iter < max_iter_market:
            market_iter += 1
            print(f"\nMarket clearing  {market_iter} iterations.")

            # Stage 1: Value Function Iteration
            W_current = self.W.copy()
            iteration = 0
            converged = False
            start_time_vf = time.time()

            os.makedirs('exports', exist_ok=True)

            vf_history = {
                'max_diff': [],
                'mean_diff': [],
                'iteration_time': []
            }

            while iteration < self.max_iter and not converged:
                iter_start = time.time()
                results = self.solver_iteration(prices, W_current)
                W_current = results['W']
                max_diff = results['max_diff']
                mean_diff = results['mean_diff']
                converged = results['converged']
                iter_time = time.time() - iter_start

                vf_history['max_diff'].append(max_diff)
                vf_history['mean_diff'].append(mean_diff)
                vf_history['iteration_time'].append(iter_time)

                iteration += 1
                if iteration % report_frequency == 0 or converged:
                    print(f"[VF] Iter {iteration}: Max diff = {max_diff:.2e}, Mean diff = {mean_diff:.2e}, Time = {iter_time:.2f}s")

                if iteration % plot_frequency == 0 or converged:
                    self.plot_functions(iteration, results)

            total_time_vf = time.time() - start_time_vf
            if converged:
                print(f"\nValue function converged in {iteration} iterations, {total_time_vf:.2f} seconds.")
            else:
                print(f"\nValue function did not converge in {self.max_iter} iterations, {total_time_vf:.2f} seconds.")

            self.W = W_current
            self.V = results['V']

            print("Exporting final V and W to Excel...")
            self.export_4d_to_excel(self.V, 'exports/V_final.xlsx', sheet_name='ValueVFunction')
            self.export_4d_to_excel(self.W, 'exports/W_final.xlsx', sheet_name='ValueWFunction')
            print("Export complete.")

            # Stage 2: Stationary Distribution Iteration
            print("\nStarting household distribution iteration...")
            dist_start = time.time()
            shape = (len(self.a_grid), len(self.b_grid), self.n_z, self.n_e)
            G = np.ones(shape) / np.prod(shape)
            dis_history = {
                'max_diff': [],
                'iteration_time': []
            }

            cm_output = {
                'policy_m': results['cm_results']['policy_m'],
                'policy_f': results['cm_results']['policy_f']
            }

            dm_output = {
                'policy_y0': results['dm_results']['policy_y0'],
                'policy_y1': results['dm_results']['policy_y1'],
                'policy_b0': results['dm_results']['policy_b0'],
                'policy_b1': results['dm_results']['policy_b1'],
                'policy_b_noshock': results['dm_results']['policy_b_noshock']
            }

            print("Exporting Policy functions Excel...")
            self.export_4d_to_excel(cm_output['policy_m'], 'exports/policy_m.xlsx', sheet_name='MoneyHolding')
            self.export_4d_to_excel(cm_output['policy_f'], 'exports/policy_f.xlsx', sheet_name='StockHolding')
            self.export_4d_to_excel(dm_output['policy_y0'], 'exports/policy_y0.xlsx', sheet_name='Consumption0')
            self.export_4d_to_excel(dm_output['policy_y1'], 'exports/policy_y1.xlsx', sheet_name='Consumption1')
            self.export_4d_to_excel(dm_output['policy_b0'], 'exports/policy_b0.xlsx', sheet_name='Borrowing0')
            self.export_4d_to_excel(dm_output['policy_b1'], 'exports/policy_b1.xlsx', sheet_name='Borrowing1')
            self.export_4d_to_excel(dm_output['policy_b_noshock'], 'exports/policy_b_noshock.xlsx', sheet_name='BorrowingNoshock')
            print("Export complete.")

            for dist_iter in range(max_iter_dist):
                G_new = self.household_transition(G, prices, cm_output, dm_output)
                error = np.max(np.abs(G_new - G))
                dis_history['max_diff'].append(error)
                iter_time = time.time() - dist_start
                dis_history['iteration_time'].append(iter_time)

                if dist_iter % report_frequency == 0 or error < tol_dist:
                    print(f"[DIST] Iter {dist_iter+1}: max error = {error:.2e}")

                if error < tol_dist:
                    print(f"\nDistribution converged in {dist_iter+1} iterations.")
                    break

                G = G_new.copy()
            else:
                raise RuntimeError("Household distribution did not converge.")

            total_time_dist = time.time() - dist_start

            print("Exporting Distribution to Excel...")
            self.export_4d_to_excel(G, 'exports/G_final.xlsx', sheet_name='HouseholdDistribution')
            print("Export complete.")

            # Market clearing evaluation
            excess_demand, demand_vector, supply_vector = self.market_clearing(prices, G, dm_output, cm_output)
            print(f"Excess demand: Goods = {excess_demand[0]:.2e}, Illiquid assets = {excess_demand[1]:.2e}, Bank = {excess_demand[2]:.2e}")
            print(f"Demand: Goods = {demand_vector[0]:.2e}, Illiquid assets = {demand_vector[1]:.2e}, Bank = {demand_vector[2]:.2e}")
            print(f"Supply: Goods = {supply_vector[0]:.2e}, Illiquid assets = {supply_vector[1]:.2e}, Bank = {supply_vector[2]:.2e}")

            # Check convergence across markets
            if np.max(np.abs(excess_demand)) < tol_market:
                print("\nMarket clearing achieved.")
                break

            # Price update using bisection
            prices, price_bounds = self.update_price_bisection(prices, excess_demand, price_bounds, lambda_=lambda_)
            print(f"Updated prices: py = {prices[0]:.4f}, Rl = {prices[1]:.4f}, i = {prices[2]:.4f}\n")

        else:
            raise RuntimeError("Equilibrium prices did not converge after maximum iterations.")

        # Plot convergence error
        plt.figure(figsize=(6, 4))
        plt.plot(dis_history['max_diff'], marker='o')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Max Error')
        plt.title('Convergence of Household Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('exports/distribution_convergence.png')
        plt.close()

        return {
            'W': self.W,
            'V': self.V,
            'G': G_new,
            'prices': prices,
            'excess_demand': excess_demand,
            'converged': converged,
            'iterations': iteration,
            'iterations_distribution': dist_iter + 1,
            'time_value': total_time_vf,
            'time_distribution': total_time_dist,
            'total_time': total_time_vf + total_time_dist,
            'history_value': vf_history,
            'history_distribution': dis_history
        }

    
    @staticmethod
    def export_4d_to_excel(array, filename, sheet_name='Sheet1'):
        """
        Flattens a 4D numpy array and exports it to an Excel file.
        Each row will contain the 4 indices and the value.
        """
        dim1, dim2, dim3, dim4 = array.shape
        idx = np.indices((dim1, dim2, dim3, dim4)).reshape(4, -1).T
        values = array.flatten()
        df = pd.DataFrame(idx, columns=['i', 'j', 'k', 'l'])
        df['value'] = values
        df.to_excel(filename, sheet_name=sheet_name, index=False)

    def plot_functions(self, iteration, results):
        """
        Plot value functions and policy functions along money and illiquid asset dimensions,
        separated by employment status and skill types.
        
        :param iteration: Current iteration number
        :param results: Results from solver_iteration
        """
        # Create directory for plots if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Extract results
        V = results['V']
        W = results['W']
        dm_results = results['dm_results']
        cm_results = results['cm_results']
        
        # 1. Value Function Plots
        # -----------------------
        
        # Plot V as a function of money holdings for different illiquid asset levels
        fig, axes = plt.subplots(self.n_z, self.n_e, figsize=(12, 14))
        fig.suptitle(f'Value Function V (Iteration {iteration})', fontsize=16)
        
        for z_idx in range(self.n_z):
            for e_idx in range(self.n_e):
                # Select a few illiquid asset levels to plot
                f_indices = [0, self.n_f//4, self.n_f//2, 3*self.n_f//4, self.n_f-1]
                
                for f_idx in f_indices:
                    f_val = self.f_grid[f_idx]
                    axes[z_idx, e_idx].plot(self.m_grid, V[:, f_idx, z_idx, e_idx], 
                                           label=f'f={f_val:.2f}')
                
                # Label plot
                employment_label = 'Employed' if e_idx == 1 else 'Unemployed'
                skill_label = f'z={self.z_grid[z_idx]:.2f}'
                axes[z_idx, e_idx].set_title(f'{skill_label}, {employment_label}')
                axes[z_idx, e_idx].set_xlabel('Money Holdings (m)')
                axes[z_idx, e_idx].set_ylabel('Value')
                axes[z_idx, e_idx].legend()
                axes[z_idx, e_idx].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/value_function_V_m_iter_{iteration}.png')
        plt.close()
        
        # Plot V as a function of illiquid assets for different money levels
        fig, axes = plt.subplots(self.n_z, self.n_e, figsize=(12, 14))
        fig.suptitle(f'Value Function V (Iteration {iteration})', fontsize=16)
        
        for z_idx in range(self.n_z):
            for e_idx in range(self.n_e):
                # Select a few money levels to plot
                m_indices = [0, self.n_m//4, self.n_m//2, 3*self.n_m//4, self.n_m-1]
                
                for m_idx in m_indices:
                    m_val = self.m_grid[m_idx]
                    axes[z_idx, e_idx].plot(self.f_grid, V[m_idx, :, z_idx, e_idx], 
                                           label=f'm={m_val:.2f}')
                
                # Label plot
                employment_label = 'Employed' if e_idx == 1 else 'Unemployed'
                skill_label = f'z={self.z_grid[z_idx]:.2f}'
                axes[z_idx, e_idx].set_title(f'{skill_label}, {employment_label}')
                axes[z_idx, e_idx].set_xlabel('Illiquid Assets (f)')
                axes[z_idx, e_idx].set_ylabel('Value')
                axes[z_idx, e_idx].legend()
                axes[z_idx, e_idx].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/value_function_V_f_iter_{iteration}.png')
        plt.close()
        
        # 2. Policy Function Plots
        # ------------------------
        
        # Plot money holdings policy for middle skill type
        z_idx = self.n_z // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Money Policy Function (z={self.z_grid[z_idx]:.2f}, Iteration {iteration})', fontsize=16)
        
        # Get slice of policy function for specific deposit and loan values
        b_idx = self.n_b // 4
        
        # For employed
        for f_idx in [0, self.n_f//4, self.n_f//2, 3*self.n_f//4, self.n_f-1]:
            f_val = self.f_grid[f_idx]
            axes[0].plot(self.m_grid, cm_results['policy_m'][:, b_idx, z_idx, 1],
                        label=f'f={f_val:.2f}')
        
        axes[0].plot(self.m_grid, self.m_grid, 'k--', label='45° line')
        axes[0].set_title('Employed')
        axes[0].set_xlabel('Current Money (m)')
        axes[0].set_ylabel('Next Period Money (m\')')
        axes[0].legend()
        axes[0].grid(True)
        
        # For unemployed
        for f_idx in [0, self.n_f//4, self.n_f//2, 3*self.n_f//4, self.n_f-1]:
            f_val = self.f_grid[f_idx]
            axes[1].plot(self.m_grid, cm_results['policy_m'][:, b_idx, z_idx, 0],
                        label=f'f={f_val:.2f}')
        
        axes[1].plot(self.m_grid, self.m_grid, 'k--', label='45° line')
        axes[1].set_title('Unemployed')
        axes[1].set_xlabel('Current Money (m)')
        axes[1].set_ylabel('Next Period Money (m\')')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/policy_money_iter_{iteration}.png')
        plt.close()
        
        # Plot illiquid asset policy for middle skill type
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Illiquid Asset Policy Function (z={self.z_grid[z_idx]:.2f}, Iteration {iteration})', fontsize=16)
        
        # For employed
        for m_idx in [0, self.n_m//4, self.n_m//2, 3*self.n_m//4, self.n_m-1]:
            m_val = self.m_grid[m_idx]
            axes[0].plot(self.f_grid, cm_results['policy_f'][:, b_idx, z_idx, 1],
                        label=f'm={m_val:.2f}')
        
        axes[0].plot(self.f_grid, self.f_grid, 'k--', label='45° line')
        axes[0].set_title('Employed')
        axes[0].set_xlabel('Current Illiquid Assets (f)')
        axes[0].set_ylabel('Next Period Illiquid Assets (f\')')
        axes[0].legend()
        axes[0].grid(True)
        
        # For unemployed
        for m_idx in [0, self.n_m//4, self.n_m//2, 3*self.n_m//4, self.n_m-1]:
            m_val = self.m_grid[m_idx]
            axes[1].plot(self.f_grid, cm_results['policy_f'][:, b_idx, z_idx, 0],
                        label=f'm={m_val:.2f}')
        
        axes[1].plot(self.f_grid, self.f_grid, 'k--', label='45° line')
        axes[1].set_title('Unemployed')
        axes[1].set_xlabel('Current Illiquid Assets (f)')
        axes[1].set_ylabel('Next Period Illiquid Assets (f\')')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/policy_illiquid_iter_{iteration}.png')
        plt.close()
        
        # 3. Portfolio Composition Plots
        # ------------------------------
        
        # Plot money ratio m/(m+f) for middle skill type
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Portfolio Composition (z={self.z_grid[z_idx]:.2f}, Iteration {iteration})', fontsize=16)
        
        # Compute portfolio compositions
        for f_idx in [0, self.n_f//4, self.n_f//2, 3*self.n_f//4, self.n_f-1]:
            f_val = self.f_grid[f_idx]
            
            # For employed
            m_next = cm_results['policy_m'][:, b_idx, z_idx, 1]
            f_next = cm_results['policy_f'][:, b_idx, z_idx, 1]
            ratio = np.divide(m_next, m_next + f_next, out=np.zeros_like(m_next), where=(m_next + f_next) > 0)
            
            axes[0].plot(self.m_grid, ratio, label=f'f={f_val:.2f}')
            
            # For unemployed
            m_next = cm_results['policy_m'][:, b_idx, z_idx, 0]
            f_next = cm_results['policy_f'][:, b_idx, z_idx, 0]
            ratio = np.divide(m_next, m_next + f_next, out=np.zeros_like(m_next), where=(m_next + f_next) > 0)
            
            axes[1].plot(self.m_grid, ratio, label=f'f={f_val:.2f}')
        
        axes[0].set_title('Employed')
        axes[0].set_xlabel('Current Money (m)')
        axes[0].set_ylabel('Money Ratio m\'/(m\'+f\')')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title('Unemployed')
        axes[1].set_xlabel('Current Money (m)')
        axes[1].set_ylabel('Money Ratio m\'/(m\'+f\')')
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'plots/portfolio_composition_iter_{iteration}.png')
        plt.close()
    

params = {
        # Preference parameters
        'beta': 0.96,      # Discount factor
        'alpha': 0.075,    # Probability of DM consumption opportunity (tested 0.3, originally works but dies overtime)
        'alpha_1': 0.06,   # Probability of accepting both money and assets
        'gamma': 1.5,      # Risk aversion parameter
        'Psi': 2.2,        # DM utility scaling parameter
        
        # Production and labor market parameters
        'psi': 0.28,       # Matching function elasticity
        'zeta': 0.75,      # Worker bargaining power
        'nu': 1.6,         # Labor supply elasticity
        'mu': 0.7,         # Matching efficiency
        'delta': 0.035,    # Job separation rate
        'kappa': 7.29,     # Vacancy posting cost
        'repl_rate': 0.4,  # Unemployment benefit replacement rate

        # DM parameters
        'c_min': 1e-2,     # minimum consumption
        
        # Grid specifications
        'n_a': 10,         # Number of asset grid points (for testing)
        'n_m': 10,         # Number of money grid points
        'n_f': 10,         # Number of illiquid asset grid points
        'n_b': 20,         # Number of bank grid points
        'a_min': 0.0,      # Minimum asset holdings
        'a_max': 20.0,     # Maximum asset holdings
        'm_min': 0.0,      # Minimum money holdings
        'm_max': 10.0,     # Maximum money holdings
        'f_min': 0.0,      # Minimum illiquid holdings
        'f_max': 10.0,     # Maximum illiquid holdings
        'b_min': -10.0,        # Minimum loan value
        'b_max': 10.0,     # Maximum loan value
        'ny': 20,          # Number of grid points for DM goods
        
        # Price parameters
        'py': 1.0,         # Price of DM goods
        'Rl': 1.03,        # Return on illiquid assets
        'i': 0.02,         # Nominal interest rate
        'Rm': (1.0-0.014)**(1.0/12.0),   # Gross return of real money balances (exogenous)

        # Government Spending
        'Ag0': 0.1,        # Exogenous government spending 

        # Convergence parameters
        'max_iter': 1000,   # Maximum number of iterations
        'tol': 1e-5        # Convergence tolerance
    }

# Initialise solver with parameters
print("Initialising LagosWrightAiyagariSolver...")
solver = LagosWrightAiyagariSolver(params)

# Set baseline prices
baseline_prices = np.array([
    params['py'],      # Price of DM goods
    params['Rl'],      # Return on illiquid assets
    params['i']        # Nominal interest rate
])

# Print initial conditions
print("\nInitial conditions:")
firm_result = solver.firm_problem(prices=baseline_prices)
print(f"  Employment rate: {firm_result['emp']:.4f}")
# print(f"  Labor market tightness: {solver.market_tightness:.4f}")
# print(f"  Job finding probability: {solver.job_finding_prob:.4f}")
# print(f"  Employment rate: {solver.emp_rate:.4f}")
print(f"  Wages (employed, z=1): {firm_result['wages'][1, 1]:.4f}")
print(f"  Wages (unemployed, z=1): {firm_result['wages'][1, 0]:.4f}")

# Solve the model
print("\nSolving model...")
# solver.compute_firm_block(prices=baseline_prices)
solution = solver.solve_model(prices=baseline_prices, plot_frequency=50, report_frequency=100)

# Print results
print("\nSolution results:")
print(f"  Converged: {solution['converged']}")
print(f"  Iterations: {solution['iterations']}")
print(f"  Total time: {solution['total_time']:.2f} seconds")
print(f"  Final max diff: {solution['history_value']['max_diff'][-1]:.6e}")

## Ensure 'plots' directory exists
os.makedirs('plots', exist_ok=True)

# --- Plot Value Function Convergence ---
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(solution['history_value']['max_diff']) + 1), 
             solution['history_value']['max_diff'], 'b-', label='Max Diff')
plt.semilogy(range(1, len(solution['history_value']['mean_diff']) + 1), 
             solution['history_value']['mean_diff'], 'r--', label='Mean Diff')
plt.axhline(y=params['tol'], color='k', linestyle=':', label='Tolerance')
plt.xlabel('Iteration')
plt.ylabel('Value Function Difference (log scale)')
plt.title('Value Function Convergence History')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/value_function_convergence.png')
plt.close()

# --- Plot Distribution Convergence ---
plt.figure(figsize=(8, 5))
plt.semilogy(range(1, len(solution['history_distribution']['max_diff']) + 1),
             solution['history_distribution']['max_diff'], marker='o', label='Max Error')
plt.axhline(y=1e-5, color='k', linestyle=':', label='Tolerance')  # match tol_dist if different
plt.xlabel('Iteration')
plt.ylabel('Distribution Difference (log scale)')
plt.title('Household Distribution Convergence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/distribution_convergence.png')
plt.close()