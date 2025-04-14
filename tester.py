import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

class LagosWrightAiyagariSolver:
    def __init__(self, params):
        """Initialize solver with model parameters"""
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
        self.n_d = params['n_d']       # Number of deposit grid points
        self.n_l = params['n_l']       # Number of loan grid points
        self.a_min = params['a_min']   # Minimum asset value
        self.a_max = params['a_max']   # Maximum asset value
        self.m_min = params['m_min']   # Minimum money holdings
        self.m_max = params['m_max']   # Maximum money holdings
        self.l_min = params['l_min']   # Minimum deposit/loan value
        self.l_max = params['l_max']   # Maximum deposit/loan value
        self.n_e = 2                   # Employment states: [0=unemployed, 1=employed]
        self.n_z = 3                   # Skill types: [0=low, 1=medium, 2=high]
        self.ny = 300                  # Number of grid points for DM goods
        self.nd = 50                   # Number of grid points for deposit/loan

        # for computation
        self.c_min = params['c_min']   # Minimum consumption
        self.Rm = params['Rm']       # Gross return of real money balances (exogenous)
        
        # Price parameters
        self.prices = np.array([
            params['py'],              # Price of DM goods
            params['Rl'],             # Return on illiquid assets
            params['i']              # Nominal rate from banks
        ])
        
        # Convergence parameters
        self.max_iter = params['max_iter']
        self.tol = params['tol']
        
        
        # Initialize grids
        self.setup_grids(params)

        
        # Initialize value and policy functions
        self.initialize_functions()

    
    def setup_grids(self, params):
        """Set up state space grids with appropriate spacing"""
        # Asset grid
        self.a_grid = np.linspace(params['a_min'], params['a_max'], self.n_a)
        
        # Money holdings grid
        self.m_grid = np.linspace(params['m_min'], params['m_max'], self.n_m)
        
        # Deposit grid (real terms)
        self.d_grid = np.linspace(params['m_min'], params['m_max'], self.n_m)

        # loan grid
        self.l_grid = np.linspace(params['l_min'], params['l_max'], self.n_l)
        
        # Employment grid
        self.e_grid = np.array([0, 1])  # 0 = unemployed, 1 = employed
        
        # Productivity/skill grid
        self.z_grid = np.array([0.58, 0.98, 2.15])  # Low, medium, high skill
        
        # Skill type distribution (fixed - doesn't change over time)
        self.z_dist = np.array([0.62, 0.28, 0.1])  # Distribution of skill types

    
    def initialize_functions(self):
        """Initialize value and policy functions"""
        # Initialize arrays
        self.W = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        self.V = np.zeros((self.n_a, self.n_m, self.n_z, self.n_e))
        
        # Initialize W with reasonable guesses using broadcasting
        a_grid_reshaped = self.a_grid.reshape(-1, 1)  # Shape: (n_a, 1)


    def initialise_labor_market(self):
        """
        Initialize labor market variables and calculate steady state values.
        """

        # Firm production and revenue calculations
        self.Ys = self.κ_prime_inv(self.prices[0])  # Optimal production of early consumption goods
        
        # Calculate firm revenue for each skill type
        self.frev = np.zeros(self.n_z)
        for z_idx in range(self.n_z):
            self.frev[z_idx] = self.z_grid[z_idx] * (1.0 + self.prices[0] * self.Ys - self.κ_fun(np.array([self.Ys]))[0])
        
        # Calculate wages and unemployment benefits (based on labor share)
        self.wages_bar = np.zeros((self.n_z, self.n_e))
        self.wages_bar[:, 1] = self.mu * self.frev  # Employed wage (labor share * revenue)
        self.wages_bar[:, 0] = self.replace_rate * self.wages_bar[:, 1]  # Unemployed benefits
        
        # Taxes and transfers
        self.Ag0 = params['Ag0']  # Government bond supply
        taulumpsum = ((1.0 / self.prices[1]) - 1.0) * self.Ag0  # Revenue from money creation
        
        # Apply lump-sum transfer to all households using broadcasting
        self.wages = self.wages_bar + taulumpsum

        # Firm profits and value
        profits = self.frev - self.wages_bar[1, :]
        self.Js = profits / (1.0 - ((1.0 - self.delta) / self.prices[1]))
        
        # Calculate steady state tightness and employment rate
        # First check if the free-entry condition can be satisfied
        entry_condition_max = np.max((self.z_grid * self.kappa * self.prices[1]) / self.Js)
        
        if entry_condition_max < 1:
            # Use the first productivity type to determine tightness (should be the same for all z)
            self.market_tightness = self.job_fill_inv((self.z_grid[0] * self.kappa * self.prices[1]) / self.Js[0])
        else:
            # If entry condition can't be satisfied, set very low tightness
            self.market_tightness = 0.0001
        
        # Calculate job finding probability and employment rate
        self.job_finding_prob = self.job_finding(self.market_tightness)
        self.emp_rate = self.job_finding_prob / (self.delta + self.job_finding_prob)
        
        # Employment transition matrix: rows = current state, cols = next state
        # [0,0] = P(unemployed → unemployed), [0,1] = P(unemployed → employed)
        # [1,0] = P(employed → unemployed), [1,1] = P(employed → employed)
        self.P = np.array([
            [1 - self.job_finding_prob, self.job_finding_prob],
            [self.delta, 1 - self.delta]
        ])
        
        
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
                    -1e5,  # penalty for non-positive consumption
                    (safe_c ** (1 - self.gamma) - 1) / (1 - self.gamma))
    
    def utility_dm(self, y):
        """
        Utility function for decentralized market -- paramters are different from the centralised market
        Generates 26% increase in consumption during preference shocks.
        """
        y = np.asarray(y)
        return np.where(y <= 0,
                    0,  # normalisation in the paper
                    self.Psi * (y ** (1 - self.psi) - 1) / (1 - self.psi))
    
    def κ_prime_inv(self, py):
        """
        Solve the firm's optimal early consumption goods production, given early consumption price.
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
        Opportunity cost of producing y.
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

    def firm_post_rev(self, prices):
        """
        Calculates the productivity-normalized value of a filled job.
        This represents the value per unit of worker productivity (z).
        Both revenue and costs scale with z, so we normalize by z to get 
        a single market tightness for all productivity types.
        
        Derived from Eq (13): ϕᶠ(z) = zq(pʸ) - zw₁ + (1-δ)ϕᶠ(z)/Rⁱ
        When normalized by z: ϕ̃ᶠ = ϕᶠ(z)/z = q(pʸ) - w₁ + (1-δ)ϕ̃ᶠ/Rⁱ
        
        Returns:
            normalized_rev: The value per unit of productivity (independent of z)
        """
        py = prices[0]
        Rl = prices[1]

        q = self.q_fun(py)
        normalized_rev = (1 - self.mu) * q / (1 - (1 - self.delta) / Rl)
        return normalized_rev
    
    def job_finding(self, θ):
        """
        Job finding probability λ for a worker.
        :param θ: tightness in the labor market:  job vacancy / seekers
        :return: vacancy filling rate between 0 and 1.
        """
        prob = 1 / (1 + θ ** (-self.zeta)) ** (1 / self.zeta)
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
        Inverse function of job filling function.
        :param x: probability of job filling
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
        # Get productivity-normalized value of a filled job
        normalized_rev = self.firm_post_rev(prices=prices)
        
        # Compare expected cost with expected benefit (both normalised by z)
        if kappa >= normalized_rev / Rl:
            θ = 1e-3  # Almost no vacancies if cost exceeds revenue
        else:
            θ = self.job_fill_inv(kappa * Rl / normalized_rev)
        
        # Compute job finding probability & vacancy filling probability
        λ = self.job_finding(θ)
        filling = self.job_filling(θ)
        return θ, λ, filling
    

    def solve_dm_problem(self, W_guess, prices):
        """
        Solve DM problem for given prices and guess for value function.
        :param W_guess: Initial guess for value function
        :param prices: Current prices [py, Rl, i]
        :return: Updated value function and policy functions
        """
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialize policy arrays
        policy_y0 = np.zeros_like(self.V)
        policy_d0 = np.zeros_like(self.V)
        policy_l0 = np.zeros_like(self.V)
        V0 = np.zeros_like(self.V)
        policy_y1 = np.zeros_like(self.V)
        policy_d1 = np.zeros_like(self.V)
        policy_l1 = np.zeros_like(self.V)
        V1 = np.zeros_like(self.V)
        
        # Policy arrays for no preference shock case
        policy_d_noshock = np.zeros_like(self.V)
        V_noshock = np.zeros_like(self.V)

        
        
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                income = self.wages[z_idx, e_idx]
                for a_idx, a in enumerate(self.a_grid):
                    for m_idx, a_m in enumerate(self.m_grid):
                        a_l = max(min(a - a_m, self.a_max), self.a_min)

                        # -------------------------------------------------------------------------
                        # Case 1: Preference shock + ω=0 (only money accepted)
                        # -------------------------------------------------------------------------
                        
                        # Case 1.1: No borrowing
                        y0_nb = np.linspace(0, a_m/py, self.ny)
                        d0 = a_m - py*y0_nb
                        w0_nb = self.interpolate_2d_vectorised(
                            a - d0 - py*y0_nb, d0, 
                            self.a_grid, self.d_grid, 
                            W_guess[:, :, 0, z_idx, e_idx]
                        )
                        v0_nb = self.utility_dm(y0_nb) + w0_nb
                        
                        # Case 1.2: With borrowing
                        # Revised effective borrowing constraint - take minimum of two constraints
                        income_based_max = (a_l + income - self.c_min)/(1 + i)
                        collateral_based_max = a_l/(1 + i)
                        effective_max_borrow = min(income_based_max, collateral_based_max)
                        
                        # Ensure non-negative borrowing
                        effective_max_borrow = max(0, effective_max_borrow)
                        
                        y0_b = np.linspace(0, (a_m + effective_max_borrow)/py, self.ny)
                        l0 = np.maximum(py*y0_b - a_m, 0)
                        l0 = np.minimum(l0, effective_max_borrow)  # Apply borrowing constraint
                        
                        w0_b = self.interpolate_2d_vectorised(
                            a + l0 - py*y0_b, l0,
                            self.a_grid, self.l_grid,
                            W_guess[:, 0, :, z_idx, e_idx]
                        )
                        v0_b = self.utility_dm(y0_b) + w0_b
                        
                        # Optimal choice for ω=0
                        if v0_nb.max() > v0_b.max():
                            opt_idx = np.argmax(v0_nb)
                            policy_y0[a_idx,m_idx,z_idx,e_idx] = y0_nb[opt_idx]
                            policy_d0[a_idx,m_idx,z_idx,e_idx] = d0[opt_idx]
                            policy_l0[a_idx,m_idx,z_idx,e_idx] = 0.0
                            V0[a_idx,m_idx,z_idx,e_idx] = v0_nb[opt_idx]
                        else:
                            opt_idx = np.argmax(v0_b)
                            policy_y0[a_idx,m_idx,z_idx,e_idx] = y0_b[opt_idx]
                            policy_d0[a_idx,m_idx,z_idx,e_idx] = 0.0
                            policy_l0[a_idx,m_idx,z_idx,e_idx] = l0[opt_idx]
                            V0[a_idx,m_idx,z_idx,e_idx] = v0_b[opt_idx]

                        # -------------------------------------------------------------------------
                        # Case 2: Preference shock + ω=1 (both assets accepted)
                        # -------------------------------------------------------------------------
                        
                        # Total liquidity available when ω=1 is the entire asset portfolio a
                        total_assets = a
                        
                        # Case 2.1: No borrowing
                        y1_nb = np.linspace(0, total_assets/py, self.ny)
                        d1 = np.maximum(0, total_assets - py*y1_nb)
                        w1_nb = self.interpolate_2d_vectorised(
                            a - d1 - py*y1_nb, d1,
                            self.a_grid, self.d_grid,
                            W_guess[:, :, 0, z_idx, e_idx]
                        )
                        v1_nb = self.utility_dm(y1_nb) + w1_nb
                        
                        # Case 2.2: With borrowing
                        # For ω=1, only consider income-based borrowing constraint
                        max_borrow = max(0, (income - self.c_min)/(1 + i))
                        
                        y1_b = np.linspace(0, (total_assets + max_borrow)/py, self.ny)
                        l1 = np.maximum(py*y1_b - total_assets, 0)
                        l1 = np.minimum(l1, max_borrow)  # Apply borrowing constraint
                        
                        w1_b = self.interpolate_2d_vectorised(
                            a + l1 - py*y1_b, l1,
                            self.a_grid, self.l_grid,
                            W_guess[:, 0, :, z_idx, e_idx]
                        )
                        v1_b = self.utility_dm(y1_b) + w1_b
                        
                        # Optimal choice for ω=1
                        if v1_nb.max() > v1_b.max():
                            opt_idx = np.argmax(v1_nb)
                            policy_y1[a_idx,m_idx,z_idx,e_idx] = y1_nb[opt_idx]
                            policy_d1[a_idx,m_idx,z_idx,e_idx] = d1[opt_idx]
                            policy_l1[a_idx,m_idx,z_idx,e_idx] = 0.0
                            V1[a_idx,m_idx,z_idx,e_idx] = v1_nb[opt_idx]
                        else:
                            opt_idx = np.argmax(v1_b)
                            policy_y1[a_idx,m_idx,z_idx,e_idx] = y1_b[opt_idx]
                            policy_d1[a_idx,m_idx,z_idx,e_idx] = 0.0
                            policy_l1[a_idx,m_idx,z_idx,e_idx] = l1[opt_idx]
                            V1[a_idx,m_idx,z_idx,e_idx] = v1_b[opt_idx]
                        
                        # -------------------------------------------------------------------------
                        # Case 3: No preference shock
                        # -------------------------------------------------------------------------
                        # When there is no preference shock, the household optimally deposits all money
                        # and does not borrow (l = 0, d > 0)
                        
                        # All money can be deposited
                        d_noshock = a_m
                        
                        # Calculate continuation value
                        w_noshock = self.interpolate_2d_vectorised(
                            a - d_noshock, d_noshock,
                            self.a_grid, self.d_grid,
                            W_guess[:, :, 0, z_idx, e_idx]
                        )
                        
                        # Store values and policies
                        policy_d_noshock[a_idx,m_idx,z_idx,e_idx] = d_noshock
                        V_noshock[a_idx,m_idx,z_idx,e_idx] = w_noshock

        # Calculate expected DM value function (alpha0*V0 + alpha1*V1 + (1-alpha)*V_noshock)
        V_dm = self.alpha * (self.alpha_0 * V0 + self.alpha_1 * V1)  + (1 - self.alpha) * V_noshock

        # Return updated value function and policy functions
        return {
            'V_dm': V_dm,
            'V0': V0,
            'V1': V1,
            'V_noshock': V_noshock,
            'policy_y0': policy_y0,
            'policy_d0': policy_d0,
            'policy_l0': policy_l0,
            'policy_y1': policy_y1,
            'policy_d1': policy_d1,
            'policy_l1': policy_l1,
            'policy_d_noshock': policy_d_noshock
        }

    def solve_dm_problem_vectorized(self, W_guess, prices):
        """
        Vectorized version of the DM problem solver.
        :param W_guess: Initial guess for value function
        :param prices: Current prices [py, Rl, i]
        :return: Updated value function and policy functions
        """
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialize arrays - using float32 for memory efficiency if precision allows
        shape = (self.n_a, self.n_m, self.n_z, self.n_e)
        policy_y0 = np.zeros(shape, dtype=np.float32)
        policy_d0 = np.zeros(shape, dtype=np.float32)
        policy_l0 = np.zeros(shape, dtype=np.float32)
        V0 = np.zeros(shape, dtype=np.float32)
        policy_y1 = np.zeros(shape, dtype=np.float32)
        policy_d1 = np.zeros(shape, dtype=np.float32)
        policy_l1 = np.zeros(shape, dtype=np.float32)
        V1 = np.zeros(shape, dtype=np.float32)
        policy_d_noshock = np.zeros(shape, dtype=np.float32)
        V_noshock = np.zeros(shape, dtype=np.float32)
        
        # Precompute y-grids for all asset levels
        y0_nb_grids = {}  # No borrowing grids for ω=0
        y1_nb_grids = {}  # No borrowing grids for ω=1
        
        for a_idx, a in enumerate(self.a_grid):
            y1_nb_grids[a_idx] = np.linspace(0, a/py, self.ny)  # Can use all assets when ω=1
            for m_idx, a_m in enumerate(self.m_grid):
                y0_nb_grids[(a_idx, m_idx)] = np.linspace(0, a_m/py, self.ny)  # Only money when ω=0
        
        # Loop over employment states and skill types
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                income = self.wages[z_idx, e_idx]
                
                # Vectorize for different asset levels
                for a_idx, a in enumerate(self.a_grid):
                    # Case 3: No preference shock (can be fully vectorized for all money levels)
                    # When there's no preference shock, optimal policy is to deposit all money
                    a_m_values = self.m_grid  # All possible money values
                    d_noshock_values = a_m_values  # Deposit all money
                    
                    # Vectorized interpolation for no-shock case
                    w_noshock_values = np.zeros(self.n_m)
                    for m_idx, a_m in enumerate(a_m_values):
                        # Calculate continuation value
                        w_noshock_values[m_idx] = self.interpolate_2d_vectorised(
                            np.array([a - a_m]), np.array([a_m]),
                            self.a_grid, self.d_grid,
                            W_guess[:, :, 0, z_idx, e_idx]
                        )[0]
                    
                    # Store no-shock policies and values for all money levels at once
                    policy_d_noshock[a_idx, :, z_idx, e_idx] = d_noshock_values
                    V_noshock[a_idx, :, z_idx, e_idx] = w_noshock_values
                    
                    # Process each money level separately for shock cases
                    for m_idx, a_m in enumerate(self.m_grid):
                        a_l = max(min(a - a_m, self.a_max), self.a_min)
                        
                        # -------------------------------------------------------------------------
                        # Case 1: Preference shock + ω=0 (only money accepted)
                        # -------------------------------------------------------------------------
                        
                        # Case 1.1: No borrowing (can use precomputed grid)
                        y0_nb = y0_nb_grids[(a_idx, m_idx)]
                        d0 = a_m - py*y0_nb
                        
                        w0_nb = self.interpolate_2d_vectorised(
                            a - d0 - py*y0_nb, d0, 
                            self.a_grid, self.d_grid, 
                            W_guess[:, :, 0, z_idx, e_idx]
                        )
                        v0_nb = self.utility_dm(y0_nb) + w0_nb
                        
                        # Case 1.2: With borrowing
                        # Efficient borrowing constraint calculation
                        income_based_max = (a_l + income - self.c_min)/(1 + i)
                        collateral_based_max = a_l/(1 + i)
                        effective_max_borrow = max(0, min(income_based_max, collateral_based_max))
                        
                        # Skip borrowing calculations if no borrowing is possible
                        if effective_max_borrow > 0:
                            y0_b = np.linspace(0, (a_m + effective_max_borrow)/py, self.ny)
                            l0 = np.clip(py*y0_b - a_m, 0, effective_max_borrow)
                            
                            w0_b = self.interpolate_2d_vectorised(
                                a + l0 - py*y0_b, l0,
                                self.a_grid, self.l_grid,
                                W_guess[:, 0, :, z_idx, e_idx]
                            )
                            v0_b = self.utility_dm(y0_b) + w0_b
                            
                            # Find optimal ω=0 policy efficiently
                            max_nb_idx = np.argmax(v0_nb)
                            max_b_idx = np.argmax(v0_b)
                            max_nb_val = v0_nb[max_nb_idx]
                            max_b_val = v0_b[max_b_idx]
                            
                            if max_nb_val >= max_b_val:
                                policy_y0[a_idx, m_idx, z_idx, e_idx] = y0_nb[max_nb_idx]
                                policy_d0[a_idx, m_idx, z_idx, e_idx] = d0[max_nb_idx]
                                policy_l0[a_idx, m_idx, z_idx, e_idx] = 0.0
                                V0[a_idx, m_idx, z_idx, e_idx] = max_nb_val
                            else:
                                policy_y0[a_idx, m_idx, z_idx, e_idx] = y0_b[max_b_idx]
                                policy_d0[a_idx, m_idx, z_idx, e_idx] = 0.0
                                policy_l0[a_idx, m_idx, z_idx, e_idx] = l0[max_b_idx]
                                V0[a_idx, m_idx, z_idx, e_idx] = max_b_val
                        else:
                            # No borrowing possible, only use no-borrowing case
                            max_idx = np.argmax(v0_nb)
                            policy_y0[a_idx, m_idx, z_idx, e_idx] = y0_nb[max_idx]
                            policy_d0[a_idx, m_idx, z_idx, e_idx] = d0[max_idx]
                            policy_l0[a_idx, m_idx, z_idx, e_idx] = 0.0
                            V0[a_idx, m_idx, z_idx, e_idx] = v0_nb[max_idx]
                        
                        # -------------------------------------------------------------------------
                        # Case 2: Preference shock + ω=1 (both assets accepted)
                        # -------------------------------------------------------------------------
                        
                        # Total liquidity available when ω=1 is the entire asset portfolio
                        total_assets = a
                        
                        # Case 2.1: No borrowing (use precomputed grid)
                        y1_nb = y1_nb_grids[a_idx]
                        d1 = np.maximum(0, total_assets - py*y1_nb)
                        
                        w1_nb = self.interpolate_2d_vectorised(
                            a - d1 - py*y1_nb, d1,
                            self.a_grid, self.d_grid,
                            W_guess[:, :, 0, z_idx, e_idx]
                        )
                        v1_nb = self.utility_dm(y1_nb) + w1_nb
                        
                        # Case 2.2: With borrowing
                        max_borrow = max(0, (income - self.c_min)/(1 + i))
                        
                        # Skip borrowing calculations if no borrowing is possible
                        if max_borrow > 0:
                            y1_b = np.linspace(0, (total_assets + max_borrow)/py, self.ny)
                            l1 = np.clip(py*y1_b - total_assets, 0, max_borrow)
                            
                            w1_b = self.interpolate_2d_vectorised(
                                a + l1 - py*y1_b, l1,
                                self.a_grid, self.l_grid,
                                W_guess[:, 0, :, z_idx, e_idx]
                            )
                            v1_b = self.utility_dm(y1_b) + w1_b
                            
                            # Find optimal ω=1 policy efficiently
                            max_nb_idx = np.argmax(v1_nb)
                            max_b_idx = np.argmax(v1_b)
                            max_nb_val = v1_nb[max_nb_idx]
                            max_b_val = v1_b[max_b_idx]
                            
                            if max_nb_val >= max_b_val:
                                policy_y1[a_idx, m_idx, z_idx, e_idx] = y1_nb[max_nb_idx]
                                policy_d1[a_idx, m_idx, z_idx, e_idx] = d1[max_nb_idx]
                                policy_l1[a_idx, m_idx, z_idx, e_idx] = 0.0
                                V1[a_idx, m_idx, z_idx, e_idx] = max_nb_val
                            else:
                                policy_y1[a_idx, m_idx, z_idx, e_idx] = y1_b[max_b_idx]
                                policy_d1[a_idx, m_idx, z_idx, e_idx] = 0.0
                                policy_l1[a_idx, m_idx, z_idx, e_idx] = l1[max_b_idx]
                                V1[a_idx, m_idx, z_idx, e_idx] = max_b_val
                        else:
                            # No borrowing possible, only use no-borrowing case
                            max_idx = np.argmax(v1_nb)
                            policy_y1[a_idx, m_idx, z_idx, e_idx] = y1_nb[max_idx]
                            policy_d1[a_idx, m_idx, z_idx, e_idx] = d1[max_idx]
                            policy_l1[a_idx, m_idx, z_idx, e_idx] = 0.0
                            V1[a_idx, m_idx, z_idx, e_idx] = v1_nb[max_idx]

        # Calculate expected value with correct probability weighting
        V_dm = self.alpha * (self.alpha_0 * V0 + self.alpha_1 * V1) + (1 - self.alpha) * V_noshock

        # Return updated value function and policy functions
        return {
            'V_dm': V_dm,
            'V0': V0,
            'V1': V1,
            'V_noshock': V_noshock,
            'policy_y0': policy_y0,
            'policy_d0': policy_d0,
            'policy_l0': policy_l0,
            'policy_y1': policy_y1,
            'policy_d1': policy_d1,
            'policy_l1': policy_l1,
            'policy_d_noshock': policy_d_noshock
        }


    def solve_cm_problem(self, V_guess, prices):
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialize value and policy functions
        w_value = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        policy_ap = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        policy_m = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        
        # Main loop for solving the CM problem
        for e_idx, e in enumerate(self.e_grid):
            for z_idx in range(self.n_z):
                income = self.wages[z_idx, e_idx]
                for a_idx, a in enumerate(self.a_grid):
                    for d_idx, d in enumerate(self.d_grid):
                        for l_idx, l in enumerate(self.l_grid):
                            # Calculate total resources available
                            total_resources = a + (1 + i) * d - (1 + i) * l + income
                            w_choice = np.zeros((self.n_a, self.n_m))
                            
                            # Create meshgrid for vectorization
                            a_prime_grid, m_prime_grid = np.meshgrid(self.a_grid, self.m_grid, indexing='ij')
                            
                            # Calculate illiquid assets for each portfolio choice (vectorized)
                            a_l_grid = np.clip(a_prime_grid - m_prime_grid, self.a_min, self.a_max)
                            
                            # Calculate consumption for each choice (vectorized)
                            c_grid = total_resources - a_l_grid / Rl - m_prime_grid / self.Rm
                            
                            # Apply utility function (vectorized)
                            utility_grid = np.where(c_grid >= self.c_min, 
                                                self.utility(c_grid), 
                                                -1e5)  # Penalty for insufficient consumption
                            
                            # Calculate continuation values (partially vectorized)
                            v_continuation = np.zeros_like(utility_grid)
                            for ap_idx in range(self.n_a):
                                for m_idx, m in enumerate(self.m_grid):
                                    # Expected value over employment transitions
                                    v_continuation[ap_idx, m_idx] = V_guess[ap_idx, m_idx, z_idx, :] @ self.P[e_idx, :]
                            
                            # Total value
                            w_choice = utility_grid + self.beta * v_continuation
                            
                            # Find the maximum value and corresponding indices
                            max_val = np.max(w_choice)
                            max_indices = np.unravel_index(np.argmax(w_choice), w_choice.shape)
                            
                            # Store the maximum value and corresponding policies
                            w_value[a_idx, d_idx, l_idx, z_idx, e_idx] = max_val
                            policy_ap[a_idx, d_idx, l_idx, z_idx, e_idx] = self.a_grid[max_indices[0]]
                            policy_m[a_idx, d_idx, l_idx, z_idx, e_idx] = self.m_grid[max_indices[1]]

        # Return updated value function and policy functions
        return {
            'W': w_value,
            'a_p': policy_ap,
            'a_m': policy_m,
        }

    def solve_cm_problem_vectorized(self, V_guess, prices):
        """
        Vectorized solution for CM problem
        :param V_guess: Initial guess for value function
        :param prices: Current prices [py, Rl, i]
        :return: Updated value function and policy functions
        """
        # Unpack prices
        py = prices[0]
        Rl = prices[1]
        i = prices[2]

        # Initialize value and policy functions
        w_value = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        policy_ap = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        policy_m = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        
        # Create meshgrids for potential portfolio choices
        a_prime_grid, m_prime_grid = np.meshgrid(self.a_grid, self.m_grid, indexing='ij')
        
        # Calculate illiquid assets for each portfolio choice
        a_l_grid = np.clip(a_prime_grid - m_prime_grid, self.a_min, self.a_max)
        
        # Precompute some portfolio costs
        portfolio_cost = a_l_grid / Rl + m_prime_grid / self.Rm
        
        # Loop over employment status and skill types
        for e_idx, e in enumerate(self.e_grid):
            for z_idx in range(self.n_z):
                income = self.wages[z_idx, e_idx]
                
                # Precompute continuation values for this (z, e) combination
                # Create an array of shape (n_a, n_m) for continuation values
                v_continuation = np.zeros((self.n_a, self.n_m))
                for ap_idx in range(self.n_a):
                    for m_idx in range(self.n_m):
                        # Expected value over employment transitions
                        v_continuation[ap_idx, m_idx] = V_guess[ap_idx, m_idx, z_idx, :] @ self.P[e_idx, :]
                
                # Vectorize over current state (a, d, l)
                for a_idx, a in enumerate(self.a_grid):
                    for d_idx, d in enumerate(self.d_grid):
                        for l_idx, l in enumerate(self.l_grid):
                            # Calculate total resources available
                            total_resources = a + (1 + i) * d - (1 + i) * l + income
                            
                            # Calculate consumption for each portfolio choice
                            c_grid = total_resources - portfolio_cost
                            
                            # Apply utility function to all consumption levels at once
                            utility_grid = np.where(c_grid >= self.c_min, 
                                                self.utility(c_grid), 
                                                -1e5)  # Penalty for insufficient consumption
                            
                            # Total value (utility + continuation value)
                            w_choice = utility_grid + self.beta * v_continuation
                            
                            # Find the maximum value and corresponding indices
                            max_val = np.max(w_choice)
                            max_indices = np.unravel_index(np.argmax(w_choice), w_choice.shape)
                            
                            # Store the maximum value and corresponding policies
                            w_value[a_idx, d_idx, l_idx, z_idx, e_idx] = max_val
                            policy_ap[a_idx, d_idx, l_idx, z_idx, e_idx] = self.a_grid[max_indices[0]]
                            policy_m[a_idx, d_idx, l_idx, z_idx, e_idx] = self.m_grid[max_indices[1]]

        # Return updated value function and policy functions
        return {
            'W': w_value,
            'a_p': policy_ap,
            'a_m': policy_m,
        }


    @staticmethod    
    def find_closest_indices(value, grid):
        """
        Find the two closest indices in the grid to the given value.
        
        Parameters:
        -----------
        value : float
            The value to find closest indices for
        grid : numpy.ndarray
            The grid of values
        
        Returns:
        --------
        tuple
            (lower_index, upper_index, lower_weight, upper_weight)
        """
        # Check if value is outside the grid
        if value <= grid[0]:
            return 0, 0, 1.0, 0.0
        elif value >= grid[-1]:
            return len(grid)-1, len(grid)-1, 1.0, 0.0
        
        # Find the index of the grid point just below the value
        lower_idx = np.searchsorted(grid, value, side='right') - 1
        upper_idx = lower_idx + 1
        
        # Calculate weights for interpolation
        grid_range = grid[upper_idx] - grid[lower_idx]
        if grid_range == 0:  # Avoid division by zero
            lower_weight = 1.0
            upper_weight = 0.0
        else:
            lower_weight = (grid[upper_idx] - value) / grid_range
            upper_weight = (value - grid[lower_idx]) / grid_range
        
        return lower_idx, upper_idx, lower_weight, upper_weight


    
    def interpolate_2d_vectorised(self, x_values, y_values, x_grid, y_grid, grid_values):
        """
        Vectorized bilinear interpolation for arrays of query points
        
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
    
    
    def find_nearest_index(self, grid, value):
        """Find index of the nearest grid point to a given value"""
        idx = np.abs(grid - value).argmin()
        return idx
    
    def find_nearest_lower_index(self, grid, value):
            """Find index of the nearest grid point less than or equal to a given value"""
            indices = np.where(grid <= value)[0]
            if len(indices) == 0:
                return 0
            return indices[-1]



if __name__ == "__main__":
    params = {
        # Preference parameters
        'beta': 0.96,      # Discount factor
        'alpha': 0.075,    # Probability of DM consumption opportunity
        'alpha_1': 0.06,   # Probability of accepting both money and assets
        'gamma': 1.5,      # Risk aversion parameter
        'Psi': 2.2,       # DM utility scaling parameter
        
        # Production and labor market parameters
        'psi': 0.28,      # Matching function elasticity
        'zeta': 0.75,     # Worker bargaining power
        'nu': 1.6,        # Labor supply elasticity
        'mu': 0.7,        # Matching efficiency
        'delta': 0.035,   # Job separation rate
        'kappa': 7.29,    # Vacancy posting cost
        'repl_rate': 0.4, # Unemployment benefit replacement rate

        # DM parameters
        'c_min': 1e-2,       # minimum consumption
        
        # Grid specifications
        'n_a': 100,       # Number of asset grid points
        'n_m': 50,        # Number of money grid points
        'n_l': 60,        # Number of deposit grid points
        'n_d': 50,        # Number of loan grid points
        'a_min': 0.0,     # Minimum asset value
        'a_max': 20.0,    # Maximum asset value
        'm_min': 0.0,     # Minimum money holdings
        'm_max': 15.0,    # Maximum money holdings
        'l_min': 0,         # Minimum loan value
        'l_max': 15.0,     # Maximum loan value
        'd_min': 0,         # Minimum deposit value
        'd_max': 15.0,     # Maximum deposit value
        'n_e': 2,         # Employment states: [0=unemployed, 1=employed]
        'n_z': 3,         # Skill types: [0=low, 1=medium, 2=high]
        'ny': 300,        # Number of grid points for DM goods
        
        # Price parameters
        'py': 1.0,        # Price of DM goods
        'Rl': 1.03,       # Return on illiquid assets
        'i': 0.02,        # Nominal interest rate
        'Rm': (1.0-0.014)**(1.0/12.0),   # Gross return of real money balances (exogenous)

        # Government Spending
        'Ag0': 0.1,      # Exogenous government spending, 

        # Convergence parameters
        'max_iter': 500,
        'tol': 1e-5
    }

    
    solver = LagosWrightAiyagariSolver(params)
    
    # Set initial market tightness (will be used to calculate job-finding probability)
    solver.market_tightness = 0.7
    
    # Solve the model for given prices
    convergence_path = solver.solve_model()
    

