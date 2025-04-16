import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

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
        self.n_m = params['n_m']       # Number of money grid points
        self.n_f = params['n_f']       # Number of illiquid asset grid points
        self.n_d = params['n_d']       # Number of deposit grid points
        self.n_l = params['n_l']       # Number of loan grid points
        self.m_min = params['m_min']   # Minimum money holdings
        self.m_max = params['m_max']   # Maximum money holdings
        self.f_min = params['f_min']   # Minimum illiquid asset holdings
        self.f_max = params['f_max']   # Maximum illiquid asset holdings
        self.l_min = params['l_min']   # Minimum loan value
        self.l_max = params['l_max']   # Maximum loan value
        self.d_min = params['d_min']   # Minimum deposit value
        self.d_max = params['d_max']   # Maximum deposit value
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
        
        # Initialize grids
        self.setup_grids(params)
        
        # Initialize labor market
        self.initialise_labor_market()

        # Initialize value and policy functions
        self.initialize_functions()
        
        # For storing value function history
        self.value_history = {
            'W': [],
            'V': []
        }

    
    def setup_grids(self, params):
        """Set up state space grids with appropriate spacing"""
        # Money holdings grid
        self.m_grid = np.linspace(params['m_min'], params['m_max'], self.n_m)

        # Illiquid asset grid
        self.f_grid = np.linspace(params['f_min'], params['f_max'], self.n_f)
        
        # Deposit grid (real terms)
        self.d_grid = np.linspace(params['d_min'], params['d_max'], self.n_d)

        # Loan grid (real terms)
        self.l_grid = np.linspace(params['l_min'], params['l_max'], self.n_l)
        
        # Employment grid
        self.e_grid = np.array([0, 1])  # 0 = unemployed, 1 = employed
        
        # Productivity/skill grid
        self.z_grid = np.array([0.58, 0.98, 2.15])  # Low, medium, high skill
        
        # Skill type distribution (fixed - doesn't change over time)
        self.z_dist = np.array([0.62, 0.28, 0.1])  # Distribution of skill types

    
    def initialize_functions(self):
        """Initialize value and policy functions with reasonable guesses"""
        # Initialize arrays using m and f as state variables
        self.W = np.zeros((self.n_m, self.n_f, self.n_d, self.n_l, self.n_z, self.n_e))
        self.V = np.zeros((self.n_m, self.n_f, self.n_z, self.n_e))
        
        # Initialize W with reasonable guesses using a utility-based approach
        for e_idx, e in enumerate(self.e_grid):
            for z_idx, z in enumerate(self.z_grid):
                # Estimate income based on skill and employment
                income_est = self.wages[z_idx, e_idx]
                
                for m_idx, m in enumerate(self.m_grid):
                    for f_idx, f in enumerate(self.f_grid):
                        for d_idx, d in enumerate(self.d_grid):
                            for l_idx, l in enumerate(self.l_grid):
                                # Estimate resources available for consumption
                                resources = m + f + (1 + self.prices[2]) * d - (1 + self.prices[2]) * l + income_est
                                
                                # Ensure minimum consumption
                                consumption = max(resources * 0.9, self.c_min)  # Consume ~90% of resources
                                
                                # Calculate utility and scale for perpetuity
                                self.W[m_idx, f_idx, d_idx, l_idx, z_idx, e_idx] = self.utility(consumption)
        
        # Initialize V based on W
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for m_idx, m in enumerate(self.m_grid):
                    for f_idx, f in enumerate(self.f_grid):
                        # Simple estimate: deposit all money, no loans
                        d_est = m
                        l_est = 0
                        
                        # Find closest grid points
                        d_idx = min(self.find_nearest_index(self.d_grid, d_est), self.n_d - 1)
                        l_idx = min(self.find_nearest_index(self.l_grid, l_est), self.n_l - 1)
                        
                        # Map to V
                        self.V[m_idx, f_idx, z_idx, e_idx] = self.W[m_idx, f_idx, d_idx, l_idx, z_idx, e_idx]


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
        profits = self.frev - self.wages_bar[:, 1]
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
        Utility function for decentralized market -- parameters are different from the centralised market
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
        
        for m_idx, a_m in enumerate(self.m_grid):
            for f_idx, a_f in enumerate(self.f_grid):
                a = min(a_m + a_f, self.a_max)  # Total assets
                y1_nb_grids[(m_idx, f_idx)] = np.linspace(0, a/py, self.ny)  # Can use all assets when ω=1
                y0_nb_grids[(m_idx, f_idx)] = np.linspace(0, a_m/py, self.ny)
        
        # Loop over employment states and skill types
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                income = self.wages[z_idx, e_idx]
                # Vectorize for different asset levels
                for f_idx, a_f in enumerate(self.f_grid):
                    # Case 3: No preference shock (can be fully vectorized for all money levels)
                    # When there's no preference shock, optimal policy is to deposit all money
                    w_noshock_values = np.zeros(self.n_m)
                    for m_idx, a_m in enumerate(self.m_grid):
                        # Total assets
                        a = min(a_m + a_f, self.a_max)
                        # Calculate continuation value
                        w_noshock_values[m_idx] = self.interpolate_2d_vectorised(
                            np.array([a - a_m]), np.array([a_m]),
                            self.a_grid, self.d_grid,
                            W_guess[:, :, 0, z_idx, e_idx]
                        )[0]
                        # Deposit all money
                        policy_d_noshock[m_idx, f_idx, z_idx, e_idx] = a_m
                        
                            
                    # Store no-shock values for all money levels at once
                    V_noshock[:, f_idx, z_idx, e_idx] = w_noshock_values
                    
                    # Process each money level separately for shock cases
                    for m_idx, a_m in enumerate(self.m_grid):
                        a = a_m + a_f
                        
                        # -------------------------------------------------------------------------
                        # Case 1: Preference shock + ω=0 (only money accepted)
                        # -------------------------------------------------------------------------
                        
                        # Case 1.1: No borrowing (can use precomputed grid)
                        y0_nb = y0_nb_grids[(m_idx, f_idx)]
                        d0 = a_m - py*y0_nb  # won't exceed the maximum
                        
                        w0_nb = self.interpolate_2d_vectorised(
                            a_f, d0, 
                            self.f_grid, self.d_grid,                # a_f is the total asset
                            W_guess[:, :, 0, z_idx, e_idx]           # when money is either consumed or deposited
                        )
                        v0_nb = self.utility_dm(y0_nb) + w0_nb
                        
                        # Case 1.2: With borrowing
                        # Efficient borrowing constraint calculation
                        income_based_max = (a_f + income - self.c_min)/(1 + i)
                        collateral_based_max = a_f/(1 + i)
                        effective_max_borrow = max(0, min(income_based_max, collateral_based_max))
                        
                        # Skip borrowing calculations if no borrowing is possible
                        if effective_max_borrow > 0:
                            y0_b = np.linspace(0, (a_m + effective_max_borrow)/py, self.ny)
                            l0 = np.clip(py*y0_b - a_m, 0, effective_max_borrow)   # borrow exactly the amount needed to consume
                            
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
                                policy_y0[m_idx, f_idx, z_idx, e_idx] = y0_nb[max_nb_idx]
                                policy_d0[m_idx, f_idx, z_idx, e_idx] = d0[max_nb_idx]
                                policy_l0[m_idx, f_idx, z_idx, e_idx] = 0.0
                                V0[m_idx, f_idx, z_idx, e_idx] = max_nb_val
                            else:
                                policy_y0[m_idx, f_idx, z_idx, e_idx] = y0_b[max_b_idx]
                                policy_d0[m_idx, f_idx, z_idx, e_idx] = 0.0
                                policy_l0[m_idx, f_idx, z_idx, e_idx] = l0[max_b_idx]
                                V0[m_idx, f_idx, z_idx, e_idx] = max_b_val
                        else:
                            # No borrowing possible, only use no-borrowing case
                            max_idx = np.argmax(v0_nb)
                            policy_y0[m_idx, f_idx, z_idx, e_idx] = y0_nb[max_idx]
                            policy_d0[m_idx, f_idx, z_idx, e_idx] = d0[max_idx]
                            policy_l0[m_idx, f_idx, z_idx, e_idx] = 0.0
                            V0[m_idx, f_idx, z_idx, e_idx] = v0_nb[max_idx]
                        
                        # -------------------------------------------------------------------------
                        # Case 2: Preference shock + ω=1 (both assets accepted)
                        # -------------------------------------------------------------------------
                        
                        
                        # Case 2.1: No borrowing (use precomputed grid)
                        y1_nb = y1_nb_grids[(m_idx, f_idx)]
                        d1 = a - py*y1_nb
                        
                        w1_nb = self.interpolate_2d_vectorised(
                            a_f, d1,       # money is either consumed or deposited; only a_f is the total asset
                            self.a_grid, self.d_grid,
                            W_guess[:, :, 0, z_idx, e_idx]
                        )
                        v1_nb = self.utility_dm(y1_nb) + w1_nb
                        
                        # Case 2.2: With borrowing
                        max_borrow = max(0, (income - self.c_min)/(1 + i))
                        
                        # Skip borrowing calculations if no borrowing is possible
                        if max_borrow > 0:
                            y1_b = np.linspace(0, (a + max_borrow)/py, self.ny)
                            l1 = np.clip(py*y1_b - a, 0, max_borrow)
                            
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
                                policy_y1[m_idx, f_idx, z_idx, e_idx] = y1_nb[max_nb_idx]
                                policy_d1[m_idx, f_idx, z_idx, e_idx] = d1[max_nb_idx]
                                policy_l1[m_idx, f_idx, z_idx, e_idx] = 0.0
                                V1[m_idx, f_idx, z_idx, e_idx] = max_nb_val
                            else:
                                policy_y1[m_idx, f_idx, z_idx, e_idx] = y1_b[max_b_idx]
                                policy_d1[m_idx, f_idx, z_idx, e_idx] = 0.0
                                policy_l1[m_idx, f_idx, z_idx, e_idx] = l1[max_b_idx]
                                V1[m_idx, f_idx, z_idx, e_idx] = max_b_val
                        else:
                            # No borrowing possible, only use no-borrowing case
                            max_idx = np.argmax(v1_nb)
                            policy_y1[m_idx, f_idx, z_idx, e_idx] = y1_nb[max_idx]
                            policy_d1[m_idx, f_idx, z_idx, e_idx] = d1[max_idx]
                            policy_l1[m_idx, f_idx, z_idx, e_idx] = 0.0
                            V1[m_idx, f_idx, z_idx, e_idx] = v1_nb[max_idx]

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
        m_prime_grid, f_prime_grid = np.meshgrid(self.m_grid, self.f_grid, indexing='ij')
        
        # Precompute some portfolio costs
        portfolio_cost = f_prime_grid / Rl + m_prime_grid / self.Rm
        
        # Loop over employment status and skill types
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                income = self.wages[z_idx, e_idx]
                
                # Precompute continuation values for this (z, e) combination
                # Create an array of shape (n_a, n_m) for continuation values
                v_continuation = np.zeros((self.n_a, self.n_m))
                for m_idx in range(self.n_m):
                    for f_idx in range(self.n_f):
                        # Expected value over employment transitions
                        v_continuation[m_idx, f_idx] = V_guess[m_idx, f_idx, z_idx, :] @ self.P[e_idx, :]
                
                # Vectorize over current state (a, d, l)
                for a_idx, a in enumerate(self.a_grid):
                    for d_idx, d in enumerate(self.d_grid):
                        for l_idx, l in enumerate(self.l_grid):
                            # Calculate total resources available
                            total_resources = a + (1 + i) * d - (1 + i) * l + income
                            
                            # Initialize choice array
                            w_choice = np.full((self.n_a, self.n_m), -1e5)  # Default to very negative
                            
                            # Calculate continuation values for valid portfolio combinations only
                            for m_idx, m_prime in enumerate(self.m_grid):
                                for f_idx, f_prime in enumerate(self.f_grid):
                                    
                                    # Calculate consumption
                                    c = total_resources - f_prime / Rl - m_prime / self.Rm
                                    
                                    if c >= self.c_min:  # Valid consumption level
                                        # Expected value over employment transitions
                                        v_cont = V_guess[m_idx, f_idx, z_idx, :] @ self.P[e_idx, :]
                                        w_choice[m_idx, f_idx] = self.utility(c) + self.beta * v_cont
                            
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
    
    def solver_iteration(self, prices, W_guess=None):
        """
        Performs one iteration of the solution algorithm:
        1. Solve DM problem given W_guess to get V
        2. Solve CM problem given V to get updated W
        
        :param prices: Current prices [py, Rl, i]
        :param W_guess: Initial guess for value function, uses self.W if None
        :return: Updated value functions, policy functions, and convergence information
        """
        # Use current W if no guess provided
        if W_guess is None:
            W_guess = self.W
            
        # Step 1: Solve DM problem to get V
        dm_results = self.solve_dm_problem_vectorized(W_guess, prices)
        V = dm_results['V_dm']
        
        # Step 2: Solve CM problem to get updated W
        cm_results = self.solve_cm_problem_vectorized(V, prices)
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

    def solve_model(self, prices=None, plot_frequency=10, export_frequency=20):
        """
        Solve the model by iterating until convergence or max iterations reached.
        Plots value and policy functions periodically during iteration.
        
        :param prices: Custom prices [py, Rl, i], uses self.prices if None
        :param plot_frequency: How often to plot (every N iterations)
        :param export_frequency: How often to export value functions (every N iterations)
        :return: Solution dictionary with value functions, policy functions, and convergence info
        """
        if prices is None:
            prices = self.prices
            
        # Initialize
        W_current = self.W.copy()
        iteration = 0
        converged = False
        start_time = time.time()
        
        # Create directory for exports if it doesn't exist
        if not os.path.exists('exports'):
            os.makedirs('exports')
        
        # Store convergence history
        history = {
            'max_diff': [],
            'mean_diff': [],
            'iteration_time': []
        }
        
        # Main solution loop
        while iteration < self.max_iter and not converged:
            iter_start = time.time()
            
            # Perform one iteration
            results = self.solver_iteration(prices, W_current)
            W_current = results['W']
            
            # Store convergence metrics
            max_diff = results['max_diff']
            mean_diff = results['mean_diff']
            converged = results['converged']
            
            # Calculate iteration time
            iter_time = time.time() - iter_start
            history['max_diff'].append(max_diff)
            history['mean_diff'].append(mean_diff)
            history['iteration_time'].append(iter_time)
            
            # Display progress
            iteration += 1
            if iteration % 5 == 0 or converged:
                print(f"Iteration {iteration}: Max diff = {max_diff:.6e}, Mean diff = {mean_diff:.6e}, Time = {iter_time:.2f}s")
            
            # Plot current value and policy functions
            if iteration % plot_frequency == 0 or converged:
                self.plot_functions(iteration, results)
            
            # Export value functions to Excel
            if iteration % export_frequency == 0 or converged:
                self.export_value_functions(iteration)
        
        # Final results
        total_time = time.time() - start_time
        
        if converged:
            print(f"\nConverged after {iteration} iterations. Total time: {total_time:.2f}s")
        else:
            print(f"\nDid not converge after {self.max_iter} iterations. Total time: {total_time:.2f}s")
        
        # Store final value functions
        self.W = W_current
        self.V = results['V']
        
        # Export final value functions
        self.export_value_functions(iteration, final=True)
        
        # Return complete solution
        return {
            'W': W_current,
            'V': results['V'],
            'dm_results': results['dm_results'],
            'cm_results': results['cm_results'],
            'converged': converged,
            'iterations': iteration,
            'total_time': total_time,
            'history': history
        }

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
        d_idx = self.n_d // 2
        l_idx = 0  # No loans
        
        # For employed
        for f_idx in [0, self.n_f//4, self.n_f//2, 3*self.n_f//4, self.n_f-1]:
            f_val = self.f_grid[f_idx]
            axes[0].plot(self.m_grid, cm_results['policy_m'][:, f_idx, d_idx, l_idx, z_idx, 1], 
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
            axes[1].plot(self.m_grid, cm_results['policy_m'][:, f_idx, d_idx, l_idx, z_idx, 0], 
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
            axes[0].plot(self.f_grid, cm_results['policy_f'][m_idx, :, d_idx, l_idx, z_idx, 1], 
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
            axes[1].plot(self.f_grid, cm_results['policy_f'][m_idx, :, d_idx, l_idx, z_idx, 0], 
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
            m_next = cm_results['policy_m'][:, f_idx, d_idx, l_idx, z_idx, 1]
            f_next = cm_results['policy_f'][:, f_idx, d_idx, l_idx, z_idx, 1]
            ratio = np.divide(m_next, m_next + f_next, out=np.zeros_like(m_next), where=(m_next + f_next) > 0)
            
            axes[0].plot(self.m_grid, ratio, label=f'f={f_val:.2f}')
            
            # For unemployed
            m_next = cm_results['policy_m'][:, f_idx, d_idx, l_idx, z_idx, 0]
            f_next = cm_results['policy_f'][:, f_idx, d_idx, l_idx, z_idx, 0]
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
    
    def export_value_functions(self, iteration, final=False):
        """
        Export value functions to Excel files
        
        :param iteration: Current iteration number
        :param final: Whether this is the final iteration
        """
        # Create directories if they don't exist
        if not os.path.exists('exports'):
            os.makedirs('exports')
        
        prefix = 'final' if final else f'iter_{iteration}'
        
        # Export V function
        V = self.V
        
        # Create multi-index DataFrame for V
        idx = pd.MultiIndex.from_product([
            self.m_grid, self.f_grid, self.z_grid, ['Unemployed', 'Employed']
        ], names=['Money', 'Illiquid', 'Skill', 'Employment'])
        
        # Reshape V for DataFrame
        V_flat = V.reshape(-1)
        df_V = pd.DataFrame({'Value': V_flat}, index=idx)
        
        # Export to Excel
        df_V.to_excel(f'exports/V_{prefix}.xlsx')
        
        # Export W function (for select combinations to manage file size)
        W = self.W
        
        # Select a subset of deposit/loan indices
        d_indices = [0, self.n_d//2, self.n_d-1]
        l_indices = [0, self.n_l//2, self.n_l-1]
        
        for d_idx in d_indices:
            for l_idx in l_indices:
                d_val = self.d_grid[d_idx]
                l_val = self.l_grid[l_idx]
                
                # Create multi-index DataFrame for this W slice
                idx = pd.MultiIndex.from_product([
                    self.m_grid, self.f_grid, self.z_grid, ['Unemployed', 'Employed']
                ], names=['Money', 'Illiquid', 'Skill', 'Employment'])
                
                # Extract slice and reshape
                W_slice = W[:, :, d_idx, l_idx, :, :]
                W_flat = W_slice.reshape(-1)
                
                df_W = pd.DataFrame({'Value': W_flat}, index=idx)
                
                # Export to Excel
                df_W.to_excel(f'exports/W_{prefix}_d{d_val:.2f}_l{l_val:.2f}.xlsx')
        
        # If we're storing value function history, export the latest entry
        if hasattr(self, 'value_history') and len(self.value_history['V']) > 0:
            # Get latest stored V
            hist_idx = len(self.value_history['V']) - 1
            V_hist = self.value_history['V'][hist_idx]
            
            # Reshape and export
            V_hist_flat = V_hist.reshape(-1)
            df_V_hist = pd.DataFrame({'Value': V_hist_flat}, index=idx)
            df_V_hist.to_excel(f'exports/V_history_{hist_idx}.xlsx')

def main():
    # Parameter definitions
    params = {
        # ...existing params...
    }
    
    # Initialize solver
    solver = LagosWrightAiyagariSolver(params)
    
    # Set baseline prices and solve
    baseline_prices = np.array([
        params['py'],
        params['Rl'],
        params['i']
    ])
    
    # Solve model
    solution = solver.solve_model(prices=baseline_prices)
    return solution

    
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
        'c_min': 1e-2,    # minimum consumption
        
        # Grid specifications
        'n_m': 20,        # Number of money grid points
        'n_f': 20,        # Number of illiquid asset grid points
        'n_d': 20,        # Number of deposit grid points
        'n_l': 20,        # Number of loan grid points
        'm_min': 0.0,     # Minimum money holdings
        'm_max': 10.0,    # Maximum money holdings
        'f_min': 0.0,     # Minimum illiquid holdings
        'f_max': 10.0,    # Maximum illiquid holdings
        'l_min': 0,       # Minimum loan value
        'l_max': 10.0,    # Maximum loan value
        'd_min': 0,       # Minimum deposit value
        'd_max': 10.0,    # Maximum deposit value
        'ny': 100,        # Number of grid points for DM goods
        
        # Price parameters
        'py': 1.0,        # Price of DM goods
        'Rl': 1.03,       # Return on illiquid assets
        'i': 0.02,        # Nominal interest rate
        'Rm': (1.0-0.014)**(1.0/12.0),   # Gross return of real money balances (exogenous)

        # Government Spending
        'Ag0': 0.1,       # Exogenous government spending 

        # Convergence parameters
        'max_iter': 100,   # Maximum number of iterations
        'tol': 1e-5       # Convergence tolerance
    }

    # Initialize solver with parameters
    print("Initializing LagosWrightAiyagariSolver...")
    solver = LagosWrightAiyagariSolver(params)
    
    # Set baseline prices
    baseline_prices = np.array([
        params['py'],      # Price of DM goods
        params['Rl'],      # Return on illiquid assets
        params['i']        # Nominal interest rate
    ])
    
    # Print initial conditions
    print("\nInitial conditions:")
    print(f"  Labor market tightness: {solver.market_tightness:.4f}")
    print(f"  Job finding probability: {solver.job_finding_prob:.4f}")
    print(f"  Employment rate: {solver.emp_rate:.4f}")
    print(f"  Wages (employed, z=1): {solver.wages[1, 1]:.4f}")
    print(f"  Wages (unemployed, z=1): {solver.wages[1, 0]:.4f}")
    
    # Solve the model
    print("\nSolving model...")
    solution = solver.solve_model(prices=baseline_prices, plot_frequency=10, export_frequency=20)
    
    # Print results
    print("\nSolution results:")
    print(f"  Converged: {solution['converged']}")
    print(f"  Iterations: {solution['iterations']}")
    print(f"  Total time: {solution['total_time']:.2f} seconds")
    print(f"  Final max diff: {solution['history']['max_diff'][-1]:.6e}")
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(solution['history']['max_diff'])+1), 
                solution['history']['max_diff'], 'b-', label='Max Diff')
    plt.semilogy(range(1, len(solution['history']['mean_diff'])+1), 
                solution['history']['mean_diff'], 'r--', label='Mean Diff')
    plt.axhline(y=params['tol'], color='k', linestyle=':', label='Tolerance')
    plt.xlabel('Iteration')
    plt.ylabel('Value Function Difference (log scale)')
    plt.title('Convergence History')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/convergence_history.png')
    
    print("\nAnalysis complete. Check the 'plots' and 'exports' directories for visualizations and data.")