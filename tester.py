import numpy as np
import matplotlib.pyplot as plt

class LagosWrightAiyagariSolver:
    def __init__(self, params):
        # Model parameters
        self.beta = params.get('beta', 0.96)
        self.alpha = params.get('alpha', 0.075)
        self.alpha_1 = params.get('alpha_1', 0.06)
        self.alpha_0 = self.alpha - self.alpha_1
        
        self.gamma = params.get('gamma', 1.5)
        self.Psi = params.get('Psi', 2.2)
        self.psi = params.get('psi', 0.28)
        self.zeta = params.get('zeta', 0.75)
        self.nu = params.get('nu', 1.6)
        self.mu = params.get('mu', 0.7)
        self.delta = params.get('delta', 0.035)
        self.kappa = params.get('kappa', 7.29)
        
        self.replace_rate = params.get('repl_rate', 0.4)
        
        # Grid specifications
        self.n_a = params.get('n_a', 100)
        self.n_m = params.get('n_m', 50)
        self.n_D = params.get('n_D', 40)  # Grid size for deposits/loans
        self.n_e = 2                      # Employment states: [0=unemployed, 1=employed]
        self.n_z = 3                      # Skill types: [0=low, 1=medium, 2=high]
        
        # Price parameters - stored as a vector for easier equilibrium solving
        self.prices = np.array([
            params.get('py', 1.0),      # Price of early consumption goods
            params.get('Rl', 1.03),     # Return on illiquid assets
            params.get('phi_m', 1.0),   # Price of money
            params.get('i', 0.02)       # Nominal interest rate
        ])
        
        # Convergence parameters
        self.max_iter = params.get('max_iter', 1000)
        self.tol = params.get('tol', 1e-6)
        
        # Initialize grids
        self.setup_grids(params)

        # Firm production and revenue calculations
        self.Ys = self.κ_prime_inv(self.prices[0])
        
        # Calculate firm revenue for each skill type
        self.frev = np.zeros(self.n_z)
        for z_idx in range(self.n_z):
            self.frev[z_idx] = self.z_grid[z_idx] * (1.0 + self.prices[0] * self.Ys - self.κ_fun(np.array([self.Ys]))[0])
        
        # Calculate wages and unemployment benefits (base on labor share)
        self.wages_bar = np.zeros((2, self.n_z))
        self.wages_bar[1, :] = self.mu * self.frev  # Employed wage (labor share * revenue)
        self.wages_bar[0, :] = self.replace_rate * self.wages_bar[1, :]  # Unemployed benefits
        
        # Taxes and transfers
        self.Ag0 = params.get('Ag0', 1.0)  # Government bond supply
        taulumpsum = ((1.0 / self.prices[1]) - 1.0) * self.Ag0  # Revenue from money creation
        
        # Apply lump-sum transfer to all households
        self.tau = np.zeros((self.n_a, 2, self.n_z))
        for z_idx in range(self.n_z):
            for e_idx in range(2):
                for a_idx in range(self.n_a):
                    self.tau[a_idx, e_idx, z_idx] = taulumpsum
        
        # Full income (wages + transfers) for all states
        self.wages = np.zeros((self.n_a, 2, self.n_z))
        for z_idx in range(self.n_z):
            for e_idx in range(2):
                self.wages[:, e_idx, z_idx] = self.wages_bar[e_idx, z_idx] + self.tau[:, e_idx, z_idx]
        
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
        self.emp_transition = np.array([
            [1 - self.job_finding_prob, self.job_finding_prob],
            [self.delta, 1 - self.delta]
        ])
    
        # Initialize value and policy functions
        self.initialize_functions()

    
    def setup_grids(self, params):
        """Set up state space grids with appropriate spacing"""
        # Asset grid (potentially non-uniform)
        a_min = params.get('a_min', 0.0)
        a_max = params.get('a_max', 20.0)
        self.a_grid = np.linspace(a_min, a_max, self.n_a)
        
        # Money holdings grid
        m_min = params.get('m_min', 0.0)
        m_max = params.get('m_max', 10.0)
        self.m_grid = np.linspace(m_min, m_max, self.n_m)
        
        # Deposit/loan grid
        D_min = params.get('D_min', -5.0)  # Negative values = loans
        D_max = params.get('D_max', 5.0)   # Positive values = deposits
        self.D_grid = np.linspace(D_min, D_max, self.n_D)
        
        # Employment grid
        self.e_grid = np.array([0, 1])  # 0 = unemployed, 1 = employed
        
        # Productivity/skill grid
        self.z_grid = np.array([0.58, 0.98, 2.15])  # Low, medium, high skill
        
        # Skill type distribution (fixed - doesn't change over time)
        self.z_dist = np.array([0.62, 0.28, 0.1])  # Distribution of skill types
    
    def initialize_functions(self):
        """Initialize value and policy functions"""
        # CM value function W: dimensions (employment, assets, deposits, skill)
        self.W = np.zeros((self.n_e, self.n_a, self.n_D, self.n_z))
        
        # DM value function V: dimensions (employment, assets, money, skill)
        self.V = np.zeros((self.n_e, self.n_a, self.n_m, self.n_z))
        
        # Initialize with reasonable guesses
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                z = self.z_grid[z_idx]
                for a_idx in range(self.n_a):
                    a = self.a_grid[a_idx]
                    for D_idx in range(self.n_D):
                        D = self.D_grid[D_idx]
                        # Initial W value
                        inc = self.wages[a_idx, e_idx, z_idx]  # Income based on skill and employment
                        c_guess = inc + 0.05 * (a + max(0, D) * self.prices[3])  # Rough consumption guess
                        self.W[e_idx, a_idx, D_idx, z_idx] = self.utility(c_guess) / (1 - self.beta)
        
        # Policy functions for CM
        self.policy_c = np.zeros((self.n_e, self.n_a, self.n_D, self.n_z))
        self.policy_a_next = np.zeros((self.n_e, self.n_a, self.n_D, self.n_z))
        self.policy_m_next = np.zeros((self.n_e, self.n_a, self.n_D, self.n_z))
        
        # Policy functions for DM
        self.policy_y = np.zeros((self.n_e, self.n_a, self.n_m, self.n_z, 2))  # DM consumption (for ω={0,1})
        self.policy_b = np.zeros((self.n_e, self.n_a, self.n_m, self.n_z, 2))  # DM borrowing (for ω={0,1})
        self.policy_d = np.zeros((self.n_e, self.n_a, self.n_m, self.n_z))     # Money deposits


    def utility(self, c):
        """
        1. Utility function for centralised market
        2. c : endogenously determined by the skill type e, which impacts the budget constraint
        """
        if c <= 0:
            return -1e10  # Large negative value for infeasible consumption
        return (c ** (1 - self.gamma) - 1) / (1 - self.gamma)
    
    def utility_dm(self, y):
        """
        Utility function for decentralized market -- paramters are different from the centralised market
        Generates 26% increase in consumption during preference shocks.
        """
        if y <= 0:
            return 0  # No early consumption gives zero utility
        return self.Psi * (y ** (1 - self.psi) - 1) / (1 - self.psi)
    
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
    
    
    def solve_model(self):
        """Main solution method that iterates until convergence"""
        # Initialize convergence tracking
        diff = np.inf
        iter_count = 0
        convergence_path = []
        
        # Run value function iteration
        while diff > self.tol and iter_count < self.max_iter:
            # Store old value function for convergence check
            W_old = self.W.copy()
            
            # Step 1: Solve the DM problem for all states
            self.solve_dm_problems()
            
            # Step 2: Use DM solutions to update CM value function
            self.update_CM_value_function()
            
            # Check convergence on W (CM value function)
            diff = np.max(np.abs(self.W - W_old))
            convergence_path.append(diff)
            
            if iter_count % 10 == 0:
                print(f"Iteration {iter_count}, max diff: {diff:.8f}")
                # Periodically visualize key functions to check for economic sense
                if iter_count % 50 == 0 and iter_count > 0:
                    self.visualize_convergence(iter_count)
            
            iter_count += 1
        
        if diff <= self.tol:
            print(f"Converged after {iter_count} iterations")
        else:
            print(f"Failed to converge after {iter_count} iterations")
        
        # Generate final visualizations
        self.visualize_solution()
        
        return convergence_path
    
    def solve_dm_problems(self):
        """Solve DM problem for all states"""
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for a_idx in range(self.n_a):
                    for m_idx in range(self.n_m):
                        self.solve_dm_problem(e_idx, a_idx, m_idx, z_idx)
        
        # After solving all DM problems, update V (DM value function)
        self.update_DM_value_function()
    
    def solve_dm_problem(self, e_idx, a_idx, m_idx, z_idx):
        """
        Solve the decentralized market problem for a given state.
        
        Parameters:
        e_idx: Index of employment state (0=unemployed, 1=employed)
        a_idx: Index of asset holdings
        m_idx: Index of money holdings
        z_idx: Index of skill type
        """
        a = self.a_grid[a_idx]
        m = self.m_grid[m_idx]
        z = self.z_grid[z_idx]
        
        # Extract prices
        py = self.prices[0]
        phi_m = self.prices[2]
        i = self.prices[3]
        
        # Case ω=0: Only money accepted
        # Maximum spending with only money (no borrowing)
        y0_max_no_borrow = m * phi_m / py
        
        # Try different consumption levels without borrowing
        y0_values_no_borrow = np.linspace(0, y0_max_no_borrow, 50)
        util0_no_borrow = np.zeros_like(y0_values_no_borrow)
        
        for i, y0 in enumerate(y0_values_no_borrow):
            # Post-DM wealth and deposit position
            remain_m = m - y0 * py / phi_m
            a_post = a
            D = 0  # No deposits or loans
            
            # Calculate utility (expected over possible next employment states)
            dm_utility = self.utility_dm(y0)
            cm_utility = self.interpolate_CM_value(e_idx, a_post, remain_m, D, z_idx)
            util0_no_borrow[i] = dm_utility + cm_utility
        
        # Find optimal y0 without borrowing
        if len(util0_no_borrow) > 0:
            best_idx = np.argmax(util0_no_borrow)
            optimal_y0_no_borrow = y0_values_no_borrow[best_idx]
            max_util0_no_borrow = util0_no_borrow[best_idx]
        else:
            optimal_y0_no_borrow = 0
            max_util0_no_borrow = self.utility_dm(0) + self.interpolate_CM_value(e_idx, a, m, 0, z_idx)
        
        # With borrowing:
        # We need to consider higher y values that require borrowing
        y0_max_with_borrow = min(3*y0_max_no_borrow, 10)  # Arbitrary upper bound
        y0_values_with_borrow = np.linspace(y0_max_no_borrow+0.01, y0_max_with_borrow, 50)
        util0_with_borrow = np.zeros_like(y0_values_with_borrow)
        b0_values = np.zeros_like(y0_values_with_borrow)
        
        for i, y0 in enumerate(y0_values_with_borrow):
            # Calculate required borrowing
            payment_needed = py * y0
            available_funds = m * phi_m
            b0 = (payment_needed - available_funds) / phi_m
            
            # Post-DM position - all money spent plus borrowing
            remain_m = 0
            a_post = a
            D = -b0  # Negative D indicates loan
            
            # Calculate utility
            dm_utility = self.utility_dm(y0)
            cm_utility = self.interpolate_CM_value(e_idx, a_post, remain_m, D, z_idx)
            util0_with_borrow[i] = dm_utility + cm_utility
            b0_values[i] = b0
        
        # Find optimal y0 with borrowing
        if len(util0_with_borrow) > 0:
            best_idx = np.argmax(util0_with_borrow)
            optimal_y0_with_borrow = y0_values_with_borrow[best_idx]
            optimal_b0_with_borrow = b0_values[best_idx]
            max_util0_with_borrow = util0_with_borrow[best_idx]
        else:
            optimal_y0_with_borrow = 0
            optimal_b0_with_borrow = 0
            max_util0_with_borrow = -np.inf
        
        # Compare utilities with and without borrowing
        if max_util0_no_borrow >= max_util0_with_borrow:
            optimal_y0 = optimal_y0_no_borrow
            optimal_b0 = 0
        else:
            optimal_y0 = optimal_y0_with_borrow
            optimal_b0 = optimal_b0_with_borrow
        
        # Case ω=1: Both money and assets can be used
        # Maximum spending with money and assets (no borrowing)
        y1_max_no_borrow = (m * phi_m + a) / py
        
        # Try different consumption levels without borrowing
        y1_values_no_borrow = np.linspace(0, y1_max_no_borrow, 50)
        util1_no_borrow = np.zeros_like(y1_values_no_borrow)
        
        for i, y1 in enumerate(y1_values_no_borrow):
            # Calculate how payment is split between money and assets
            payment = py * y1
            
            # Use money first, then assets if needed
            money_payment = min(m * phi_m, payment)
            asset_payment = payment - money_payment
            
            remain_m = m - money_payment / phi_m
            remain_a = a - asset_payment
            D = 0  # No borrowing
            
            # Calculate utility
            dm_utility = self.utility_dm(y1)
            cm_utility = self.interpolate_CM_value(e_idx, remain_a, remain_m, D, z_idx)
            util1_no_borrow[i] = dm_utility + cm_utility
        
        # Find optimal y1 without borrowing
        if len(util1_no_borrow) > 0:
            best_idx = np.argmax(util1_no_borrow)
            optimal_y1_no_borrow = y1_values_no_borrow[best_idx]
            max_util1_no_borrow = util1_no_borrow[best_idx]
        else:
            optimal_y1_no_borrow = 0
            max_util1_no_borrow = self.utility_dm(0) + self.interpolate_CM_value(e_idx, a, m, 0, z_idx)
        
        # With borrowing:
        y1_max_with_borrow = min(3*y1_max_no_borrow, 10)
        y1_values_with_borrow = np.linspace(y1_max_no_borrow+0.01, y1_max_with_borrow, 50)
        util1_with_borrow = np.zeros_like(y1_values_with_borrow)
        b1_values = np.zeros_like(y1_values_with_borrow)
        
        for i, y1 in enumerate(y1_values_with_borrow):
            # Calculate required borrowing
            payment_needed = py * y1
            available_funds = m * phi_m + a
            b1 = (payment_needed - available_funds) / phi_m
            
            # Post-DM position - all money and assets spent plus borrowing
            remain_m = 0
            remain_a = 0
            D = -b1  # Negative D indicates loan
            
            # Calculate utility
            dm_utility = self.utility_dm(y1)
            cm_utility = self.interpolate_CM_value(e_idx, remain_a, remain_m, D, z_idx)
            util1_with_borrow[i] = dm_utility + cm_utility
            b1_values[i] = b1
        
        # Find optimal y1 with borrowing
        if len(util1_with_borrow) > 0:
            best_idx = np.argmax(util1_with_borrow)
            optimal_y1_with_borrow = y1_values_with_borrow[best_idx]
            optimal_b1_with_borrow = b1_values[best_idx]
            max_util1_with_borrow = util1_with_borrow[best_idx]
        else:
            optimal_y1_with_borrow = 0
            optimal_b1_with_borrow = 0
            max_util1_with_borrow = -np.inf
        
        # Compare utilities with and without borrowing
        if max_util1_no_borrow >= max_util1_with_borrow:
            optimal_y1 = optimal_y1_no_borrow
            optimal_b1 = 0
        else:
            optimal_y1 = optimal_y1_with_borrow
            optimal_b1 = optimal_b1_with_borrow
        
        # No preference shock case (depositor)
        # Try different deposit amounts
        d_max = m  # Can't deposit more money than you have
        d_values = np.linspace(0, d_max, 50)
        util_d = np.zeros_like(d_values)
        
        for i, d in enumerate(d_values):
            # Calculate post-DM position
            remain_m = m - d
            D = d  # Positive D indicates deposit
            
            # Calculate utility (no DM utility for depositors)
            util_d[i] = self.interpolate_CM_value(e_idx, a, remain_m, D, z_idx)
        
        # Find optimal d
        if len(util_d) > 0:
            best_idx = np.argmax(util_d)
            optimal_d = d_values[best_idx]
        else:
            optimal_d = 0
        
        # Store optimal policies
        self.policy_y[e_idx, a_idx, m_idx, z_idx, 0] = optimal_y0
        self.policy_y[e_idx, a_idx, m_idx, z_idx, 1] = optimal_y1
        self.policy_b[e_idx, a_idx, m_idx, z_idx, 0] = optimal_b0
        self.policy_b[e_idx, a_idx, m_idx, z_idx, 1] = optimal_b1
        self.policy_d[e_idx, a_idx, m_idx, z_idx] = optimal_d
    
    def update_DM_value_function(self):
        """Update the DM value function V based on optimal DM policies"""
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for a_idx in range(self.n_a):
                    for m_idx in range(self.n_m):
                        # Get state variables
                        a = self.a_grid[a_idx]
                        m = self.m_grid[m_idx]
                        
                        # Expected utility from preference shock cases
                        expected_dm_utility = 0
                        
                        # Case ω=0 (only money accepted) with probability alpha_0
                        y0 = self.policy_y[e_idx, a_idx, m_idx, z_idx, 0]
                        b0 = self.policy_b[e_idx, a_idx, m_idx, z_idx, 0]
                        
                        # Post-DM position for case ω=0
                        remain_m0 = m - y0 * self.prices[0] / self.prices[2]
                        remain_a0 = a
                        D0 = -b0  # Negative D indicates loan
                        
                        dm_utility0 = self.utility_dm(y0)
                        cm_utility0 = self.interpolate_CM_value(e_idx, remain_a0, remain_m0, D0, z_idx)
                        
                        # Case ω=1 (both accepted) with probability alpha_1
                        y1 = self.policy_y[e_idx, a_idx, m_idx, z_idx, 1]
                        b1 = self.policy_b[e_idx, a_idx, m_idx, z_idx, 1]
                        
                        # Calculate how payment is split between money and assets
                        payment1 = self.prices[0] * y1
                        money_payment1 = min(m * self.prices[2], payment1)
                        asset_payment1 = min(a, payment1 - money_payment1)
                        borrow_payment1 = payment1 - money_payment1 - asset_payment1
                        
                        remain_m1 = m - money_payment1 / self.prices[2]
                        remain_a1 = a - asset_payment1
                        D1 = -b1  # Negative D indicates loan
                        
                        dm_utility1 = self.utility_dm(y1)
                        cm_utility1 = self.interpolate_CM_value(e_idx, remain_a1, remain_m1, D1, z_idx)
                        
                        # Depositor case with probability (1-alpha)
                        d = self.policy_d[e_idx, a_idx, m_idx, z_idx]
                        remain_m_d = m - d
                        D_d = d  # Positive D indicates deposit
                        
                        cm_utility_d = self.interpolate_CM_value(e_idx, a, remain_m_d, D_d, z_idx)
                        
                        # Combine all cases to get expected value
                        expected_value = \
                            self.alpha_0 * (dm_utility0 + cm_utility0) + \
                            self.alpha_1 * (dm_utility1 + cm_utility1) + \
                            (1 - self.alpha) * cm_utility_d
                        
                        # Update DM value function
                        self.V[e_idx, a_idx, m_idx, z_idx] = expected_value
    
    def update_CM_value_function(self):
        """Update the CM value function W based on optimal CM policies and DM value function"""
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for a_idx in range(self.n_a):
                    for D_idx in range(self.n_D):
                        # Get state variables
                        a = self.a_grid[a_idx]
                        D = self.D_grid[D_idx]
                        z = self.z_grid[z_idx]
                        
                        # Solve CM optimization problem
                        optimal_c, optimal_a_next, optimal_m_next = self.solve_cm_problem(e_idx, a_idx, D_idx, z_idx)
                        
                        # Store policy functions
                        self.policy_c[e_idx, a_idx, D_idx, z_idx] = optimal_c
                        self.policy_a_next[e_idx, a_idx, D_idx, z_idx] = optimal_a_next
                        self.policy_m_next[e_idx, a_idx, D_idx, z_idx] = optimal_m_next
                        
                        # Update CM value function
                        # CM utility
                        cm_utility = self.utility(optimal_c)
                        
                        # Expected continuation value (accounting for employment transitions)
                        continuation_value = 0
                        for next_e_idx in range(self.n_e):
                            # Probability of transitioning to employment state next_e_idx
                            prob = self.emp_transition[e_idx, next_e_idx]
                            
                            # Find indices for next-period asset and money values
                            a_next_idx = self.find_nearest_index(self.a_grid, optimal_a_next)
                            m_next_idx = self.find_nearest_index(self.m_grid, optimal_m_next)
                            
                            # Add continuation value weighted by transition probability
                            continuation_value += prob * self.V[next_e_idx, a_next_idx, m_next_idx, z_idx]
                        
                        # Update W (CM value function)
                        self.W[e_idx, a_idx, D_idx, z_idx] = cm_utility + self.beta * continuation_value
    
    def solve_cm_problem(self, e_idx, a_idx, D_idx, z_idx):
        """
        Solve the centralized market problem for a given state.
        
        Parameters:
        e_idx: Index of employment state (0=unemployed, 1=employed)
        a_idx: Index of asset holdings
        D_idx: Index of deposit/loan position
        z_idx: Index of skill type
        
        Returns:
        optimal_c: Optimal consumption
        optimal_a_next: Optimal next-period asset holdings
        optimal_m_next: Optimal next-period money holdings
        """
        a = self.a_grid[a_idx]
        D = self.D_grid[D_idx]
        
        # Extract prices
        Rl = self.prices[1]
        phi_m = self.prices[2]
        i = self.prices[3]
        
        # Calculate total resources available
        resources = a
        
        # Add income based on employment and skill (using the calculated wages)
        # Note: in the wage matrix, e_idx is the employment state (0=unemployed, 1=employed)
        resources += self.wages[a_idx, e_idx, z_idx]
        
        # Add/subtract deposit/loan amount with interest
        if D > 0:  # Deposit
            resources += phi_m * (1 + i) * D
        else:  # Loan (D < 0)
            resources += phi_m * (1 + i) * D  # D is negative, so this subtracts
        
        # Grid of possible portfolio choices
        a_next_values = np.linspace(0, max(0.95 * resources, 0.1), 20)
        m_next_values = np.linspace(0, max(0.95 * resources, 0.1), 20)
        
        # Initialize for storing best choice
        max_utility = -np.inf
        best_a_next = 0
        best_m_next = 0
        best_c = 0
        
        # Search over portfolio choices
        for a_next in a_next_values:
            for m_next in m_next_values:
                # Calculate implied consumption (using budget constraint)
                portfolio_cost = a_next / Rl + m_next * phi_m
                if portfolio_cost > resources:
                    continue  # Skip infeasible choices
                
                c = resources - portfolio_cost
                
                # Calculate current utility
                current_utility = self.utility(c)
                
                # Calculate expected continuation value (across future employment states)
                continuation_value = 0
                for next_e_idx in range(self.n_e):
                    # Probability of transitioning to employment state next_e_idx
                    prob = self.emp_transition[e_idx, next_e_idx]
                    
                    # Find indices for next-period values
                    a_next_idx = self.find_nearest_index(self.a_grid, a_next)
                    m_next_idx = self.find_nearest_index(self.m_grid, m_next)
                    
                    # Add continuation value weighted by transition probability
                    continuation_value += prob * self.V[next_e_idx, a_next_idx, m_next_idx, z_idx]
                
                # Total utility including continuation value
                total_utility = current_utility + self.beta * continuation_value
                
                # Update if better
                if total_utility > max_utility:
                    max_utility = total_utility
                    best_a_next = a_next
                    best_m_next = m_next
                    best_c = c
        
        return best_c, best_a_next, best_m_next
    
    def interpolate_CM_value(self, e_idx, a, m, D, z_idx):
        """
        Interpolate the CM value function for given state.
        Used in the DM problem to evaluate post-DM positions.
        
        Parameters:
        e_idx: Employment state index
        a: Asset holdings
        m: Money holdings (not used directly in CM, but affects portfolio choice)
        D: Deposit/loan position
        z_idx: Skill type index
        
        Returns:
        Interpolated CM value
        """
        # Find indices in the grids
        a_idx_low = self.find_nearest_lower_index(self.a_grid, a)
        D_idx_low = self.find_nearest_lower_index(self.D_grid, D)
        
        # Bound indices to valid range
        a_idx_low = max(0, min(a_idx_low, self.n_a - 2))
        D_idx_low = max(0, min(D_idx_low, self.n_D - 2))
        
        a_idx_high = a_idx_low + 1
        D_idx_high = D_idx_low + 1
        
        # Calculate weights for interpolation
        a_low = self.a_grid[a_idx_low]
        a_high = self.a_grid[a_idx_high]
        D_low = self.D_grid[D_idx_low]
        D_high = self.D_grid[D_idx_high]
        
        wa = (a - a_low) / (a_high - a_low) if a_high > a_low else 0
        wD = (D - D_low) / (D_high - D_low) if D_high > D_low else 0
        
        # Bilinear interpolation
        value = (1-wa)*(1-wD)*self.W[e_idx, a_idx_low, D_idx_low, z_idx] + \
                wa*(1-wD)*self.W[e_idx, a_idx_high, D_idx_low, z_idx] + \
                (1-wa)*wD*self.W[e_idx, a_idx_low, D_idx_high, z_idx] + \
                wa*wD*self.W[e_idx, a_idx_high, D_idx_high, z_idx]
        
        return value
    
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
    

    def visualize_convergence(self, iter_count):
        """Generate visualizations to track convergence progress"""
        # Only show for selected states to avoid clutter
        e_idx = 1  # Employed
        z_idx = 1  # Medium skill
        
        plt.figure(figsize=(15, 10))
        
        # Plot W value function for different asset levels
        plt.subplot(2, 2, 1)
        D_idx = self.n_D // 2  # Mid-point of deposit grid
        plt.plot(self.a_grid, self.W[0, :, D_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.W[1, :, D_idx, z_idx], 'r-', label='Employed')
        plt.title(f'CM Value Function (Iteration {iter_count})')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot V value function
        plt.subplot(2, 2, 2)
        m_idx = self.n_m // 2  # Mid-point of money grid
        plt.plot(self.a_grid, self.V[0, :, m_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.V[1, :, m_idx, z_idx], 'r-', label='Employed')
        plt.title('DM Value Function')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot consumption policy
        plt.subplot(2, 2, 3)
        plt.plot(self.a_grid, self.policy_c[0, :, D_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.policy_c[1, :, D_idx, z_idx], 'r-', label='Employed')
        plt.title('Consumption Policy')
        plt.xlabel('Assets')
        plt.ylabel('Consumption')
        plt.legend()
        
        # Plot next period asset policy
        plt.subplot(2, 2, 4)
        plt.plot(self.a_grid, self.policy_a_next[0, :, D_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.policy_a_next[1, :, D_idx, z_idx], 'r-', label='Employed')
        plt.plot(self.a_grid, self.a_grid, 'k--', label='45-degree line')
        plt.title('Next Period Assets Policy')
        plt.xlabel('Current Assets')
        plt.ylabel('Next Period Assets')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'convergence_iter_{iter_count}.png')
        plt.close()

    def visualize_solution(self):
        """Generate comprehensive visualizations to verify economic sensibility of the solution"""
        # Create a folder for visualizations if it doesn't exist
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Value Functions
        plt.figure(figsize=(15, 12))
        
        # Plot CM value function (W) by employment and skill
        for z_idx in range(self.n_z):
            plt.subplot(3, 2, z_idx*2 + 1)
            D_idx = self.n_D // 2  # Middle of deposit grid
            plt.plot(self.a_grid, self.W[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.W[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.title(f'CM Value Function (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Value')
            plt.legend()
            
            # Plot DM value function (V) by employment and skill
            plt.subplot(3, 2, z_idx*2 + 2)
            m_idx = self.n_m // 2  # Middle of money grid
            plt.plot(self.a_grid, self.V[0, :, m_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.V[1, :, m_idx, z_idx], 'r-', label='Employed')
            plt.title(f'DM Value Function (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/value_functions.png')
        plt.close()
        
        # 2. CM Policy Functions
        plt.figure(figsize=(15, 12))
        
        # Plot consumption policy
        for z_idx in range(self.n_z):
            plt.subplot(3, 2, z_idx*2 + 1)
            D_idx = self.n_D // 2  # Middle of deposit grid
            plt.plot(self.a_grid, self.policy_c[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.policy_c[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.title(f'Consumption Policy (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Consumption')
            plt.legend()
            
            # Plot asset policy with 45-degree line
            plt.subplot(3, 2, z_idx*2 + 2)
            plt.plot(self.a_grid, self.policy_a_next[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.policy_a_next[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.plot(self.a_grid, self.a_grid, 'k--', label='45-degree line')
            plt.title(f'Next Period Assets (Skill Level {z_idx})')
            plt.xlabel('Current Assets')
            plt.ylabel('Next Period Assets')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/cm_policies.png')
        plt.close()
        
        # 3. Money Policies
        plt.figure(figsize=(15, 12))
        
        # Plot money holding policy
        for z_idx in range(self.n_z):
            plt.subplot(3, 2, z_idx*2 + 1)
            D_idx = self.n_D // 2  # Middle of deposit grid
            plt.plot(self.a_grid, self.policy_m_next[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.policy_m_next[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.title(f'Money Holding Policy (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Next Period Money')
            plt.legend()
            
            # Plot money-to-asset ratio
            plt.subplot(3, 2, z_idx*2 + 2)
            money_ratio_unemployed = self.policy_m_next[0, :, D_idx, z_idx] / (self.policy_a_next[0, :, D_idx, z_idx] + 1e-8)
            money_ratio_employed = self.policy_m_next[1, :, D_idx, z_idx] / (self.policy_a_next[1, :, D_idx, z_idx] + 1e-8)
            plt.plot(self.a_grid, money_ratio_unemployed, 'b-', label='Unemployed')
            plt.plot(self.a_grid, money_ratio_employed, 'r-', label='Employed')
            plt.title(f'Money-to-Asset Ratio (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Money/Assets Ratio')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/money_policies.png')
        plt.close()
        
        # 4. DM Policy Functions by Skill Type
        for z_idx in range(self.n_z):
            plt.figure(figsize=(15, 12))
            
            # Early consumption when only money accepted (ω=0)
            plt.subplot(3, 2, 1)
            a_idx = self.n_a // 2  # Middle of asset grid
            plt.plot(self.m_grid, self.policy_y[0, a_idx, :, z_idx, 0], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_y[1, a_idx, :, z_idx, 0], 'r-', label='Employed')
            plt.title(f'Early Consumption (ω=0, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Consumption in DM')
            plt.legend()
            
            # Early consumption when both assets accepted (ω=1)
            plt.subplot(3, 2, 2)
            plt.plot(self.m_grid, self.policy_y[0, a_idx, :, z_idx, 1], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_y[1, a_idx, :, z_idx, 1], 'r-', label='Employed')
            plt.title(f'Early Consumption (ω=1, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Consumption in DM')
            plt.legend()
            
            # Borrowing when only money accepted (ω=0)
            plt.subplot(3, 2, 3)
            plt.plot(self.m_grid, self.policy_b[0, a_idx, :, z_idx, 0], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_b[1, a_idx, :, z_idx, 0], 'r-', label='Employed')
            plt.title(f'Borrowing (ω=0, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Borrowing Amount')
            plt.legend()
            
            # Borrowing when both assets accepted (ω=1)
            plt.subplot(3, 2, 4)
            plt.plot(self.m_grid, self.policy_b[0, a_idx, :, z_idx, 1], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_b[1, a_idx, :, z_idx, 1], 'r-', label='Employed')
            plt.title(f'Borrowing (ω=1, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Borrowing Amount')
            plt.legend()
            
            # Deposit policy
            plt.subplot(3, 2, 5)
            plt.plot(self.m_grid, self.policy_d[0, a_idx, :, z_idx], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_d[1, a_idx, :, z_idx], 'r-', label='Employed')
            plt.title(f'Deposit Policy (Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Deposit Amount')
            plt.legend()
            
            # Deposit-to-money ratio
            plt.subplot(3, 2, 6)
            deposit_ratio_unemployed = self.policy_d[0, a_idx, :, z_idx] / (self.m_grid + 1e-8)
            deposit_ratio_employed = self.policy_d[1, a_idx, :, z_idx] / (self.m_grid + 1e-8)
            plt.plot(self.m_grid, deposit_ratio_unemployed, 'b-', label='Unemployed')
            plt.plot(self.m_grid, deposit_ratio_employed, 'r-', label='Employed')
            plt.title(f'Deposit-to-Money Ratio (Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Deposit/Money Ratio')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'visualizations/dm_policies_skill_{z_idx}.png')
            plt.close()
        
        # 5. Heat Maps for Policy Functions
        # This helps visualize how policies change across both asset and money dimensions
        for z_idx in range(self.n_z):
            for e_idx in range(self.n_e):
                employment_status = "Employed" if e_idx == 1 else "Unemployed"
                
                # Early consumption heat maps
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # For ω=0
                im0 = axes[0].imshow(
                    self.policy_y[e_idx, :, :, z_idx, 0],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axes[0].set_title(f'Early Consumption (ω=0, {employment_status}, Skill {z_idx})')
                axes[0].set_xlabel('Money Holdings')
                axes[0].set_ylabel('Asset Holdings')
                plt.colorbar(im0, ax=axes[0])
                
                # For ω=1
                im1 = axes[1].imshow(
                    self.policy_y[e_idx, :, :, z_idx, 1],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axes[1].set_title(f'Early Consumption (ω=1, {employment_status}, Skill {z_idx})')
                axes[1].set_xlabel('Money Holdings')
                axes[1].set_ylabel('Asset Holdings')
                plt.colorbar(im1, ax=axes[1])
                
                plt.tight_layout()
                plt.savefig(f'visualizations/heatmap_consumption_{employment_status}_skill_{z_idx}.png')
                plt.close()
                
                # Borrowing and deposit heat maps
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Borrowing for ω=0
                im0 = axes[0].imshow(
                    self.policy_b[e_idx, :, :, z_idx, 0],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='coolwarm'
                )
                axes[0].set_title(f'Borrowing (ω=0, {employment_status}, Skill {z_idx})')
                axes[0].set_xlabel('Money Holdings')
                axes[0].set_ylabel('Asset Holdings')
                plt.colorbar(im0, ax=axes[0])
                
                # Borrowing for ω=1
                im1 = axes[1].imshow(
                    self.policy_b[e_idx, :, :, z_idx, 1],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='coolwarm'
                )
                axes[1].set_title(f'Borrowing (ω=1, {employment_status}, Skill {z_idx})')
                axes[1].set_xlabel('Money Holdings')
                axes[1].set_ylabel('Asset Holdings')
                plt.colorbar(im1, ax=axes[1])
                
                # Deposits
                im2 = axes[2].imshow(
                    self.policy_d[e_idx, :, :, z_idx],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axes[2].set_title(f'Deposits ({employment_status}, Skill {z_idx})')
                axes[2].set_xlabel('Money Holdings')
                axes[2].set_ylabel('Asset Holdings')
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(f'visualizations/heatmap_borrowing_deposit_{employment_status}_skill_{z_idx}.png')
                plt.close()


if __name__ == "__main__":
    params = {
        'beta': 0.96,
        'alpha': 0.075,
        'alpha_1': 0.06,
        'gamma': 1.5,
        'Psi': 2.2,
        'psi': 0.28,
        'zeta': 0.75,
        'nu': 1.6,
        'mu': 0.7,
        'delta': 0.035,
        'kappa': 7.29,
        'repl_rate': 0.4,
        
        # Grid specifications
        'n_a': 100,
        'n_m': 50,
        'n_D': 40,
        'a_min': 0.0,
        'a_max': 20.0,
        'm_min': 0.0,
        'm_max': 10.0,
        'D_min': -5.0,
        'D_max': 5.0,
        
        # Initial prices
        'py': 1.2,      # Price of early consumption
        'Rl': 1.03,     # Illiquid asset return
        'phi_m': 1.0,   # Money price
        'i': 0.02,      # Nominal interest rate
        
        # Additional needed parameters
        'w1': 1.0,      # Wage for employed workers (base wage)
        'w0': 0.4,      # Unemployment benefits (as percentage of w1)
        
        # Convergence parameters
        'max_iter': 500,
        'tol': 1e-5
    }
    
    solver = LagosWrightAiyagariSolver(params)
    
    # Set initial market tightness (will be used to calculate job-finding probability)
    solver.market_tightness = 0.7
    
    # Solve the model for given prices
    convergence_path = solver.solve_model()
    
    # Plot convergence path
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_path)
    plt.yscale('log')
    plt.title('Value Function Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Max Absolute Difference')
    plt.grid(True)
    plt.savefig('convergence_path.png')
    plt.show()
    
    print("Model solution complete. Visualizations saved to 'visualizations' folder.")