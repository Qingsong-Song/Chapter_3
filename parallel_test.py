from multiprocessing import Pool

def solve_one_mf_pair(m_idx, f_idx, a_m, a_f):
    # Replace with real logic later
    result = (a_m + a_f)**2
    return (m_idx, f_idx, result)

if __name__ == "__main__":
    m_grid = [1.0, 2.0, 3.0]
    f_grid = [0.5, 1.5]

    args = [
        (m_idx, f_idx, m_grid[m_idx], f_grid[f_idx])
        for m_idx in range(len(m_grid))
        for f_idx in range(len(f_grid))
    ]

    with Pool(processes=4) as pool:
        results = pool.starmap(solve_one_mf_pair, args)

    # Unpack results
    for m_idx, f_idx, value in results:
        print(f"(m={m_idx}, f={f_idx}) => {value}")
