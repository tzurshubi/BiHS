def count_simple_paths(n, m):
    dp = [[0] * m for _ in range(n)]
    dp[0][0] = 1  # Starting point

    for i in range(n):
        for j in range(m):
            if i > 0:
                dp[i][j] += dp[i-1][j]
            if j > 0:
                dp[i][j] += dp[i][j-1]

    return dp[-1][-1]

# Example usage for a 3x3 grid
n, m = 3, 3
print(count_simple_paths(n, m))  # Output: 6