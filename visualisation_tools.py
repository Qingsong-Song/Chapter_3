# For comparison:
psi1 = 0.28
psi2 = 0.1
Psi = 2.2
y = 2.5

u1 = Psi * ((y**(1 - psi1) - 1) / (1 - psi1))
u2 = Psi * ((y**(1 - psi2) - 1) / (1 - psi2))

print(u1, u2)  # u2 > u1
