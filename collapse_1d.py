---

### **КОД `collapse_1d.py

```python
"""
Gravitational Collapse of Universal Wavefunction (1D)
Denis Bykov, November 2025
arXiv:2511.XXXXX (pending)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.integrate import simpson

# === PARAMETERS ===
L = 100.0          # Grid size
N = 1024           # Grid points
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)

# Physical constants (Planck units)
hbar = 1.0
m = 1.0
G = 1.0

# Initial state: superposition of two Gaussians
sigma = 2.0
d = 5.0
psi0_left = np.exp(-(x + d)**2 / (2*sigma**2))
psi0_right = np.exp(-(x - d)**2 / (2*sigma**2))
psi = (psi0_left + psi0_right)
psi /= np.sqrt(simpson(np.abs(psi)**2, x))  # Normalize

# FFT setup
k = np.fft.fftfreq(N, dx/(2*np.pi))
exp_kin = np.exp(-1j * hbar * (k**2) / (2*m) * 0.1)  # dt = 0.1

# === GRAVITATIONAL POTENTIAL ===
def compute_potential(psi):
    rho = np.abs(psi)**2
    rho_k = fft(rho)
    phi_k = -4*np.pi*G * rho_k / (k**2 + 1e-10)
    phi = np.real(ifft(phi_k))
    return phi

# === TIME EVOLUTION ===
dt = 0.1
steps = 200
density_evolution = []

psi_t = psi.copy()
for step in range(steps):
    # Half-step kinetic
    psi_k = fft(psi_t)
    psi_k *= exp_kin
    psi_t = ifft(psi_k)
    
    # Full potential step
    phi = compute_potential(psi_t)
    psi_t *= np.exp(-1j * m * phi * dt / hbar)
    
    # Half-step kinetic
    psi_k = fft(psi_t)
    psi_k *= exp_kin
    psi_t = ifft(psi_k)
    
    # Save density
    if step % 20 == 0:
        density_evolution.append(np.abs(psi_t)**2)

# === PLOT ===
plt.figure(figsize=(10, 6))
for i, dens in enumerate(density_evolution):
    plt.plot(x, dens + i*0.5, label=f't={i*2.0:.1f}')
plt.xlabel('x (Planck length)')
plt.ylabel('|ψ|² (arbitrary offset)')
plt.title('1D Gravitational Collapse: From Superposition to Localization')
plt.legend()
plt.tight_layout()
plt.savefig('collapse_plot.png', dpi=300)
plt.show()

print("Simulation complete. Plot saved as 'collapse_plot.png'")
