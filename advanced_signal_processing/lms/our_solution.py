import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-M", help="Cheat by setting M directly", type=int)
parser.add_argument("-u", help="Cheat by setting mu directly", type=float)
parser.add_argument("-N", help="Number of samples", type=int)
parser.add_argument("--snilter", help="Run with snilter", action="store_true")
args = parser.parse_args()

if args.snilter:
    # Snilter
    fs = 96000
    noverb = wavfile.read("../../../esd6_project/system_design/soundclips/fcpremix.wav")
    verb = wavfile.read("../../../esd6_project/system_design/reverb/reverb_with_inputdelayline.wav")
    if args.N == None:
        N = fs
    else:
        N = args.N
    primary = noverb[1][:N]
    local = np.copy(primary)
    remote = verb[1][:N]
else:
    # Actual exercise
    primary = np.loadtxt("signal.asc")
    remote = np.loadtxt("remota.asc")
    local = np.loadtxt("local.asc")
    fs = 8000

# Actual computations
interference = primary-local
T = np.linspace(0, len(primary), len(primary))/fs

# 2
def snr_array(x, v):
    return 10 * np.log10(np.sum(x**2) / np.sum(v**2))

snr_priori = snr_array(local, interference)
print(f"2) SNR of priori: {snr_priori}")

# 3 Echo canceler

def NLMS(U, d, M, mu, delta=0.0000001):
    assert len(U) == len(d)
    # Weights
    w = np.zeros(M)
    # Number of samples
    N = len(U)
    # u is Mx1 tap-input vector
    u = np.zeros(M)
    # output array
    d_hat_seq = np.zeros(N)
    e_seq = np.zeros(N)

    for n in range(N):
        # Update taps
        # u[i] = u(n-i)
        u[1:] = u[:-1]
        u[0] = U[n]
        # Calculate FIR
        d_hat = np.sum(w*u)
        d_hat_seq[n] = d_hat # for tracing and plotting
        # Update weights
        e = d[n] - d_hat
        e_seq[n] = e # for tracing and plotting
        factor = mu/(np.linalg.norm(u)**2+delta)
        w = w + factor * e * u

    return d_hat_seq, e_seq

# from scipy.io import wavfile
# fuck = wavfile.read("../../../esd6_project/system_design/soundclips/pen15.wav")
# sd.play(fuck[1], fuck[0])

# Construct mu sequence
if args.u == None:
    mu_seq = []
    mu = 0.0001
    while mu <= 0.0512:
        mu_seq.append(mu)
        mu *= 2
else:
    mu_seq = [args.u]

# Construct M sequence
if args.M == None:
    M_seq = [i for i in range(1,11)]
else:
    M_seq = [args.M]

# Search for the best params
mu_hat = None
M_hat = None
SNR_best = -999
#for mu in tqdm(mu_seq):
for mu in mu_seq:
    #print(f"{mu = }")
    for M in tqdm(M_seq):
        #print(f"{M = }")
        d_hat, e = NLMS(remote,primary,M,mu)
        snr = snr_array(local, e-local)
        if snr > SNR_best:
            SNR_best = snr
            mu_hat = mu
            M_hat = M

print(f"Best SNR: {SNR_best:.3f}, mu_hat: {mu_hat:.5f}, M_hat: {M_hat}")

d_hat, e = NLMS(remote,primary, M_hat, mu_hat)

fig, ax = plt.subplots(3, 1)
ax[0].plot(T, primary)
ax[0].set_title("primary")
ax[1].plot(T, remote)
ax[1].set_title("remote")
ax[2].plot(T, d_hat)
ax[2].set_title("d_hat")
plt.show()

input("play?")
sd.play(primary, fs)
input("next?")
sd.play(remote, fs)
input("next?")
sd.play(d_hat, fs)
input("next?")
