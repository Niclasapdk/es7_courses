import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Load the MATLAB file
data = sio.loadmat('RawMRI.mat')

# Extract the k-space data
fbrain1 = data['fbrain1']
fbrain2 = data['fbrain2']

brain1 = np.fft.ifft2(fbrain1)
brain2 = np.fft.ifft2(fbrain2)

rows, cols = fbrain1.shape
center_y = rows // 2
center_x = cols // 2
y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
r1 = 40

mask1 = np.ones_like(fbrain1)
masksquare = np.ones_like(fbrain1)
masksquare[center_y - r1: center_y + r1, center_x - r1: center_x + r1] = 0
mask1[distance < r1] = 0
masked_kspace1 = fbrain1 * mask1
masksquare_kspace1 = fbrain1 * masksquare
# Function to plot k-space data
def plot_kspace(kspace_data, title):
    fig = plt.figure(figsize=(14, 12))
    
    # Create subplot 1
    plt.subplot(2, 2, 1)
    
    # Create subplot 1
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(np.real(kspace_data), cmap='gray', aspect='auto')
    plt.title(f'{title} - Real Part (Linear Scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.colorbar(im1)
    
    # Create subplot 2
    plt.subplot(2, 2, 2)
    real_part = np.real(kspace_data)
    log_real = np.sign(real_part) * np.log10(np.abs(real_part) + 1)
    im2 = plt.imshow(log_real, cmap='gray', aspect='auto')
    plt.title(f'{title} - Real Part (Log Scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.colorbar(im2)
    
    # Create subplot 3
    plt.subplot(2, 2, 3)
    im3 = plt.imshow(np.imag(kspace_data), cmap='gray', aspect='auto')
    plt.title(f'{title} - Imaginary Part (Linear Scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.colorbar(im3)
    
    # Create subplot 4
    plt.subplot(2, 2, 4)
    imag_part = np.imag(kspace_data)
    log_imag = np.sign(imag_part) * np.log10(np.abs(imag_part) + 1)
    im4 = plt.imshow(log_imag, cmap='gray', aspect='auto')
    plt.title(f'{title} - Imaginary Part (Log Scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.colorbar(im4)
    
    plt.tight_layout()
    return fig

# Plot both datasets
fig1 = plot_kspace(masksquare_kspace1, 'fbrain1 (288×170)')

# Optional: Plot magnitude in log scale
def plot_magnitude(kspace_data, title):
    fig = plt.figure(figsize=(14, 6))
    
    # Magnitude - linear scale
    plt.subplot(1, 2, 1)
    magnitude = np.abs(kspace_data)
    im1 = plt.imshow(magnitude, cmap='viridis', aspect='auto')
    plt.title(f'{title} - Magnitude (Linear Scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.colorbar(im1)
    
    # Magnitude - log scale
    plt.subplot(1, 2, 2)
    log_magnitude = np.log10(magnitude + 1)
    im2 = plt.imshow(log_magnitude, cmap='viridis', aspect='auto')
    plt.title(f'{title} - Magnitude (Log Scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.colorbar(im2)
    
    plt.tight_layout()
    return fig

fig3 = plot_magnitude(brain1, 'brain1 (288×170)')
fig4 = plot_magnitude(brain2, 'brain2 (288×288)')
fig5 = plot_magnitude(np.fft.ifft2(masksquare_kspace1), 'masksqaurebrain')
plt.show()

