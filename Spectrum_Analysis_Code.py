import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import re
import os
from scipy.optimize import curve_fit
from collections import defaultdict

rcParams['font.family'] = 'sans-serif'
plt.rc('font', family='Arial', size=10)
plt.rc('axes', linewidth=1.0)
rcParams['lines.linewidth'] = 1.5
rcParams['figure.dpi'] = 300
rcParams["mathtext.fontset"] = 'dejavusans'

THESIS_COLOURS = {
    'purple': '#4e1fb4',      # rgb(78,31,180)
    'light_blue': '#8bcaf7',  # rgb(139,202,247)
    'coral': '#eb737f',       # rgb(235,115,127)
    'gold': '#e4cc7a',        # rgb(228,204,122)
    'orange': '#ed975a',       # rgb(237,151,90)
    'teal': '#54a391',
    'dark_blue': '#446982'   # rgb(0,63,92)
}

class ThesisSpectroscopyAnalyser:
    def __init__(self, roi, roi_sum=0, normalisation=1, save_data=0, save_plots=0, fit_fano=True):
        """
        Initialise the analyser for thesis figure generation.

        Args:
            roi: Region of interest [start, end]
            roi_sum: If 1, sum the ROI; if 0, average the ROI
            normalisation: If 1, normalise intensity to [0,1]
            save_data: If 1, save extracted data to CSV
            save_plots: If 1, save plots as PNG files
            fit_fano: If True, fit Fano resonance in addition to Gaussian
        """
        self.roi = roi
        self.roi_sum = roi_sum
        self.normalisation = normalisation
        self.save_data = save_data
        self.save_plots = save_plots
        self.fit_fano = fit_fano
        self.results = []
        
    def read_file_header(self, filepath):
        header_dict = {}
        with open(filepath, 'r', encoding='ISO-8859-1') as txtfile:
            lines = txtfile.readlines()
        
        pattern = re.compile(r'^(.*?):\s*(.*)$')
        for line in lines:
            match = pattern.match(line)
            if match:
                key, value = match.groups()
                header_dict[key.strip()] = value.strip()
        
        return header_dict
    
    def load_spectral_data(self, filepath):

        header_dict = self.read_file_header(filepath)
        exposure_time = float(header_dict['Exposure Time'].split()[0])
        
        data = np.loadtxt(filepath, skiprows=19, delimiter="\t", encoding='ISO-8859-1')
        
        n_strip, n_pixel = data.shape[1], data.shape[0]
        wavelengths = data[:, 0]
        intensities = data[:, 1:n_strip-1].transpose()
        
        return wavelengths, intensities, exposure_time
    
    def extract_spectrum(self, intensities):
        #Extract range of interest
        if self.roi_sum == 1:
            spectrum = np.sum(intensities[self.roi[0]:self.roi[1], :], axis=0)
        else:
            spectrum = np.average(intensities[self.roi[0]:self.roi[1], :], axis=0) / (self.roi[1] - self.roi[0] + 1)
        
        return spectrum
    
    def normalise_spectrum(self, spectrum):
        #normalising spectrum 
        if self.normalisation == 1:
            min_val = np.min(spectrum)
            max_val = np.max(spectrum)
            if max_val > min_val:
                return (spectrum - min_val) / (max_val - min_val)
            else:
                return spectrum
        else:
            return spectrum
    
    def normalise_intensities(self, intensities):
        #Normalising intensity matrix
        if self.normalisation == 1:
            min_val = np.min(intensities)
            max_val = np.max(intensities)
            if max_val > min_val:
                return (intensities - min_val) / (max_val - min_val)
            else:
                return intensities
        else:
            return intensities
    
    def gaussian(self, x, a, x0, sigma):
        #Defining the gaussian function for fitting
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    def fano_lineshape(self, wavelength, amplitude, lambda_0, gamma, q, offset=0):
        #Defining fano function for fitting
        # Convert to energy-like parameter for better fitting
        epsilon = 2 * (wavelength - lambda_0) / gamma
        return offset + amplitude * (q + epsilon)**2 / (1 + epsilon**2)
    
    def calculate_fwhm_from_gaussian(self, sigma):
        # Calculating FWHM from sigma fitting from Gaussian function
        return 2 * np.sqrt(2 * np.log(2)) * sigma
    
    def calculate_fwhm_numerical(self, wavelengths, spectrum):
        # Numerically calculating FWHM
        max_intensity = np.max(spectrum)
        half_max = max_intensity / 2
        
        indices = np.where(spectrum >= half_max)[0]
        
        if len(indices) == 0:
            return None
        
        left_idx = indices[0]
        right_idx = indices[-1]
        
        if left_idx > 0:
            y1, y2 = spectrum[left_idx-1], spectrum[left_idx]
            x1, x2 = wavelengths[left_idx-1], wavelengths[left_idx]
            left_wavelength = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        else:
            left_wavelength = wavelengths[left_idx]
        
        if right_idx < len(spectrum) - 1:
            y1, y2 = spectrum[right_idx], spectrum[right_idx+1]
            x1, x2 = wavelengths[right_idx], wavelengths[right_idx+1]
            right_wavelength = x2 + (half_max - y2) * (x1 - x2) / (y1 - y2)
        else:
            right_wavelength = wavelengths[right_idx]
        
        fwhm = right_wavelength - left_wavelength
        return fwhm if fwhm > 0 else None
    
    def find_fano_peak_wavelength(self, wavelengths, y_fano):
        #Finding fano peak (numerical)
        peak_idx = np.argmax(y_fano)
        return wavelengths[peak_idx]
    
    def calculate_fano_fwhm_analytical(self, gamma, q):
        # For asymmetric Fano resonances, FWHM depends on q
        # This is an approximation - exact formula is more complex
        return gamma * np.sqrt(1 + 1/q**2) if q != 0 else gamma
    
    def fit_spectrum(self, wavelengths, spectrum, fit_range=[1240, 1290]):
        #Fitting spectrum with gaussian and fano resonances
        # Find indices corresponding to the fit range
        fit_mask = (wavelengths >= fit_range[0]) & (wavelengths <= fit_range[1])
        x_fit = wavelengths[fit_mask]
        y_fit = spectrum[fit_mask]
        
        if len(x_fit) == 0:
            print(f"Warning: No data points in fit range {fit_range}")
            lambda_peak_idx = np.argmax(spectrum)
            return wavelengths, spectrum, None, None, wavelengths[lambda_peak_idx], None, None, None
        
        # Initialise return values
        y_gauss = None
        y_fano = None
        fwhm_gaussian = None
        fwhm_numerical = None
        fwhm_fano = None
        
        # Initial guess for parameters
        peak_wavelength_guess = x_fit[np.argmax(y_fit)]
        amplitude_guess = np.max(y_fit)
        sigma_guess = 5  # Initial width guess in nm
        
        # Gaussian fit
        p0_gauss = [amplitude_guess, peak_wavelength_guess, sigma_guess]
        
        try:
            # Fit Gaussian
            popt_gauss, _ = curve_fit(self.gaussian, x_fit, y_fit, p0=p0_gauss)
            y_gauss = self.gaussian(x_fit, *popt_gauss)
            fitted_peak_wavelength = popt_gauss[1]
            
            # Calculate FWHM from Gaussian fit
            sigma = abs(popt_gauss[2])
            fwhm_gaussian = self.calculate_fwhm_from_gaussian(sigma)
            
            print(f"Gaussian fit successful in range {fit_range[0]}-{fit_range[1]} nm")
            print(f"  Peak: {fitted_peak_wavelength:.1f} nm")
            print(f"  FWHM: {fwhm_gaussian:.1f} nm")
        #Define failiure to exclude silent failiure    
        except Exception as e:
            print(f"Gaussian fit failed: {e}")
            y_gauss = None
            fitted_peak_wavelength = x_fit[np.argmax(y_fit)]
        
        # Fano fit (if enabled)
        if self.fit_fano:
            # Initial parameters for Fano fit
            offset_guess = np.min(y_fit)
            
            p0_fano = [
                amplitude_guess - offset_guess,  # amplitude
                peak_wavelength_guess,           # lambda_0
                10.0,                            # gamma (linewidth)
                1.0,                             # q (Fano parameter)
                offset_guess                     # offset
            ]
            
            # Set bounds to ensure physical parameters
            bounds = (
                [0, x_fit[0], 0.1, -10, 0],           # lower bounds
                [np.inf, x_fit[-1], 100, 10, 1]       # upper bounds
            )
            
            try:
                popt_fano, pcov_fano = curve_fit(
                    self.fano_lineshape, 
                    x_fit, 
                    y_fit, 
                    p0=p0_fano,
                    bounds=bounds,
                    maxfev=10000
                )
                y_fano = self.fano_lineshape(x_fit, *popt_fano)
                
                # Extract Fano parameters
                amplitude = popt_fano[0]
                lambda_0 = popt_fano[1]  # Resonance wavelength (not necessarily the peak!)
                gamma = popt_fano[2]     # Linewidth parameter
                q = popt_fano[3]          # Fano asymmetry parameter
                offset = popt_fano[4]
                
                # Find the actual peak of the Fano curve
                fano_peak_wavelength = self.find_fano_peak_wavelength(x_fit, y_fano)
                
                # Calculate FWHM: use numerical method on the fitted curve
                fwhm_fano = self.calculate_fwhm_numerical(x_fit, y_fano)
            
                print(f"  Resonance λ₀: {lambda_0:.1f} nm")
                print(f"  Peak position: {fano_peak_wavelength:.1f} nm")
                print(f"  Fano parameter q: {q:.2f}")
                print(f"  Linewidth γ: {gamma:.1f} nm")
                print(f"  FWHM (Fano): {fwhm_fano:.1f} nm" if fwhm_fano else "  FWHM: N/A")
                
                # Calculate R-squared for fit quality
                residuals = y_fit - y_fano
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                r_squared = 1 - (ss_res / ss_tot)
                print(f"  R² = {r_squared:.4f}")
                
            except Exception as e:
                print(f"Fano fit failed: {e}")
                y_fano = None
                fwhm_fano = None
                fano_peak_wavelength = None
        
        # Calculate numerical FWHM from raw data
        if y_gauss is not None:
            fwhm_numerical = self.calculate_fwhm_numerical(x_fit, y_fit)
        else:
            fwhm_numerical = self.calculate_fwhm_numerical(wavelengths, spectrum)
        
        # Use Fano peak if available, otherwise use Gaussian peak
        if self.fit_fano and y_fano is not None and 'fano_peak_wavelength' in locals():
            final_peak_wavelength = fano_peak_wavelength
        else:
            final_peak_wavelength = fitted_peak_wavelength
        
        return x_fit, y_fit, y_gauss, y_fano, final_peak_wavelength, fwhm_gaussian, fwhm_numerical, fwhm_fano
    
    def parse_filename(self, filename):
        #Parse filename to extract metadata
        name = os.path.splitext(filename)[0]
        
        metalens = re.search(r'metalens(\d+)', name)
        zoom = re.search(r'(\d+)x', name)
        centre = re.search(r'_c_(\d+)', name)
        exposure = re.search(r'_exp(\d+)', name)
        
        return {
            'metalens': metalens.group(1) if metalens else '',
            'zoom': zoom.group(1) if zoom else '',
            'centre': centre.group(1) if centre else '',
            'exposure': exposure.group(1) if exposure else '',
            'filename': filename
        }
    
    def process_files(self, file_list):
        target_file = None
        
        for filepath in file_list:
            filename = os.path.basename(filepath)
            metadata = self.parse_filename(filename)
            
            if metadata['metalens'] == '4' and metadata['zoom'] == '4':
                target_file = {'filepath': filepath, 'metadata': metadata}
                break
        
        if target_file is None:
            print("ERROR: No file found for Metalens 4, Zoom 4x")
            return
        
        self.process_target_file(target_file)
    
    def process_target_file(self, file_info):
        filepath = file_info['filepath']
        metadata = file_info['metadata']
        
        print(f'Processing Metalens 4, Zoom 4x: {os.path.basename(filepath)}')
        
        # Load and process data
        wavelengths, intensities, exposure_time = self.load_spectral_data(filepath)
        spectrum = self.extract_spectrum(intensities)
        spectrum_normalised = self.normalise_spectrum(spectrum)
        intensities_display = self.normalise_intensities(intensities)
        
        # Fit spectrum in specified range (now returns 8 values)
        x_fit, y_fit, y_gauss, y_fano, fitted_peak, fwhm_gaussian, fwhm_numerical, fwhm_fano = \
            self.fit_spectrum(wavelengths, spectrum_normalised, fit_range=[1240, 1290])
        
        # Create figure
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Plot 1: Pixel-strip intensity map
        n_strip = intensities.shape[0]
        X, Y = np.meshgrid(wavelengths, range(n_strip))
        
        img = ax1.pcolormesh(X, Y, intensities_display, cmap='plasma', 
                            shading='auto', rasterized=True)
        ax1.set_aspect('auto')
        ax1.set_xlabel('Wavelength (nm)', fontsize=11, fontweight='medium')
        ax1.set_ylabel('Pixel Strip', fontsize=11, fontweight='medium')
        ax1.set_title('(a) Spatial-Spectral Intensity Distribution', 
                     fontsize=12, fontweight='bold', pad=10)
        

        cbar = plt.colorbar(img, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Normalised Intensity', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        ax1.axhline(y=self.roi[0], color='white', 
                   linestyle='--', linewidth=1.5, alpha=0.9, label='ROI')
        ax1.axhline(y=self.roi[1], color='white', 
                   linestyle='--', linewidth=1.5, alpha=0.9)
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.3, labelcolor='white')
        
        # Plot 2: Spectrum with Fano fit only
        ax2.plot(wavelengths, spectrum_normalised, linewidth=2, 
                color=THESIS_COLOURS['purple'], label='Measured Spectrum', alpha=0.8)
        
        # Plot Fano fit if available
        if y_fano is not None:
            ax2.plot(x_fit, y_fano, linewidth=2.5, 
                    color=THESIS_COLOURS['teal'], label='Fano Fit', 
                    linestyle='-', alpha=0.9)
            
            # Peak wavelength line from Fano fit
            ax2.axvline(fitted_peak, color=THESIS_COLOURS['coral'], 
                       linestyle='--', linewidth=1.5, alpha=0.8, label='Peak')
            
            # Add text annotation with Fano results
            text_str = f'Peak: {fitted_peak:.0f} nm'
            if fwhm_fano is not None:
                text_str += f'\nFWHM: {fwhm_fano:.0f} nm'
            
            ax2.text(0.98, 0.95, text_str, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', 
                             edgecolor=THESIS_COLOURS['teal'], alpha=0.9, linewidth=1.5))
        else:
            # Fallback to Gaussian if Fano failed
            if y_gauss is not None:
                ax2.plot(x_fit, y_gauss, linewidth=2.5, 
                        color=THESIS_COLOURS['orange'], label='Gaussian Fit', 
                        linestyle='--', alpha=0.9)
                
                ax2.axvline(fitted_peak, color=THESIS_COLOURS['coral'], 
                           linestyle='--', linewidth=1.5, alpha=0.8, label='Peak')
                
                text_str = f'Peak: {fitted_peak:.0f} nm'
                if fwhm_gaussian is not None:
                    text_str += f'\nFWHM: {fwhm_gaussian:.0f} nm'
                
                ax2.text(0.98, 0.95, text_str, transform=ax2.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', 
                                 edgecolor=THESIS_COLOURS['orange'], alpha=0.9, linewidth=1.5))
        
        ax2.set_xlabel('Wavelength (nm)', fontsize=11, fontweight='medium')
        ax2.set_ylabel('Normalised Intensity', fontsize=11, fontweight='medium')
        ax2.set_title('(b) Spectra of Transmitted Light', 
                     fontsize=12, fontweight='bold', pad=10)
        ax2.legend(fontsize=10, framealpha=0.95, loc='upper left')
        ax2.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
        ax2.set_ylim(0, 1.05)
        
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=10, 
                          direction='in', length=4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Format x-axis to show whole numbers only
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
            
            # Format y-axis to show whole numbers only
            if ax == ax1:
                # For pixel strip plot (y-axis is already integers)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

        
        plt.tight_layout()
        
        # Save if requested
        if self.save_plots:
            file_directory = os.path.dirname(filepath)
            plot_filename = "/Users/oliviaberti/Desktop/Research/Thesis/Thesis Figures/Resonance_spectra.png"
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Thesis figure saved to: {plot_filename}")
            
            # Also save as PDF for latex
            pdf_filename = os.path.join(file_directory, "Metalens4_Zoom4x_thesis_figure.pdf")
            fig.savefig(pdf_filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"PDF version saved to: {pdf_filename}")
        
        plt.show()
        
        # Store results (prioritise Fano values)
        result_dict = {
            'Metalens': '4',
            'Zoom': '4',
            'Fitted Peak Wavelength (nm)': f"{fitted_peak:.1f}",
            'FWHM (nm)': f"{fwhm_fano:.1f}" if fwhm_fano else (f"{fwhm_gaussian:.1f}" if fwhm_gaussian else "N/A"),
            'Fit Type': 'Fano' if fwhm_fano else ('Gaussian' if fwhm_gaussian else 'None')
        }
        
        self.results.append(result_dict)
        
        # Print summary
        print("METALENS 4, ZOOM 4X - FANO RESONANCE ANALYSIS")
        print("="*60)
        print(f"Peak Wavelength: {fitted_peak:.1f} nm")
        if fwhm_fano:
            print(f"FWHM (Fano Fit): {fwhm_fano:.1f} nm")
        elif fwhm_gaussian:
            print(f"FWHM (Gaussian Fit): {fwhm_gaussian:.1f} nm [Fano fit failed]")
        
        # Save data if requested
        if self.save_data:
            df = pd.DataFrame({
                'Wavelength (nm)': wavelengths,
                'Normalised Intensity': spectrum_normalised
            })
            
            file_directory = os.path.dirname(filepath)
            csv_filename = os.path.join(file_directory, "Metalens4_Zoom4x_data.csv")
            df.to_csv(csv_filename, index=False)
            print(f"Extracted data saved to: {csv_filename}")

def main(file_sample, roi=[10, 50], roi_sum=0, normalisation=1, save_data=0, save_plots=1, fit_fano=True):
    """
    Main function for thesis figure generation.
    
    Args:
        file_sample: List of file paths to analyse
        roi: Region of interest [start, end]
        roi_sum: If 1, sum the ROI; if 0, average the ROI
        normalisation: If 1, normalise intensity to [0,1]
        save_data: If 1, save extracted data to CSV
        save_plots: If 1, save plots as PNG and PDF files
        fit_fano: If True, fit Fano resonance in addition to Gaussian
    """
    analyser = ThesisSpectroscopyAnalyser(roi=roi, roi_sum=roi_sum, 
                                         normalisation=normalisation, 
                                         save_data=save_data, 
                                         save_plots=save_plots,
                                         fit_fano=fit_fano)
    
    analyser.process_files(file_sample)
    
    return analyser


# Main call function
if __name__ == "__main__":
    folder_path = '/Users/oliviaberti/Desktop/Lens Characterisation/NTC_DataprocessCodes/Olivia/Data/Spectra'
    file_sample = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.txt')
    ]
    
    print(f"Files in folder: {folder_path}")
    print(f"Total files found: {len(file_sample)}\n")
    #Defining region of interest
    ROI = [100, 350]
    ROI_sum = 0 
    normalisation = 1
    save_data = 1
    save_plots = 1
    
    analyser = main(file_sample, 
                    roi=ROI, 
                    roi_sum=ROI_sum, 
                    normalisation=normalisation, 
                    save_data=save_data,
                    save_plots=save_plots,
                    fit_fano=True)  # Enable Fano fitting