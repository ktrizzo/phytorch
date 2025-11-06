"""
Example: LI-600 Stomatal Conductance Correction

This script demonstrates how to use PhyTorch's LI-600 correction utility
to apply the Rizzo & Bailey (2025) correction to porometer measurements.
"""

from phytorch.utilities import correct_li600, plot_correction

# Path to LI-600 data file
data_path = 'phytorch/data/correction/walnut.csv'

# Apply correction for hypostomatous leaf (stomata only on lower surface)
print("Applying LI-600 correction...")
corrected_data = correct_li600(
    filepath=data_path,
    stomatal_sidedness=1.0,  # 1.0 = hypostomatous, 2.0 = amphistomatous
    thermal_conductance=0.007,  # Default calibrated value (W/°C)
    save_output=True  # Saves to walnut_corrected.csv
)

print(f"\nProcessed {len(corrected_data)} measurements")
print(f"Mean original gsw: {corrected_data['gsw'].mean():.4f} mol m⁻² s⁻¹")
print(f"Mean corrected gsw: {corrected_data['gsw_corrected'].mean():.4f} mol m⁻² s⁻¹")
print(f"Average correction: {(corrected_data['gsw_corrected'] - corrected_data['gsw']).mean():.4f} mol m⁻² s⁻¹")

# Generate diagnostic plots
print("\nGenerating diagnostic plots...")
fig, axes = plot_correction(
    data=corrected_data,
    save_path='walnut_correction_plot.png',
    show=False  # Set to True to display plot
)

print("✓ Correction complete!")
print("  - Corrected data saved to: phytorch/data/correction/walnut_corrected.csv")
print("  - Diagnostic plot saved to: walnut_correction_plot.png")
