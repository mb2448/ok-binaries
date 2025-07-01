#!/usr/bin/env python3
"""
WDS Binary Star Plotter - MODIFIED VERSION WITH LINE NUMBER SUPPORT AND INSET PLOT

Plots binary star positions and orbits from WDS spreadsheet data.
Usage: python wds_binary_plotter.py <csv_file> <identifier>

The identifier can be:
- Line number (e.g., '77' or 'line:77')
- WDS designation (e.g., '00155-1608')
- Discoverer designation (e.g., 'STF2272AB')
- HIP number (e.g., 'HIP 165341')
- HD number (e.g., 'HD 103400')

MODIFICATIONS:
1. Added support for line number as identifier
2. Added warnings for non-unique identifiers
3. When line number is provided, uses that specific row directly
4. Added inset plot showing current position distribution with 1, 2, 3 sigma contours
5. Added dark mode theme for better website integration
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from scipy import stats
import binary_calculator as bc

# Set dark theme
plt.style.use('dark_background')

# Additional settings for crisper rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.antialiased'] = True
plt.rcParams['patch.antialiased'] = True
plt.rcParams['text.antialiased'] = True

def format_error_value(value, threshold=0.001):
    """
    Format error values appropriately, showing small values in scientific notation.
    """
    if value < threshold:
        return f"{value:.2e}"
    elif value < 0.01:
        return f"{value:.4f}"
    elif value < 0.1:
        return f"{value:.3f}"
    elif value < 1:
        return f"{value:.2f}"
    else:
        return f"{value:.1f}"

def plot_position_scatter(separations, position_angles, title="Binary Star Position",
                         save_fig=False):
    """
    Plot separations and position angles as a simple scatter plot.
    """
    # Convert to numpy arrays
    separations = np.asarray(separations)
    position_angles = np.asarray(position_angles)

    # Calculate statistics
    sep_median = np.median(separations)
    sep_std = np.std(separations)

    # Use circular statistics for position angle
    pa_rad = np.radians(position_angles)
    sin_mean = np.mean(np.sin(pa_rad))
    cos_mean = np.mean(np.cos(pa_rad))
    pa_median = np.degrees(np.arctan2(sin_mean, cos_mean))
    if pa_median < 0:
        pa_median += 360

    # Calculate circular standard deviation
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    if R > 0.99999:
        pa_std = 0.0
    else:
        pa_std = np.degrees(np.sqrt(-2 * np.log(R)))

    # Convert to Cartesian coordinates
    theta_rad = np.radians(90 - position_angles)
    x = separations * np.cos(theta_rad)
    y = separations * np.sin(theta_rad)

    # Median position
    theta_median_rad = np.radians(90 - pa_median)
    x_median = sep_median * np.cos(theta_median_rad)
    y_median = sep_median * np.sin(theta_median_rad)

    # Create plot - compact size for position scatter
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    # Scatter plot - using cyan for better visibility on dark background
    ax.scatter(x, y, alpha=0.6, s=15, c='cyan', edgecolors='none')

    # Median position
    ax.plot(x_median, y_median, 'ro', markersize=8, label='Median')

    # Skip plotting primary star at origin

    ax.set_xlabel('East ← Δα cos(δ) (arcsec) → West')
    ax.set_ylabel('South ← Δδ (arcsec) → North')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()

    # Print statistics with proper formatting
    print(f"Separation: {sep_median:.3f} ± {format_error_value(sep_std)} arcsec")
    print(f"Position Angle: {pa_median:.2f} ± {format_error_value(pa_std)}° (circular stats)")
    print(f"Sample size: {len(separations)}")

    if save_fig:
        plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight', facecolor='#0d1117')
        print(f"Figure saved")

    plt.show()


def plot_orbit_ensemble(orbit_data, current_positions=None, title="Binary Star Orbit Ensemble", save_fig=False):
    """
    Plot multiple orbital tracks from compute_orbit_ensemble with transparency.
    Optionally overlay current position uncertainty cloud with inset showing contours.
    """
    separations = orbit_data['separations']
    position_angles = orbit_data['position_angles']
    epochs = orbit_data['epochs']

    n_samples, n_epochs = separations.shape

    # Create figure with specific size and DPI
    # Smaller figure size but higher DPI for web display
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    # Increase font sizes for better readability
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Calculate plot limits based on maximum separation
    max_sep = np.nanmax(separations)
    # Check if current positions would extend the range
    if current_positions is not None:
        max_current = np.max(current_positions['separation'])
        max_sep = max(max_sep, max_current)

    # Add padding
    max_range = max_sep * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    # Set alpha based on number of samples for good visibility
    alpha = min(0.3, 100 / n_samples)

    # Plot each orbit - using cyan for better visibility on dark background
    for i in range(n_samples):
        # Handle position angle wrapping for smooth curves
        pa = position_angles[i, :].copy()

        # Unwrap position angles to avoid jumps across 0°/360°
        pa_unwrapped = np.unwrap(np.radians(pa)) * 180 / np.pi

        # Convert to Cartesian coordinates
        theta_rad = np.radians(90 - pa_unwrapped)
        x = separations[i, :] * np.cos(theta_rad)
        y = separations[i, :] * np.sin(theta_rad)

        # Only plot if we have valid data points
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid_mask) > 2:
            ax.plot(x[valid_mask], y[valid_mask], 'c-', alpha=alpha, linewidth=1.0, antialiased=True)

    # Overlay current position uncertainty cloud if provided
    if current_positions is not None:
        current_sep = current_positions['separation']
        current_pa = current_positions['position_angle']
        current_epoch = current_positions['epoch']

        # Calculate current position statistics
        sep_median = np.median(current_sep)
        sep_std = np.std(current_sep)

        # Use circular statistics for position angle
        # Convert to radians for circular mean calculation
        pa_rad = np.radians(current_pa)
        # Calculate circular mean
        sin_mean = np.mean(np.sin(pa_rad))
        cos_mean = np.mean(np.cos(pa_rad))
        pa_median = np.degrees(np.arctan2(sin_mean, cos_mean))
        if pa_median < 0:
            pa_median += 360

        # Calculate circular standard deviation
        R = np.sqrt(sin_mean**2 + cos_mean**2)  # Mean resultant length
        # Remove the artificial threshold - let the formula work
        if R >= 1.0:  # Only avoid log(0)
            pa_std = 0.0
        else:
            # Circular standard deviation in degrees
            pa_std = np.degrees(np.sqrt(-2 * np.log(R)))

        # Alternative: use median and MAD for more robust estimates
        # For median, we need to handle wraparound properly
        pa_shifted = current_pa.copy()
        # Find the mean angle and shift data to center around 180
        initial_mean = np.degrees(np.arctan2(sin_mean, cos_mean))
        pa_shifted = (current_pa - initial_mean + 180) % 360
        pa_median_shifted = np.median(pa_shifted)
        pa_median = (pa_median_shifted + initial_mean - 180) % 360
        if pa_median < 0:
            pa_median += 360

        # Convert current positions to Cartesian coordinates
        theta_current_rad = np.radians(90 - current_pa)
        x_current = current_sep * np.cos(theta_current_rad)
        y_current = current_sep * np.sin(theta_current_rad)

        # Plot current position cloud - using orange for good contrast
        ax.scatter(x_current, y_current, alpha=0.5, s=15, c='orange',
                  label=f'Current position ({current_epoch:.3f})', zorder=5, edgecolors='none', rasterized=False)

        # Add text box with current position statistics outside the plot
        # Convert decimal year to calendar date
        year = int(current_epoch)
        year_fraction = current_epoch - year
        # Calculate day of year
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        day_of_year = int(year_fraction * days_in_year) + 1
        # Convert to calendar date
        from datetime import datetime, timedelta
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        date_str = date.strftime("%Y %B %d")

        # Get current time for calculation timestamp
        calc_time = datetime.now().strftime("%H:%M:%S")

        # Use the errors from the CSV if provided, otherwise calculate them
        if 'csv_sep_error' in current_positions and 'csv_pa_error' in current_positions:
            # Use the pre-calculated errors from the CSV
            sep_error_str = format_error_value(current_positions['csv_sep_error'])
            pa_error_str = format_error_value(current_positions['csv_pa_error'])
        else:
            # Fall back to calculated errors
            sep_error_str = format_error_value(sep_std)
            pa_error_str = format_error_value(pa_std)

        stats_text = (f'Current Position\n({date_str}):\n'
                     f'Sep: {sep_median:.3f} ± {sep_error_str}" \n'
                     f'PA: {pa_median:.2f} ± {pa_error_str}°\n\n'
                     f'Calculated: {calc_time}')

        # Position text box in bottom right outside the plot
        # Using dark background for the text box
        ax.text(1.02, 0.02, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e2329', edgecolor='#3d4248', alpha=0.9))

        # Create inset axes for separation vs PA plot
        # Position it between the legend and the stats text box
        # Moved further right to avoid overlap with main plot
        ax_inset = fig.add_axes([0.72, 0.4, 0.25, 0.25], facecolor='#161b22')

        # Handle wraparound: shift data to center around 180° to avoid edge effects
        # This prevents artificial gaps at 0°/360°
        pa_shifted = (current_pa + 180) % 360
        pa_median_shifted = (pa_median + 180) % 360

        # Plot points in separation/PA space (shifted)
        ax_inset.scatter(pa_shifted, current_sep, alpha=0.6, s=20, c='orange', edgecolors='none')

        # Add median point
        ax_inset.plot(pa_median_shifted, sep_median, 'wo', markersize=8, label='Median', markeredgecolor='none')

        # Set inset labels and styling
        ax_inset.set_xlabel('Position Angle (°)', fontsize=10)
        ax_inset.set_ylabel('Separation (arcsec)', fontsize=10)
        ax_inset.set_title('Current Position Distribution', fontsize=11)
        ax_inset.tick_params(axis='both', which='major', labelsize=9)
        ax_inset.grid(True, alpha=0.3, linewidth=0.5)

        # Calculate x-axis limits to zoom in on the data
        pa_range = pa_shifted.max() - pa_shifted.min()
        pa_center = (pa_shifted.max() + pa_shifted.min()) / 2

        # Add padding - at least 20% of range or 10 degrees, whichever is larger
        pa_padding = max(pa_range * 0.2, 10)

        # Set x limits, but ensure we don't go beyond 0-360
        x_min = max(0, pa_shifted.min() - pa_padding)
        x_max = min(360, pa_shifted.max() + pa_padding)
        ax_inset.set_xlim(x_min, x_max)

        # Create custom tick labels that map back to original PA values
        # Generate ticks within the visible range
        n_ticks = 5
        xticks = np.linspace(x_min, x_max, n_ticks)
        xtick_labels = []
        for tick in xticks:
            # Convert from shifted to original PA
            original_pa = (tick - 180) % 360
            xtick_labels.append(f'{original_pa:.0f}°')

        ax_inset.set_xticks(xticks)
        ax_inset.set_xticklabels(xtick_labels, fontsize=9)

        # Set reasonable y-axis limits with some padding
        sep_range = current_sep.max() - current_sep.min()
        sep_padding = sep_range * 0.2 if sep_range > 0 else 0.001
        ax_inset.set_ylim(current_sep.min() - sep_padding, current_sep.max() + sep_padding)

        # Add border to inset with subtle color
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('#3d4248')
            spine.set_linewidth(1)

    # Primary star at origin - using yellow/gold color
    ax.plot(0, 0, 'o', color='gold', markersize=14, label='Primary', markeredgecolor='none', markeredgewidth=0)

    ax.set_xlabel('East ← Δα cos(δ) (arcsec) → West', fontsize=14, fontweight='medium')
    ax.set_ylabel('South ← Δδ (arcsec) → North', fontsize=14, fontweight='medium')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 0.98), loc='upper left', facecolor='#1e2329',
             edgecolor='#3d4248', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.8, antialiased=True)
    ax.invert_xaxis()

    # Adjust layout to make room for text boxes and inset on the right
    plt.subplots_adjust(left=0.1, right=0.62, top=0.95, bottom=0.05)

    # Set DPI for better scaling on screen
    fig.set_dpi(100)  # Use matplotlib default for display

    print(f"Plotted {n_samples} orbital tracks over {n_epochs} epochs")
    if current_positions is not None:
        current_sep = current_positions['separation']
        current_pa = current_positions['position_angle']
        current_epoch = current_positions['epoch']

        # Calculate and print current position statistics
        sep_median = np.median(current_sep)
        sep_std = np.std(current_sep)

        # Use circular statistics for position angle
        pa_rad = np.radians(current_pa)
        sin_mean = np.mean(np.sin(pa_rad))
        cos_mean = np.mean(np.cos(pa_rad))
        pa_mean = np.degrees(np.arctan2(sin_mean, cos_mean))
        if pa_mean < 0:
            pa_mean += 360

        # Calculate circular standard deviation
        R = np.sqrt(sin_mean**2 + cos_mean**2)  # Mean resultant length
        if R > 0.99999:  # Avoid numerical issues
            pa_std = 0.0
        else:
            # Circular standard deviation in degrees
            pa_std = np.degrees(np.sqrt(-2 * np.log(R)))

        # Alternative approach for very small errors:
        # When R is very close to 1, use regular standard deviation
        # but account for wraparound
        if R > 0.9999:  # Very tightly clustered
            # Shift angles to be centered around the mean
            pa_shifted = position_angles - pa_median
            # Wrap to [-180, 180]
            pa_shifted = ((pa_shifted + 180) % 360) - 180
            # Now calculate standard deviation
            pa_std_alt = np.std(pa_shifted)
            # Use the larger of the two calculations
            pa_std = max(pa_std, pa_std_alt)
            print(f"Using alternative PA error calculation: {pa_std_alt:.6f}° (R={R:.10f})")

        # For display, use circular mean instead of median
        pa_median = pa_mean

        print(f"Current position cloud: {len(current_sep)} samples at epoch {current_epoch:.1f}")
        print(f"Current Position Statistics:")
        print(f"  Separation: {sep_median:.3f} ± {format_error_value(sep_std)} arcsec")
        print(f"  Position Angle: {pa_median:.2f} ± {format_error_value(pa_std)}° (circular stats)")

    if save_fig:
        # Save as SVG for perfect scaling
        plt.savefig(f'{title.replace(" ", "_").lower()}.svg', format='svg', bbox_inches='tight',
                   facecolor='#0d1117', edgecolor='none', pad_inches=0.1)
        print(f"Figure saved as SVG")

    plt.show()


def plot_wds_binary(csv_file, identifier, n_samples=200, epoch=None):
    """
    Plot binary star from WDS spreadsheet data.

    Parameters:
    -----------
    csv_file : str
        Path to CSV file (e.g., 'current_binary_positions_2025-06-08.csv')
    identifier : str
        Star identifier - can be:
        - Line number (e.g., '77' or 'line:77')
        - WDS designation (e.g., '00182+7257')
        - Discoverer designation (e.g., 'STF2272AB')
        - HIP number (e.g., 'HIP 165341' or 'hip 165341' or just '165341' if numeric)
        - HD number (e.g., 'HD 103400' or 'hd 103400')
    n_samples : int
        Number of Monte Carlo samples for plotting
    epoch : float, optional
        Decimal year for the plot. If None, uses current date/time
    """

    # Load the spreadsheet
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Helper function to normalize identifiers for comparison
    def normalize_identifier(s):
        """Remove extra spaces and convert to uppercase for comparison"""
        if pd.isna(s):
            return ""
        # Replace multiple spaces with single space and strip
        return ' '.join(str(s).upper().split())

    # Check if identifier is a line number
    star_data = pd.DataFrame()
    identifier_normalized = normalize_identifier(identifier)

    # First check if it's a line number
    if identifier.lower().startswith('line:'):
        # Extract line number after 'line:'
        try:
            line_num = int(identifier[5:])
            star_data = df[df['line_number'] == line_num]
            if not star_data.empty:
                print(f"Found star by line number: {line_num}")
        except ValueError:
            pass
    elif identifier.isdigit():
        # Check if it's just a number - could be line number
        try:
            line_num = int(identifier)
            star_data = df[df['line_number'] == line_num]
            if not star_data.empty:
                print(f"Found star by line number: {line_num}")
            else:
                # Check if it's a HIP number without prefix
                star_data = df[df['hip_number'] == float(identifier)]
                if not star_data.empty:
                    print(f"Found star by HIP number: {identifier}")
        except ValueError:
            pass

    # If not found as line number, try other identifiers
    if star_data.empty:
        # Normalize the column values for comparison
        df['wds_norm'] = df['wds_designation'].apply(normalize_identifier)
        df['disc_norm'] = df['discoverer_designation'].apply(normalize_identifier)

        # Try WDS designation
        star_data = df[df['wds_norm'] == identifier_normalized]
        if not star_data.empty:
            print(f"Found star by WDS designation: {identifier}")

        # Try discoverer designation
        if star_data.empty:
            star_data = df[df['disc_norm'] == identifier_normalized]
            if not star_data.empty:
                print(f"Found star by discoverer designation: {identifier}")

        # Try HIP number (with or without prefix)
        if star_data.empty:
            if identifier_normalized.startswith('HIP'):
                hip_num = identifier_normalized.replace('HIP', '').strip()
                try:
                    star_data = df[df['hip_number'] == float(hip_num)]
                    if not star_data.empty:
                        print(f"Found star by HIP number: {hip_num}")
                except ValueError:
                    pass

        # Try HD number (with or without prefix)
        if star_data.empty:
            if identifier_normalized.startswith('HD'):
                hd_num = identifier_normalized.replace('HD', '').strip()
                try:
                    star_data = df[df['hd_number'] == float(hd_num)]
                    if not star_data.empty:
                        print(f"Found star by HD number: {hd_num}")
                except ValueError:
                    pass

    if star_data.empty:
        print(f"Error: Could not find star with identifier '{identifier}'")
        print("\nTry using:")
        print("  - Line number (e.g., '77' or 'line:77')")
        print("  - WDS designation (e.g., '00155-1608')")
        print("  - Discoverer designation (e.g., 'STF2272AB')")
        print("  - HIP number (e.g., 'HIP 165341' or just '165341')")
        print("  - HD number (e.g., 'HD 103400')")
        return

    # Check if identifier is unique (except for line numbers which are always unique)
    if len(star_data) > 1 and not (identifier.isdigit() or identifier.lower().startswith('line:')):
        print(f"\nWarning: Found {len(star_data)} systems matching '{identifier}':")
        for idx, row in star_data.iterrows():
            print(f"  Line {row['line_number']}: {row['wds_designation']} - {row['discoverer_designation']}")
        print(f"\nUsing first match (line {star_data.iloc[0]['line_number']}). To select a specific system, use its line number.")

    # Use the first (or only) match
    star = star_data.iloc[0]

    # Extract the data
    wds_designation = star['wds_designation']
    discoverer_designation = star['discoverer_designation']

    # Extract orbital elements with error checking
    def get_value(col):
        val = star.get(col, np.nan)
        return val if pd.notna(val) else 0.0

    # Create Monte Carlo samples
    np.random.seed(42)  # For reproducibility

    period = np.random.normal(star['period_years'], star['period_error_years'], n_samples)
    periastron_date = np.random.normal(star['periastron_time_years'], star['time_error_years'], n_samples)
    semimajor_axis = np.random.normal(star['semimajor_axis_arcsec'], star['axis_error_arcsec'], n_samples)
    eccentricity = np.random.normal(star['eccentricity'], star['eccentricity_error'], n_samples)
    inclination = np.random.normal(star['inclination'], star['inclination_error'], n_samples)
    argument_periastron = np.random.normal(star['periastron_longitude'], star['periastron_longitude_error'], n_samples)
    ascending_node = np.random.normal(star['ascending_node'], star['node_error'], n_samples)

    # Apply physical constraints
    period = np.maximum(period, 0.0001)
    eccentricity = np.clip(eccentricity, 0, 0.9999)
    semimajor_axis = np.maximum(semimajor_axis, 0.000001)
    inclination = np.clip(inclination, 0, 180)

    # Use provided epoch or current time
    if epoch is None:
        current_epoch = bc.get_current_decimal_year()
    else:
        current_epoch = epoch

    # Plot 1: Current position scatter
    print(f"\nTest 1: Current position scatter plot (epoch {current_epoch:.1f})")
    result = bc.calculate_binary_position(
        period=period,
        periastron_date=periastron_date,
        semimajor_axis=semimajor_axis,
        eccentricity=eccentricity,
        inclination=inclination,
        argument_periastron=argument_periastron,
        ascending_node=ascending_node,
        epoch=current_epoch
    )

    separations = result['separation']
    position_angles = result['position_angle']

    # Debug: print statistics about the position angle distribution
    print(f"\nDEBUG: Position angle distribution:")
    print(f"  Min: {np.min(position_angles):.6f}°")
    print(f"  Max: {np.max(position_angles):.6f}°")
    print(f"  Range: {np.max(position_angles) - np.min(position_angles):.6f}°")
    print(f"  Std dev: {np.std(position_angles):.6f}°")

    # Calculate circular statistics
    pa_rad = np.radians(position_angles)
    sin_mean = np.mean(np.sin(pa_rad))
    cos_mean = np.mean(np.cos(pa_rad))
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    print(f"  R (mean resultant length): {R:.10f}")
    if R > 0.99999:
        circ_std = 0.0
    else:
        circ_std = np.degrees(np.sqrt(-2 * np.log(R)))
    print(f"  Circular std dev: {circ_std:.6f}°")
    #plot_position_scatter(separations, position_angles, f"{wds_designation} Current Position")

    # Plot 2: Orbit ensemble
    print(f"\nTest 2: Computing {n_samples} orbital tracks...")
    orbit_data = bc.compute_orbit_ensemble(
        period, periastron_date, semimajor_axis, eccentricity,
        inclination, argument_periastron, ascending_node,
        n_epochs=200
    )

    # Plot the ensemble with current position overlay
    current_positions = {
        'separation': separations,
        'position_angle': position_angles,
        'epoch': current_epoch
    }

    plot_orbit_ensemble(orbit_data, current_positions, f"{wds_designation} Orbit Uncertainty", save_fig=True)


def main():
    """Main function to handle command line arguments."""

    if len(sys.argv) < 3:
        print("Usage: python wds_binary_plotter.py <csv_file> <identifier> [--epoch <decimal_year>]")
        print("\nExamples:")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv 77")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv 77 --epoch 2025.5")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv line:77")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv 00155-1608")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv STF2272AB")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv 'HIP 165341'")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv 'HD 103400'")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv BEU   3")
        print("  python wds_binary_plotter.py current_binary_positions_2025-06-13.csv 'GAA 6Aa,Ab'")
        return

    csv_file = sys.argv[1]

    # Find where the identifier ends and optional arguments begin
    identifier_parts = []
    epoch = None
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--epoch' and i + 1 < len(sys.argv):
            epoch = float(sys.argv[i + 1])
            break
        else:
            identifier_parts.append(sys.argv[i])
        i += 1

    identifier = ' '.join(identifier_parts)

    try:
        plot_wds_binary(csv_file, identifier, epoch=epoch)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
