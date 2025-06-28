#!/usr/bin/env python3
"""
Streamlit Binary Star Catalog Web Application

A simple web interface for browsing and plotting binary star data.
Run with: streamlit run binary_star_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
from PIL import Image
import glob
import re

# Page configuration
st.set_page_config(
    page_title="OK Binary Star Catalog",
    page_icon="⭐⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better table display and compact columns
st.markdown("""
<style>
    .stDataFrame {
        font-size: 11px;
    }
    .row_heading {
        display: none;
    }
    .blank {
        display: none;
    }
    /* Make the dataframe more compact */
    .dataframe td, .dataframe th {
        padding: 2px 5px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    /* Style for orbital elements table */
    .orbital-element {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .orbital-element:last-child {
        border-bottom: none;
    }
    .element-name {
        font-weight: 600;
    }
    .element-value {
        text-align: right;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the CSV data and cache it"""
    # Look for files matching the pattern binary_positions_YYYY-MM-DD.csv
    pattern = re.compile(r'binary_positions_(\d{4}-\d{2}-\d{2})\.csv')

    # Search in multiple possible locations
    search_dirs = ['.', '..', './data']
    matching_files = []

    for directory in search_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                match = pattern.match(file)
                if match:
                    file_path = os.path.join(directory, file)
                    date_str = match.group(1)
                    try:
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        matching_files.append((file_path, file_date, file))
                    except ValueError:
                        continue

    # If no matching files found, try the old filename pattern as fallback
    if not matching_files:
        possible_paths = [
            'current_binary_positions.csv',
            './current_binary_positions.csv',
            '../current_binary_positions.csv',
            './data/current_binary_positions.csv'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                mtime = os.path.getmtime(path)
                last_update = datetime.fromtimestamp(mtime)
                return df, last_update, path

    # Sort by date and get the most recent file
    if matching_files:
        matching_files.sort(key=lambda x: x[1], reverse=True)
        most_recent_path, file_date, filename = matching_files[0]

        df = pd.read_csv(most_recent_path)
        # Use the date from filename for display
        return df, file_date, most_recent_path

    st.error("Could not find any binary positions CSV file!")
    return None, None, None

def format_value_with_error(value, error, format_spec=".2f", unit=""):
    """Format a value with its error in a nice way"""
    if pd.isna(value):
        return "—"

    if pd.isna(error) or error == 0:
        return f"{value:{format_spec}}{unit}"

    # Determine significant figures based on error
    if error < 0.001:
        return f"{value:{format_spec}}{unit} ± {error:.2e}"
    elif error < 0.01:
        return f"{value:{format_spec}}{unit} ± {error:.4f}"
    elif error < 0.1:
        return f"{value:{format_spec}}{unit} ± {error:.3f}"
    elif error < 1:
        return f"{value:{format_spec}}{unit} ± {error:.2f}"
    else:
        return f"{value:{format_spec}}{unit} ± {error:.1f}"

def display_orbital_elements(star_data):
    """Display orbital elements in a formatted table"""
    st.markdown("### Orbital Elements")

    # Create HTML for nice formatting - no background color, using default theme
    html_content = '<div style="padding: 10px 0;">'

    # Period
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Period (P)</span>
        <span class="element-value">{format_value_with_error(star_data['period_years'], star_data['period_error_years'], ".2f", "y")}</span>
    </div>'''

    # Periastron Time
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Periastron (T)</span>
        <span class="element-value">{format_value_with_error(star_data['periastron_time_years'], star_data['time_error_years'], ".2f", "")}</span>
    </div>'''

    # Semi-major axis
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Semi-major axis (a)</span>
        <span class="element-value">{format_value_with_error(star_data['semimajor_axis_arcsec'], star_data['axis_error_arcsec'], ".3f", "″")}</span>
    </div>'''

    # Eccentricity
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Eccentricity (e)</span>
        <span class="element-value">{format_value_with_error(star_data['eccentricity'], star_data['eccentricity_error'], ".4f", "")}</span>
    </div>'''

    # Inclination
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Inclination (i)</span>
        <span class="element-value">{format_value_with_error(star_data['inclination'], star_data['inclination_error'], ".2f", "°")}</span>
    </div>'''

    # Longitude of periastron
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Longitude of periastron (ω)</span>
        <span class="element-value">{format_value_with_error(star_data['periastron_longitude'], star_data['periastron_longitude_error'], ".1f", "°")}</span>
    </div>'''

    # Node
    html_content += f'''<div class="orbital-element">
        <span class="element-name">Node (Ω)</span>
        <span class="element-value">{format_value_with_error(star_data['ascending_node'], star_data['node_error'], ".2f", "°")}</span>
    </div>'''

    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

    # Additional information in a separate section
    st.markdown("### Additional Information")
    info_cols = st.columns(2)

    with info_cols[0]:
        st.write(f"**Grade:** {int(star_data['grade']) if pd.notna(star_data['grade']) else '—'} (1=Definitive, 9=Indeterminate)")
        st.write(f"**Equinox:** {int(star_data['equinox']) if pd.notna(star_data['equinox']) else '—'}")

    with info_cols[1]:
        st.write(f"**Last observation:** {int(star_data['last_observation']) if pd.notna(star_data['last_observation']) else '—'}")
        st.write(f"**Reference:** {star_data['reference'] if pd.notna(star_data['reference']) else '—'}")

    # Display notes if they exist
    if pd.notna(star_data.get('notes')) and star_data['notes']:
        st.markdown("### Notes")
        st.write(star_data['notes'])

def plot_binary_star(row, csv_path):
    """Generate orbit plot by calling wds_binary_plotter.py script"""

    # Use line number as the identifier since it's guaranteed to be unique
    identifier = str(int(row['line_number']))

    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        'wds_binary_plotter.py',
        csv_path,  # Use the actual CSV file path
        identifier
    ]

    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # The script creates a file like "00182+7257_orbit_uncertainty.png"
        # Look for the most recently created PNG file
        png_files = glob.glob('*.png')
        if png_files:
            # Get the most recent PNG file
            latest_png = max(png_files, key=os.path.getctime)

            # Load and return the image
            image = Image.open(latest_png)

            # Optionally clean up the PNG file after loading
            # os.remove(latest_png)

            return image
        else:
            st.error("No plot file was generated")
            if result.stderr:
                st.error(f"Script output: {result.stderr}")
            return None

    except subprocess.CalledProcessError as e:
        st.error(f"Error running plotter: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Main app
def main():
    st.title("🌟🌟 OK Binary Star Catalog")

    # Load data
    df, last_update, file_path = load_data()

    if df is None:
        st.stop()

    # Display last update time and filename
    if last_update:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📊 Data from: {os.path.basename(file_path)}")
        with col2:
            st.info(f"📅 Date: {last_update.strftime('%Y-%m-%d')}")

    # Initialize counter for widget keys
    if 'key_counter' not in st.session_state:
        st.session_state.key_counter = 0

    # Sidebar filters
    st.sidebar.header("Filters")

    # Search box
    search_term = st.sidebar.text_input("Search", placeholder="WDS, discoverer, HIP#, HD#...", key=f"search_box_{st.session_state.key_counter}")

    # RA/Dec filters
    st.sidebar.subheader("Position")
    ra_hours = df['ra_j2000_deg'] / 15  # Convert degrees to hours
    ra_range = st.sidebar.slider(
        "RA (hours)",
        min_value=0.0,
        max_value=24.0,
        value=(0.0, 24.0),
        step=0.1,
        format="%.1fh",
        key=f"ra_slider_{st.session_state.key_counter}"
    )

    dec_range = st.sidebar.slider(
        "Dec (degrees)",
        min_value=-90.0,
        max_value=90.0,
        value=(-90.0, 90.0),
        step=1.0,
        format="%.0f°",
        key=f"dec_slider_{st.session_state.key_counter}"
    )

    # Period filter with logarithmic scale
    st.sidebar.subheader("Orbital Properties")
    period_min = float(df['period_years'].min())
    period_max = float(df['period_years'].max())

    # Create logarithmically spaced values
    period_values = np.logspace(
        np.log10(max(period_min, 0.1)),
        np.log10(min(period_max, 10000.0)),
        num=100
    )
    period_values = np.unique(np.round(period_values, 2))  # Round and remove duplicates

    # Find closest values to min and max for default
    default_min_idx = np.argmin(np.abs(period_values - period_min))
    default_max_idx = np.argmin(np.abs(period_values - min(period_max, 10000.0)))

    period_range = st.sidebar.select_slider(
        "Period (years)",
        options=period_values,
        value=(period_values[default_min_idx], period_values[default_max_idx]),
        format_func=lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}",
        key=f"period_slider_{st.session_state.key_counter}"
    )

    # Magnitude filters
    st.sidebar.subheader("Magnitudes")
    v1_range = st.sidebar.slider(
        "V₁ magnitude",
        min_value=float(df['v1_mag'].min()),
        max_value=float(df['v1_mag'].max()),
        value=(float(df['v1_mag'].min()), float(df['v1_mag'].max())),
        step=0.1,
        key=f"v1_slider_{st.session_state.key_counter}"
    )

    v2_range = st.sidebar.slider(
        "V₂ magnitude",
        min_value=float(df['v2_mag'].min()),
        max_value=float(df['v2_mag'].max()),
        value=(float(df['v2_mag'].min()), float(df['v2_mag'].max())),
        step=0.1,
        key=f"v2_slider_{st.session_state.key_counter}"
    )

    # Separation filter with logarithmic scale
    st.sidebar.subheader("Current Position")
    sep_min = max(df['separation_current'].min(), 0.001)
    sep_max = float(df['separation_current'].max())

    # Create logarithmically spaced values
    sep_values = np.logspace(
        np.log10(sep_min),
        np.log10(min(sep_max, 100.0)),
        num=100
    )
    sep_values = np.unique(np.round(sep_values, 4))  # Round to 4 decimals for arcsec

    # Find closest values for default
    default_min_idx = np.argmin(np.abs(sep_values - sep_min))
    default_max_idx = np.argmin(np.abs(sep_values - min(sep_max, 100.0)))

    sep_range = st.sidebar.select_slider(
        "Separation (arcsec)",
        options=sep_values,
        value=(sep_values[default_min_idx], sep_values[default_max_idx]),
        format_func=lambda x: f"{x:.3f}″" if x < 1 else f"{x:.1f}″",
        key=f"sep_slider_{st.session_state.key_counter}"
    )

    # Position angle error filter
    pa_error_max = st.sidebar.slider(
        "Max PA error (°)",
        min_value=0.0,
        max_value=180.0,
        value=180.0,
        step=1.0,
        help="Maximum position angle uncertainty (0 = no limit)",
        key=f"pa_error_slider_{st.session_state.key_counter}"
    )

    # Separation error filter
    sep_error_max = st.sidebar.slider(
        "Max separation error (″)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        format="%.2f″",
        help="Maximum separation uncertainty (0 = no limit)",
        key=f"sep_error_slider_{st.session_state.key_counter}"
    )

    # Grade filter
    st.sidebar.subheader("Quality")
    unique_grades = sorted(df['grade'].dropna().unique().astype(int))
    selected_grades = st.sidebar.multiselect(
        "Orbit grades",
        options=unique_grades,
        default=unique_grades,
        help="1=definitive, 2=good, 3=reliable, 4=preliminary, 5=indeterminate",
        key=f"grade_select_{st.session_state.key_counter}"
    )

    # Reset filters button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", type="secondary", use_container_width=True):
        # Increment the key counter to force all widgets to reset
        st.session_state.key_counter += 1
        st.rerun()

    # Apply filters
    filtered_df = df.copy()

    # Period filter
    filtered_df = filtered_df[
        (filtered_df['period_years'] >= period_range[0]) &
        (filtered_df['period_years'] <= period_range[1])
    ]

    # Magnitude filters
    filtered_df = filtered_df[
        (filtered_df['v1_mag'] >= v1_range[0]) &
        (filtered_df['v1_mag'] <= v1_range[1])
    ]
    filtered_df = filtered_df[
        (filtered_df['v2_mag'] >= v2_range[0]) &
        (filtered_df['v2_mag'] <= v2_range[1])
    ]

    # Separation filter
    filtered_df = filtered_df[
        (filtered_df['separation_current'] >= sep_range[0]) &
        (filtered_df['separation_current'] <= sep_range[1])
    ]

    # Error filters
    if pa_error_max > 0:  # 0 means no limit
        filtered_df = filtered_df[filtered_df['position_angle_error'] <= pa_error_max]
    if sep_error_max > 0:  # 0 means no limit
        filtered_df = filtered_df[filtered_df['separation_error'] <= sep_error_max]

    # Grade filter
    if 'grade' in df.columns and df['grade'].notna().any():
        filtered_df = filtered_df[filtered_df['grade'].isin(selected_grades)]

    # RA/Dec filter (always applied)
    # Convert RA hours to degrees for filtering
    ra_min_deg = ra_range[0] * 15
    ra_max_deg = ra_range[1] * 15

    # Handle RA wrap-around at 0/360 degrees
    if ra_min_deg <= ra_max_deg:
        filtered_df = filtered_df[
            (filtered_df['ra_j2000_deg'] >= ra_min_deg) &
            (filtered_df['ra_j2000_deg'] <= ra_max_deg)
        ]
    else:  # Wraps around 0
        filtered_df = filtered_df[
            (filtered_df['ra_j2000_deg'] >= ra_min_deg) |
            (filtered_df['ra_j2000_deg'] <= ra_max_deg)
        ]

    filtered_df = filtered_df[
        (filtered_df['dec_j2000_deg'] >= dec_range[0]) &
        (filtered_df['dec_j2000_deg'] <= dec_range[1])
    ]

    # Apply search
    if search_term:
        search_upper = search_term.upper()
        mask = (
            filtered_df['wds_designation'].astype(str).str.contains(search_upper, na=False) |
            filtered_df['discoverer_designation'].astype(str).str.contains(search_upper, na=False) |
            filtered_df['hip_number'].astype(str).str.contains(search_term, na=False) |
            filtered_df['hd_number'].astype(str).str.contains(search_term, na=False)
        )
        filtered_df = filtered_df[mask]

    # Display summary
    st.info(f"Showing {len(filtered_df)} of {len(df)} binary stars")

    # Create two columns for table and details
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Binary Star Table")

        # Select columns to display
        display_columns = [
            'ra_j2000_hms',
            'dec_j2000_dms',
            'wds_designation',
            'hd_number',
            'v1_mag',
            'v2_mag',
            'separation_current',
            'separation_error',
            'position_angle_current',
            'position_angle_error',
            'reference'
        ]

        # Also need the degree columns for sorting
        sort_columns = [
            'ra_j2000_deg',
            'dec_j2000_deg'
        ]

        # Create a simplified dataframe for display
        display_df = filtered_df[display_columns + sort_columns].copy()

        # Shorten RA/Dec display for compactness
        def format_ra(ra_str):
            if pd.isna(ra_str) or not ra_str:
                return ''
            parts = str(ra_str).split(' ')
            if len(parts) == 3:
                h, m, s = parts
                try:
                    s_float = float(s)
                    if s_float == int(s_float):
                        s = f"{int(s_float):02d}"
                    else:
                        s = f"{s_float:.2f}".rstrip('0').rstrip('.')
                except:
                    pass
                return f"{h} {m} {s}"
            return ra_str

        def format_dec(dec_str):
            if pd.isna(dec_str) or not dec_str:
                return ''
            parts = str(dec_str).split(' ')
            if len(parts) == 3:
                d, m, s = parts
                try:
                    s_float = float(s)
                    if s_float == int(s_float):
                        s = f"{int(s_float):02d}"
                    else:
                        s = f"{s_float:.1f}".rstrip('0').rstrip('.')
                except:
                    pass
                return f"{d} {m} {s}"
            return dec_str

        display_df['ra_j2000_hms'] = display_df['ra_j2000_hms'].apply(format_ra)
        display_df['dec_j2000_dms'] = display_df['dec_j2000_dms'].apply(format_dec)

        # Drop the degree columns from display
        display_df = display_df.drop(columns=['ra_j2000_deg', 'dec_j2000_deg'])

        # Rename columns for display
        display_df.columns = ['RA (J2000)', 'Dec (J2000)', 'WDS', 'HD', 'V₁', 'V₂',
                             'ρ (″)', 'σ_ρ (″)', 'θ (°)', 'σ_θ (°)', 'Ref']

        # Display as interactive dataframe
        selected_indices = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=600
        )

        # Get selected row
        selected_star = None
        if selected_indices and selected_indices.selection.rows:
            selected_index = filtered_df.index[selected_indices.selection.rows[0]]
            selected_star = df.loc[selected_index]

    with col2:
        st.subheader("Star Details")

        # Fixed position for Generate Plot button at the top
        button_container = st.container()
        with button_container:
            if selected_star is not None:
                if st.button("Generate Plot", type="primary", use_container_width=True):
                    st.session_state.show_plot = True
                    st.session_state.plot_star_line = int(selected_star['line_number'])
            else:
                st.button("Generate Plot", type="primary", use_container_width=True, disabled=True)

        # Create a scrollable container for details with fixed height
        details_container = st.container(height=520)  # Match table height
        with details_container:
            if selected_star is not None:
                # Display star info
                st.write(f"**{selected_star['wds_designation']}** - {selected_star['discoverer_designation']}")

                # Additional identifiers
                identifiers = []
                if pd.notna(selected_star.get('hip_number')):
                    identifiers.append(f"HIP {int(selected_star['hip_number'])}")
                if pd.notna(selected_star.get('hd_number')):
                    identifiers.append(f"HD {int(selected_star['hd_number'])}")
                if identifiers:
                    st.write(" • ".join(identifiers))

                # Display orbital elements
                display_orbital_elements(selected_star)
            else:
                st.info("Select a star from the table to view its details")

    # Plot section - completely outside the columns
    if hasattr(st.session_state, 'show_plot') and st.session_state.show_plot:
        if (hasattr(st.session_state, 'plot_star_line') and
            selected_star is not None and
            int(selected_star['line_number']) == st.session_state.plot_star_line):

            st.markdown("---")
            st.subheader("Orbit Plot")

            # Current position
            pos_col1, pos_col2 = st.columns(2)
            with pos_col1:
                st.write(f"**Current Separation:** {selected_star['separation_current']:.3f} ± {selected_star['separation_error']:.3f} arcsec")
            with pos_col2:
                st.write(f"**Current Position Angle:** {selected_star['position_angle_current']:.1f} ± {selected_star['position_angle_error']:.1f}°")

            # Generate plot
            with st.spinner("Generating orbit plot..."):
                try:
                    image = plot_binary_star(selected_star, file_path)
                    if image:
                        st.image(image, use_container_width=True)

                        if st.button("Hide Plot"):
                            st.session_state.show_plot = False
                            st.rerun()
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
                    if st.button("Dismiss Error"):
                        st.session_state.show_plot = False
                        st.rerun()

    # Footer
    st.markdown("---")
    st.caption("Binary Star Catalog Viewer - Using WDS Sixth Orbit Catalog")

if __name__ == "__main__":
    main()
