
#!/usr/bin/env python3
"""
WDS Position Calculator - FINAL VERSION WITH CIRCULAR STATISTICS

Correctly parses WDS catalog where values are pre-converted to standard units.
The unit codes indicate what units the values are already in, not conversion needed.
Uses circular statistics for position angle calculations to handle wraparound at 0°/360°.

Usage:
    python final_wds_parser.py                    # Process all systems
    python final_wds_parser.py test              # Test specific system
    python final_wds_parser.py debug 00084+2905  # Debug specific system
"""

import numpy as np
import pandas as pd
import binary_calculator as bc
import re
from typing import Optional, Tuple
from tqdm import tqdm
from datetime import datetime

# ADD THESE TWO FUNCTIONS TO YOUR EXISTING wds_parser.py FILE:

def parse_notes_file(notes_filename='./orb6/orb6notes.txt'):
    """
    Parse the WDS notes file and return a dictionary mapping WDS designations to notes.

    Parameters:
    -----------
    notes_filename : str
        Path to the notes file

    Returns:
    --------
    dict : Dictionary with WDS designations as keys and notes as values
    """
    notes_dict = {}
    current_wds = None
    current_note_lines = []

    try:
        with open(notes_filename, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header lines
            for line in f:
                if line.strip().startswith('Sixth Catalog'):
                    continue
                if line.strip() == '':
                    continue
                break

            # Process the rest of the file
            for line in f:
                # Check if this line starts with a WDS designation
                # WDS format: NNNNN±NNNN (10 chars) at the beginning of the line
                if len(line) >= 10:
                    potential_wds = line[0:10].strip()
                    # Check if it matches WDS pattern (5 digits, +/-, 4 digits)
                    if (len(potential_wds) == 10 and
                        potential_wds[5] in ['+', '-'] and
                        potential_wds[0:5].replace(' ', '').isdigit() and
                        potential_wds[6:10].replace(' ', '').isdigit()):

                        # Save previous note if exists
                        if current_wds and current_note_lines:
                            # Join lines and clean up extra spaces
                            note_text = ' '.join(current_note_lines)
                            note_text = ' '.join(note_text.split())  # Normalize whitespace
                            notes_dict[current_wds] = note_text

                        # Start new note
                        current_wds = potential_wds
                        # Extract the note text (everything after the discoverer designation)
                        # The discoverer designation typically ends around position 25-30
                        note_start = line[25:].strip() if len(line) > 25 else ''
                        current_note_lines = [note_start] if note_start else []
                    else:
                        # This is a continuation line
                        if current_wds and line.strip():
                            current_note_lines.append(line.strip())

            # Don't forget the last note
            if current_wds and current_note_lines:
                note_text = ' '.join(current_note_lines)
                note_text = ' '.join(note_text.split())
                notes_dict[current_wds] = note_text

    except FileNotFoundError:
        print(f"Warning: Notes file not found at {notes_filename}")
        return {}
    except Exception as e:
        print(f"Warning: Error reading notes file: {e}")
        return {}

    print(f"Parsed {len(notes_dict)} notes from {notes_filename}")
    return notes_dict


def add_notes_to_dataframe(df, notes_filename='./orb6/orb6notes.txt'):
    """
    Add a notes column to the dataframe by matching WDS designations.

    Parameters:
    -----------
    df : pd.DataFrame
        The WDS dataframe with a 'wds_designation' column
    notes_filename : str
        Path to the notes file

    Returns:
    --------
    pd.DataFrame : The input dataframe with an added 'notes' column
    """
    # Parse the notes file
    notes_dict = parse_notes_file(notes_filename)

    if not notes_dict:
        print("No notes found or could not parse notes file")
        df['notes'] = ''
        return df

    # Add notes column
    df['notes'] = df['wds_designation'].map(notes_dict).fillna('')

    # Count how many systems have notes
    notes_count = (df['notes'] != '').sum()
    print(f"Added notes to {notes_count} out of {len(df)} systems ({notes_count/len(df)*100:.1f}%)")

    return df




class WDSParser:
    def __init__(self):
        """Initialize parser with unit indicators."""

        # The unit codes in the file indicate what units the values are ALREADY IN
        # No conversion needed when the unit matches our target
        self.period_target = 'years'
        self.axis_target = 'arcseconds'

        # These conversions are only used if we need to convert TO our target units
        # But in this catalog, values are already in the units indicated
        JULIAN_YEAR_DAYS = 365.25

        self.period_conversions = {
            'c': 100.0,      # centuries to years
            'd': 1/JULIAN_YEAR_DAYS,   # days to years
            'h': 1/(JULIAN_YEAR_DAYS*24), # hours to years
            'm': 1/(JULIAN_YEAR_DAYS*24*60), # minutes to years
            'y': 1.0,        # years (no conversion needed)
            '': 1.0          # assume years if no unit
        }

        self.axis_conversions = {
            'a': 1.0,        # arcseconds (no conversion needed)
            'm': 0.001,      # milliarcseconds to arcseconds
            'M': 60.0,       # arcminutes to arcseconds
            'u': 0.000001,   # microarcseconds to arcseconds
            '': 1.0          # assume arcseconds if no unit
        }

    def _clean_field(self, value: str) -> Optional[str]:
        """Clean and standardize field values."""
        if not value:
            return None
        cleaned = value.strip()
        if cleaned in ['.', '', '...', '....']:
            return None
        return cleaned if cleaned else None

    def _parse_numeric_field(self, value: str) -> Optional[float]:
        """Parse numeric fields, handling various formats and missing values."""
        cleaned = self._clean_field(value)
        if cleaned is None:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_coordinates(self, line: str) -> tuple:
        """Parse RA,Dec coordinates from the first 18 characters."""
        try:
            # RA: positions 1-9 (hours, minutes, seconds)
            ra_hours = float(line[0:2])
            ra_minutes = float(line[2:4])
            ra_seconds = float(line[4:9])

            # Dec sign: position 10
            dec_sign = line[9:10]

            # Dec: positions 11-18 (degrees, minutes, seconds)
            dec_degrees = float(line[10:12])
            dec_minutes = float(line[12:14])
            dec_seconds = float(line[14:18])

            # Convert RA to degrees
            ra_deg = (ra_hours + ra_minutes/60 + ra_seconds/3600) * 15

            # Convert Dec to degrees
            dec_deg = dec_degrees + dec_minutes/60 + dec_seconds/3600
            if dec_sign == '-':
                dec_deg = -dec_deg

            return ra_deg, dec_deg
        except:
            return None, None

    def _deg_to_hms(self, ra_deg: float) -> str:
        """Convert RA in decimal degrees to hours:minutes:seconds format."""
        if ra_deg is None:
            return None
        ra_hours = ra_deg / 15.0
        hours = int(ra_hours)
        minutes_float = (ra_hours - hours) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        return f"{hours:02d} {minutes:02d} {seconds:09.6f}"

    def _deg_to_dms(self, dec_deg: float) -> str:
        """Convert Dec in decimal degrees to degrees:arcminutes:arcseconds format."""
        if dec_deg is None:
            return None
        sign = '+' if dec_deg >= 0 else '-'
        dec_deg = abs(dec_deg)
        degrees = int(dec_deg)
        arcminutes_float = (dec_deg - degrees) * 60
        arcminutes = int(arcminutes_float)
        arcseconds = (arcminutes_float - arcminutes) * 60
        return f"{sign}{degrees:02d} {arcminutes:02d} {arcseconds:08.5f}"

    def _convert_time_to_year(self, time_value: float, time_unit: str) -> float:
        """Convert time value to decimal year based on unit code."""
        if time_value is None:
            return None

        if time_unit == 'y':
            # Already in fractional Besselian year
            return time_value
        elif time_unit == 'd':
            # Truncated Julian Date (JD - 2,400,000)
            jd = time_value + 2400000.0
            # Convert to decimal year
            return 2000.0 + (jd - 2451545.0) / 365.25
        elif time_unit == 'm':
            # Modified Julian Date (JD - 2,400,000.5)
            jd = time_value + 2400000.5
            return 2000.0 + (jd - 2451545.0) / 365.25
        elif time_unit == 'c':
            # Centuries (fractional year / 100)
            return time_value * 100
        else:
            # Assume already in years
            return time_value

    def parse_line(self, line: str, debug=False) -> dict:
        """Parse a single data line from the WDS orbit catalog."""

        # Ensure line is long enough
        if len(line) < 264:
            line = line.ljust(264)

        result = {}

        if debug:
            print(f"\nDEBUG: Parsing line of length {len(line)}")

        # Column 1: RA,Dec (J2000) - positions 1-18
        ra, dec = self._parse_coordinates(line[0:18])
        result['ra_j2000_deg'] = ra
        result['dec_j2000_deg'] = dec
        result['ra_dec_j2000'] = self._clean_field(line[0:18])
        if ra is not None:
            result['ra_j2000_hms'] = self._deg_to_hms(ra)
        if dec is not None:
            result['dec_j2000_dms'] = self._deg_to_dms(dec)

        # Column 2: WDS designation - T20,A10 (positions 20-29)
        result['wds_designation'] = self._clean_field(line[19:29])

        # Column 3: Discoverer designation - T31,A14 (positions 31-44)
        result['discoverer_designation'] = self._clean_field(line[30:44])

        # Column 4: ADS number - T46,I5 (positions 46-50)
        result['ads_number'] = self._parse_numeric_field(line[45:50])
        if result['ads_number'] is not None:
            result['ads_number'] = int(result['ads_number'])

        # Column 5: HD number - T52,I6 (positions 52-57)
        result['hd_number'] = self._parse_numeric_field(line[51:57])
        if result['hd_number'] is not None:
            result['hd_number'] = int(result['hd_number'])

        # Column 6: Hipparcos number - T59,I6 (positions 59-64)
        result['hip_number'] = self._parse_numeric_field(line[58:64])
        if result['hip_number'] is not None:
            result['hip_number'] = int(result['hip_number'])

        # Column 7: Primary magnitude - T67,F5.2,A1 (positions 67-72)
        result['v1_mag'] = self._parse_numeric_field(line[66:71])
        result['v1_mag_flag'] = self._clean_field(line[71:72])

        # Column 8: Secondary magnitude - T74,F5.2,A1 (positions 74-79)
        result['v2_mag'] = self._parse_numeric_field(line[73:78])
        result['v2_mag_flag'] = self._clean_field(line[78:79])

        # Column 9: Period - T82,F11.6,A1 (positions 82-93)
        result['period_value'] = self._parse_numeric_field(line[81:92])
        result['period_unit_code'] = self._clean_field(line[92:93]) or ''

        # Column 10: Period error - T95,F10.6 (positions 95-104)
        result['period_error_value'] = self._parse_numeric_field(line[94:104])

        # Column 11: Semi-major axis - T106,F9.5,A1 (positions 106-115)
        result['axis_value'] = self._parse_numeric_field(line[105:114])
        result['axis_unit_code'] = self._clean_field(line[114:115]) or ''

        # Column 12: Axis error - T117,F8.5 (positions 117-124)
        result['axis_error_value'] = self._parse_numeric_field(line[116:124])

        # Column 13: Inclination - T126,F8.4 (positions 126-133)
        result['inclination'] = self._parse_numeric_field(line[125:133])

        # Column 14: Inclination error - T135,F8.4 (positions 135-142)
        result['inclination_error'] = self._parse_numeric_field(line[134:142])

        # Column 15: Node - T144,F8.4,A1 (positions 144-152)
        result['ascending_node'] = self._parse_numeric_field(line[143:151])
        result['node_flag'] = self._clean_field(line[151:152])

        # Column 16: Node error - T154,F8.4 (positions 154-161)
        result['node_error'] = self._parse_numeric_field(line[153:161])

        # Column 17: Periastron time - T163,F12.6,A1 (positions 163-175)
        result['periastron_time_value'] = self._parse_numeric_field(line[162:174])
        result['time_unit_code'] = self._clean_field(line[174:175]) or ''

        # Column 18: Time error - T177,F10.6 (positions 177-186)
        result['time_error'] = self._parse_numeric_field(line[176:186])

        # Column 19: Eccentricity - T188,F8.6 (positions 188-195)
        result['eccentricity'] = self._parse_numeric_field(line[187:195])

        # Column 20: Eccentricity error - T197,F8.6 (positions 197-204)
        result['eccentricity_error'] = self._parse_numeric_field(line[196:204])

        # Column 21: Periastron longitude - T206,F8.4,A1 (positions 206-214)
        result['periastron_longitude'] = self._parse_numeric_field(line[205:213])
        result['periastron_longitude_flag'] = self._clean_field(line[213:214])

        # Column 22: Periastron longitude error - T215,F8.4 (positions 215-222)
        result['periastron_longitude_error'] = self._parse_numeric_field(line[214:222])

        # Column 23: Equinox - T224,I4 (positions 224-227)
        result['equinox'] = self._parse_numeric_field(line[223:227])
        if result['equinox'] is not None:
            result['equinox'] = int(result['equinox'])

        # Column 24: Last observation - T229,I4 (positions 229-232)
        result['last_observation'] = self._parse_numeric_field(line[228:232])
        if result['last_observation'] is not None:
            result['last_observation'] = int(result['last_observation'])

        # Column 25: Grade - T234,I1 (position 234)
        result['grade'] = self._parse_numeric_field(line[233:234])
        if result['grade'] is not None:
            result['grade'] = int(result['grade'])

        # Column 26: Notes flag - T236,A1 (position 236)
        result['notes_flag'] = self._clean_field(line[235:236])

        # Column 27: Reference - T238,A8 (positions 238-245)
        result['reference'] = self._clean_field(line[237:245])

        # Column 28: PNG file - T247,A18 (positions 247-264)
        result['png_file'] = self._clean_field(line[246:264])

        # Convert units if needed (but in this catalog, they're already in target units)
        # Period
        if result['period_value'] is not None:
            if result['period_unit_code'] == 'y':
                result['period_years'] = result['period_value']  # Already in years
            else:
                # Apply conversion if needed
                conversion = self.period_conversions.get(result['period_unit_code'], 1.0)
                result['period_years'] = result['period_value'] * conversion

        # Period error (same units as period)
        if result['period_error_value'] is not None:
            if result['period_unit_code'] == 'y':
                result['period_error_years'] = result['period_error_value']
            else:
                conversion = self.period_conversions.get(result['period_unit_code'], 1.0)
                result['period_error_years'] = result['period_error_value'] * conversion

        # Semi-major axis
        if result['axis_value'] is not None:
            if result['axis_unit_code'] == 'a':
                result['semimajor_axis_arcsec'] = result['axis_value']  # Already in arcsec
            else:
                conversion = self.axis_conversions.get(result['axis_unit_code'], 1.0)
                result['semimajor_axis_arcsec'] = result['axis_value'] * conversion

        # Axis error (same units as axis)
        if result['axis_error_value'] is not None:
            if result['axis_unit_code'] == 'a':
                result['axis_error_arcsec'] = result['axis_error_value']
            else:
                conversion = self.axis_conversions.get(result['axis_unit_code'], 1.0)
                result['axis_error_arcsec'] = result['axis_error_value'] * conversion

        # Periastron time
        result['periastron_time_years'] = self._convert_time_to_year(
            result['periastron_time_value'], result['time_unit_code'])

        # Time error
        if result['time_error'] is not None:
            if result['time_unit_code'] in ['d', 'm']:
                # Error in days for Julian dates
                result['time_error_years'] = result['time_error'] / 365.25
            elif result['time_unit_code'] == 'c':
                # Error in centuries
                result['time_error_years'] = result['time_error'] * 100
            elif result['time_unit_code'] == 'y':
                # Already in years
                result['time_error_years'] = result['time_error']
            else:
                result['time_error_years'] = result['time_error']

        if debug:
            print("\nDEBUG: Parsed values:")
            print(f"  Period: {result.get('period_value')} {result.get('period_unit_code')} → {result.get('period_years')} years")
            print(f"  Axis: {result.get('axis_value')} {result.get('axis_unit_code')} → {result.get('semimajor_axis_arcsec')} arcsec")
            print(f"  Time: {result.get('periastron_time_value')} {result.get('time_unit_code')} → {result.get('periastron_time_years')} years")

        return result

    def parse_file(self, filename: str, debug_wds=None):
        """Parse the entire catalog file."""
        data_rows = []

        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line_num <= 6:  # Skip header lines
                    continue
                if not line.strip():
                    continue

                # Check if this is the debug line
                debug_this_line = False
                if debug_wds and len(line) > 29:
                    wds = line[19:29].strip()
                    if wds == debug_wds:
                        debug_this_line = True
                        print(f"\nDEBUG: Found {debug_wds} at line {line_num}")

                parsed_data = self.parse_line(line, debug=debug_this_line)
                if parsed_data:
                    parsed_data['line_number'] = line_num
                    data_rows.append(parsed_data)

        if not data_rows:
            return pd.DataFrame()

        df = pd.DataFrame(data_rows)
        return df

    def filter_systems_with_complete_errors(self, df):
        """Filter for systems with complete error measurements."""
        required_error_fields = [
            'period_error_years', 'axis_error_arcsec', 'inclination_error',
            'node_error', 'time_error_years', 'eccentricity_error', 'periastron_longitude_error'
        ]

        existing_error_fields = [field for field in required_error_fields if field in df.columns]

        if not existing_error_fields:
            print("Warning: No error fields found in DataFrame")
            return pd.DataFrame()

        print(f"Filtering based on {len(existing_error_fields)} error fields:")

        complete_errors_mask = pd.Series(True, index=df.index)
        for field in existing_error_fields:
            field_mask = pd.notna(df[field]) & (df[field] > 0)  # Check for positive errors
            complete_errors_mask = complete_errors_mask & field_mask
            count_with_field = field_mask.sum()
            print(f"  {field}: {count_with_field}/{len(df)} systems ({count_with_field/len(df)*100:.1f}%)")

        filtered_df = df[complete_errors_mask].copy()

        print(f"\nFiltering results:")
        print(f"Original systems: {len(df)}")
        print(f"Systems with complete errors: {len(filtered_df)}")
        print(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")

        return filtered_df


def compute_wds_positions(filename='./orb6/orb6orbits.txt', n_samples=2000, debug_wds=None):
    """
    Load WDS data, filter for complete errors, compute current positions.
    Uses circular statistics for position angle calculations.

    Parameters:
    -----------
    filename : str
        Path to the WDS catalog file
    n_samples : int
        Number of Monte Carlo samples for uncertainty estimation
    debug_wds : str, optional
        WDS designation to debug (e.g., '00084+2905')

    Returns:
    --------
    pd.DataFrame with new position columns added
    """

    # 1. Load and filter WDS data
    parser = WDSParser()
    df = parser.parse_file(filename, debug_wds=debug_wds)

    if len(df) == 0:
        print("No data loaded!")
        return df

    print(f"\nLoaded {len(df)} systems from the catalog")

    # Filter for complete errors
    df = parser.filter_systems_with_complete_errors(df)

    if len(df) == 0:
        print("No systems with complete errors found!")
        return df

    # 2. Get current epoch
    current_epoch = bc.get_current_decimal_year()
    print(f"\nComputing positions for epoch {current_epoch:.2f}")

    # 3. Initialize new columns
    df['separation_current'] = np.nan
    df['separation_error'] = np.nan
    df['position_angle_current'] = np.nan
    df['position_angle_error'] = np.nan
    df['separation_error_dimensionless'] = np.nan  # Add new column

    # 4. Compute positions for each system
    np.random.seed(42)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing positions"):
        try:
            # Create Monte Carlo samples
            period = np.random.normal(row['period_years'], row['period_error_years'], n_samples)
            periastron_date = np.random.normal(row['periastron_time_years'], row['time_error_years'], n_samples)
            semimajor_axis = np.random.normal(row['semimajor_axis_arcsec'], row['axis_error_arcsec'], n_samples)
            eccentricity = np.random.normal(row['eccentricity'], row['eccentricity_error'], n_samples)
            inclination = np.random.normal(row['inclination'], row['inclination_error'], n_samples)
            argument_periastron = np.random.normal(row['periastron_longitude'], row['periastron_longitude_error'], n_samples)
            ascending_node = np.random.normal(row['ascending_node'], row['node_error'], n_samples)

            # Apply constraints
            period = np.maximum(period, 0.0001)
            eccentricity = np.clip(eccentricity, 0, 0.9999)
            semimajor_axis = np.maximum(semimajor_axis, 0.000001)
            inclination = np.clip(inclination, 0, 180)

            # Compute positions
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

            # Store separation results (regular statistics)
            sep_current = np.median(result['separation'])
            sep_error = np.std(result['separation'])

            df.loc[idx, 'separation_current'] = sep_current
            df.loc[idx, 'separation_error'] = sep_error

            # Calculate circular statistics for position angle
            pa_values = result['position_angle']
            pa_rad = np.radians(pa_values)
            sin_mean = np.mean(np.sin(pa_rad))
            cos_mean = np.mean(np.cos(pa_rad))

            # Circular mean
            pa_mean = np.degrees(np.arctan2(sin_mean, cos_mean))
            if pa_mean < 0:
                pa_mean += 360

            # Circular standard deviation
            R = np.sqrt(sin_mean**2 + cos_mean**2)  # Mean resultant length

            # Don't artificially set to zero - calculate the actual error
            # even for very small values
            if R >= 1.0:  # Protect against numerical issues at exactly 1
                R = 0.999999999  # Use a value very close to 1

            # Circular standard deviation in degrees
            if R > 0:
                pa_error = np.degrees(np.sqrt(-2 * np.log(R)))
            else:
                pa_error = 180.0  # Maximum possible error when R = 0

            df.loc[idx, 'position_angle_current'] = pa_mean
            df.loc[idx, 'position_angle_error'] = pa_error

            # Calculate dimensionless separation error quality metric
            if sep_current > 0:  # Avoid division by zero
                df.loc[idx, 'separation_error_dimensionless'] = sep_error / sep_current

        except Exception as e:
            print(f"\nError computing position for {row['wds_designation']}: {e}")
            continue

    # Count successful calculations
    successful = (~df['separation_current'].isna()).sum()
    print(f"\nSuccessfully computed positions for {successful}/{len(df)} systems")

    # Print note about circular statistics
    print("\nNote: Position angle errors calculated using circular statistics")

    df = add_notes_to_dataframe(df)
    # Save to CSV
    current_date = datetime.now().strftime('%Y-%m-%d')
    filename_out = f'binary_positions_{current_date}.csv'

    df.to_csv(filename_out, index=False)
    print(f"\nResults saved to '{filename_out}'")

    return df


def test_specific_system(filename='./orb6/orb6orbits.txt'):
    """Test parsing of specific systems."""

    parser = WDSParser()

    test_systems = ['00084+2905', '00155-1608']

    for wds in test_systems:
        print(f"\nTesting {wds}...")
        print("="*60)

        df = parser.parse_file(filename, debug_wds=wds)
        system = df[df['wds_designation'] == wds]

        if not system.empty:
            row = system.iloc[0]
            print("\nParsed values:")
            print(f"Period: {row['period_value']} {row.get('period_unit_code', '')} = {row['period_years']:.6f} years")
            print(f"Axis: {row['axis_value']} {row.get('axis_unit_code', '')} = {row['semimajor_axis_arcsec']:.6f} arcsec")
            print(f"Eccentricity: {row['eccentricity']} ± {row['eccentricity_error']}")
            print(f"Inclination: {row['inclination']}° ± {row['inclination_error']}°")
            print(f"Node: {row['ascending_node']}° ± {row['node_error']}°")
            print(f"Omega: {row['periastron_longitude']}° ± {row['periastron_longitude_error']}°")
            print(f"T0: {row['periastron_time_value']} {row.get('time_unit_code', '')} = {row['periastron_time_years']:.3f}")
        else:
            print("System not found!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Run test on specific systems
            test_specific_system()
        elif sys.argv[1] == 'debug' and len(sys.argv) > 2:
            # Debug specific system
            df = compute_wds_positions(debug_wds=sys.argv[2])
        else:
            print("Usage:")
            print("  python final_wds_parser.py              # Process all systems")
            print("  python final_wds_parser.py test         # Test specific systems")
            print("  python final_wds_parser.py debug WDS    # Debug specific system")
    else:
        # Run normal processing
        df = compute_wds_positions()

        if len(df) > 0:
            print(f"\nProcessed {len(df)} systems")
            print(f"Successful calculations: {(~df['separation_current'].isna()).sum()}")

            # Show sample results
            print("\nSample results:")
            sample_systems = ['00084+2905', '00155-1608', '00003-4417', '00021-6817']
            for wds in sample_systems:
                system = df[df['wds_designation'] == wds]
                if not system.empty:
                    row = system.iloc[0]
                    print(f"\n{wds}:")
                    print(f"  Period: {row['period_years']:.4f} years")
                    print(f"  Sep: {row['separation_current']:.3f} ± {row['separation_error']:.3f} arcsec")
                    print(f"  PA: {row['position_angle_current']:.1f} ± {row['position_angle_error']:.1f}° (circular stats)")
                    if not pd.isna(row['separation_error_dimensionless']):
                        print(f"  Dimensionless error: {row['separation_error_dimensionless']:.4f}")
