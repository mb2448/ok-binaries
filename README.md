# OK Binary Star Catalog - Offline Usage

This guide explains how to run the OK Binary Star Catalog application on your local computer.

## Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning the repository)

## Installation

### 1. Clone or Download the Repository

**Option A: Using Git**
```bash
git clone https://github.com/mb2448/ok-binaries.git
cd ok-binaries
```

**Option B: Download ZIP**
- Go to https://github.com/mb2448/ok-binaries
- Click "Code" → "Download ZIP"
- Extract the ZIP file
- Navigate to the extracted folder

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install streamlit pandas numpy scipy matplotlib pillow
```

## Running the Application

### 1. Start the Streamlit App

```bash
streamlit run app.py
```

This will:
- Start a local web server (usually on port 8501)
- Open your default web browser automatically
- Display the URL (typically http://localhost:8501)

### 2. Using the Application

**Browse and Search:**
- The main table shows all binary stars
- Use the search box to find specific stars (searches WDS, HD, HIP numbers and notes)
- Click column headers to sort

**Filters (Left Sidebar):**
- **Position**: Filter by Right Ascension and Declination
- **Orbital Properties**: Filter by orbital period
- **Magnitudes**: Filter by primary (V₁) and secondary (V₂) magnitudes
- **Current Position**: Filter by current separation
- **Quality**: Filter by orbit grades (1=best, 5=worst)
- **Reset All Filters**: Return all filters to default values

**View Star Details:**
1. Click any row in the table to select a star
2. Details appear in the right panel
3. Shows orbital elements and additional information

**Generate Orbit Plots:**
1. Select a star from the table
2. Click "Generate Plot"
3. Optionally set a custom date/time (default is now)
4. Click "Reset to Now" to return to current time
5. Plot shows orbital path with uncertainty

### 3. Updating the Data

The application includes a data updater that can fetch the latest binary star positions:

```bash
python wds_parser.py
```

This will:
- Download the latest data from the WDS catalog
- Process orbital elements and calculate current positions
- Create a new CSV file with today's date
- Keep only the 7 most recent data files

**Note**: The automated GitHub Actions workflow updates data daily online, but for offline use, run the parser manually when you want fresh data.

## Command Line Orbit Plotting

You can generate orbit plots directly from the command line without using the Streamlit interface:

### Basic Usage

```bash
python wds_binary_plotter.py <csv_file> <identifier>
```

### Examples

**Using line number:**
```bash
python wds_binary_plotter.py binary_positions_2025-06-30.csv 77
```

**Using WDS designation:**
```bash
python wds_binary_plotter.py binary_positions_2025-06-30.csv 00022+2705
```

**Using discoverer designation:**
```bash
python wds_binary_plotter.py binary_positions_2025-06-30.csv STF2272AB
```

**Using HIP number:**
```bash
python wds_binary_plotter.py binary_positions_2025-06-30.csv "HIP 165341"
# or just the number if unique:
python wds_binary_plotter.py binary_positions_2025-06-30.csv 165341
```

**Using HD number:**
```bash
python wds_binary_plotter.py binary_positions_2025-06-30.csv "HD 103400"
```

**With custom epoch (decimal year):**
```bash
python wds_binary_plotter.py binary_positions_2025-06-30.csv 77 --epoch 2025.5
python wds_binary_plotter.py binary_positions_2025-06-30.csv STF2272AB --epoch 2030.0
```

### Output

The script will:
1. Load the star's orbital elements from the CSV file
2. Calculate 200 Monte Carlo samples for uncertainty analysis
3. Generate an orbit plot showing:
   - The orbital path (cyan lines showing uncertainty)
   - Current position cloud (orange points)
   - Primary star at origin (gold)
   - Statistics box with position and uncertainties
   - Inset plot showing position angle vs separation distribution
4. Save the plot as an SVG file (e.g., `00022+2705_orbit_uncertainty.svg`)

The plot uses a dark theme optimized for screen viewing.

## File Structure

```
ok-binaries/
├── app.py                      # Main Streamlit application
├── wds_parser.py              # Data fetcher and processor
├── wds_binary_plotter.py      # Orbit plotting module
├── binary_calculator.py       # Orbital calculations
├── binary_positions_*.csv     # Data files (up to 7 most recent)
└── requirements.txt           # Python dependencies
```

## Offline Limitations

When running offline:
- No automatic daily data updates (run `wds_parser.py` manually)
- Initial data download requires internet connection
- No access to cloud deployment features