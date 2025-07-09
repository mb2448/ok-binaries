# OK Binary Star Catalog - Online Usage
https://ok-binaries.streamlit.app/

Click *About* for instructions on how to use.

# OK Binary Star Catalog - Offline Usage

This guide explains how to run the OK Binary Star Catalog application on your local computer.  This can be faster than online.

## Running the Application Locally

### 1. Start the Streamlit App
Navigate to the folder you cloned/downloaded the directory and execute
```bash
streamlit run app.py
```

This will:
- Start a local web server (usually on port 8501)
- Open your default web browser automatically
- Display the URL (typically http://localhost:8501)


### 2. Updating the Data

The application includes a data updater that can fetch the latest binary star positions:

```bash
python wds_parser.py
```

This will:
- Process orbital elements and calculate current positions
- Create a new CSV file with today's date
- Keep only the 7 most recent data files

**Note**: The automated GitHub Actions workflow updates data daily online, but for offline use, run the parser manually when you want more up to date orbits.

## Command Line Orbit Plotting

You can generate orbit plots directly from the command line without using the Streamlit interface:

### Basic Usage
First run:
```bash
python wds_parser.py
```
this will generate a csv file with the current positions.  Then run:
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
