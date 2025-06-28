import numpy as np
import datetime
from scipy import optimize


def date_to_decimal_year(year, month, day):
    """
    Convert a calendar date to decimal year format.

    Parameters:
    -----------
    year : int
        Calendar year (e.g., 2026)
    month : int
        Month (1-12)
    day : int
        Day of month (1-31)

    Returns:
    --------
    float : Decimal year (e.g., 2026.2342)
    """

    # Create date object for the given date
    given_date = datetime.date(year, month, day)

    # Create date objects for start and end of the year
    year_start = datetime.date(year, 1, 1)
    year_end = datetime.date(year + 1, 1, 1)

    # Calculate fraction of year elapsed
    days_elapsed = (given_date - year_start).days
    total_days_in_year = (year_end - year_start).days
    year_fraction = days_elapsed / total_days_in_year

    return year + year_fraction

def get_current_decimal_year():
    """
    Get the current date and time as a decimal year.

    Returns:
    --------
    float : Current decimal year (e.g., 2025.4567)
    """

    now = datetime.datetime.now()
    year = now.year

    # Start and end of current year
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year + 1, 1, 1)

    # Calculate fraction of year elapsed (including time of day)
    time_elapsed = now - year_start
    total_time_in_year = year_end - year_start
    year_fraction = time_elapsed.total_seconds() / total_time_in_year.total_seconds()

    return year + year_fraction


def solve_kepler_brentq(M, e):
    """
    Solve Kepler's equation M = E - e*sin(E) for E using Brent's method.
    
    Parameters:
    -----------
    M : float or array
        Mean anomaly in radians
    e : float or array  
        Eccentricity
        
    Returns:
    --------
    E : float or array
        Eccentric anomaly in radians
    """
    def kepler_eq(E, M, e):
        return E - e * np.sin(E) - M
    
    # Handle scalar case
    if np.isscalar(M) and np.isscalar(e):
        # Brent's method needs a bracket that contains the root
        # For Kepler's equation, E is always within [M-e, M+e]
        E = optimize.brentq(kepler_eq, M - np.pi, M + np.pi, args=(M, e))
        return E
    
    # Handle array case
    M = np.asarray(M)
    e = np.asarray(e)
    
    # Ensure arrays have the same shape
    M, e = np.broadcast_arrays(M, e)
    
    # Initialize output array
    E = np.zeros_like(M, dtype=float)
    
    # Solve element by element
    for idx in np.ndindex(M.shape):
        E[idx] = optimize.brentq(kepler_eq, 
                                M[idx] - np.pi, 
                                M[idx] + np.pi, 
                                args=(M[idx], e[idx]))
    
    return E


def calculate_binary_position(period, periastron_date, semimajor_axis, eccentricity,
                            inclination, argument_periastron, ascending_node, epoch):
    """
    Calculate the position angle and separation of a binary star system.

    Based on algorithms from Jean Meeus' "Astronomical Algorithms" and the method
    used in Roger Wesson's binary star calculator.

    This function supports both scalar inputs and NumPy arrays.

    Parameters:
    -----------
    period : float or array-like
        Orbital period in years
    periastron_date : float or array-like
        Date of periastron passage in years (decimal)
    semimajor_axis : float or array-like
        Semimajor axis in arcseconds
    eccentricity : float or array-like
        Orbital eccentricity (0 to 1)
    inclination : float or array-like
        Orbital inclination in degrees
    argument_periastron : float or array-like
        Argument of periastron in degrees
    ascending_node : float or array-like
        Position angle of ascending node in degrees
    epoch : float or array-like
        Epoch of observation in years (decimal)

    Returns:
    --------
    dict : Dictionary containing:
        - 'position_angle': Position angle in degrees (0-360)
        - 'separation': Separation in arcseconds
        - 'radius_vector': Radius vector in arcseconds
        - 'true_anomaly': True anomaly in degrees

    Note: If any input is an array, all outputs will be arrays of the same shape.
    """

    # Convert inputs to numpy arrays for vectorized operations
    period = np.asarray(period)
    periastron_date = np.asarray(periastron_date)
    semimajor_axis = np.asarray(semimajor_axis)
    eccentricity = np.asarray(eccentricity)
    inclination = np.asarray(inclination)
    argument_periastron = np.asarray(argument_periastron)
    ascending_node = np.asarray(ascending_node)
    epoch = np.asarray(epoch)

    # Convert angles to radians
    inclination_rad = np.radians(inclination)
    argument_rad = np.radians(argument_periastron)
    ascending_node_rad = np.radians(ascending_node)

    # Calculate mean motion (average annual angular motion in radians)
    n = 2 * np.pi / period

    # Calculate mean anomaly in radians
    M = n * (epoch - periastron_date)
    
    # Normalize M to [-pi, pi] range for better numerical behavior
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi

    # Solve for eccentric anomaly using Brent's method
    E = solve_kepler_brentq(M, eccentricity)

    # Calculate radius vector
    r = semimajor_axis * (1 - eccentricity * np.cos(E))

    # Calculate true anomaly
    v = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(E / 2))

    # Calculate position angle
    pa_rad = ascending_node_rad + np.arctan2(
        np.sin(v + argument_rad) * np.cos(inclination_rad),
        np.cos(v + argument_rad)
    )

    # Calculate separation
    sep = r * np.cos(v + argument_rad) / np.cos(pa_rad - ascending_node_rad)

    # Convert position angle to degrees and normalize to [0, 360)
    pa_deg = np.degrees(pa_rad) % 360
    pa_deg = np.where(pa_deg < 0, pa_deg + 360, pa_deg)

    # Convert back to scalars if all inputs were scalars
    if all(np.isscalar(x) or (hasattr(x, 'ndim') and x.ndim == 0)
           for x in [period, periastron_date, semimajor_axis,
                    eccentricity, inclination, argument_periastron,
                    ascending_node, epoch]):
        pa_deg = float(pa_deg)
        sep = float(np.abs(sep))
        r = float(r)
        v_deg = float(np.degrees(v))
    else:
        sep = np.abs(sep)
        v_deg = np.degrees(v)

    return {
        'position_angle': pa_deg,
        'separation': sep,
        'radius_vector': r,
        'true_anomaly': v_deg
    }


def get_current_epoch():
    """
    Get the current epoch as a decimal year.

    Returns:
    --------
    float : Current year as decimal (e.g., 2024.456)
    """
    import datetime
    now = datetime.datetime.utcnow()
    year = now.year
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year + 1, 1, 1)
    year_fraction = (now - year_start).total_seconds() / (year_end - year_start).total_seconds()
    return year + year_fraction

def compute_single_orbit(period, periastron_date, semimajor_axis, eccentricity,
                        inclination, argument_periastron, ascending_node, n_epochs=100):
    """
    Compute orbital positions for a single binary star system over one full period.

    Parameters:
    -----------
    period : float
        Orbital period in years
    periastron_date : float
        Date of periastron passage in years (decimal)
    semimajor_axis : float
        Semimajor axis in arcseconds
    eccentricity : float
        Orbital eccentricity (0 to 1)
    inclination : float
        Orbital inclination in degrees
    argument_periastron : float
        Argument of periastron in degrees
    ascending_node : float
        Position angle of ascending node in degrees
    n_epochs : int
        Number of epochs across one full orbit (default: 100)

    Returns:
    --------
    dict : Dictionary containing:
        - 'epochs': Array of epoch values (years)
        - 'separations': Array of separations (arcsec)
        - 'position_angles': Array of position angles (degrees)
    """
    # Create epochs spanning one full orbit
    start_epoch = periastron_date
    end_epoch = start_epoch + period
    epochs = np.linspace(start_epoch, end_epoch, n_epochs)

    # Initialize output arrays
    separations = np.zeros(n_epochs)
    position_angles = np.zeros(n_epochs)

    # Compute positions at each epoch
    for i, epoch in enumerate(epochs):
        result = calculate_binary_position(
            period=period,
            periastron_date=periastron_date,
            semimajor_axis=semimajor_axis,
            eccentricity=eccentricity,
            inclination=inclination,
            argument_periastron=argument_periastron,
            ascending_node=ascending_node,
            epoch=epoch
        )
        separations[i] = result['separation']
        position_angles[i] = result['position_angle']

    return {
        'epochs': epochs,
        'separations': separations,
        'position_angles': position_angles
    }

def compute_orbit_ensemble(period_array, periastron_date_array, semimajor_axis_array,
                          eccentricity_array, inclination_array, argument_periastron_array,
                          ascending_node_array, n_epochs=100):
    """
    Compute orbital positions for multiple binary star systems (uncertainty ensemble).
    Fixed version that doesn't interpolate between different orbital periods.

    Parameters:
    -----------
    period_array : array-like
        Array of orbital periods in years
    periastron_date_array : array-like
        Array of periastron dates in years (decimal)
    semimajor_axis_array : array-like
        Array of semimajor axes in arcseconds
    eccentricity_array : array-like
        Array of eccentricities (0 to 1)
    inclination_array : array-like
        Array of inclinations in degrees
    argument_periastron_array : array-like
        Array of arguments of periastron in degrees
    ascending_node_array : array-like
        Array of position angles of ascending node in degrees
    n_epochs : int
        Number of epochs across one full orbit (default: 100)

    Returns:
    --------
    dict : Dictionary containing:
        - 'epochs': Array of epoch values (years) [n_epochs] (based on mean period)
        - 'separations': 2D array of separations (arcsec) [n_samples, n_epochs]
        - 'position_angles': 2D array of position angles (degrees) [n_samples, n_epochs]
    """
    # Convert inputs to arrays
    period_array = np.asarray(period_array)
    periastron_date_array = np.asarray(periastron_date_array)
    semimajor_axis_array = np.asarray(semimajor_axis_array)
    eccentricity_array = np.asarray(eccentricity_array)
    inclination_array = np.asarray(inclination_array)
    argument_periastron_array = np.asarray(argument_periastron_array)
    ascending_node_array = np.asarray(ascending_node_array)

    n_samples = len(period_array)

    # Use mean values for the common epoch grid
    mean_period = np.mean(period_array)
    mean_periastron = np.mean(periastron_date_array)

    # Create epochs spanning one full orbit (based on mean values)
    start_epoch = mean_periastron
    end_epoch = start_epoch + mean_period
    epochs = np.linspace(start_epoch, end_epoch, n_epochs)

    # Initialize output arrays
    separations = np.zeros((n_samples, n_epochs))
    position_angles = np.zeros((n_samples, n_epochs))

    # Compute orbit for each sample - but use phase instead of interpolation
    for i in range(n_samples):
        # For each sample, create epochs that span exactly one orbital period
        sample_start = periastron_date_array[i]
        sample_end = sample_start + period_array[i]
        sample_epochs = np.linspace(sample_start, sample_end, n_epochs)

        # Compute positions at these sample-specific epochs
        for j, epoch in enumerate(sample_epochs):
            result = calculate_binary_position(
                period=period_array[i],
                periastron_date=periastron_date_array[i],
                semimajor_axis=semimajor_axis_array[i],
                eccentricity=eccentricity_array[i],
                inclination=inclination_array[i],
                argument_periastron=argument_periastron_array[i],
                ascending_node=ascending_node_array[i],
                epoch=epoch
            )

            separations[i, j] = result['separation']
            position_angles[i, j] = result['position_angle']

    return {
        'epochs': epochs,  # Reference epochs (mean period)
        'separations': separations,
        'position_angles': position_angles
    }


# Example usage
if __name__ == "__main__":
    # Test case: 99 Herculis (99 her) - single values
    result = calculate_binary_position(
        period=55.91,          # years
        periastron_date=1997.8,  # year of periastron passage
        semimajor_axis=1.05,   # arcseconds
        eccentricity=0.761,    # dimensionless
        inclination=36.1,      # degrees
        argument_periastron=295.0,  # degrees
        ascending_node=223.0,  # degrees
        epoch=2026.0          # observation epoch
    )

    print("99 Herculis Binary Star Calculation for epoch 2026.0:")
    print("=" * 55)
    print(f"Position Angle: {result['position_angle']:.1f}°")
    print(f"Separation: {result['separation']:.3f} arcsec")
    print(f"Radius Vector: {result['radius_vector']:.3f} arcsec")
    print(f"True Anomaly: {result['true_anomaly']:.1f}°")
    print()

    # Example with arrays - multiple epochs
    epochs = 2020

    array_result = calculate_binary_position(
        period=np.random.normal(loc=55.91, scale=0.1, size=100),
        periastron_date=1997.8,
        semimajor_axis=1.05,
        eccentricity=0.761,
        inclination=36.18,
        argument_periastron=295.38,
        ascending_node=223.62,
        epoch=epochs  # This is an array
    )

    print("Array calculation for multiple epochs:")
    print("Epochs:", epochs)
    print("Position Angles:", np.round(array_result['position_angle'], 1))
    print("Separations:", np.round(array_result['separation'], 3))
