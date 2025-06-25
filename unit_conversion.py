def ugm3_to_ppb(ug_per_m3, gas, temperature_c=25):
    """
    Converts Gas concentration from ug_per_m3 to ppb

    Args:
        ppb: Concentration in parts per billion.
        Gas: which gas we are attempting to convert
        temperature_c: Temperature in Celsius (default is 25°C).

    Returns:
        Gas concentration in ppb
    """

    #https://www.engineeringtoolbox.com/molecular-weight-gas-vapor-d_1156.html
    mol_weights = {
        'NO2' : 44.013,
        'O3' : 47.998,
        'NO' : 30.006,
        'CO' : 28.011,
        'CO2' : 44.01
    }

    M = mol_weights[gas]
    temperature_k = temperature_c + 273.15

    ppb = temperature_k*ug_per_m3 / (12.187*M)
    return ppb