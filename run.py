from conehead.conehead import Conehead

settings = {
    'gantryAngle': 0,
    'collimatorAngle': 0,
    'SAD': 100,
    'stepSize': 0.1,  # Stepsize when raytracing effective depth
    'sPri': 1.0,  # Primary source strength (photons/mm^2)
    'softRatio': 0.0025,  # mm^-1
    'softLimit': 20,  # cm
    'eLow': 0.01,  # MeV
    'eHigh': 7.0,  # MeV
    'eNum': 500,  # Spectrum samples
}

calculation = Conehead()
calculation.calculate(settings)
calculation.plot()
