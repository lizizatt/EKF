Dependencies:
numpy (http://www.numpy.org/)
matplotlib (http://matplotlib.org/)
numdifftools (https://pypi.python.org/pypi/Numdifftools)

To run:
Execute ekf.py, which will output three charts in the current configuration; voltage, current, resistance.
The easiest way to modify the input data is to modify the voltmeter/ammeter statewise models in sensors.py, and the matricies in ekf.py used as initialization parameters for the ExtendedKalman object. 