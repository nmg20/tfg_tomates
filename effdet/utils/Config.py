import configparser
file = "model.config"
config = configparser.ConfigParser()
config.read(file)
config.sections()

"""
Una config para cada dataset (?).
Also para cada modelo.
"""

def get_params(section):
	if section not in config.sections():
		print(f"No section {section} on {file}.\n")
	else:
		d = {}
		for opt in cf.options(section):
			r[opt]=cf.get(section,opt)
		return r


def __init__():
	config = configparser.ConfigParser()
	config.read_file(file)
