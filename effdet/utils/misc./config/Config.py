import configparser
file = "model.config"
config = configparser.ConfigParser()
config.read(file)
config.sections()