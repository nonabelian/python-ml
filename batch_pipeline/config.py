import logging


VERSION = '0.0.1'
PROJECT_NAME = 'batch_pipeline' + '-' + VERSION


class Config(object):
    def __init__(self):
        self.DEBUG = True
        self.TESTING = False
        self.LOGGING_LEVEL = logging.DEBUG
        self.FORMAT_STRING = '%(asctime)s {}: '.format(PROJECT_NAME)\
                             + '%(name)-12s %(levelname)-8s %(message)s'
        self.LOGGING_CONFIG = {
            'version': 1,
            'formatters': {'f': {'format': self.FORMAT_STRING} },
            'handlers': {'h': {'class': 'logging.StreamHandler',
                               'formatter': 'f',
                               'level': self.LOGGING_LEVEL}
                        },
            'root': {'handlers': ['h'],
                     'level': self.LOGGING_LEVEL
                    }
        }


class ProductionConfig(Config):
    def __init__(self):
        super().__init__()

        self.DEBUG = False
        self.TESTING = False
        self.LOGGING_LEVEL = logging.WARNING
        self.LOGGING_CONFIG['handlers']['h']['level'] = self.LOGGING_LEVEL
        self.LOGGING_CONFIG['root']['level'] = self.LOGGING_LEVEL


class TestingConfig(Config):
    def __init__(self):
        super().__init__()

        self.DEBUG = True
        self.TESTING = True
        self.LOGGING_LEVEL = logging.DEBUG
        self.LOGGING_CONFIG['handlers']['h']['level'] = self.LOGGING_LEVEL
        self.LOGGING_CONFIG['root']['level'] = self.LOGGING_LEVEL
