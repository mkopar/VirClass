"""Central place for package metadata."""

# NOTE: We use __title__ instead of simply __name__ since the latter would
#       interfere with a global variable __name__ denoting object's name.
__title__ = 'VirClass'
__summary__ = 'Tool for classifying virus samples into virus classes'
__url__ = 'https://github.com/mkopar/VirClass'

# Semantic versioning is used. For more information see:
# https://packaging.python.org/en/latest/distributing/#semantic-versioning-preferred
__version__ = '0.3.0-alpha'

__author__ = 'Matej Kopar'
__email__ = 'matej@kopar.si'

__license__ = 'MIT'
__copyright__ = '2015, ' + __author__

__all__ = (
    '__title__', '__summary__', '__url__', '__version__', '__author__',
    '__email__', '__license__', '__copyright__',
)
