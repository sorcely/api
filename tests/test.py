# This runs all tests
# Mainly used for GitHub workflows

import unittest

from test_answer_engine import *
from test_data_engine import *
from test_run import *
from test_server import *

if __name__ == '__main__':
    unittest.main()
