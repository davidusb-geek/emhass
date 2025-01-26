import os
from distutils.util import strtobool

bind = f"{os.getenv('IP', '0.0.0.0')}:{os.getenv('PORT', '5000')}"

workers = int(os.getenv("WEB_CONCURRENCY", 1))
threads = int(os.getenv("PYTHON_MAX_THREADS", 8))

reload = bool(strtobool(os.getenv("WEB_RELOAD", "true")))

timeout = int(os.getenv("WEB_TIMEOUT", 240))
