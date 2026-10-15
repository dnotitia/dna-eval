# The environment SciCode's tests run in: Python 3.11 and the numerical packages the benchmark's
# solutions and tests import (SciCode's README names Python 3.11; its package list is unpinned,
# so the versions below are those of the benchmark's era, because model code calls functions
# that later releases removed, such as scipy.integrate.simps and np.float_).
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 sympy==1.13.1 h5py==3.11.0 matplotlib==3.8.4
ENV MPLBACKEND=Agg
WORKDIR /work
