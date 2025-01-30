import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_output(verbose: bool = False):
    """Context manager that redirects stdout and stderr to null to suppress console output.

    :param verbose: If False, suppresses output. If True, allows output.
    """
    if verbose:
        yield
        return

    devnull = os.open(os.devnull, os.O_RDWR)
    stdout_fileno = sys.stdout.fileno()
    stderr_fileno = sys.stderr.fileno()

    original_stdout = os.dup(stdout_fileno)
    original_stderr = os.dup(stderr_fileno)

    try:
        os.dup2(devnull, stdout_fileno)
        os.dup2(devnull, stderr_fileno)
        yield
    finally:
        os.dup2(original_stdout, stdout_fileno)
        os.dup2(original_stderr, stderr_fileno)
        os.close(original_stdout)
        os.close(original_stderr)
        os.close(devnull)
