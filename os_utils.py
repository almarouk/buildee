import os
import sys


def redirect_output_to_null():
    """
    Redirects the system's stdout and stderr to null to suppress console output.
    Returns the original file descriptors for restoration later.
    """
    # Open the null device
    devnull = os.open(os.devnull, os.O_RDWR)

    # Get the file descriptors for stdout and stderr
    stdout_fileno = sys.stdout.fileno()
    stderr_fileno = sys.stderr.fileno()

    # Save the original file descriptors
    original_stdout = os.dup(stdout_fileno)
    original_stderr = os.dup(stderr_fileno)

    # Redirect stdout and stderr to devnull
    os.dup2(devnull, stdout_fileno)
    os.dup2(devnull, stderr_fileno)

    # Return devnull and the original file descriptors for later restoration
    return devnull, original_stdout, original_stderr


def restore_output(devnull, original_stdout, original_stderr):
    """
    Restores the system's stdout and stderr to their original file descriptors.
    Closes any temporary file descriptors used during the redirection.
    """
    # Get the file descriptors for stdout and stderr
    stdout_fileno = sys.stdout.fileno()
    stderr_fileno = sys.stderr.fileno()

    # Restore the original stdout and stderr
    os.dup2(original_stdout, stdout_fileno)
    os.dup2(original_stderr, stderr_fileno)

    # Close the duplicated file descriptors and the null device
    os.close(original_stdout)
    os.close(original_stderr)
    os.close(devnull)
