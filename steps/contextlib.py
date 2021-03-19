import contextlib


@contextlib.contextmanager
def config_test():
    print("START")
    try:
        yield
    finally:
        print("DONE")

with config_test():
    print('process...')
