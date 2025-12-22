import time

'''
run 
with Timer("name"):
    code
'''

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
    def __exit__(self, *args):
        self.end = time.perf_counter()
        print(f"[{self.name}] executed in {self.end - self.start:.4f}s")