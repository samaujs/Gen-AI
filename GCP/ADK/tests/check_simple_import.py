import sys
print("sys.path:", sys.path)
try:
    import simple
    print("Imported simple from:", simple.__file__)
except Exception as e:
    print("Could not import simple:", e)
