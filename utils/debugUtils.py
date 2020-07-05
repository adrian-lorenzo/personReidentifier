verbose = True

def printv(str, endJump=True):
    if verbose:
        if endJump:
            print(str)
        else:
            print(str, end="")

def printDone():
    printv("Done! ✅\n")
