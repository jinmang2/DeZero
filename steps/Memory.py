class obj:
    pass


def f(x):
    print(x)


# Example 1 ==============================================
a = obj() # Assign to variable: Reference Count 1
f(a) # Pass to variable: Reference Count 2 in function
# Complete the function!: Reference Count 1 when existing the function
a = None # Assignment canceled: Reference Count 0


# Example 2 ==============================================
a = obj()
b = obj()
c = obj()

a.b = b
b.c = c

a = b = c = None


# Example 3 ==============================================
# Circular Reference
a = obj()
b = obj()
c = obj()

a.b = b
b.c = c
c.a = a

a = b = c = None
