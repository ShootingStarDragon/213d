
def smart_divide(func):
    print("wtf?", func)
    def inner(*args):
        print("wtf2", *args)
        a = args[0]
        b = args[1]
        print("I am going to divide", a, "and", b)
        if b == 0:
            print("Whoops! cannot divide")
            return

        return func(*args)
    return inner

@smart_divide
def divide(a,b):
    print(a/b)
'''
@ syntax is the same as:
divide = smart_divide(divide)
https://realpython.com/primer-on-python-decorators/#syntactic-sugar
'''

# @smart_divide
# def divide(*args):
#     print(args[0]/args[1])

divide(2,5)


# #reference: https://www.programiz.com/python-programming/decorator
# def cvfunction(func, *args):
#     print("decorator args", func, *args)
#     def inner(*args):
#         print("what are args?", args)
#         return func(args)
#     return inner

# def cvbasic(*args):
#     return "basic return"

# @cvfunction
# cvbasic(3)