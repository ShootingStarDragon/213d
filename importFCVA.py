import FastCVApp

app = FastCVApp.FCVA()

def my_cv_function(inputframe):
    return inputframe
app.appliedcv = my_cv_function

if __name__ == '__main__' :
    app.run()
