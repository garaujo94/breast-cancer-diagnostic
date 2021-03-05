import os.path


def check_if_model_exists():
    if os.path.isfile('models/trained_model.sav'):
        print ("File exist")
        return True
    else:
        print ("File not exist")
        return False