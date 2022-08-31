import pickle


def save_model(model, name, model_type=""):
    with open("trained_models/" + model_type + name, "wb") as file:
        pickle.dump(model, file)


def load_model(name, model_type):
    with open("trained_models/" + model_type + name, "rb") as file:
        model = pickle.load(file)
    return model
