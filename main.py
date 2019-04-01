from model import Model
from utli import load_data
from default import get_config


def main():
    x_train, x_test, y_train, y_test = load_data()
    para = get_config()
    model = Model(**para)
    model.train(x_train, y_train)

if __name__ == "__main__":
    main()
