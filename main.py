from model1.default import get_config
from model1.model import Model
from utli import load_data, get_session, update_para
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_units", type=int, default=100)

    parser.add_argument("--pi", type=float, default=0.25)
    parser.add_argument("--mu1", type=float, default=0.0)
    parser.add_argument("--std1", type=float, default=0.5)
    parser.add_argument("--mu2", type=float, default=0.0)
    parser.add_argument("--std2", type=float, default=1.5)

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default="logs/model1/")

    return parser.parse_args()


def main(args):
    sess = get_session()
    default_para = get_config()
    para = update_para(default_para, args)
    model = Model(sess, para)

    x_train, x_test, y_train, y_test = load_data()

    x_train_ = x_train[:-5000]
    y_train_ = y_train[:-5000]
    x_valid = x_train[-5000:]
    y_valid = y_train[-5000:]
    if args.train:
        model.train(x_train_, y_train_, x_valid, y_valid)
    else:
        model.load(args.load_path)
    model.test(x_test, y_test)

if __name__ == "__main__":
    args = arg_parse()
    main(args)
