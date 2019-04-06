import json
import time
import os
import pandas as pd
import pickle


class Logger:
    def __init__(self, save_path=None, file_type="json"):
        assert file_type in ["json", "csv"]
        self.file_type = file_type

        if save_path is None:
            save_path = "logs/"
        self.save_path = os.path.join(save_path, time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.content = dict()

    def write(self, content, content_type, verbose=True):
        assert isinstance(content, dict)
        if verbose:
            memory_str = ""
            for key, value in content.items():
                memory_str += "{}:{}|".format(key, value)
            print(memory_str)

        if content_type not in self.content:
            self.content[content_type] = dict()
        # create dict_list
        for key in content:
            if key not in self.content[content_type]:
                self.content[content_type][key] = []
            else:
                self.content[content_type][key].append(content[key])

    def dump(self):
        if self.file_type == "json":
            for content_type in self.content:
                file = open(os.path.join(self.save_path, "{}.json".format(content_type)), "wb")
                json.dump(self.content[content_type], file)
                file.close()
        else:
            for content_type in self.content:
                if content_type == "stats":
                    file = open(os.path.join(self.save_path, "{}.pkl".format(content_type)), "wb")
                    pickle.dump(self.content[content_type], file)
                    file.close()
                else:
                    file = open(os.path.join(self.save_path, "{}.csv".format(content_type)), "w")
                    df = pd.DataFrame(self.content[content_type])
                    df.to_csv(file)
