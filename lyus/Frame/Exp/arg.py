# decollator
__all__ = ["ArgTool", "ArgumentParser"]

import re
import os

import argparse


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        # allow the key to contain  ".",like --data.name
        kwargs["fromfile_prefix_chars"] = ""
        super(ArgumentParser, self).__init__(**kwargs)


from collections  import  OrderedDict

class ArgTool(object):
    def __init__(self,base_name, separator=",", connector="-", abbr=False):
        self.base_name=base_name
        self.sepa = separator
        self.con = connector
        self.abbr = abbr
        self._alias_dict = {}  # {alias:key}
        self._param = None

    def init_from_arg(self,arg:OrderedDict):
        if self.abbr:
            for k, v in arg.items():
                self._alias_dict[self.abbreviate(k)] = k

        self._param = arg
        # print(self._alias_dict)
        return self

    def init_from_folder(self, folder):
        assert os.path.exists(folder)

        def read_dicts_from_txt(filename):
            original_dict = {}
            alias_dict = {}
            with open(filename, 'r') as f:
                for line in f:
                    key, alias, value = line.strip().split(' ')
                    original_dict[key] = value
                    alias_dict[alias] = key
            return original_dict, alias_dict

        txt_path = os.path.join(folder, "arg.txt")
        self._param, self._alias_dict = read_dicts_from_txt(txt_path)
        return self

    def create_folder(self, root):
        assert (self._param is not None)
        key_alias = {v: k for k, v in self._alias_dict.items()}
        folder_name = self.get_arg_name()
        folder = os.path.join(root, folder_name)
        assert not os.path.exists(folder), folder
        os.makedirs(folder)

        def save_dicts_to_txt(original_dict, alias_dict, filename):
            with open(filename, 'w') as f:
                for key, value in original_dict.items():
                    alias = key_alias.get(key, key)
                    f.write(f"{key} {alias} {value}\n")

        txt_path = os.path.join(folder, "arg.txt")
        save_dicts_to_txt(self._param, self._alias_dict, txt_path)
        return folder

    def get_arg_name(self):
        assert (self._param is not None)
        key_alias = {v: k for k, v in self._alias_dict.items()}
        segments = [self.base_name]
        for key, val in self._param.items():
            key = key_alias.get(key, key)
            seg = "{}{}{}".format(key, self.con, val)
            segments.append(seg)
        return ",".join(segments)

    def get_value(self, key):
        assert (self._param is not None)
        key = self._alias_dict.get(key, key)
        return self._param[key]

    def __str__(self):
        assert (self._param is not None)
        key_alias = {v: k for k, v in self._alias_dict.items()}
        stri = ""
        for key, value in self._param.items():
            alias = key_alias.get(key, key)
            stri += (f"{key} {alias} {value}\n")
        return stri

    @staticmethod
    def abbreviate(s, connector=""):
        substrings = re.split(r'[^a-zA-Z0-9]+', s)  # 按照非字母数字的字符进行分割
        result = ""

        def abbr_chars(substring):  # 如果子串只包含字母，则保留首位和其他大写字母
            return "".join([substring[0].upper()] + [s for s in substring[1:] if s.isupper()])

        def split_char_digit(s):  # 把一串字符串分割为片段，每个片段全是字母或者全是数字
            fragments = []
            current_fragment = ""
            last_char_type = None

            for char in s:
                if char.isalpha() and last_char_type != "alpha":
                    # Start a new fragment consisting of only letters
                    fragments.append(current_fragment)
                    current_fragment = char
                    last_char_type = "alpha"
                elif char.isdigit() and last_char_type != "digit":
                    # Start a new fragment consisting of only digits
                    fragments.append(current_fragment)
                    current_fragment = char
                    last_char_type = "digit"
                else:
                    # Add the character to the current fragment
                    current_fragment += char

            # Append the final fragment to the list
            fragments.append(current_fragment)

            # Remove any empty fragments from the list
            fragments = [fragment for fragment in fragments if fragment]

            return fragments

        new_string = []
        for substring in substrings:
            if substring.isalpha():
                substring = abbr_chars(substring)
            else:  # 如果子串包含数字或其他字符，则保留数字并将字母全部大写
                segments = split_char_digit(substring)
                segments = [abbr_chars(seg) if seg.isalpha() else seg for seg in segments]
                substring = "".join(segments)
            new_string.append(substring)
        return connector.join(new_string)


import copy

