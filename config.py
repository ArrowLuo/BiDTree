import os

class Config(object):
    def __init__(self):
        self.current_path = "."
        self.data_sets = "laptops_2014"

        # setting
        self.nepochs = 100
        self.dropout = 0.4
        self.batch_size = 20
        self.lr = 0.001
        self.lr_decay = 0.9
        self.nepoch_no_imprv = 10
        self.show_test_results = False

        self.dim = 300
        self.dim_char = 100
        self.dim_rel = 300
        self.hidden_size = 300
        self.char_hidden_size = 100

        # default
        self.lowercase = True
        self.train_embeddings = True
        self.crf = True
        self.chars = True

        # auto setting
        self.test_filename = ""
        self.dev_filename = ""
        self.train_filename = ""

        self.words_filename = ""
        self.tags_filename = ""
        self.chars_filename = ""
        self.relations_filename = ""

        self.test_deps_filename = ""
        self.dev_deps_filename = ""
        self.train_deps_filename = ""

        self.trimmed_filename = ""

        self.output_path = ""
        self.model_output = ""
        self.log_path = ""

        # derivative variable
        self.ntags = 0
        self.nwords = 0
        self.nchars = 0
        self.nrels = 0
        self.max_sentence_size = 0
        self.max_word_size = 0
        self.max_btup_deps_len = 0
        self.max_upbt_deps_len = 0

        # train for default, this value will update in model
        self.istrain = True
    def auto_config(self):

        data_sets_name = self.data_sets.split("_")[0]
        assert data_sets_name in ['laptops', 'restaurants']

        if data_sets_name == "laptops":
            self.glove_filename = "{}/data/amazon/amazon_reviews_small.{}d.txt".format(self.current_path, self.dim)
        elif data_sets_name == "restaurants":
            self.glove_filename = "{}/data/yelp/yelp_reviews_small.{}d.txt".format(self.current_path, self.dim)
        else:
            raise ValueError("{} doesn't exsits.".format(data_sets_name))

        model_data_path = "{}/data/model_data".format(self.current_path)
        if data_sets_name == "laptops":
            self.trimmed_filename = "{}/amazon_reviews.{}.{}d.trimmed.npz".format(model_data_path, self.data_sets, self.dim)
        elif data_sets_name == "restaurants":
            self.trimmed_filename = "{}/yelp_reviews.{}.{}d.trimmed.npz".format(model_data_path, self.data_sets, self.dim)

        self.words_filename = "{}/words_{}.txt".format(model_data_path, self.data_sets)
        self.tags_filename = "{}/tags_{}.txt".format(model_data_path, self.data_sets)
        self.chars_filename = "{}/chars_{}.txt".format(model_data_path, self.data_sets)
        self.relations_filename = "{}/relations_{}.txt".format(model_data_path, self.data_sets)

        self.test_filename = "{}/data/{}/{}_test.gold.txt".format(self.current_path, data_sets_name, self.data_sets)
        self.dev_filename = "{}/data/{}/{}_trial.txt".format(self.current_path, data_sets_name, self.data_sets)
        self.train_filename = "{}/data/{}/{}_train.txt".format(self.current_path, data_sets_name, self.data_sets)

        # dependency tree
        self.test_deps_filename = "{}/data/{}/{}_test.gold.deps".format(self.current_path, data_sets_name, self.data_sets)
        self.dev_deps_filename = "{}/data/{}/{}_trial.deps".format(self.current_path, data_sets_name, self.data_sets)
        self.train_deps_filename = "{}/data/{}/{}_train.deps".format(self.current_path, data_sets_name, self.data_sets)

        output_root = "{}/results".format(self.current_path)
        self.output_path = "{}/{}/".format(output_root, self.data_sets)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"

        if os.path.exists(model_data_path) is False:
            os.mkdir(model_data_path)
        if os.path.exists(output_root) is False:
            os.mkdir(output_root)
        if os.path.exists(self.output_path) is False:
            os.mkdir(self.output_path)
        if os.path.exists(self.model_output) is False:
            os.mkdir(self.model_output)
