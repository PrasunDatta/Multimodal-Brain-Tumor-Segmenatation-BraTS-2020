
config = dict()

config["base_path"] = "/content/gdrive/MyDrive/PrasunFolder/BraTS2020/2Stage_VAE/Dataset/BraTS2020_TrainingData"
# config["base_path"] = "/content/gdrive/MyDrive/2Stage_VAE/Dataset/BraTS2020_TrainingData"
config["initial_learning_rate"] = 1e-4
# config["input_/shape"] = (1,4,128, 192, 160)
config["normalizaiton"] = "group_normalization"
config["mode"] = "trilinear"
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["loss_k1_weight"] = 0.1
config["loss_k2_weight"] = 0.1
config["focal_alpha"] = 0.99
config["focal_gamma"] = 2
# config["data_path"] = config["base_path"] + "data/MICCAI_BraTS_2018_Data_Training"
config["data_path"] = config["base_path"] + "/MICCAI_BraTS2020_TrainingData"
config["training_patients"] = []
config["validation_patients"] = []
config["activation"]= "relu"
#Transformer related

config["img_dim"] = 128
config["embedding_dim"] = 512
config["num_heads"] = 8
config["patch_dim"] = 8
config["num_channels"] = 4
config["dropout_rate"] = 0.1
config["attn_dropout_rate"] = 0.1
config["num_layers"] =4
config["hidden_dim"]= 7680
config["positional_encoding_type"] = "learned"

# augmentation
config["intensity_shift"] = True
config["scale"] = True
config["flip"] = True
config["L2_norm"] = 1e-5
config["patience"] = 5
config["lr_decay"] = 0.7
config["checkpoint"] = True  # Boolean. If True, will save the best model as checkpoint.
config["label_containing"] = True  # Boolean. If True, will generate label with overlapping.
config["VAE_enable"] = True  # Boolean. If True, will enable the VAE module.
config["focal_enable"] = False  # Boolean. If True, will enable the focal loss.
if config["focal_enable"]:
    config["initial_learning_rate"] *= 2




####Attempt

# config = dict()

# # config["base_path"] = "/content/gdrive/MyDrive/PrasunFolder/BraTS2020/2Stage_VAE/Dataset/BraTS2020_TrainingData"
# # config["base_path"] 
# config["root"]= "/content/gdrive/MyDrive/2Stage_VAE/Dataset/BraTS2020_TrainingData"
# config["train_dir"] = config['root']] + "/MICCAI_BraTS2020_TrainingData"
# config["train_file"] = 'train_list.txt'
# config["valid_file"] = 'valid_list.txt'
# config["initial_learning_rate"] = 2e-4
# # config["input_/shape"] = (1,4,128, 192, 160)
# config["normalizaiton"] = "group_normalization"
# # config["mode"] = "trilinear"
# config["mode"] = 'train'

# config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
# config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
# config["nb_channels"] = len(config["training_modalities"])
# config["loss_k1_weight"] = 0.1
# config["loss_k2_weight"] = 0.1
# config["focal_alpha"] = 0.99
# config["focal_gamma"] = 2
# # config["data_path"] = config["base_path"] + "data/MICCAI_BraTS_2018_Data_Training"
# # config["data_path"]

# config["training_patients"] = []
# config["validation_patients"] = []
# config["activation"]= "relu"
# #Transformer related

# config["img_dim"] = 128
# config["embedding_dim"] = 512
# config["num_heads"] = 8
# config["patch_dim"] = 8
# config["num_channels"] = 4
# config["dropout_rate"] = 0.1
# config["attn_dropout_rate"] = 0.1
# config["num_layers"] =4
# config["hidden_dim"]= 7680
# config["positional_encoding_type"] = "learned"

# # augmentation
# config["intensity_shift"] = True
# config["scale"] = True
# config["flip"] = True
# config["L2_norm"] = 1e-5
# config["patience"] = 5
# config["lr_decay"] = 0.7
# config["checkpoint"] = True  # Boolean. If True, will save the best model as checkpoint.
# config["label_containing"] = True  # Boolean. If True, will generate label with overlapping.
# config["VAE_enable"] = True  # Boolean. If True, will enable the VAE module.
# config["focal_enable"] = False  # Boolean. If True, will enable the focal loss.
# if config["focal_enable"]:
#     config["initial_learning_rate"] *= 2



