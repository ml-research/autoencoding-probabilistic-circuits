from enum import Enum


class PcEncoderType(str, Enum):
    EINET = "einet"
    EINET_ACT = "einet_act"
    EINET_CAT = "einet_cat"
    CONV_PC = "conv_pc"
    CONV_PC_SPNAE_ACT = "conv_pc_spnae_act"
    CONV_PC_SPNAE_CAT = "conv_pc_spnae_cat"

    def __str__(self):
        return self.value

class ModelName(str, Enum):
    VAE = "vae"
    AE = "ae"
    APC = "apc"
    VAEM = "vaem"
    MIWAE = "miwae"
    MICE = "mice"
    MISSFOREST = "missforest"
    HIVAE = "hivae"

    def __str__(self):
        return self.value

class DecoderType(str, Enum):
    NN = "nn"
    PC = "pc"

    def __str__(self):
        return self.value
