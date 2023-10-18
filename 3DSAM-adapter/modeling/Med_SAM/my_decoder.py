import torch
import torch.nn as nn
from modeling.Med_SAM.text2img_trans import Text2ImageTransformer
from modeling.Med_SAM.foo_copy import VIT_MLAHead_h

class MyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trans = Text2ImageTransformer()
        self.mlahead = VIT_MLAHead_h(img_size=96, num_classes=2)
        self.text_embedding = None
        self.load_text("txt_encoding.pth")

    def load_text(self, filepath):
        word_embedding = torch.load(filepath)
        self.text_embedding = word_embedding.float()
    
    def forward(self, feature_list, image_feature):
        print(self.text_embedding.size())
        feature_list[-1] = self.trans(feature_list[-1], self.text_embedding)
        feature_list.append(image_feature)
        x_out = self.mlahead(feature_list, 2, 128//32)
        return x_out