from sklearn.preprocessing import normalize
from torchvision import transforms

import modules.embeddingGenerators.alignedreid.models as models
from modules.embeddingGenerators.alignedreid.util.FeatureExtractor import FeatureExtractor
from modules.embeddingGenerators.alignedreid.util.utils import *
from modules.embeddingGenerators.bodyEmbeddingGenerator import BodyEmbeddingGenerator


def pool2d(tensor, type='max'):
    sz = tensor.size()
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(int(sz[2] / 8), sz[3]))
    elif type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(int(sz[2] / 8), sz[3]))
    else:
        return None
    x = x[0].cpu().data.numpy()
    x = np.transpose(x, (2, 1, 0))[0]
    return x


class AlignedReIdEmbeddingGenerator(BodyEmbeddingGenerator):

    def __init__(self, weights="../pretrained_models/alignedreid/checkpoint_ep300.pth.tar", classes=1041):
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.use_gpu = torch.cuda.is_available()
        self.model = models.init_model(name='resnet50', num_classes=classes, loss={'softmax', 'metric'},
                                       use_gpu=self.use_gpu, aligned=True)

        checkpoint = torch.load(weights, encoding='latin-1', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.use_gpu:
            self.model = self.model.cuda()

        self.exact_list = ['7']
        self.featureExtractor = FeatureExtractor(self.model, self.exact_list)

    def getEmbedding(self, bodyImage):
        image = img_to_tensor(Image.fromarray(bodyImage).convert('RGB'), self.img_transform)
        if self.use_gpu:
            image = image.cuda()
        self.model.eval()
        f1 = self.featureExtractor(image)
        return (normalize(pool2d(f1[0], type='max'))).flatten()
